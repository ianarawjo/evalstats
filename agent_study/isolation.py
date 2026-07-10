"""Filesystem isolation for agent runs.

Context: a real run leaked in two ways. (1) The agent ran `cat
~/.claude/projects/.../memory/key_facts.md` via Bash and successfully read a
Claude Code memory file that names evalstats and its venv -- in the
"baseline" (no-evalstats) condition. (2) It then `cd`-ed into the actual repo
and invoked `.pkgtest-venv/bin/python` (a venv with evalstats installed),
bypassing the "baseline" venv entirely. Separately, the workspace path itself
(nested inside the `prompt-stats` repo, named e.g. `borderline_improvement__
baseline`) may have hinted at both the domain (statistics) and the
experimental condition.

The SDK's `sandbox` option (see run_agent.py) only restricts network access
for Bash-spawned processes -- it does not restrict filesystem reads, by
design (ordinary coding tasks need broad read access). This module adds a
second, independent layer:

1. workspace.py places each run's workspace under the system temp directory
   with an opaque random name, never inside this repo, so there's nothing
   experiment-relevant nearby for incidental `ls`/`cat` exploration to find,
   and no directory name that hints at the scenario or condition.
2. This module's PreToolUse hook denies any Read/Write/Bash tool call that
   touches a path under the user's home directory other than the run's own
   workspace or its assigned condition venv -- so even a deliberate absolute
   path reference (like the `.claude` memory read that actually happened)
   is blocked. Paths outside the home directory (system libraries the Python
   interpreter needs, e.g. under /usr, /opt/homebrew, /System) are left
   alone, since those aren't sensitive to this study and enumerating every
   legitimate one would be fragile.

Caveat: the Bash check is a best-effort scan for path-like tokens in the
command string, not a formal sandbox -- it catches the incidental-exploration
threat model this was built for (an agent naturally poking around, as
observed), not a maximally adversarial agent deliberately obfuscating a path
to escape it. The Read/Write check is exact (single structured path argument,
no shell parsing involved).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

HOME_DIR = Path.home().resolve()


def _resolve(path_str: str, cwd: Path) -> Path | None:
    try:
        expanded = os.path.expandvars(os.path.expanduser(path_str))
        p = Path(expanded)
        if not p.is_absolute():
            p = cwd / p
        return p.resolve()
    except (OSError, ValueError):
        return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _path_allowed(path: Path, allowed_roots: list[Path]) -> bool:
    if not _is_within(path, HOME_DIR):
        return True  # system/language-runtime paths outside the home dir: not sensitive here.
    return any(_is_within(path, root) for root in allowed_roots)


# Path-like tokens in a shell command: anything containing a "/". A bare,
# slash-less "~" is deliberately NOT matched here -- it's indistinguishable
# by regex alone from a "~" used as ordinary text inside a heredoc/string
# (e.g. `print(f"p ~ {x}")` for "approximately", which a real run hit and
# got wrongly denied). "~/foo" still matches (has a "/"); bare "~" as an
# actual cd target is still caught separately by _CD_RE below.
_PATH_TOKEN_RE = re.compile(r"[^\s'\"|&;()<>]*/[^\s'\"|&;()<>]*")

# `cd`/`pushd` to a forbidden directory invalidates the "resolve relative
# paths against the original cwd" assumption for everything after it in the
# same command -- e.g. `cd <repo> && .pkgtest-venv/bin/python x.py` slips
# past a plain token scan because the venv reference is relative and only
# resolves to something forbidden once the preceding `cd` is accounted for.
# So any `cd`/`pushd` to a forbidden target denies the whole command outright
# rather than trying to track cwd state through the rest of the string.
_CD_RE = re.compile(r"(?:^|&&|\|\||;|\n)\s*(?:cd|pushd)\s+(\S+)")


def build_pre_tool_use_hook(workspace_dir: Path, venv_dir: Path, violations: list[dict]):
    """Returns a PreToolUse hook denying Read/Write/Bash access to anything
    under the home directory except workspace_dir and venv_dir. Appends a
    record to `violations` (caller-owned list) every time it denies
    something, so a run that tried to escape its workspace -- even if
    ultimately blocked -- is visible in that run's results rather than
    silently indistinguishable from a run that never tried."""
    allowed_roots = [workspace_dir.resolve(), venv_dir.resolve()]

    async def hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict:
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input") or {}
        cwd = workspace_dir

        violation: str | None = None
        if tool_name in ("Read", "Write", "Edit"):
            file_path = tool_input.get("file_path")
            if isinstance(file_path, str):
                resolved = _resolve(file_path, cwd)
                if resolved is not None and not _path_allowed(resolved, allowed_roots):
                    violation = file_path
        elif tool_name == "Bash":
            command = tool_input.get("command")
            if isinstance(command, str):
                for cd_target in _CD_RE.findall(command):
                    resolved = _resolve(cd_target, cwd)
                    if resolved is not None and not _path_allowed(resolved, allowed_roots):
                        violation = f"cd {cd_target}"
                        break
                if violation is None:
                    for token in _PATH_TOKEN_RE.findall(command):
                        resolved = _resolve(token, cwd)
                        if resolved is not None and not _path_allowed(resolved, allowed_roots):
                            violation = token
                            break

        if violation is not None:
            violations.append({"tool_name": tool_name, "path": violation, "tool_input": tool_input})
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Path outside the assigned workspace/venv is not allowed: {violation!r}"
                    ),
                }
            }
        return {}

    return hook
