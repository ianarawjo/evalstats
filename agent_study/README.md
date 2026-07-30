# Agent Study v0

Minimal harness measuring whether access to `evalstats` changes a coding
agent's deployment recommendation on a prompt A/B comparison, under three
statistical regimes with analytically known correct answers. See
`../DESIGN.md`-adjacent plan discussion for the full rationale; this README
just covers running it.

## One-time setup

1. Three long-lived venvs at the repo root (already created):
   - `.agent-study-venv-full` -- evalstats + numpy/scipy/pandas/statsmodels.
     This is the Python environment the `claude_agent_sdk` backend's Bash
     tool uses in the "full" condition.
   - `.agent-study-venv-baseline` -- numpy/scipy/pandas/statsmodels only,
     evalstats absent. Used in the "baseline" condition.
   - `.agent-study-venv-runner` -- the orchestrator's own environment
     (claude-agent-sdk, openhands-sdk, evalstats, mcp, pandas). Run
     everything below with this venv's Python. **Must be Python 3.12, not
     whatever `python3` defaults to** -- `openhands-sdk`'s `litellm`
     dependency has a Rust extension (via PyO3) that doesn't build on 3.14
     yet. Built via `uv venv --python 3.12 .agent-study-venv-runner`, not
     plain `python3 -m venv`.

2. Copy `.env.example` to `.env` and fill in an API key:

   ```
   cp agent_study/.env.example agent_study/.env
   # edit agent_study/.env, set ANTHROPIC_API_KEY=...
   ```

   `.env` is gitignored. The harness spawns a separate `claude` CLI
   subprocess per agent run, which needs its own auth -- it does not
   inherit an interactive Claude Code session's login.

## Backends

Two interchangeable agent harness backends, selected via `--backend`
(default `claude_agent_sdk`). Both implement the same contract and produce
the same archived artifacts -- `results.csv` gets a `backend` column so a
batch can mix both for comparison.

### `claude_agent_sdk` (default)

The original backend -- `run_agent_claude_agent_sdk.py`. Fully validated
(isolation-hardened, both scenario families, the MCP evalstats tool). See
Isolation below for how it's sandboxed. Setup: steps 1-2 above.

### `openhands`

`run_agent_openhands.py` -- built for lower per-run cost and open/local
model support (the original motivation: see the harness-comparison
discussion in conversation history). Uses `openhands.sdk`'s own Python API
(`Agent`/`Conversation`/`DockerWorkspace`) directly, **not** the `openhands`
CLI's `--headless` mode -- that path runs completely unsandboxed on the host
by default (confirmed by reading `openhands_cli/setup.py`: it constructs a
bare `Workspace(working_dir=os.getcwd())`, no container at all), which would
have reintroduced the exact leak this project already fixed once. Real
isolation only comes from `DockerWorkspace`, which is Python-API-only.

**Live-validated** (2026-07-12): one full `clear_improvement`/`full` episode
via Claude Haiku -- correct decision, `compare_prompts` MCP tool called
successfully from inside the container, cost $0.04. Getting there needed 4
real fixes past what source-reading alone predicted (all against
`openhands-sdk` 1.35.0, whose own docstrings/examples were stale relative to
that installed version) -- see `run_agent_openhands.py`'s module docstring
for the specifics (platform detection, `mcp_config`'s actual shape, tool
registration names, and where the MCP server process actually has to run).
Worth internalizing before extending this backend: source reading got the
shape right, but several exact names/formats only surfaced by actually
running it.

Setup:
1. Install the CLI (used only to source the SDK packages' pinned, mutually
   compatible versions -- see below) and Docker Desktop:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv tool install openhands --python 3.12
   ```
   Docker Desktop: install separately (`brew install --cask docker`, then
   launch it once yourself -- first launch needs your password for its
   privileged networking helper, which can't be scripted).
2. Install the matching SDK packages into `.agent-study-venv-runner`
   (plain `pip install openhands-sdk` resolves an incompatible version
   combination -- use `uv pip install`, which resolves the same
   coordinated set the CLI install above uses). Pinned to what was actually
   validated (1.35.0) rather than left floating, since an unpinned install
   is exactly what produced the stale-docstring surprises above in the
   first place:
   ```
   uv pip install --python .agent-study-venv-runner/bin/python \
     "openhands-sdk==1.35.0" "openhands-workspace==1.35.0" "openhands-tools==1.35.0"
   ```
3. Build the two condition images (from the repo root, so the Dockerfiles'
   COPY paths resolve):
   ```
   docker build -f agent_study/docker/Dockerfile.full -t agent-study-openhands-full:latest .
   docker build -f agent_study/docker/Dockerfile.baseline -t agent-study-openhands-baseline:latest .
   ```
4. Same `agent_study/.env` as the `claude_agent_sdk` backend --
   `ANTHROPIC_API_KEY` is reused (Claude-via-OpenHands for this first
   validation pass, not a cheap/open model yet, so the harness itself gets
   validated independent of a model swap).
5. Run with `--backend openhands`.

Isolation model differs structurally from `claude_agent_sdk`'s: Docker's own
filesystem namespace is the boundary (nothing outside `/workspace` is even
mounted into the container) rather than a hook that can deny and log an
attempt, so `n_containment_violations` is `None` (not 0) for this backend --
"not applicable," not "zero attempts."

## Running

Sanity-check data generation only (no API calls, no cost):

```
.venv/bin/python -m agent_study.scenarios
```

Build all 6 workspaces without calling the agent:

```
.agent-study-venv-runner/bin/python -m agent_study.runner --dry-run
```

Run the full v0 batch (3 phenomena x {full, baseline} = 6 real agent
episodes, each capped at $1 / 40 turns):

```
.agent-study-venv-runner/bin/python -m agent_study.runner
```

Run a subset, e.g. just the null case:

```
.agent-study-venv-runner/bin/python -m agent_study.runner --scenarios true_null
```

Output lands in `agent_study/runs/<timestamp>/`:
- `<scenario>__<condition>/` -- an *archived copy* of the agent's workspace
  (data.csv, PROMPT.md, transcript.txt, transcript.jsonl, recommendation.json,
  run_meta.json, containment_violations.json if any). Copied here only after
  the run finishes -- see Isolation below for where the agent actually ran.
- `_ground_truth/<scenario>/ground_truth.json` -- true effect, statistical
  power, and correct decision(s), kept outside every agent workspace
- `results.csv`, `manifest.json` -- aggregated scoring

## Isolation

A real run leaked: the agent read one of this machine's Claude Code memory
files via `cat ~/.claude/.../key_facts.md` (which names evalstats and its
venv), then `cd`-ed into the actual repo and invoked a different venv
(`.pkgtest-venv`) that has evalstats installed, bypassing the "baseline"
condition entirely. Separately, the original workspace path was nested
inside this repo and named after the scenario/condition (e.g.
`.../prompt-stats/agent_study/runs/.../borderline_improvement__baseline`),
which may itself have shaped behavior by hinting at both the domain
(statistics) and the experimental condition.

Two independent fixes, both in `isolation.py` / `workspace.py`:

1. **Blinded, external workspace.** The agent's actual cwd is
   `$TMPDIR/agent_study_workspaces/<random-hex>` -- never inside this repo,
   never named after the scenario or condition. The real-named
   `agent_study/runs/<timestamp>/<scenario>__<condition>/` directory is a
   harness-side archive, populated by copying artifacts out *after* the run
   completes; the agent never sees that path.
2. **PreToolUse hook (`isolation.py`).** Denies any Read/Write/Bash call
   that touches a path under the user's home directory other than the run's
   own workspace or its assigned condition venv -- this is what actually
   blocks the `.claude` memory read and the wrong-venv invocation, including
   via `cd` into a forbidden directory followed by a relative reference.
   Denied attempts are recorded (not just silently blocked) in
   `run_meta.json`'s `n_containment_violations` and
   `containment_violations.json`, so a run that tried to escape is visible
   in the results rather than indistinguishable from a clean one. Paths
   outside the home directory (e.g. `/usr`, `/opt/homebrew`, `/System` --
   what the Python interpreter itself needs) are left alone.

This is a best-effort mitigation for *incidental* exploration (an agent
naturally poking around, which is what was observed), not a formally
verified sandbox against a maximally adversarial agent deliberately
obfuscating a path (e.g. constructing it from concatenated string parts at
runtime rather than a literal shell token). The SDK's own `sandbox` option
(also enabled, in `run_agent_claude_agent_sdk.py`) separately blocks network
access for Bash-spawned processes. This is specific to the `claude_agent_sdk`
backend -- the `openhands` backend (see Backends above) gets its isolation
from a real Docker container boundary instead, a structurally stronger
(adversarial-proof, not just incidental-exploration-proof) guarantee, at
the cost of that backend being newer and not yet live-validated.

## What's in scope for v0 (and what isn't)

One scenario family (paired prompt A/B comparison, continuous scores),
three phenomena (true null, clear improvement, borderline improvement),
one factor (evalstats on/off), implicit prompt only, decision-correctness
scoring only, no reps per cell. Explicit-prompt variant, other scenario
families (LLM judge/PPI, ordinal, binary, multi-arm), multiple LLMs, and
statistical-power reps are deliberately deferred until this loop is
validated.
