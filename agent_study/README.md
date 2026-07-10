# Agent Study v0

Minimal harness measuring whether access to `evalstats` changes a coding
agent's deployment recommendation on a prompt A/B comparison, under three
statistical regimes with analytically known correct answers. See
`../DESIGN.md`-adjacent plan discussion for the full rationale; this README
just covers running it.

## One-time setup

1. Two long-lived venvs at the repo root (already created):
   - `.agent-study-venv-full` -- evalstats + numpy/scipy/pandas/statsmodels.
     This is the Python environment the agent's Bash tool uses in the
     "full" condition.
   - `.agent-study-venv-baseline` -- numpy/scipy/pandas/statsmodels only,
     evalstats absent. Used in the "baseline" condition.
   - `.agent-study-venv-runner` -- the orchestrator's own environment
     (claude-agent-sdk + numpy/scipy/pandas). Run everything below with
     this venv's Python.

2. Copy `.env.example` to `.env` and fill in an API key:

   ```
   cp agent_study/.env.example agent_study/.env
   # edit agent_study/.env, set ANTHROPIC_API_KEY=...
   ```

   `.env` is gitignored. The harness spawns a separate `claude` CLI
   subprocess per agent run, which needs its own auth -- it does not
   inherit an interactive Claude Code session's login.

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
(also enabled, in `run_agent.py`) separately blocks network access for
Bash-spawned processes. For a stronger, adversarial-proof guarantee, actual
container/VM isolation (e.g. Docker, matching the long-term OpenHands plan)
would be the next step; that wasn't pursued for v0 since Docker isn't
installed on this machine and installing it is a bigger call than fixing
the harness's own logic.

## What's in scope for v0 (and what isn't)

One scenario family (paired prompt A/B comparison, continuous scores),
three phenomena (true null, clear improvement, borderline improvement),
one factor (evalstats on/off), implicit prompt only, decision-correctness
scoring only, no reps per cell. Explicit-prompt variant, other scenario
families (LLM judge/PPI, ordinal, binary, multi-arm), multiple LLMs, and
statistical-power reps are deliberately deferred until this loop is
validated.
