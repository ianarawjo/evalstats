"""How much did the uncentred k-group lambda actually move things IN THE
HARNESS'S OWN EFFECT UNITS?

The drift measurement used raw per-step mean offsets on unit-variance data.
The sweeps use _jb_effect_magnitude(eval_type, frac), a different scale, so
"39% drift at effect=2.0" says nothing until it is expressed in the fracs the
sweeps actually run: PPI_POWER_EFFECT_FRACS goes to 1.2 and
PPI_FACTORIAL_EFFECT_FRACS to 0.8.

Builds cells with the harness's own generator at those fracs and reports the
shipped-vs-centred lambda for the ANOVA path.
"""
import sys, warnings, inspect, textwrap
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")

import evalstats.ppi as _ppi
from simulations.harness.scenarios.synthetic import (
    _ppi_power_baseline, _jb_effect_magnitude, JudgeBiasSource,
    generate_judge_bias_cell, PPI_POWER_EFFECT_FRACS, PPI_FACTORIAL_EFFECT_FRACS,
)

CENTRED = _ppi._pooled_k_group_lambda            # shipped is now the centred one
_src = inspect.getsource(CENTRED)
_OLD = """    Y_lab = np.concatenate([_c(g) for g in Y_lab_groups])
    Y_hat_lab = np.concatenate([_c(g) for g in Y_hat_lab_groups])
    Y_hat_unlab = np.concatenate([_c(g) for g in Y_hat_unlab_groups])"""
_NEW = """    Y_lab = np.concatenate(Y_lab_groups)
    Y_hat_lab = np.concatenate(Y_hat_lab_groups)
    Y_hat_unlab = np.concatenate(Y_hat_unlab_groups)"""
assert _src.count(_OLD) == 1
_ns = dict(_ppi.__dict__); _ns["np"] = np
exec(compile(textwrap.dedent(_src.replace(_OLD, _NEW)
                             .replace("def _pooled_k_group_lambda(", "def _old(")),
             "<old>", "exec"), _ns)
OLD = _ns["_old"]

ET = "continuous"
def lams(frac, reps=200, seed=5):
    base = _ppi_power_baseline(ET)
    sc = JudgeBiasSource(name=f"imp.{frac}", tag="power",
                         effect_size=_jb_effect_magnitude(ET, frac), **base)
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(reps):
        c = generate_judge_bias_cell(sc, rng)
        gs = [c.truth_a3, c.truth_b3, c.truth_c3]
        ls = [c.llm_a3, c.llm_b3, c.llm_c3]
        lb = [c.lab_a3, c.lab_b3, c.lab_c3]
        YL, YHL, YHU = [], [], []
        ok = True
        for g, l, labv in zip(gs, ls, lb):
            m = ~np.isnan(np.asarray(labv, float))
            if m.sum() < 2 or (~m).sum() < 2: ok = False; break
            YL.append(np.asarray(labv, float)[m]); YHL.append(np.asarray(l, float)[m])
            YHU.append(np.asarray(l, float)[~m])
        if not ok: continue
        a.append(OLD(YL, YHL, YHU)[0]); b.append(CENTRED(YL, YHL, YHU)[0])
    return (np.mean(a) if a else np.nan), (np.mean(b) if b else np.nan)

print(f"ANOVA path lambda, harness generator, eval_type={ET}, k=3")
print(f"{'frac':>7s} {'lam OLD':>9s} {'lam NEW':>9s} {'drift':>8s}  note")
base_new = None
for frac in (0.0, 0.15, 0.3, 0.5, 0.8, 1.0, 1.2):
    a, b = lams(frac)
    if base_new is None: base_new = b
    note = ""
    if frac in (0.5, 0.8): note = "<- factorial sweep"
    if frac == 1.2: note = "<- power sweep max"
    print(f"{frac:>7.2f} {a:>9.4f} {b:>9.4f} {a-b:>+8.4f}  {note}")
