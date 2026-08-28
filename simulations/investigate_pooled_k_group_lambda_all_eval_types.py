"""The k-group lambda centering fix (775ab43) was measured on CONTINUOUS only.
Binary and likert have different judge quality and different truth supports,
so the drift could differ. Repeat the drift measurement on all three, in the
harness's own effect units.

Shipped code is now the CENTRED one, so the uncentred version is rebuilt from
its source for comparison.
"""
import sys, warnings, inspect, textwrap
import numpy as np
sys.path.insert(0, "/Users/ianarawjo/Documents/prompt-stats")
warnings.filterwarnings("ignore")

import evalstats.ppi as _ppi
from simulations.harness.scenarios.synthetic import (
    _ppi_power_baseline, _ppi_power_baseline_binary, _jb_effect_magnitude,
    _jb_effect_magnitude_binary, JudgeBiasSource, generate_judge_bias_cell,
)

CENTRED = _ppi._pooled_k_group_lambda
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


def lams(et, frac, reps=250, seed=5):
    if et == "binary":
        base = _ppi_power_baseline_binary(); mag = _jb_effect_magnitude_binary(frac)
    else:
        base = _ppi_power_baseline(et); mag = _jb_effect_magnitude(et, frac)
    sc = JudgeBiasSource(name=f"k.{et}.{frac}", tag="power", effect_size=mag, **base)
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(reps):
        c = generate_judge_bias_cell(sc, rng)
        gs = [c.llm_a3, c.llm_b3, c.llm_c3]; lb = [c.lab_a3, c.lab_b3, c.lab_c3]
        YL, YHL, YHU = [], [], []
        ok = True
        for g, labv in zip(gs, lb):
            m = ~np.isnan(np.asarray(labv, float))
            if m.sum() < 2 or (~m).sum() < 2: ok = False; break
            YL.append(np.asarray(labv, float)[m])
            YHL.append(np.asarray(g, float)[m]); YHU.append(np.asarray(g, float)[~m])
        if not ok: continue
        a.append(OLD(YL, YHL, YHU)[0]); b.append(CENTRED(YL, YHL, YHU)[0])
    return (np.mean(a) if a else np.nan), (np.mean(b) if b else np.nan)

for et in ("continuous", "likert", "binary"):
    print(f"\n=== {et} ===")
    print(f"{'frac':>7s} {'lam UNCENTRED':>14s} {'lam CENTRED':>12s} {'drift':>9s}  note")
    for frac in (0.0, 0.15, 0.30, 0.50, 0.80, 1.00, 1.20):
        try:
            a, b = lams(et, frac)
        except Exception as e:
            print(f"{frac:>7.2f}   ERROR {type(e).__name__}: {str(e)[:40]}"); continue
        note = "<- factorial" if frac in (0.5, 0.8) else ("<- power max" if frac == 1.2 else "")
        print(f"{frac:>7.2f} {a:>14.4f} {b:>12.4f} {a-b:>+9.4f}  {note}")
