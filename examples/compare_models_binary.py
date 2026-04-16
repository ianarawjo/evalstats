"""Binary (pass/fail) model comparison example for evalstats.compare_models.

Scores are 0/1 values (e.g. a judge labelling each response as correct or not).
Uses the Newcombe interval method, which is designed for binary proportions.

Usage:
    python examples/compare_models_binary.py
"""

import numpy as np

import evalstats as estats


rng = np.random.default_rng(7)

# Synthetic per-input binary scores (0 = fail, 1 = pass) for three models.
# model_A and model_B have similar pass rates; model_C is clearly weaker.
n_inputs = 120

# A shared per-input latent difficulty drives model_A and model_C together,
# so they tend to pass and fail on the same items (high behavioral agreement).
# model_B is drawn independently — it has no correlation with the others.
z = rng.standard_normal(n_inputs)

# Thresholds chosen so pass rates match the target proportions:
#   norm.ppf(1 - 0.82) ≈ -0.915  →  ~82% pass rate for model_A
#   norm.ppf(1 - 0.65) ≈ -0.385  →  ~65% pass rate for model_C
scores_dict = {
    "model_A": (z > -0.915).astype(float),
    "model_B": rng.binomial(1, p=0.80, size=n_inputs).astype(float),
    "model_C": (z > -0.385).astype(float),
}

report = estats.compare_models(
    scores_dict,
    method="auto",
    statistic="mean",
    n_bootstrap=10_000,
)

report.summary()
print()
