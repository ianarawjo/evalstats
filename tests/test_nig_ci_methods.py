import numpy as np

from evalstats.core.resampling import nig_ci_1d, nig_ci_nested


def test_nig_ci_1d_empirical_coverage_is_reasonable():
    """Battle test: NIG 95% CI should have sane coverage on bounded data."""
    rng = np.random.default_rng(20260505)
    alpha = 0.05
    n_rep = 650
    n = 45

    # Beta truth in (0,1), with known population mean.
    a_true, b_true = 3.0, 4.0
    true_mean = a_true / (a_true + b_true)

    covered = 0
    widths: list[float] = []
    for _ in range(n_rep):
        x = rng.beta(a_true, b_true, size=n)
        lo, hi = nig_ci_1d(x, alpha=alpha)
        widths.append(hi - lo)
        covered += int(lo <= true_mean <= hi)

    coverage = covered / n_rep
    mean_width = float(np.mean(widths))

    # Keep bounds broad enough for stable CI runs while still catching regressions.
    assert 0.89 <= coverage <= 0.995, f"unexpected NIG coverage={coverage:.3f}"
    assert 0.0 < mean_width < 1.0


def _sample_nested_beta(
    n_items: int,
    n_runs: int,
    mu: float,
    item_concentration: float,
    run_concentration: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate hierarchical bounded data and per-item latent means."""
    alpha_item = mu * item_concentration
    beta_item = (1.0 - mu) * item_concentration
    theta = rng.beta(alpha_item, beta_item, size=n_items)

    vals = np.empty((n_items, n_runs), dtype=float)
    for i in range(n_items):
        t = float(np.clip(theta[i], 1e-4, 1.0 - 1e-4))
        vals[i] = rng.beta(t * run_concentration, (1.0 - t) * run_concentration, size=n_runs)
    return vals, theta


def test_nig_ci_nested_empirical_coverage_is_reasonable():
    """Battle test: nested NIG should cover the latent item-mean target."""
    rng = np.random.default_rng(20260506)
    alpha = 0.05
    n_rep = 500
    n_items = 70
    n_runs = 5
    true_mean = 0.62

    covered = 0
    widths: list[float] = []
    for _ in range(n_rep):
        vals, _theta = _sample_nested_beta(
            n_items=n_items,
            n_runs=n_runs,
            mu=true_mean,
            item_concentration=18.0,
            run_concentration=40.0,
            rng=rng,
        )
        lo, hi = nig_ci_nested(vals, alpha=alpha)
        widths.append(hi - lo)
        covered += int(lo <= true_mean <= hi)

    coverage = covered / n_rep
    mean_width = float(np.mean(widths))

    assert 0.89 <= coverage <= 0.995, f"unexpected nested-NIG coverage={coverage:.3f}"
    assert 0.0 < mean_width < 1.0
