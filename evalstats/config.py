"""Global configuration for evalstats."""

# We use a global variable to store the default alpha for CI analyses, 
# which can be set by the user via set_alpha_ci() and is used by default in 
# all CI analyses across the library (but can be overridden on a per-analysis basis 
# by passing an explicit alpha).
_alpha: float = 0.05

# Alpha levels used to build the gradient CI bands in terminal plots.
# Ordered narrowest→widest: 90%, 95%, 99%, 99.9% CI.
GRADIENT_CI_ALPHAS: tuple[float, ...] = (0.32, 0.10, 0.05, 0.01)

def set_alpha_ci(alpha: float) -> None:
    """Set the default significance level used across all CI analyses.

    Parameters
    ----------
    alpha:
        Significance level (e.g. 0.05 for 95% CI, 0.01 for 99% CI).
        Must be in the open interval (0, 1).
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    global _alpha
    _alpha = alpha


def get_alpha_ci() -> float:
    """Return the current default significance level."""
    return _alpha
