"""Build a patched _ppi_bootstrap_t_joint_stats WITHOUT editing api.py.

Takes the shipped function's exact source, applies one textual change (a
RELATIVE floor on boot_se), and execs it in api.py's own module globals so
every helper it calls resolves identically. Guarantees the only difference
under test is the floor itself.

Current line:
    se_boot_safe = np.where(boot_se > 1e-12, boot_se, 1.0)
An ABSOLUTE 1e-12 floor cannot catch a boot_se that is small-but-nonzero.
T is studentized, so the meaningful scale is the OBSERVED se: a replicate
whose se collapses far below obs_se is a degenerate resample (e.g. an
all-identical draw from a near-constant vector), not evidence about the
sampling distribution.
"""
import inspect, textwrap
import numpy as np
import evalstats.api as _api

_SRC = inspect.getsource(_api._ppi_bootstrap_t_joint_stats)

_OLD = "    se_boot_safe = np.where(boot_se > 1e-12, boot_se, 1.0)"
assert _SRC.count(_OLD) == 1, "anchor line not found -- api.py changed"

_NEW = """    # RELATIVE floor: a bootstrap SE far below the observed SE means the
    # replicate degenerated, not that the statistic is that precise.
    _floor = _SE_FLOOR_C * obs_se[np.newaxis, :]
    boot_se = np.maximum(boot_se, _floor)
    se_boot_safe = np.where(boot_se > 1e-12, boot_se, 1.0)"""


def make(c: float):
    """Return a joint-stats function with relative floor coefficient `c`.
    c=0.0 reproduces the shipped behaviour exactly."""
    src = _SRC.replace(_OLD, _NEW)
    src = src.replace("def _ppi_bootstrap_t_joint_stats(", "def _patched_joint(")
    ns = dict(_api.__dict__)
    ns["_SE_FLOOR_C"] = float(c)
    ns["np"] = np
    exec(compile(textwrap.dedent(src), "<patched>", "exec"), ns)
    fn = ns["_patched_joint"]
    fn._floor_c = float(c)
    return fn
