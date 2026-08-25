"""Formatting helpers behind the harness's --latex table output."""

import collections
import math
import statistics
import sys

from numpy import percentile as np_percentile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulations.harness.latex_tables import (  # noqa: E402
    coverage_cell,
    error_rate_cell,
    mark_best_and_runnerup,
)

ALPHA = 0.05
TARGET = 1 - ALPHA


def shade(cell):
    """(colour, percent) of a cell's \\cellcolor, or None if unshaded."""
    if not cell.startswith("\\cellcolor{"):
        return None
    colour, _, pct = cell[len("\\cellcolor{"):cell.index("}")].partition("!")
    return colour, int(pct)


class TestErrorRateCell:
    def test_on_target_is_unshaded(self):
        assert error_rate_cell(0.05, ALPHA) == "0.050"

    def test_inflated_shades_red_and_conservative_shades_blue(self):
        assert shade(error_rate_cell(0.12, ALPHA))[0] == "red"
        assert shade(error_rate_cell(0.005, ALPHA))[0] == "blue"

    def test_intensity_grows_with_distance(self):
        worse = shade(error_rate_cell(0.15, ALPHA))[1]
        milder = shade(error_rate_cell(0.07, ALPHA))[1]
        assert worse > milder

    def test_intensity_is_capped(self):
        assert shade(error_rate_cell(0.99, ALPHA))[1] == 65
        assert shade(error_rate_cell(0.0, ALPHA))[1] == 65

    @pytest.mark.parametrize(
        "rate", [0.05, 0.051, 0.030, 0.029, 0.07, 0.20, 0.0, 1.0]
    )
    def test_dual_of_coverage_cell(self, rate):
        """The whole point: rate r must shade identically to coverage 1-r.
        Colour carries meaning, not direction -- inflated error and
        under-coverage are the same failure (anti-conservative) and must
        both read red, or the CI tables and the p-value tables would teach
        the reader two different colour languages."""
        assert shade(error_rate_cell(rate, ALPHA)) == shade(
            coverage_cell(1 - rate, TARGET)
        )

    def test_boundary_uses_displayed_rounding(self):
        """A value that prints as the threshold must not shade -- shading a
        cell that visibly reads 0.051 as if it were above 0.051 looks like a
        bug to a reader checking the numbers."""
        assert error_rate_cell(0.05099, ALPHA) == "0.051"

    def test_non_finite_renders_as_dash(self):
        assert error_rate_cell(float("nan"), ALPHA) == "-"
        assert error_rate_cell(None, ALPHA) == "-"

    def test_respects_non_default_alpha(self):
        assert error_rate_cell(0.10, 0.10) == "0.100"
        assert shade(error_rate_cell(0.10, 0.01))[0] == "red"


class TestMarkBestAndRunnerup:
    def test_lower_is_better_by_default(self):
        out = mark_best_and_runnerup(["0.1", "0.3", "0.2"], [0.1, 0.3, 0.2])
        assert out == ["\\textbf{0.1}", "0.3", "\\underline{0.2}"]

    def test_higher_is_better_flips_ranking(self):
        out = mark_best_and_runnerup(
            ["0.1", "0.3", "0.2"], [0.1, 0.3, 0.2], higher_is_better=True
        )
        assert out == ["0.1", "\\textbf{0.3}", "\\underline{0.2}"]

    def test_non_finite_excluded_from_ranking_but_kept(self):
        out = mark_best_and_runnerup(
            ["-", "0.9", "0.4"], [math.nan, 0.9, 0.4], higher_is_better=True
        )
        assert out == ["-", "\\textbf{0.9}", "\\underline{0.4}"]

    def test_single_ranked_value_gets_no_runner_up(self):
        out = mark_best_and_runnerup(["0.5", "-"], [0.5, math.nan])
        assert out == ["\\textbf{0.5}", "-"]

    def test_all_non_finite_is_left_alone(self):
        assert mark_best_and_runnerup(["-", "-"], [math.nan, math.nan]) == ["-", "-"]


class FakeAx:
    """Records fill_between calls so band geometry can be asserted."""

    def __init__(self):
        self.calls = []

    def fill_between(self, xs, los, his, **kw):
        self.calls.append((list(xs), list(los), list(his)))


class TestScenarioBands:
    """Bands treat the scenario as the unit of replication, not the rep."""

    @staticmethod
    def _mods():
        from simulations.harness.cases import pvalues

        return pvalues

    def test_inner_band_is_a_ci_on_the_mean_across_scenarios(self):
        pv = self._mods()
        vals = [0.90, 0.94, 0.95, 0.96, 1.00]
        ax = FakeAx()
        pv._scenario_bands(ax, [1], [0.95], [vals], color="k", style="both")
        (_, o_lo, o_hi), (_, i_lo, i_hi) = ax.calls  # outer drawn first
        sd = statistics.stdev(vals)
        assert i_hi[0] - i_lo[0] == pytest.approx(2 * 1.96 * sd / math.sqrt(len(vals)))

    def test_inner_band_is_centred_on_the_plotted_point(self):
        """Not on the scenario mean -- an off-centre band around the drawn
        line reads as a bug when the suite is unbalanced."""
        pv = self._mods()
        ax = FakeAx()
        pv._scenario_bands(ax, [1], [0.80], [[0.90, 0.94, 0.98]], color="k", style="both")
        (_, _, _), (_, i_lo, i_hi) = ax.calls
        assert (i_lo[0] + i_hi[0]) / 2 == pytest.approx(0.80)

    def test_outer_band_is_the_10_90_percentile_of_scenarios(self):
        pv = self._mods()
        vals = list(range(101))  # 0..100
        ax = FakeAx()
        pv._scenario_bands(ax, [1], [50], [vals], color="k", style="both")
        (_, o_lo, o_hi), _ = ax.calls
        assert o_lo[0] == pytest.approx(10) and o_hi[0] == pytest.approx(90)

    def test_outer_band_is_wider_than_inner_under_heterogeneity(self):
        """The whole reason for two bands: a method that is unreliable
        across scenarios must not look precise."""
        pv = self._mods()
        vals = [0.60, 0.75, 0.95, 0.99, 1.00] * 8
        ax = FakeAx()
        pv._scenario_bands(ax, [1], [0.85], [vals], color="k", style="both")
        (_, o_lo, o_hi), (_, i_lo, i_hi) = ax.calls
        assert (o_hi[0] - o_lo[0]) > 3 * (i_hi[0] - i_lo[0])

    def test_inner_band_does_not_shrink_with_reps_only(self):
        """Scenario-level SD is unchanged by how many reps produced each
        scenario value -- that is the point of moving off a per-rep MC error."""
        pv = self._mods()
        vals = [0.90, 0.94, 0.98]
        a, b = FakeAx(), FakeAx()
        pv._scenario_bands(a, [1], [0.94], [vals], color="k", style="ci")
        pv._scenario_bands(b, [1], [0.94], [vals], color="k", style="ci")
        assert a.calls[0][2][0] == pytest.approx(b.calls[0][2][0])

    def test_too_few_scenarios_yields_a_gap_not_a_fake_band(self):
        pv = self._mods()
        ax = FakeAx()
        pv._scenario_bands(ax, [1], [0.95], [[0.95]], color="k", style="both")
        for _, lo, hi in ax.calls:
            assert math.isnan(lo[0]) and math.isnan(hi[0])

    def test_default_draws_exactly_one_band(self):
        """Two translucent fills per method stack into a wash once a panel
        carries a dozen curves."""
        pv = self._mods()
        ax = FakeAx()
        pv._scenario_bands(ax, [1], [0.95], [[0.9, 0.95, 1.0]], color="k")
        assert len(ax.calls) == 1

    def test_default_band_is_the_ci_not_the_spread(self):
        """The paper figures use CI-on-the-mean; with 4-10 methods per panel
        the percentile spread overlaps into mud, and the conditional detail
        it compensated for is carried by the tables' per-n/per-k columns."""
        pv = self._mods()
        vals = [0.60, 0.95, 1.00] * 10
        ax = FakeAx()
        pv._scenario_bands(ax, [1], [0.85], [vals], color="k")
        (_, lo, hi), = ax.calls
        half = 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))
        assert lo[0] == pytest.approx(0.85 - half)
        assert hi[0] == pytest.approx(0.85 + half)
        # and it is NOT the percentile band
        assert lo[0] != pytest.approx(np_percentile(vals, 10))


    def test_scenario_values_group_by_eval_type_and_label(self):
        pv = self._mods()
        Row = collections.namedtuple("Row", "eval_type label rejects n_reps")
        rows = [
            Row("binary", "a", 5, 100), Row("binary", "a", 15, 100),  # -> 20/200
            Row("binary", "b", 40, 100),                              # -> 0.40
            Row("likert", "a", 0, 100),                               # -> 0.00
        ]
        got = sorted(pv._scenario_values(rows, lambda r: r.rejects))
        assert got == pytest.approx([0.0, 0.10, 0.40])

    def test_scenario_values_skips_empty_denominators(self):
        pv = self._mods()
        Row = collections.namedtuple("Row", "eval_type label rejects n_reps")
        assert pv._scenario_values([Row("binary", "a", 0, 0)], lambda r: r.rejects) == []


class TestWidthNormalization:
    """Pooling widths across eval types onto one axis requires dividing out
    each type's scale first, or the largest-scale type dominates."""

    @staticmethod
    def _scale(et):
        from simulations.harness.cases.pvalues import _width_scale

        return _width_scale(et)

    def test_spans_match_the_simulation_scale_bounds(self):
        assert self._scale("binary") == 1.0
        assert self._scale("continuous") == 1.0
        assert self._scale("likert") == 4.0  # 1-5
        assert self._scale("grades") == 100.0  # 0-100

    def test_normalizing_makes_likert_comparable_to_continuous(self):
        """A 1.24-wide Likert interval and a 0.31-wide continuous one are
        the same fraction of their scales -- unnormalized, Likert would look
        4x worse purely from its 1-5 range."""
        assert 1.24 / self._scale("likert") == pytest.approx(0.31)

    def test_unknown_eval_type_falls_back_to_unit_scale(self):
        assert self._scale("something_new") == 1.0

    def test_grades_is_out_of_the_default_sweep(self):
        """grades is continuous rescaled, and sidak/boot have no canonical
        CI for it -- leaving it in the default made the pooled width curves
        average over different eval-type mixes per method."""
        from simulations.harness.cases.pvalues import DEFAULT_EVAL_TYPES

        assert DEFAULT_EVAL_TYPES == ["binary", "continuous", "likert"]


class TestPowerRankingGate:
    """Power is only comparable between tests that hold their nominal level,
    so the p-value/FWER tables must not crown an uncorrected procedure."""

    @staticmethod
    def _rank(powers, rates, alpha=0.05):
        from simulations.harness.cases.pvalues import _power_ranking_values

        return _power_ranking_values(powers, rates, alpha)

    def test_inflated_method_is_excluded_from_ranking(self):
        # the uncorrected 0.22-FWER row has the highest raw power
        out = self._rank([0.85, 0.78, 0.60], [0.220, 0.049, 0.011])
        assert math.isnan(out[0])
        assert out[1:] == [0.78, 0.60]

    def test_conservative_method_stays_eligible(self):
        """Over-conservative is a real trade-off, not disqualifying -- an
        honest low-power result should still be rankable."""
        assert self._rank([0.60], [0.011]) == [0.60]

    def test_cutoff_is_bradleys_liberal_upper_bound(self):
        """Bradley (1978): calibrated means empirical alpha in
        [0.5a, 1.5a] -- so at a=0.05 anything above 0.075 is disqualified."""
        assert self._rank([0.9], [0.075]) == [0.9]
        assert math.isnan(self._rank([0.9], [0.0751])[0])

    def test_cutoff_scales_with_alpha(self):
        """The point of swapping off a fixed +-0.02 band: at alpha=0.01 that
        band would have waved through a rate three times nominal."""
        assert math.isnan(self._rank([0.9], [0.03], alpha=0.01)[0])
        assert self._rank([0.9], [0.03], alpha=0.10) == [0.9]

    def test_bradley_bounds_are_free_of_float_artifacts(self):
        from simulations.harness.cases.pvalues import bradley_bounds

        assert bradley_bounds(0.05) == (0.025, 0.075)
        assert bradley_bounds(0.01) == (0.005, 0.015)

    def test_non_finite_rate_is_excluded(self):
        assert math.isnan(self._rank([0.9], [math.nan])[0])

    def test_marking_end_to_end_skips_the_uncalibrated_winner(self):
        powers, rates = [0.85, 0.78, 0.60], [0.220, 0.049, 0.011]
        out = mark_best_and_runnerup(
            ["0.850", "0.780", "0.600"], self._rank(powers, rates),
            higher_is_better=True,
        )
        assert out == ["0.850", "\\textbf{0.780}", "\\underline{0.600}"]


def test_co_plotted_methods_have_distinct_colors():
    """Methods drawn on the same figure must not share a colour.

    Each group below is a set the ci_paired case plots together. A shared
    colour makes two lines indistinguishable in the paper's figures, which
    is how bootstrap_diff_nested and the multi-run mj_floor variant silently collided.
    """
    import collections
    from simulations.harness import methods as M

    color = {}
    for obj in vars(M).values():
        if isinstance(obj, M.Method):
            color.setdefault(obj.name, obj.color)

    groups = {
        "single-run binary": [
            "bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t",
            "newcombe_mover", "mj_floor", "tango_scc", "bayes_indep_comp",
            "bayes_paired_comp", "wald_indep", "tango_exact", "mj_unfloored",
            "bonett_price",
        ],
        "nested binary": [
            "bootstrap", "bca", "bayes_bootstrap", "smooth_bootstrap", "bootstrap_t",
            "t_interval", "bayes_indep_comp", "bayes_paired_comp", "wald_indep",
            "bootstrap_diff_nested", "bayes_diff_nested", "smooth_diff_nested",
            "mj_floor_flat", "newcombe_flat", "mj_floor_cluster",
            "clustered_score",
            "bonett_price_flat", "bonett_price_cluster",
        ],
    }
    for group, names in groups.items():
        by_color = collections.defaultdict(list)
        for name in names:
            assert name in color, f"{name!r} is not a registered Method"
            by_color[color[name]].append(name)
        clashes = {c: v for c, v in by_color.items() if len(v) > 1}
        assert not clashes, f"{group}: methods sharing a colour: {clashes}"

        # Exact equality is not enough -- bonett_price and bayes_indep_comp
        # were distinct hex values but only deltaE 9 apart, which reads as the
        # same pale orange in a legend. Deliberately-related families
        # (bayes_*/smooth_* variants) sit around deltaE 14-17, so the floor is
        # set below those.
        import itertools

        for m1, m2 in itertools.combinations(names, 2):
            de = _cielab_distance(color[m1], color[m2])
            assert de >= 12.0, (
                f"{group}: {m1} ({color[m1]}) and {m2} ({color[m2]}) are only "
                f"deltaE={de:.1f} apart -- indistinguishable in a plot"
            )


def _cielab_distance(hex1, hex2):
    """CIE76 colour difference between two hex colours."""
    import numpy as np
    from matplotlib.colors import to_rgb

    def to_lab(hexc):
        r, g, b = to_rgb(hexc)
        lin = lambda u: u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4
        r, g, b = lin(r), lin(g), lin(b)
        x = r * 0.4124 + g * 0.3576 + b * 0.1805
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = r * 0.0193 + g * 0.1192 + b * 0.9505
        pivot = lambda v: v ** (1 / 3) if v > 0.008856 else 7.787 * v + 16 / 116
        fx, fy, fz = pivot(x / 0.95047), pivot(y / 1.0), pivot(z / 1.08883)
        return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])

    return float(np.linalg.norm(to_lab(hex1) - to_lab(hex2)))
