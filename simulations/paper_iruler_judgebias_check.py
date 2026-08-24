"""Case study: how much undetected LLM-judge bias would it take to overturn
one of iRULER's reported significant results?

iRULER (Bai, Cheong, Muller, Lim; CHI 2026; arXiv:2602.12779) evaluates its
writing-revision system against two baselines (Text-LLM, Rubric-LLM) using
GPT-4.1 as an LLM-as-a-judge to score writing quality improvement (DScore,
0-100 scale). This is exactly the practice evalstats is built to correct
for: statistics computed directly on LLM-judge scores, with no PPI-style
adjustment for judge/human disagreement.

We do NOT have iRULER's raw data. Every number below is taken directly from
the published paper (means, N, exact p-values, QWK validation figures) or
read off Figure 7's plotted 95% CIs via pixel measurement (see
_read_figure7_ci() below for the extraction method -- calibrated against the
y-axis tick labels, verified to reproduce the paper's reported means to
within ~0.1 points). This is a plausibility / fragility analysis: given the
paper's own numbers treated as fixed, how much systematic judge bias would
be needed to erase a "significant" result, and is that amount plausible
given what the paper itself reports about its judge?

Target result (S6.1.1, "Writing Quality (H1)"): iRULER's writing-quality
improvement (Delta-Score) was reported as significantly higher than the
Rubric-LLM baseline's -- but *only* when scored by the Text-LLM judge
(M=26.7 vs M=19.2, p=.0003), NOT when scored by the Rubric-LLM judge itself
(M=30.8 vs M=23.8, p=n.s.). The paper's own two judges disagree about
whether this comparison is significant. We interrogate the one that says
yes.

Run:
    .venv/bin/python -m simulations.paper_iruler_judgebias_check
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Numbers taken directly from the paper (all cited to section/figure/table)
# ---------------------------------------------------------------------------

# S6.1.1: Delta-Score (Text-LLM-scored), Writing Revision experiment, Main task.
# Between-subjects, N=16 per condition (N=48 total, S6.1).
N_PER_CONDITION = 16
MEAN_IRULER_TEXT = 26.7
MEAN_RUBRIC_TEXT = 19.2
DIFF_REPORTED = MEAN_IRULER_TEXT - MEAN_RUBRIC_TEXT  # 7.5
P_REPORTED = 0.0003  # exact p from the paper's post-hoc contrast t-test
ALPHA_PAPER = 0.002  # paper's own Bonferroni-corrected significance threshold (0.05/25)
ALPHA_CONVENTIONAL = 0.05

# Figure 7 pixel-read 95% CIs (Text-LLM score / red series), calibrated
# against the y-axis tick labels (0/10/20/30/40 at pixel rows 1299.5/1039.5/
# 781.5/522.0/261.5 in a 400dpi render). Reproduces the paper's stated means
# (26.7, 19.2) to within 0.1-0.2 points, confirming the calibration.
FIG7_CI_IRULER_TEXT = (21.37, 32.20)   # read off Fig. 7
FIG7_CI_RUBRIC_TEXT = (13.97, 24.64)   # read off Fig. 7

# Table 1 (technical validation, ICNALE dataset, N=640, 5 repeated GPT-4.1
# runs/essay): pure aleatoric judge noise on the total score, after
# averaging 5 runs (the paper's own scoring protocol, S6.1.1).
INTRA_MODEL_SD_TOTAL = 1.69  # SD across 5 repeated runs, single essay, 0-100 scale
N_JUDGE_RUNS_AVERAGED = 5

# Table 3 (S6.5, human expert validation on 96 essays spanning all 3
# feedback conditions of the actual Writing Revision experiment):
QWK_TEXT_LLM_VS_EXPERTS = 0.76   # "substantial"
QWK_RUBRIC_LLM_VS_EXPERTS = 0.88  # "almost perfect"


def marginal_se_from_ci(ci: tuple[float, float]) -> float:
    """SE implied by a plotted 95% CI, assuming a normal approximation."""
    half_width = (ci[1] - ci[0]) / 2
    return half_width / 1.96


def implied_se_from_p(diff: float, p: float, df: float) -> float:
    """Back out the SE the paper's own contrast test must have used, given
    the reported mean difference and exact p-value, for an assumed df."""
    t_crit = stats.t.ppf(1 - p / 2, df)
    return diff / t_crit


def breakeven_bias(diff: float, se: float, alpha: float) -> float:
    """How much of `diff` would need to be attributable to bias (not true
    effect) for the comparison to drop to exactly `alpha` significance."""
    z_or_t_crit = stats.norm.ppf(1 - alpha / 2)
    max_defensible_diff = z_or_t_crit * se
    return diff - max_defensible_diff


def implied_disagreement_sd(sd_item: float, qwk: float) -> float:
    """Convert a quadratic-weighted kappa into an implied item-level SD of
    (judge score - true score), in the metric's own units.

    Two assumptions, stacked:
    1. QWK ~ rho (Pearson correlation / ICC). Quadratic-weighted kappa is
       asymptotically equivalent to the intraclass correlation coefficient
       when the two raters' marginal distributions are similar (Fleiss &
       Cohen, 1973, "The equivalence of weighted kappa and the intraclass
       correlation coefficient as measures of reliability"). We cannot
       verify the similar-marginals condition without iRULER's raw score
       distributions, so this is a standard approximation, not a check.
    2. Judge and true scores share a common variance V = sd_item**2
       (classical test theory: X = judge score, Y = true score). Then
       Var(X - Y) = Var(X) + Var(Y) - 2*Cov(X,Y) = 2V(1 - rho), so
       SD(X - Y) = sqrt(2) * sd_item * sqrt(1 - rho).

    `sd_item` must come from an independent source, not from QWK itself --
    here, back-solved from Figure 7's plotted 95% CIs (SE = SD / sqrt(N)
    for a group mean), since QWK is dimensionless and carries no
    information about the metric's own scale.
    """
    return np.sqrt(2) * sd_item * np.sqrt(1 - qwk)


def main() -> None:
    print("=" * 78)
    print("Target: iRULER vs Rubric-LLM, Delta-Score, Text-LLM-scored (S6.1.1)")
    print("=" * 78)
    print(f"Reported means: iRULER={MEAN_IRULER_TEXT}, Rubric-LLM={MEAN_RUBRIC_TEXT}, "
          f"diff={DIFF_REPORTED:.1f}")
    print(f"Reported p={P_REPORTED} (Bonferroni alpha={ALPHA_PAPER})")
    print(f"N={N_PER_CONDITION}/condition, between-subjects")
    print()

    # --- Two independent readings of "how uncertain is this diff?" ---
    se_iruler_marginal = marginal_se_from_ci(FIG7_CI_IRULER_TEXT)
    se_rubric_marginal = marginal_se_from_ci(FIG7_CI_RUBRIC_TEXT)
    se_diff_marginal = np.sqrt(se_iruler_marginal**2 + se_rubric_marginal**2)
    z_marginal = DIFF_REPORTED / se_diff_marginal
    p_marginal = 2 * (1 - stats.norm.cdf(z_marginal))

    print("--- Reading 1: Figure 7's plotted 95% CIs, as independent groups ---")
    print(f"  SE(iRULER)={se_iruler_marginal:.2f}, SE(Rubric-LLM)={se_rubric_marginal:.2f}")
    print(f"  SE(diff)={se_diff_marginal:.2f} -> implied p={p_marginal:.4f}")
    print(f"  (This is what a reader sees just from the figure: the CIs")
    print(f"   visibly overlap, and the implied p is borderline, not .0003.)")
    print()

    print("--- Reading 2: SE implied by the paper's own reported p=.0003 ---")
    for df in (14, 30, 46, 200):
        se = implied_se_from_p(DIFF_REPORTED, P_REPORTED, df)
        print(f"  assumed df={df:>4d}: implied SE(diff)={se:.2f}")
    print(f"  (~3-4x smaller than Reading 1's SE. The mixed model with")
    print(f"   Participant + Task ID random effects is drawing on more than")
    print(f"   N=16/condition worth of information -- consistent with")
    print(f"   multiple revision iterations per participant feeding the model.)")
    print()

    # --- Breakeven bias needed to erase significance, under each reading ---
    print("--- Breakeven: how much of the 7.5-point gap would need to be")
    print("    judge bias (not true iRULER superiority) to lose significance? ---")
    for label, se in [
        ("Reading 1 (marginal CI)", se_diff_marginal),
        ("Reading 2 (df=46, implied)", implied_se_from_p(DIFF_REPORTED, P_REPORTED, 46)),
    ]:
        for alpha_label, alpha in [
            ("conventional alpha=.05", ALPHA_CONVENTIONAL),
            ("paper's own alpha=.002", ALPHA_PAPER),
        ]:
            bias = breakeven_bias(DIFF_REPORTED, se, alpha)
            if bias <= 0:
                print(f"  {label:<32s} @ {alpha_label:<24s}: "
                      f"already NOT significant at this alpha under this SE (no bias needed)")
            else:
                pct = 100 * bias / DIFF_REPORTED
                print(f"  {label:<32s} @ {alpha_label:<24s}: "
                      f"bias >= {bias:+.2f} pts ({pct:.0f}% of the 7.5-pt gap)")
    print()

    # --- Ground the breakeven bias in the judge's own reported QWK ---
    # "21% of the gap" is an abstract fraction with no anchor to what the
    # judge is actually capable of. Instead, ask: how large is that
    # breakeven bias relative to how much this judge is already known to
    # disagree with human experts, item by item?
    print("--- Grounding the breakeven bias in the judge's own reported QWK ---")
    se_pooled_marginal = (se_iruler_marginal + se_rubric_marginal) / 2
    sd_pop_item_level = se_pooled_marginal * np.sqrt(N_PER_CONDITION)
    print(f"  Step 1: Fig. 7's marginal SEs imply an item-level SD(DeltaScore)")
    print(f"          of ~{sd_pop_item_level:.1f} pts (SE={se_pooled_marginal:.2f} x sqrt(N={N_PER_CONDITION})).")

    rho = QWK_TEXT_LLM_VS_EXPERTS  # QWK ~ Pearson r under similar marginals (Fleiss & Cohen 1973)
    sd_disagreement = implied_disagreement_sd(sd_pop_item_level, rho)
    print(f"  Step 2: treating QWK={rho} as an approximate correlation with the true")
    print(f"          score (Fleiss & Cohen 1973), a signal+noise model implies the")
    print(f"          judge disagrees with the true score by ~{sd_disagreement:.1f} pts SD,")
    print(f"          essay to essay -- almost as large as the entire 7.5-pt gap.")

    breakeven_002 = breakeven_bias(DIFF_REPORTED, implied_se_from_p(DIFF_REPORTED, P_REPORTED, 46), ALPHA_PAPER)
    frac_sd = breakeven_002 / sd_disagreement
    print(f"  Step 3: the {breakeven_002:.2f}-pt breakeven bias is only "
          f"{frac_sd:.2f} SD of that")
    print(f"          per-essay disagreement -- i.e., the judge doesn't need to be")
    print(f"          unusually wrong; an ordinary-sized slice of its already-")
    print(f"          acknowledged imperfection, systematically correlated with")
    print(f"          condition (e.g. via length/verbosity) rather than random,")
    print(f"          fully accounts for the observed gap.")
    print(f"          (Note: {frac_sd:.0%} SD and the {100*breakeven_002/DIFF_REPORTED:.0f}% -of-the-gap figure above are two")
    print(f"          different quantities that happen to land close together --")
    print(f"          not the same calculation.)")

    rho_rubric = QWK_RUBRIC_LLM_VS_EXPERTS
    sd_disagreement_rubric = implied_disagreement_sd(sd_pop_item_level, rho_rubric)
    print(f"  For contrast: the OTHER judge (Rubric-LLM, QWK={rho_rubric}) has an implied")
    print(f"  disagreement SD of only ~{sd_disagreement_rubric:.1f} pts -- tighter, and it is the")
    print(f"  judge that finds this comparison non-significant.")
    print()

    print("--- Context for plausibility: what does the paper say about its judge? ---")
    print(f"  Intra-model SD (5 repeated GPT-4.1 runs, averaged): {INTRA_MODEL_SD_TOTAL}")
    print(f"    -> on the mean-of-5 score actually used: "
          f"{INTRA_MODEL_SD_TOTAL/np.sqrt(N_JUDGE_RUNS_AVERAGED):.2f} "
          f"(small: rules out *random* judge noise as the source)")
    print(f"  Expert-LLM QWK: Text-LLM={QWK_TEXT_LLM_VS_EXPERTS} (substantial), "
          f"Rubric-LLM={QWK_RUBRIC_LLM_VS_EXPERTS} (almost perfect)")
    print(f"    -> the LESS-validated judge (Text-LLM) is the one that finds")
    print(f"       iRULER significantly beats Rubric-LLM; the MORE-validated")
    print(f"       judge (Rubric-LLM) finds no significant difference (n.s.)")
    print(f"       for the exact same comparison (M=30.8 vs 23.8, S6.1.1).")
    print()
    print("  NOT a same-model self-preference confound: GPT-4.1 writes the")
    print("  revisions in ALL THREE conditions, not just iRULER's -- per")
    print("  Appendix A.2.3, 'the following general-purpose revision prompt")
    print("  was used in the chatbot for both Text-LLM and Rubric-LLM")
    print("  conditions.' So GPT-4.1 scoring GPT-4.1-written text is true of")
    print("  Rubric-LLM (the condition that LOST this comparison) too, and")
    print("  can't by itself explain a differential advantage for iRULER.")
    print()
    print("  Workflow asymmetry (the more precise mechanism): Rubric-LLM's")
    print("  revisions are narrow and user-initiated -- the participant reads")
    print("  the rubric, manually types one instruction into a generic chat")
    print("  (A.2.3, told to return 'no explanatory rationale'). iRULER's")
    print("  'How-To' feature (A.1.2) instead auto-generates a revision aimed")
    print("  at hitting a specific TARGET rubric score across potentially")
    print("  several criteria at once -- a structurally different generation")
    print("  task, even though its own prompt explicitly instructs 'minimal")
    print("  modification.' Raising multiple criteria simultaneously plausibly")
    print("  requires adding more material than a single user-typed ask does.")
    print()
    print('  Direct qualitative evidence (S6.4.4, participant W48, iRULER')
    print('  condition): "I feel it might keep getting longer and longer')
    print('  texts, but there\'s no content that\'s being added" and the')
    print('  system "over-generate[s]... You make it too sophisticated."')
    print("  Verbosity/length is one of the most replicated LLM-judge biases")
    print("  in the literature, independent of who wrote the text -- and this")
    print("  is a participant, in this specific paper, describing exactly")
    print("  that surface feature in exactly the condition (iRULER) whose")
    print("  advantage is in question, consistent with the workflow asymmetry")
    print("  above rather than model self-preference.")


if __name__ == "__main__":
    main()
