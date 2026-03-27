#!/usr/bin/env python3
"""Generate investigation stub HTML pages."""

import os

INVESTIGATIONS = [
    {
        "slug": "model-vs-model",
        "tier": "Foundations",
        "title": "Model A vs. Model B: Is the Gap Real?",
        "subtitle": "You ran both models on your eval set. Model A scored 64%, Model B scored 72%. Is that a real difference? Here&rsquo;s how to find out.",
        "intro": "The most fundamental question in LLM evaluation: does the observed score difference between two models reflect a genuine capability gap, or is it within the noise of your sample? This investigation walks through the complete statistical workflow &mdash; from raw scores to a defensible conclusion.",
        "learns": [
            ("How to compute a CI on the score difference", "Using Newcombe (binary) or smooth bootstrap (numeric), and how to interpret the resulting interval."),
            ("What &ldquo;statistically distinguishable&rdquo; means", "Why two models can have different means while still being statistically tied, and how to communicate that honestly."),
            ("How sample size limits your conclusions", "The minimum N needed to detect a 2-point gap, a 5-point gap, or a 10-point gap at 95% confidence."),
            ("Wilson vs. smooth bootstrap", "When to use each method based on your score type, and what you give up by choosing the wrong one."),
        ],
    },
    {
        "slug": "best-prompt",
        "tier": "Foundations",
        "title": "Finding Your Best Prompt",
        "subtitle": "You&rsquo;ve tested 8 prompt variants. The best scores 82%, the worst 71%. But comparing 8 variants inflates your false-positive risk by up to 8&times;. Here&rsquo;s how to find the real winner.",
        "intro": "Prompt comparison is the most common eval task &mdash; and one of the most statistically fraught. When you run k comparisons against a baseline (or all-pairs), you need multiple comparison correction or your &ldquo;best prompt&rdquo; is probably just the luckiest one.",
        "learns": [
            ("Why multiple comparisons inflate false positives", "How running k tests at &alpha;=0.05 gives you roughly k&times;0.05 expected false discoveries without correction."),
            ("Holm correction in practice", "How Holm-Bonferroni works step by step, and why it gives more statistical power than plain Bonferroni without increasing false positives."),
            ("Building CIs for each variant", "How to compute smooth bootstrap or Wilson CIs for each prompt and visualize the full uncertainty picture."),
            ("Reporting honestly", "How to say &ldquo;Prompt C wins&rdquo; with appropriate hedging, and when the data says &ldquo;we can&rsquo;t tell.&rdquo;"),
        ],
    },
    {
        "slug": "multi-metric",
        "nav_label": "Comparing Multiple Metrics",
        "tier": "Foundations",
        "title": "Comparing Models Across Multiple Metrics",
        "subtitle": "Model A wins on quality, Model B wins on safety, and they tie on cost. There&rsquo;s no universal &ldquo;best.&rdquo; Here&rsquo;s how to reason about multi-metric results without cherry-picking the metric that flatters your preferred model.",
        "intro": "Real-world LLM deployments are evaluated on several dimensions simultaneously &mdash; quality, safety, latency, cost, refusal rate, and more. When models don&rsquo;t agree across metrics, picking a winner requires making values explicit. This investigation covers the statistical and conceptual tools for multi-metric comparison: how to carry uncertainty through composite scores, when Pareto dominance lets you sidestep weighting choices, and how to present tradeoffs honestly to stakeholders.",
        "learns": [
            ("Why composite scores can mislead", "How the choice of weights drives the composite winner, and why hiding that choice inside an aggregate obscures a values decision that should be made explicitly."),
            ("Pareto dominance as a weight-free criterion", "What it means for one model to dominate another across all metrics simultaneously, how to identify the Pareto frontier, and when it is (and isn&rsquo;t) small enough to be useful."),
            ("Carrying uncertainty through multiple metrics", "How to compute and visualize per-metric confidence intervals, detect when metric CIs are too wide to support any conclusion, and report results without false precision."),
            ("Presenting multi-metric results to stakeholders", "Chart patterns and table formats that surface tradeoffs rather than a pre-digested verdict, and how to make weighting assumptions visible rather than baked in."),
        ],
    },
    {
        "slug": "model-prompt-grid",
        "tier": "Foundations",
        "title": "Finding the Best Model-Prompt Combo: What Actually Wins?",
        "subtitle": "Three models, five prompt templates, fifteen combinations. Which pairing genuinely wins? And does Model A&rsquo;s lead hold across all prompts &mdash; or only some of them?",
        "intro": "The 2D model&times;prompt comparison is where most real-world eval work lives. It&rsquo;s also where naive statistics fail most spectacularly: picking the highest cell in a 3&times;5 matrix without correction virtually guarantees a spurious winner.",
        "learns": [
            ("All-pairs comparison with family-wise correction", "How to test all 15 model&times;prompt combinations and apply Holm correction across the full family of tests."),
            ("Detecting interaction effects", "How to check whether Model A&rsquo;s advantage over Model B is consistent across prompt templates, or only appears with certain phrasings."),
            ("Visualizing the score matrix with uncertainty", "How to build a score heatmap that shows CI width alongside means, so narrow wins are visually distinct from robust ones."),
            ("Reporting the winning combination", "How to report a model+prompt winner while being honest about whether the win generalizes beyond these specific templates."),
        ],
    },
    {
        "slug": "before-after",
        "nav_label": "Before / After Fine-Tune",
        "tier": "Going Deeper",
        "title": "Before / After: Did My Fine-Tune Actually Help?",
        "subtitle": "Your fine-tuned checkpoint scores 2 points higher on average. But is that real signal or sampling noise? CI-based before/after analysis for deployment decisions.",
        "intro": "Fine-tuning comparisons are high-stakes: a spurious positive could mean shipping a regression, and a spurious negative could mean discarding real progress. The paired structure of before/after evals (same inputs, two checkpoints) is statistically powerful when used correctly.",
        "learns": [
            ("Framing fine-tuning as a paired test", "Why the &ldquo;before&rdquo; and &ldquo;after&rdquo; scores should be treated as paired differences, not independent samples, and how this dramatically reduces required N."),
            ("Setting a deployment CI threshold", "How to decide in advance what CI gap justifies shipping &mdash; and how to frame this as a statistical decision rule rather than a subjective call."),
            ("Detecting category-level regressions", "How to check whether an overall improvement masks regressions on specific task subsets, and how to correct for multiple category comparisons."),
            ("Minimum detectable effect at your sample size", "How to know before running the eval whether your N is sufficient to detect the effect size you care about."),
        ],
    },
    {
        "slug": "prompt-sensitivity",
        "nav_label": "Prompt Sensitivity",
        "tier": "Going Deeper",
        "title": "Prompt Sensitivity Analysis: Model Strength or Phrasing Luck?",
        "subtitle": "The elephant in the room: model rankings often flip when you change prompt wording. This investigation decomposes score variance into what&rsquo;s the model and what&rsquo;s the prompt.",
        "intro": "A &ldquo;model comparison&rdquo; that uses a single prompt template per model is not really comparing models &mdash; it&rsquo;s comparing model+prompt pairs. Prompt sensitivity analysis is the corrective: run each model on k semantically equivalent phrasings, then ask how stable the ranking is across phrasings.",
        "learns": [
            ("Running a prompt sensitivity study", "How to design a suite of k paraphrase variants that test the same capability while varying surface-level phrasing."),
            ("Decomposing variance: model vs. prompt", "How to partition the total score variance into a model component and a prompt component using a simple variance decomposition."),
            ("Computing a &ldquo;ranking stability score&rdquo;", "How to estimate the probability that the observed Model A &gt; Model B ranking would hold under a new, unseen prompt &mdash; and what that probability looks like in practice."),
            ("Reporting rankings that are robust to prompt choice", "How to report model comparisons that are backed by multi-prompt evidence, and how to flag comparisons that are not."),
        ],
    },
    {
        "slug": "response-consistency",
        "nav_label": "Checking Response Consistency",
        "tier": "Going Deeper",
        "title": "Checking Response Consistency",
        "subtitle": "You run the same prompt twice and get different scores. Is that model variance, judge variance, or both? Without knowing how much outputs vary across runs, a single-pass eval may be a snapshot of noise &mdash; not a stable estimate of capability.",
        "intro": "Most LLM eval pipelines run each test item once. But at temperature &gt; 0, the same prompt produces different outputs &mdash; and different scores &mdash; on every call. This variance has two sources: the model&rsquo;s stochastic generation and the judge&rsquo;s stochastic scoring. Until you measure it, you can&rsquo;t know whether your eval score is a stable estimate or a lucky draw. This investigation shows how to quantify run-to-run variance, separate its two sources, and decide how many runs per item your pipeline actually needs.",
        "learns": [
            ("Quantifying run-to-run variance", "How to score the same inputs multiple times and estimate within-item standard deviation. When is variance small enough to trust a single pass, and when does it swamp your signal?"),
            ("Separating model variance from judge variance", "How to design a replicated experiment &mdash; scoring fixed outputs multiple times, and generating multiple outputs per item &mdash; to isolate how much variability comes from the model vs. the scorer."),
            ("Computing the minimum runs needed per item", "The calculation for deciding how many samples per item are required to keep your aggregate score reliable to within a target margin, given your measured within-item variance."),
            ("When consistency is itself a quality signal", "How to report response consistency as a standalone metric: a model that scores 80% with low variance may be more deployable than one that scores 85% with high variance on the same items."),
        ],
    },
    {
        "slug": "rag-factorial",
        "nav_label": "Stats for RAG Pipelines",
        "tier": "Going Deeper",
        "title": "Which Chunker and Retrieval Method Wins? Stats for RAG Pipelines",
        "subtitle": "Your dataset is labeled with the chunker and retrieval method used for each row. Fixed-512 + BM25, semantic + dense &mdash; four combinations, one score column. Here&rsquo;s how to find out which pipeline configuration genuinely wins, and whether the factors interact.",
        "intro": "RAG pipelines have multiple independently-tunable components &mdash; chunking strategy, retrieval method, reranker, context window size &mdash; and it&rsquo;s tempting to tune them one at a time. But factors interact: dense retrieval may only outperform BM25 when paired with semantic chunking, while the combination with fixed-size chunks shows no difference. A one-factor-at-a-time analysis misses this. Factorial evaluation with a mixed model captures main effects and interactions simultaneously, accounts for per-question variation in difficulty, and applies the right multiple-comparison correction across all pairwise tests. If your eval dataset already records which configuration produced each row, you&rsquo;re one <code>analyze_factorial()</code> call away from a rigorous answer.",
        "learns": [
            ("Structuring your dataset for factorial analysis", "How to format a tagged RAG eval dataset &mdash; one row per (question, configuration) with the factor columns and a score column &mdash; so it can be passed directly to <code>analyze_factorial()</code>."),
            ("What the mixed model is doing", "How fitting <code>score&nbsp;~&nbsp;chunker&nbsp;*&nbsp;retrieval&nbsp;+&nbsp;(1|question)</code> separates configuration effects from question-difficulty effects, and why this gives tighter CIs than ignoring the random intercept."),
            ("Reading main effects vs. interactions", "How to interpret the Wald &chi;&sup2; tests per factor: a significant interaction means the best chunker depends on which retrieval method you use, and you can&rsquo;t report a single &ldquo;best chunker&rdquo; without qualification."),
            ("Reporting the winning configuration honestly", "How to present estimated marginal means, pairwise CIs, and Holm-corrected p-values across all combinations &mdash; and how to flag when no combination is statistically distinguishable from the others."),
        ],
    },
    {
        "slug": "sample-size",
        "nav_label": "Sample Size Planning",
        "tier": "Going Deeper",
        "title": "How Many Eval Items Do I Actually Need?",
        "subtitle": "Statistical power analysis for LLM evals. Given the effect size you want to detect, how many items do you need &mdash; before you run the eval, not after.",
        "intro": "Most eval datasets are sized by convenience (&ldquo;I had 50 good test cases&rdquo;) rather than by statistical need. Power analysis flips this: given the minimum difference you care about detecting, compute how large N must be to detect it reliably.",
        "learns": [
            ("Framing eval design as a power problem", "How to specify minimum detectable effect (MDE), desired power (1&minus;&beta;), and confidence level (&alpha;) before choosing N."),
            ("How N scales with effect size and variance", "The key relationships: halving the MDE quadruples required N; doubling variance roughly doubles N; binary data often requires fewer samples than continuous."),
            ("Using pilot data to estimate variance", "How to run a small pilot (N=20&ndash;30) to estimate score variance, then use that to compute required N for the full study."),
            ("Power curves for common eval scenarios", "Reference curves for binary pass/fail (Wilson-based power), numeric scores (bootstrap-based power), and paired vs. unpaired designs."),
        ],
    },
    {
        "slug": "regression-guard",
        "tier": "Advanced",
        "title": "Regression Guard: CI-Based Safety Net for Model Updates",
        "subtitle": "You just shipped a new model version. How do you know it didn&rsquo;t regress anywhere? CI-based regression detection, category-level analysis, and statistical thresholds for your release checklist.",
        "intro": "Model updates are an ongoing source of silent regressions. A new checkpoint that improves summarization may degrade instruction-following. A fine-tune that improves math may hurt creative writing. Regression Guard is a systematic, statistics-first approach to catching these before they ship.",
        "learns": [
            ("Defining regression statistically", "Why &ldquo;lower mean score&rdquo; is not the right definition of regression, and how to define it instead as a CI lower bound crossing a threshold."),
            ("Setting CI-based release gates", "How to specify a release criterion (&ldquo;ship if the CI lower bound on difference is above &minus;2 points&rdquo;) and how to calibrate the threshold for your use case."),
            ("Category-level regression detection with correction", "How to test all task categories simultaneously and apply Holm correction so that a regression anywhere in the eval is flagged reliably."),
            ("Running regression guard in CI/CD", "How to integrate statistical regression testing into a continuous delivery pipeline so every checkpoint is evaluated automatically."),
        ],
    },
    {
        "slug": "irr-human-annotations",
        "nav_label": "Auditing Your LLM Judge",
        "tier": "Advanced",
        "title": "Auditing Your LLM Judge",
        "subtitle": "You&rsquo;ve gathered human annotations on a sample of outputs. Now you want to know: does your LLM judge agree with humans? And by how much must it agree before you can trust it as a proxy? This investigation quantifies judge&ndash;human alignment with the right statistics.",
        "intro": "LLM-as-judge pipelines are trusted at scale, but the key question is rarely asked rigorously: how well does the judge actually agree with humans? Simple accuracy against a held-out set understates the problem. Agreement metrics must account for scale type (binary, ordinal, continuous), chance-level agreement, and the difference between systematic bias and random disagreement. This investigation covers how to design a human annotation study for judge validation, compute the right agreement statistics with confidence intervals, and interpret the results in terms of when automated scoring is &mdash; and isn&rsquo;t &mdash; an adequate proxy.",
        "learns": [
            ("Choosing the right agreement metric", "Cohen&rsquo;s &kappa;, Krippendorff&rsquo;s &alpha;, weighted &kappa;, and intraclass correlation: which to use based on your scale type (binary, ordinal, continuous) and whether you have two raters or many."),
            ("Structuring your annotation sample", "How many items to annotate, how to stratify across score levels to avoid ceiling/floor effects, and how to handle disagreements between human annotators before using them as a gold standard."),
            ("The minimum agreement threshold", "What &kappa; or ICC value is required before automated judge scores can substitute for human labels? How to set your own threshold based on the consequences of a wrong call in your deployment context."),
            ("Diagnosing disagreement patterns", "When the judge and humans diverge, is it systematic (a consistent directional bias you can correct) or idiosyncratic (random noise that requires more human labels to average out)?"),
        ],
    },
    {
        "slug": "ranking-noise",
        "nav_label": "When Rankings Flip",
        "tier": "Advanced",
        "title": "When Model Rankings Are Just Noise",
        "subtitle": "Leaderboard A says Model X is ranked #3. Leaderboard B says #7. This investigation shows how to compute the probability that any ranking is stable &mdash; and estimates how many published benchmark &ldquo;improvements&rdquo; are indistinguishable from chance.",
        "intro": "Leaderboards present rankings as if they were precise measurements. They are not. Any ranking computed from a finite eval set carries sampling uncertainty. Two models separated by less than the CI width of their score difference are statistically tied &mdash; regardless of which number is larger.",
        "learns": [
            ("Computing confidence intervals on ranks, not just scores", "How to bootstrap the full ranking and compute a CI on each model&rsquo;s rank position, separate from its score CI."),
            ("Testing whether adjacent rankings are distinguishable", "How to directly test whether Model #3 and Model #4 are actually distinguishable at the stated confidence level."),
            ("Estimating the fraction of &ldquo;real&rdquo; improvements in a benchmark history", "Given a leaderboard&rsquo;s history of score improvements, how many are outside the CI noise floor? What does this look like for a typical academic NLP benchmark?"),
            ("How N and leaderboard size interact", "How the required N to reliably rank k models scales with k, and why large leaderboards with small eval sets produce mostly noise."),
        ],
    },
    {
        "slug": "benchmark-distillation",
        "nav_label": "Distilling a Benchmark",
        "tier": "Advanced",
        "title": "Distilling a Benchmark",
        "subtitle": "You have 2,000 eval items but budget for 200. Which 200 carry the most statistical information? Naive random sampling wastes coverage. This investigation covers principled methods for building compact, representative eval subsets that preserve the full benchmark&rsquo;s discriminative power.",
        "intro": "Large eval sets are expensive to run and maintain. But shrinking them arbitrarily &mdash; by grabbing a random slice &mdash; risks discarding the items that most sharply differentiate models. Benchmark distillation is the principled alternative: using pilot data to identify which items contribute independent statistical information, preserve difficulty coverage, and discriminate between models that would otherwise appear identical. This investigation covers the core techniques, from stratified reduction through IRT-inspired item selection, and shows how to validate that a compact benchmark is trustworthy.",
        "learns": [
            ("Identifying redundant items", "How to detect items whose outcomes are highly correlated with other items across a model population, and how removing them shrinks the benchmark without losing coverage or discriminative power."),
            ("Stratified reduction strategies", "How to preserve the difficulty distribution and category coverage of the full benchmark while dramatically reducing item count &mdash; so the compact set doesn&rsquo;t accidentally over-represent easy or hard cases."),
            ("IRT-inspired item selection", "A practical introduction to item response theory: using pilot data to estimate item discrimination and difficulty, then selecting the items that best differentiate model capabilities across the ability range."),
            ("Validating the distilled benchmark", "How to measure whether your reduced set produces scores that correlate tightly with full-benchmark scores across a range of held-out models, and the minimum correlation you should require before relying on the compact version."),
        ],
    },
]

BOOK_SVG       = '<svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.5A1.5 1.5 0 012.5 1h11A1.5 1.5 0 0115 2.5v11a1.5 1.5 0 01-1.5 1.5h-11A1.5 1.5 0 011 13.5v-11zm1.5 0v11h11v-11h-11zM4 4h8v1H4V4zm0 3h8v1H4V7zm0 3h5v1H4v-1z"/></svg>'
WHICH_METHOD_SVG = '<svg width="11" height="11" viewBox="0 0 16 16"><rect x="2" y="2" width="12" height="12" fill="#e0e7ef"/><path d="M4 4h8v1H4V4zm0 3h8v1H4V7zm0 3h8v1H4v-1z" fill="#3b4a6b"/><path d="M8 1v14" stroke="#3b4a6b" stroke-width="0.7"/></svg>'
PAPERS_SVG     = '<svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor"><path d="M5 0h8a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2V2a2 2 0 012-2zm0 1a1 1 0 00-1 1v10a1 1 0 001 1h8a1 1 0 001-1V2a1 1 0 00-1-1H5zM1 4a1 1 0 00-1 1v10a1 1 0 001 1h8a1 1 0 001-1v-1h1v1a2 2 0 01-2 2H1a2 2 0 01-2-2V5a2 2 0 012-2h1v1H1z"/></svg>'
PRINCIPLES_SVG = '<svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor"><path d="M5 3.5h8a.5.5 0 0 1 0 1H5a.5.5 0 0 1 0-1zm0 4h8a.5.5 0 0 1 0 1H5a.5.5 0 0 1 0-1zm0 4h8a.5.5 0 0 1 0 1H5a.5.5 0 0 1 0-1zM2.5 4a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1zm0 4a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1zm0 4a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1z"/></svg>'
ROADMAP_SVG    = '<svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor"><path d="M8 16s6-5.686 6-10A6 6 0 0 0 2 6c0 4.314 6 10 6 10zm0-7a3 3 0 1 1 0-6 3 3 0 0 1 0 6z"/></svg>'

SLUGS = [inv["slug"] for inv in INVESTIGATIONS]

TIER_ORDER = ["Foundations", "Going Deeper", "Advanced"]


def _default_disabled_slugs():
    """Return investigation slugs that do not yet have notebook content."""
    notebooks_dir = os.path.join(os.path.dirname(__file__), "notebooks")
    disabled = set()
    for inv in INVESTIGATIONS:
        slug = inv["slug"]
        nb_path = os.path.join(notebooks_dir, f"{slug}.ipynb")
        if not os.path.exists(nb_path):
            disabled.add(slug)
    return disabled

def make_nav(active_slug, prefix="../", disabled_slugs=None):
    """Build the left investigations sidebar.

    active_slug: slug of the current investigation, OR one of
                 "index" | "resources" | "choose" to highlight a guide link.
    prefix:      relative path prefix to reach the site root.
                 "../" for investigation pages, "./" for top-level pages.
    disabled_slugs: optional set of investigation slugs to render as disabled.
    """
    if disabled_slugs is None:
        disabled_slugs = _default_disabled_slugs()

    groups = {}
    for inv in INVESTIGATIONS:
        groups.setdefault(inv["tier"], []).append(inv)

    # Guide links: (key, svg, href, label)
    guide_links = [
        ("index",      BOOK_SVG,           f"{prefix}index.html",       "About"),
        ("principles", PRINCIPLES_SVG,     f"{prefix}principles.html",  "Principles"),
        ("which-method", WHICH_METHOD_SVG, f"{prefix}which-method.html", "Which Method?"),
        ("roadmap",    ROADMAP_SVG,        f"{prefix}roadmap.html",     "Roadmap"),
        ("resources",  PAPERS_SVG,         f"{prefix}resources.html",   "Resources"),
    ]

    # Investigation link path depends on depth
    def inv_href(slug):
        if prefix == "../":
            return f"./{slug}.html"
        return f"{prefix}investigations/{slug}.html"

    lines = []
    lines.append('  <nav class="inv-nav" aria-label="Investigations">')
    lines.append('    <div class="inv-nav-header">Guide</div>')

    for key, svg, href, label in guide_links:
        active_cls = " active" if (active_slug == key or (key == "which-method" and active_slug == "which-method")) else ""
        lines.append(f'    <a class="inv-nav-guide-link{active_cls}" href="{href}">')
        lines.append(f'      {svg}')
        lines.append(f'      {label}')
        lines.append(f'    </a>')

    lines.append('    <div class="inv-nav-guide-divider"></div>')
    lines.append('    <div class="inv-nav-header inv-nav-header-section">Investigations</div>')

    for tier in TIER_ORDER:
        if tier not in groups:
            continue
        lines.append('')
        lines.append('    <div class="inv-group">')
        lines.append(f'      <div class="inv-group-label">{tier}</div>')
        lines.append('      <ul>')
        for inv in groups[tier]:
            active = ' class="active"' if inv["slug"] == active_slug else ''
            label = inv.get("nav_label", inv["title"].split(":")[0])
            if inv["slug"] in disabled_slugs:
                disabled_cls = ' class="inv-link-disabled active"' if inv["slug"] == active_slug else ' class="inv-link-disabled"'
                lines.append(
                    f'        <li><span{disabled_cls} data-tooltip="Coming soon" '
                    f'aria-label="{label} (coming soon)">{label}</span></li>'
                )
            else:
                lines.append(f'        <li><a href="{inv_href(inv["slug"])}"{active}>{label}</a></li>')
        lines.append('      </ul>')
        lines.append('    </div>')

    lines.append('  </nav>')
    return "\n".join(lines)


def make_page(inv):
    slug = inv["slug"]
    tier = inv["tier"]
    title = inv["title"]
    subtitle = inv["subtitle"]
    intro = inv["intro"]
    learns = inv["learns"]

    # Build learn grid
    learn_items = []
    for i, (label, desc) in enumerate(learns, 1):
        learn_items.append(f"""\
        <div class="learn-item">
          <div class="learn-item-num">{i:02d}</div>
          <p><strong>{label}</strong> &mdash; {desc}</p>
        </div>""")
    learn_grid = "\n".join(learn_items)

    nav_html = make_nav(slug)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title} &mdash; Stats for LLM Evals</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Serif:ital,wght@0,300;0,400;0,600;0,700;1,300&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="../inv.css" />
</head>
<body>

<!-- NAV -->
<nav class="site-nav">
  <div class="nav-inner">
    <a class="nav-brand" href="../index.html">Stats for LLM Evals</a>
    <ul class="nav-links">
      <li><a href="../index.html#why-statistics">Why Statistics</a></li>
      <li><a href="../index.html#principles">Core Principles</a></li>
      <li><a href="../index.html#simulation">Simulation Study</a></li>
      <li><a href="../index.html#recommendations">Recommendations</a></li>
      <li><a href="../choose.html">Choose a Method</a></li>
      <li><a href="../index.html#promptstats">promptstats</a></li>
    </ul>
  </div>
</nav>

<!-- INVESTIGATION HEADER -->
<div class="inv-header">
  <a class="inv-back" href="../index.html">&larr; Stats Reference Guide</a>
  <div class="inv-eyebrow">{tier} Investigation</div>
  <h1>{title}</h1>
  <p class="inv-subtitle">{subtitle}</p>
</div>

<!-- MAIN LAYOUT -->
<div class="page-layout">

{nav_html}

  <!-- Content -->
  <article class="article-body">
    <section>
      <h2>What This Investigation Covers</h2>
      <p>{intro}</p>

      <h3>What you&rsquo;ll learn</h3>
      <div class="learn-grid">
{learn_grid}
      </div>

      <div class="coming-soon-card">
        <div class="cs-icon">&#9879;</div>
        <div class="cs-label">Investigation in progress</div>
        <p class="cs-text">Full worked examples, interactive code, and simulation-backed results are coming soon. <a href="https://github.com/ianarawjo/promptstats">Follow on GitHub</a> for updates.</p>
      </div>
    </section>
  </article>

</div><!-- /page-layout -->

<!-- FOOTER -->
<footer class="site-footer">
  <div class="footer-inner">
    <p>Statistics for LLM Evals &middot; A living document from the <a href="https://github.com/ianarawjo/promptstats">promptstats</a> project</p>
    <ul class="footer-links">
      <li><a href="https://github.com/ianarawjo/promptstats">GitHub</a></li>
      <li><a href="https://pypi.org/project/promptstats/">PyPI</a></li>
    </ul>
  </div>
</footer>

</body>
</html>
"""


import os

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "investigations")
    os.makedirs(out_dir, exist_ok=True)

    for inv in INVESTIGATIONS:
        html = make_page(inv)
        path = os.path.join(out_dir, f"{inv['slug']}.html")
        with open(path, "w") as f:
            f.write(html)
        print(f"Wrote {path}")

    print("Done.")
