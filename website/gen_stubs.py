#!/usr/bin/env python3
"""Generate investigation stub HTML pages."""

INVESTIGATIONS = [
    {
        "slug": "model-vs-model",
        "tier": "Foundations",
        "title": "Model A vs. Model B: Is the Gap Real?",
        "subtitle": "You ran both models on your eval set. Model A scored 78%, Model B scored 75%. Is that a real difference? With N=50, the 95% CI on that 3-point gap often spans zero. Here&rsquo;s how to find out &mdash; and what to do when it does.",
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
        "slug": "model-prompt-grid",
        "tier": "Foundations",
        "title": "Model &times; Prompt Grid: What Actually Wins?",
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
        "slug": "judge-audit",
        "tier": "Advanced",
        "title": "Auditing Your LLM Judge",
        "subtitle": "Your eval pipeline uses an LLM to score outputs. But how consistent is it? How much variance does it add? This investigation measures judge reliability and shows when judge noise dominates sample noise.",
        "intro": "LLM-as-judge is now the dominant eval paradigm, yet most pipelines treat judge scores as ground truth. They are not. The judge is a stochastic scorer with its own variance, biases, and failure modes. Auditing the judge is a prerequisite for trusting any eval result that depends on it.",
        "learns": [
            ("Measuring judge consistency", "How to estimate judge variance by scoring the same outputs multiple times under identical prompts, and how to compute an intra-class correlation (ICC) for the judge."),
            ("Computing judge-induced CI inflation", "How to propagate judge variance into your model comparison CIs &mdash; the extra uncertainty that comes from the scorer, not the sample."),
            ("Detecting systematic judge biases", "Common biases to test for: position bias (favoring the first response in pairwise comparison), verbosity bias (preferring longer responses), and self-enhancement bias (a model judging its own outputs favorably)."),
            ("When to trust judge scores vs. fall back to human annotation", "How to compute the minimum ICC threshold above which automated scoring is a reliable proxy for human judgment."),
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
]

BOOK_SVG = '<svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor"><path d="M1 2.5A1.5 1.5 0 012.5 1h11A1.5 1.5 0 0115 2.5v11a1.5 1.5 0 01-1.5 1.5h-11A1.5 1.5 0 011 13.5v-11zm1.5 0v11h11v-11h-11zM4 4h8v1H4V4zm0 3h8v1H4V7zm0 3h5v1H4v-1z"/></svg>'

SLUGS = [inv["slug"] for inv in INVESTIGATIONS]

TIER_ORDER = ["Foundations", "Going Deeper", "Advanced"]

def make_nav(active_slug):
    groups = {}
    for inv in INVESTIGATIONS:
        groups.setdefault(inv["tier"], []).append(inv)

    lines = []
    lines.append('  <nav class="inv-nav" aria-label="Investigations">')
    lines.append('    <div class="inv-nav-header">Investigations</div>')
    lines.append(f'    <a class="inv-nav-guide-link" href="../index.html">')
    lines.append(f'      {BOOK_SVG}')
    lines.append(f'      Stats Reference Guide')
    lines.append(f'    </a>')

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
            lines.append(f'        <li><a href="./{inv["slug"]}.html"{active}>{label}</a></li>')
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
        <p class="cs-text">Full worked examples, interactive code, and simulation-backed results are coming soon. <a href="https://github.com/ianarawjo/prompt-stats">Follow on GitHub</a> for updates.</p>
      </div>
    </section>
  </article>

</div><!-- /page-layout -->

<!-- FOOTER -->
<footer class="site-footer">
  <div class="footer-inner">
    <p>Statistics for LLM Evals &middot; A living document from the <a href="https://github.com/ianarawjo/prompt-stats">promptstats</a> project</p>
    <ul class="footer-links">
      <li><a href="https://github.com/ianarawjo/prompt-stats">GitHub</a></li>
      <li><a href="https://pypi.org/project/promptstats/">PyPI</a></li>
    </ul>
  </div>
</footer>

</body>
</html>
"""


import os

out_dir = os.path.join(os.path.dirname(__file__), "investigations")
os.makedirs(out_dir, exist_ok=True)

for inv in INVESTIGATIONS:
    html = make_page(inv)
    path = os.path.join(out_dir, f"{inv['slug']}.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"Wrote {path}")

print("Done.")
