#!/usr/bin/env python3
"""
Build investigation pages from Jupyter notebooks (or stubs for pages without one).

For each investigation slug:
  - If website/notebooks/<slug>.ipynb exists → convert with nbconvert → full page
  - Otherwise → render coming-soon stub

Usage:
    python website/build.py                        # rebuild all pages
    python website/build.py model-vs-model         # rebuild one page
    python website/build.py --execute              # also run notebooks before building
"""

import os
import sys
import argparse

WEBSITE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(WEBSITE_DIR, "notebooks")
OUT_DIR = os.path.join(WEBSITE_DIR, "investigations")

sys.path.insert(0, WEBSITE_DIR)
from gen_stubs import INVESTIGATIONS, make_nav

GITHUB_BASE = "https://github.com/ianarawjo/prompt-stats/blob/main/website/notebooks"
COLAB_BASE  = "https://colab.research.google.com/github/ianarawjo/prompt-stats/blob/main/website/notebooks"

# ---------------------------------------------------------------------------
# Notebook → HTML fragment
# ---------------------------------------------------------------------------

def nb_to_html(nb_path, execute=False):
    """Return (html_fragment, had_error)."""
    import nbformat
    from nbconvert import HTMLExporter

    nb = nbformat.read(nb_path, as_version=4)

    if execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": NOTEBOOKS_DIR}})
        # Save executed notebook back
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

    exporter = HTMLExporter(template_name="basic")
    body, _resources = exporter.from_notebook_node(nb)
    return body


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

SITE_NAV_HTML = """\
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
    <button class="dark-toggle" id="dark-toggle" aria-label="Toggle dark mode">
      <svg id="dark-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
      </svg>
    </button>
  </div>
</nav>"""

FOOTER_HTML = """\
<footer class="site-footer">
  <div class="footer-inner">
    <p>Statistics for LLM Evals &middot; A living document from the
       <a href="https://github.com/ianarawjo/prompt-stats">promptstats</a> project</p>
    <ul class="footer-links">
      <li><a href="https://github.com/ianarawjo/prompt-stats">GitHub</a></li>
      <li><a href="https://pypi.org/project/promptstats/">PyPI</a></li>
    </ul>
  </div>
</footer>"""


def make_stub_content(inv):
    learns = inv["learns"]
    learn_items = []
    for i, (label, desc) in enumerate(learns, 1):
        learn_items.append(f"""\
      <div class="learn-item">
        <div class="learn-item-num">{i:02d}</div>
        <p><strong>{label}</strong> &mdash; {desc}</p>
      </div>""")
    learn_grid = "\n".join(learn_items)

    return f"""
    <section>
      <h2>What This Investigation Covers</h2>
      <p>{inv["intro"]}</p>

      <h3>What you&rsquo;ll learn</h3>
      <div class="learn-grid">
{learn_grid}
      </div>

      <div class="coming-soon-card">
        <div class="cs-icon">&#9879;</div>
        <div class="cs-label">Investigation in progress</div>
        <p class="cs-text">Full worked examples, interactive code, and simulation-backed
          results are coming soon.
          <a href="https://github.com/ianarawjo/prompt-stats">Follow on GitHub</a>
          for updates.</p>
      </div>
    </section>"""


def make_notebook_content(nb_html, slug):
    nb_url    = f"{GITHUB_BASE}/{slug}.ipynb"
    colab_url = f"{COLAB_BASE}/{slug}.ipynb"
    return f"""
    <section>
      <div class="nb-toolbar">
        <a class="nb-badge" href="{colab_url}" target="_blank" rel="noopener">
          <img src="https://colab.research.google.com/assets/colab-badge.svg"
               alt="Open In Colab" height="20" />
        </a>
        <a class="nb-badge-link" href="{nb_url}" target="_blank" rel="noopener"
           >View .ipynb on GitHub</a>
      </div>
      <div class="nb-content">
        {nb_html}
      </div>
    </section>"""


def make_page(inv, content_html, has_notebook=False):
    slug     = inv["slug"]
    tier     = inv["tier"]
    title    = inv["title"]
    subtitle = inv["subtitle"]
    nav_html = make_nav(slug)
    nb_css   = '  <link rel="stylesheet" href="../nb.css" />' if has_notebook else ""

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
{nb_css}
  <script>(function(){{try{{var t=localStorage.getItem('theme');if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme: dark)').matches))document.documentElement.setAttribute('data-theme','dark')}}catch(e){{}}}})()</script>
</head>
<body>

{SITE_NAV_HTML}

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
{content_html}
  </article>

</div><!-- /page-layout -->

{FOOTER_HTML}

<script src="../dark.js"></script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(slugs=None, execute=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

    inv_by_slug = {inv["slug"]: inv for inv in INVESTIGATIONS}
    targets = {s: inv_by_slug[s] for s in slugs if s in inv_by_slug} if slugs else inv_by_slug

    for slug, inv in targets.items():
        nb_path      = os.path.join(NOTEBOOKS_DIR, f"{slug}.ipynb")
        has_notebook = os.path.exists(nb_path)

        if has_notebook:
            print(f"  [notebook] {slug}")
            try:
                nb_html  = nb_to_html(nb_path, execute=execute)
                content  = make_notebook_content(nb_html, slug)
            except Exception as e:
                print(f"    WARNING: nbconvert failed ({e}), falling back to stub")
                content      = make_stub_content(inv)
                has_notebook = False
        else:
            print(f"  [stub]     {slug}")
            content = make_stub_content(inv)

        html     = make_page(inv, content, has_notebook=has_notebook)
        out_path = os.path.join(OUT_DIR, f"{slug}.html")
        with open(out_path, "w") as f:
            f.write(html)

    print(f"Built {len(targets)} page(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("slugs", nargs="*", help="Slugs to build (default: all)")
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute each notebook before converting (requires kernel + deps)",
    )
    args = parser.parse_args()
    build(args.slugs or None, execute=args.execute)
