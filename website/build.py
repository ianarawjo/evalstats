#!/usr/bin/env python3
"""
Build all site pages from source files in website/src/ and website/notebooks/.

Top-level pages (index, choose, resources, principles, roadmap, …):
  - Source body lives in website/src/<slug>.html
  - Shared nav/footer/head are injected automatically
  - Add a new page: create src/<slug>.html + entry in PAGE_CONFIGS

Investigation pages (website/investigations/<slug>.html):
  - If website/notebooks/<slug>.ipynb exists → convert with nbconvert → full page
  - Otherwise → render coming-soon stub

Usage:
    python website/build.py                        # rebuild everything
    python website/build.py model-vs-model         # rebuild one investigation
    python website/build.py --pages                # rebuild top-level pages only
    python website/build.py --execute              # also run notebooks before building
"""

import os
import sys
import argparse
import re
import html as html_lib
import shutil

WEBSITE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(WEBSITE_DIR, "notebooks")
BUILD_DIR = os.path.join(WEBSITE_DIR, "build")
OUT_DIR = os.path.join(BUILD_DIR, "investigations")

STATIC_FILES = ["choose.css", "index.css", "inv.css", "nb.css", "dark.js"]
STATIC_DIRS = ["media"]

sys.path.insert(0, WEBSITE_DIR)
from gen_stubs import INVESTIGATIONS, make_nav

GITHUB_BASE = "https://github.com/ianarawjo/promptstats/blob/main/website/notebooks"
COLAB_BASE  = "https://colab.research.google.com/github/ianarawjo/promptstats/blob/main/website/notebooks"

# ---------------------------------------------------------------------------
# Notebook → HTML fragment
# ---------------------------------------------------------------------------

def nb_to_html(nb_path, execute=False):
    """Return (html_fragment, had_error)."""
    from traitlets.config import Config
    import nbformat
    from nbconvert import HTMLExporter

    nb = nbformat.read(nb_path, as_version=4)

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", "")
        if "ci_widget_html = r'''<!DOCTYPE html>" not in source:
            continue

        metadata = cell.setdefault("metadata", {})
        tags = metadata.setdefault("tags", [])
        if "remove-input" not in tags:
            tags.append("remove-input")

        match = re.search(r"ci_widget_html\s*=\s*r'''(.*?)'''", source, flags=re.DOTALL)
        if not match:
            continue

        widget_html = match.group(1)
        iframe_srcdoc = html_lib.escape(widget_html, quote=True)
        iframe_html = (
            "<iframe "
            "title=\"Confidence interval visualizer\" "
            "style=\"width: 100%; max-width: 860px; height: 620px; border: 1px solid #d1d5db; border-radius: 8px;\" "
            "sandbox=\"allow-scripts\" "
            f"srcdoc=\"{iframe_srcdoc}\">"
            "</iframe>"
        )

        cell["outputs"] = [
            {
                "output_type": "display_data",
                "data": {
                    "text/html": iframe_html,
                },
                "metadata": {},
            }
        ]

    if execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": NOTEBOOKS_DIR}})
        # Save executed notebook back
        with open(nb_path, "w") as f:
            nbformat.write(nb, f)

    config = Config()
    config.HTMLExporter.preprocessors = [
        "nbconvert.preprocessors.TagRemovePreprocessor",
    ]
    config.TagRemovePreprocessor.enabled = True
    config.TagRemovePreprocessor.remove_input_tags = ["remove-input"]
    exporter = HTMLExporter(template_name="basic", config=config)
    body, _resources = exporter.from_notebook_node(nb)
    return body


# ---------------------------------------------------------------------------
# Shared HTML fragments — edit here to change every page at once
# ---------------------------------------------------------------------------

IBM_PLEX_FONTS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400"
    "&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400"
    "&family=IBM+Plex+Serif:ital,wght@0,300;0,400;0,600;0,700;1,300&display=swap"
)

DARK_MODE_DETECT = (
    "(function(){try{var t=localStorage.getItem('theme');"
    "if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme: dark)').matches))"
    "document.documentElement.setAttribute('data-theme','dark')}catch(e){}})()"
)

FOOTER_HTML = """\
<footer class="site-footer">
  <div class="footer-inner">
    <p>Statistics for LLM Evals &middot; A living document from the
       <a href="https://github.com/ianarawjo/promptstats">promptstats</a> project</p>
    <ul class="footer-links">
      <li><a href="https://github.com/ianarawjo/promptstats">GitHub</a></li>
      <li><a href="https://pypi.org/project/promptstats/">PyPI</a></li>
    </ul>
  </div>
</footer>"""

# Canonical top-navigation links.
# To add a new nav entry: append (display_text, href_template, active_key) here.
# href_template uses {p} as the relative-path prefix ("./" or "../").
# active_key matches the page config's "active_nav" field; use None for anchor-only links.
_NAV_LINKS = [
    ("Why Statistics",   "{p}index.html#why-statistics",  None),
    ("Core Principles",  "{p}index.html#principles",      None),
    ("Simulation Study", "{p}index.html#simulation",      None),
    ("Recommendations", "{p}index.html#recommendations", None),
    ("Choose a Method",  "{p}choose.html",                "choose"),
    ("Which Method?",    "{p}which-method.html",          "which-method"),
    ("Resources",        "{p}resources.html",             "resources"),
    ("promptstats",      "{p}index.html#promptstats",     None),
]


def make_site_nav_html(prefix="./", active=None):
    """Build the top navigation bar HTML.

    prefix: "./" for top-level pages, "../" for investigation pages.
    active: active_key string for the current page (e.g. "choose", "resources").
    """
    items = []
    for text, href_tmpl, key in _NAV_LINKS:
        href = href_tmpl.replace("{p}", prefix)
        active_cls = ' class="active"' if (key and key == active) else ""
        items.append(f'      <li><a href="{href}"{active_cls}>{text}</a></li>')
    items_html = "\n".join(items)
    return f"""\
<nav class="site-nav">
  <div class="nav-inner">
    <a class="nav-brand" href="{prefix}index.html">Stats for LLM Evals</a>
    <ul class="nav-links">
{items_html}
    </ul>
    <button class="dark-toggle" id="dark-toggle" aria-label="Toggle dark mode">
      <svg id="dark-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
      </svg>
    </button>
  </div>
</nav>"""


def make_head(title_tag, css_file, prefix="./", extra_css=""):
    """Build the <head> block for any page."""
    extra = f"\n  {extra_css}" if extra_css else ""
    return f"""\
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title_tag}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="{IBM_PLEX_FONTS}" rel="stylesheet" />{extra}
  <link rel="stylesheet" href="{prefix}{css_file}" />
  <script>{DARK_MODE_DETECT}</script>
</head>"""


# ---------------------------------------------------------------------------
# Top-level page configs — add new pages here
# ---------------------------------------------------------------------------
# Each dict describes one page built from website/src/<slug>.html.
#
# Required keys:
#   slug        Output filename (website/build/<slug>.html) and src/<slug>.html source.
#   title_tag   Contents of <title>. For "article" type also used as <h1>.
#   type        "full"    → src content placed directly between nav and footer.
#               "article" → src content placed inside an inv-header + page-layout.
#   css         CSS file (relative to website/), e.g. "inv.css" or "index.css".
#
# Optional keys (all types):
#   active_nav  Matches _NAV_LINKS active_key to highlight a top nav link.
#
# Optional keys (article type only):
#   title       <h1> text if different from title_tag.
#   eyebrow     Small label above the title. Default: "Reference".
#   subtitle    Subtitle paragraph. Default: "".
#   active_sidebar  Passed to make_nav() as active_slug for the left sidebar.
PAGE_CONFIGS = [
    {
        "slug":       "index",
        "title_tag":  "Statistics for LLM Evals",
        "type":       "full",
        "css":        "index.css",
        "active_nav": None,
    },
    {
        "slug":       "which-method",
        "title_tag":  "Which Method? — Stats for LLM Evals",
        "title":      "Which Method?",
        "type":       "article",
        "css":        "inv.css",
        "active_nav": "which-method",
        "eyebrow":    "Guide",
        "subtitle":   "A concise summary of recommended statistical methods for LLM evaluation, organized by data type and analysis type.",
        "active_sidebar": "which-method",
    },
    {
        "slug":       "choose",
        "title_tag":  "Choose Your Statistical Method \u2014 Stats for LLM Evals",
        "type":       "full",
        "css":        "choose.css",
        "active_nav": "choose",
    },
    {
        "slug":          "resources",
        "title_tag":     "Resources \u2014 Stats for LLM Evals",
        "title":         "Resources",
        "type":          "article",
        "css":           "inv.css",
        "active_nav":    "resources",
        "eyebrow":       "Reference",
        "subtitle":      ("A curated survey of the key papers behind the methods and arguments "
                          "in this guide \u2014 from foundational bootstrap theory to LLM eval "
                          "methodology and statistical pitfalls."),
        "active_sidebar": "resources",
    },
    {
        "slug":           "principles",
        "title_tag":      "Principles \u2014 Stats for LLM Evals",
        "title":          "Principles",
        "type":           "article",
        "css":            "inv.css",
        "active_nav":     None,
        "eyebrow":        "Guide",
        "subtitle":       "Declaring some principles and philosophy to guide our choices.",
        "active_sidebar": "principles",
    },
    {
        "slug":           "roadmap",
        "title_tag":      "Roadmap \u2014 Stats for LLM Evals",
        "title":          "Roadmap",
        "type":           "article",
        "css":            "inv.css",
        "active_nav":     None,
        "eyebrow":        "Project",
        "subtitle":       "Planned additions to the guide and the promptstats library.",
        "active_sidebar": "roadmap",
    },
]


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
          <a href="https://github.com/ianarawjo/promptstats">Follow on GitHub</a>
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
    nav_html = make_nav(slug, prefix="../")
    nb_css   = '<link rel="stylesheet" href="../nb.css" />' if has_notebook else ""
    head_html = make_head(
        title_tag=f"{title} \u2014 Stats for LLM Evals",
        css_file="inv.css",
        prefix="../",
        extra_css=nb_css,
    )
    top_nav = make_site_nav_html(prefix="../", active=None)

    return f"""<!DOCTYPE html>
<html lang="en">
{head_html}
<body>

{top_nav}

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
# Top-level page builder
# ---------------------------------------------------------------------------

def build_pages(slugs=None):
    """Build top-level HTML pages from website/src/<slug>.html source files."""
    src_dir = os.path.join(WEBSITE_DIR, "src")
    configs = {p["slug"]: p for p in PAGE_CONFIGS}
    targets = {s: configs[s] for s in slugs if s in configs} if slugs else configs

    for slug, page in targets.items():
        src_path = os.path.join(src_dir, f"{slug}.html")
        if not os.path.exists(src_path):
            print(f"  [skip]     {slug} (no src/{slug}.html)")
            continue

        with open(src_path) as f:
            src_content = f.read()

        # Keep sidebar content centralized in gen_stubs.py for index as well.
        if slug == "index":
            src_content = src_content.replace(
                "{{INVESTIGATIONS_NAV}}",
                make_nav(active_slug="index", prefix="./"),
            )

        prefix    = "./"
        active    = page.get("active_nav")
        top_nav   = make_site_nav_html(prefix=prefix, active=active)
        head_html = make_head(page["title_tag"], page["css"], prefix=prefix)

        if page["type"] == "full":
            html = f"""<!DOCTYPE html>
<html lang="en">
{head_html}
<body>

{top_nav}

{src_content}
{FOOTER_HTML}

<script src="{prefix}dark.js"></script>
</body>
</html>
"""
        else:  # "article"
            sidebar  = make_nav(page.get("active_sidebar"), prefix=prefix)
            eyebrow  = page.get("eyebrow", "Reference")
            subtitle = page.get("subtitle", "")
            title    = page.get("title", page["title_tag"])
            html = f"""<!DOCTYPE html>
<html lang="en">
{head_html}
<body>

{top_nav}

<!-- HEADER -->
<div class="inv-header">
  <a class="inv-back" href="{prefix}index.html">&larr; Stats Reference Guide</a>
  <div class="inv-eyebrow">{eyebrow}</div>
  <h1>{title}</h1>
  <p class="inv-subtitle">{subtitle}</p>
</div>

<!-- MAIN LAYOUT -->
<div class="page-layout">

{sidebar}

  <!-- Content -->
  <article class="article-body">
{src_content}
  </article>

</div><!-- /page-layout -->

{FOOTER_HTML}

<script src="{prefix}dark.js"></script>
</body>
</html>
"""

        out_path = os.path.join(BUILD_DIR, f"{slug}.html")
        with open(out_path, "w") as f:
            f.write(html)
        print(f"  [page]     {slug}")

    print(f"Built {len(targets)} top-level page(s).")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def prepare_build_dir():
    """Create build directory and copy static assets needed by generated pages."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    for filename in STATIC_FILES:
        src = os.path.join(WEBSITE_DIR, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(BUILD_DIR, filename))

    for dirname in STATIC_DIRS:
        src = os.path.join(WEBSITE_DIR, dirname)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(BUILD_DIR, dirname), dirs_exist_ok=True)

def build(slugs=None, execute=False):
    prepare_build_dir()
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
    parser.add_argument(
        "slugs", nargs="*",
        help="Slugs to build — investigation slugs and/or top-level page slugs (default: all)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute each notebook before converting (requires kernel + deps)",
    )
    parser.add_argument(
        "--pages", action="store_true",
        help="Build top-level pages only (skip investigation pages)",
    )
    args = parser.parse_args()

    page_slugs = {p["slug"] for p in PAGE_CONFIGS}

    if args.slugs:
        inv_slugs  = [s for s in args.slugs if s not in page_slugs]
        top_slugs  = [s for s in args.slugs if s in page_slugs]
        if inv_slugs:
            build(inv_slugs, execute=args.execute)
        if top_slugs:
            prepare_build_dir()
            build_pages(top_slugs)
    elif args.pages:
        prepare_build_dir()
        build_pages()
    else:
        build(execute=args.execute)
        build_pages()
