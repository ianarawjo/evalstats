# Website Build Structure

This file describes the build structure of the Stats for Evals website.

## Overview

Custom Python-based static site generator. No framework (no Next.js, Jekyll, Hugo).
- **Build script**: `website/build.py` (~843 lines)
- **Output**: `website/build/` (do NOT hand-edit — regenerated on each build)
- **Dev server**: `python3 -m http.server 8743` (configured in `.claude/launch.json`)

## Build Commands

```bash
# From repo root (prompt-stats/)
python website/build.py                    # Rebuild everything
python website/build.py --pages            # Top-level pages only
python website/build.py model-vs-model     # One investigation by slug
python website/build.py resources          # One top-level page by name
python website/build.py --execute          # Also re-execute notebooks
```

## Directory Layout

```
website/
├── build.py            # Main build script — edit this for layout/template changes
├── gen_stubs.py        # INVESTIGATIONS list + nav sidebar generation
├── BUILDING.md         # Human-readable build docs
│
├── src/                # Source HTML body fragments (one per top-level page)
│   ├── index.html
│   ├── choose.html
│   ├── resources.html
│   ├── principles.html
│   ├── roadmap.html
│   ├── usage.html
│   └── which-method.html
│
├── notebooks/          # Jupyter notebooks → converted to investigation pages
│   ├── model-vs-model.ipynb
│   ├── best-prompt.ipynb
│   └── *.csv           # Eval data for notebooks
│
├── media/              # Source images/video (copied to build/media/)
│
├── *.css               # Stylesheets (root level, copied to build/)
│   ├── index.css       # Index page
│   ├── choose.css      # Interactive choice tool
│   ├── inv.css         # Investigation + article pages (shared)
│   └── nb.css          # Notebook content styles
│
├── dark.js             # Dark mode toggle (copied to build/)
│
└── build/              # GENERATED OUTPUT — do not edit, do not read
    ├── *.html          # Top-level pages
    ├── investigations/ # One .html per investigation
    ├── *.css, dark.js  # Copied static assets
    ├── media/          # Copied media files
    ├── robots.txt
    └── sitemap.xml
```

## Key Concepts

### Page Types
- **"full"** pages: Hero layout with optional sidebar (index, choose)
- **"article"** pages: Header + left sidebar + main content (resources, principles, roadmap, usage, which-method)
- **Investigation pages**: Built from Jupyter notebooks or generated as "coming soon" stubs

### Shared Components (defined in build.py)
- `make_head()` — `<head>` block with fonts, meta tags, JSON-LD
- `make_site_nav_html()` — top navigation bar
- `FOOTER_HTML` — shared footer
- `_NAV_LINKS` — canonical nav links (single source of truth)

### Investigation Pages
- Slugs and metadata defined in `gen_stubs.py` → `INVESTIGATIONS` list
- If `notebooks/<slug>.ipynb` exists → converted via `nbconvert` and embedded
- If no notebook → "coming soon" stub is generated
- `gen_stubs.make_nav()` generates the left sidebar for investigation pages

### Static Files
Defined in `build.py`:
- `STATIC_FILES = ["choose.css", "index.css", "inv.css", "nb.css", "dark.js"]`
- `STATIC_DIRS = ["media"]`

## Common Tasks

| Task | Where to look |
|------|--------------|
| Change page content | `website/src/<page>.html` |
| Change nav links | `_NAV_LINKS` in `build.py` |
| Add/edit investigation | `INVESTIGATIONS` in `gen_stubs.py`; add notebook to `notebooks/` |
| Change page layout/template | `build.py` (`make_head`, `make_site_nav_html`, `FOOTER_HTML`) |
| Change styles | `website/*.css` files |
| Add a new top-level page | Add entry to `PAGE_CONFIGS` in `build.py` + new file in `src/` |
| Check build output | `website/build/` |
