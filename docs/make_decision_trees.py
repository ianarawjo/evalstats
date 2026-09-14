"""Render docs/decision-tree-{ci,pvalue}.png from the paper's decision-tree figure.

The two forest trees are copied verbatim out of the paper source, so the README
images match the paper. Needs pdflatex (with the forest package) and pdftoppm.

    python docs/make_decision_trees.py path/to/sample-sigconf.tex
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Citations become gray leaf notes, as there is no bibliography here.
CITATIONS = {
    r"~\cite{bonett2012adjusted}": r"\leafnote{Bonett \& Price, 2012}",
    r"~\cite{romano2005exact}": r"\leafnote{Romano \& Wolf, 2005}",
}

# The paper's tree style, with leaves widened for Computer Modern, which runs
# wider than the paper's font.
PREAMBLE = r"""\documentclass[border=10pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{xcolor}
\usepackage{forest}
\newlength{\leafwidth}\setlength{\leafwidth}{3.4cm}
\newcommand{\leafnote}[1]{\\{\scriptsize\textcolor{black!55}{\parbox{\leafwidth}{#1}}}}
\forestset{
  cmptree/.style={
    for tree={grow'=east, parent anchor=east, child anchor=west, anchor=west,
      align=left, font=\footnotesize, inner xsep=2.5pt, inner ysep=2pt,
      l sep=2.2mm, s sep=1.1mm, edge={gray!70, semithick},
      edge path={\noexpand\path[\forestoption{edge}]
        (!u.parent anchor) -- +(2mm,0) |- (.child anchor)\forestoption{edge label};},
    },
    where n children=0{draw=black!45, rounded corners=2pt, fill=blue!6,
      text width=\leafwidth, tier=rec}{font=\footnotesize\itshape},
  }
}
\begin{document}
"""


def extract_trees(tex: str) -> list[tuple[str, str]]:
    start = tex.index(r"{\footnotesize\textbf{(a) 95\% confidence interval}}")
    figure = tex[start:tex.index(r"\label{fig:ci-decision-tree}")]
    titles = re.findall(r"\{\\footnotesize\\textbf\{(.*?)\}\}\\\\\[3pt\]", figure)
    bodies = re.findall(r"\\begin\{forest\} cmptree\n(.*?)\\end\{forest\}", figure, flags=re.S)
    if len(titles) != 2 or len(bodies) != 2:
        raise ValueError(f"expected two titled trees, found {len(titles)} titles and {len(bodies)} trees")
    return list(zip(titles, bodies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", type=Path, help="the paper's main .tex file")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    parser.add_argument("--dpi", type=int, default=720)
    args = parser.parse_args()

    trees = extract_trees(args.paper.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        for name, (title, body) in zip(["decision-tree-ci", "decision-tree-pvalue"], trees):
            for cite, note in CITATIONS.items():
                body = body.replace(cite, note)
            if r"\cite" in body:
                raise ValueError(f"{name}: a citation has no leaf note in CITATIONS")
            doc = (PREAMBLE + "\\begin{tabular}{@{}l@{}}\n"
                   f"{{\\footnotesize\\textbf{{{title}}}}}\\\\[6pt]\n"
                   f"\\begin{{forest}} cmptree\n{body}\\end{{forest}}\n"
                   "\\end{tabular}\n\\end{document}\n")
            (Path(tmp) / f"{name}.tex").write_text(doc)
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
                           cwd=tmp, check=True, capture_output=True)
            subprocess.run(["pdftoppm", "-r", str(args.dpi), "-png", "-singlefile", f"{name}.pdf", name],
                           cwd=tmp, check=True)
            shutil.copy(Path(tmp) / f"{name}.png", args.out / f"{name}.png")
            print(f"wrote {args.out / f'{name}.png'}")


if __name__ == "__main__":
    main()
