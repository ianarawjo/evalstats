"""Rebuild compare_e2e's calibration plot from one or more saved results CSVs.

compare_e2e's plot is normally drawn from in-memory CompareE2EResult objects at
the end of a run, so there was no way to redraw it later -- or to draw ONE
figure from a run that was split across several invocations. The results CSV is
a lossless serialization of CompareE2EResult (every one of its fields is a
column; the rate columns are derived and simply recomputed), so both are just a
matter of reading it back.

The splitting case is the reason this exists. k=10 is ~76% of the grid's
pair-work and continuous is ~30% of it, so running one eval type at a time
makes the first panel available far sooner than a single all-eval-type run
would. Cells are independent Monte-Carlo estimates, so concatenating runs is
statistically sound; the only thing lost is reproducibility from a single
--seed, since each invocation spawns its own per-cell seeds (see run_case's
SeedSequence).

    python -m simulations.replot_compare_e2e a_results.csv b_results.csv -o fig.png

Duplicate cell keys across files are refused rather than silently pooled --
that would double-count a cell and quietly distort every rate it feeds.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulations.harness.cases.compare_e2e import (  # noqa: E402
    CompareE2EResult, save_calibration_plot,
)

_KEY = ("eval_type", "shape_label", "k", "n_items", "ppi_config", "is_null", "icc")


def _coerce(raw: str, typ):
    if typ is bool:
        return raw.strip().lower() in ("true", "1", "yes")
    if typ is int:
        return int(float(raw))
    if typ is float:
        return float(raw)
    return raw


def load_results(paths) -> list[CompareE2EResult]:
    fields = {f.name: f.type for f in dataclasses.fields(CompareE2EResult)}
    resolved = {n: {"int": int, "float": float, "bool": bool, "str": str}.get(
        t if isinstance(t, str) else getattr(t, "__name__", "str"), str)
        for n, t in fields.items()}
    out: list[CompareE2EResult] = []
    seen: dict[tuple, str] = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                missing = [n for n in fields if n not in row]
                if missing:
                    raise SystemExit(
                        f"{p}: not a compare_e2e results CSV -- missing {missing[:4]}"
                        f"{'...' if len(missing) > 4 else ''}. A CSV written before a"
                        " field was added cannot be replotted against current code."
                    )
                kwargs = {n: _coerce(row[n], resolved[n]) for n in fields}
                r = CompareE2EResult(**kwargs)
                key = tuple(getattr(r, k) for k in _KEY)
                if key in seen:
                    raise SystemExit(
                        f"duplicate cell {key}\n  first seen in: {seen[key]}\n  again in: {p}\n"
                        "Concatenating overlapping runs would double-count this cell. Re-run"
                        " with non-overlapping --eval-types/--k-values, or pass only one file."
                    )
                seen[key] = str(p)
                out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="compare_e2e *_results.csv file(s)")
    ap.add_argument("-o", "--out", default="compare_e2e_calibration.png")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    results = load_results(args.csv)
    ets = sorted({r.eval_type for r in results})
    print(f"loaded {len(results)} cells from {len(args.csv)} file(s); eval_types: {ets}")
    out = save_calibration_plot(results=results, alpha=args.alpha, out_path=args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
