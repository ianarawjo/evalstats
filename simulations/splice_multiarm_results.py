"""Replace one eval type's rows in a multiarm results CSV with a fresh run's.

The multiarm FWER sweep's base p-value is per eval type -- McNemar mid-p on
binary, Wilcoxon otherwise (see cases/pvalues.py::_compute_multiarm_metrics).
A change confined to one type therefore only invalidates that type's rows, and
re-running the whole grid to refresh them wastes the rest. Binary is ~29% of
the synthetic grid's rows, so the saving is real.

This is only sound while the code paths behind the RETAINED rows are unchanged
since the run that produced them. Verify that before trusting a splice: for the
Aug-23 baseline, `_compute_multiarm_metrics`, `_run_multiarm_cell`,
`_max_stat_simultaneous_cis` and `correct_pvalues` were all untouched between
that run and the mid-p dispatch commit, which is what made the splice valid
there. `git log -S <fn>` is the check.

Cells are independent Monte Carlo estimates, so mixing runs is statistically
fine. What it costs is reproducibility from a single --seed: cell seeds come
from SeedSequence.spawn(len(cells)), so a binary-only run draws different
seeds than the same cells inside a full run.

    python -m simulations.splice_multiarm_results OLD.csv NEW_BINARY.csv -o OUT.csv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

KEY = ["eval_type", "label", "n", "k", "correction", "condition"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", help="previous full-grid results CSV")
    ap.add_argument("new", help="fresh run covering the eval type(s) to replace")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--replace", default=None,
                    help="comma-separated eval types to replace; default: whatever "
                         "the NEW file contains")
    a = ap.parse_args()

    # keep_default_na=False: the `condition` column's values are the literal
    # strings "null" and "alt", and pandas' default NA handling silently turns
    # "null" into NaN -- which then round-trips back out as an empty field,
    # destroying half the rows' condition labels.
    rd = lambda f: pd.read_csv(f, keep_default_na=False, na_values=[""])
    old = rd(a.old)
    new = rd(a.new)
    if list(old.columns) != list(new.columns):
        raise SystemExit(
            f"column mismatch -- these are not the same schema.\n  old: {list(old.columns)}\n"
            f"  new: {list(new.columns)}")

    types = ([t.strip() for t in a.replace.split(",")] if a.replace
             else sorted(new.eval_type.unique()))
    missing = [t for t in types if t not in set(new.eval_type)]
    if missing:
        raise SystemExit(f"NEW file has no rows for {missing}; nothing to splice in")

    extra = sorted(set(new.eval_type) - set(types))
    if extra:
        raise SystemExit(
            f"NEW file also contains {extra}, which --replace does not list. Refusing to "
            "guess: pass --replace explicitly or trim the file.")

    # Reps must match or the spliced table mixes Monte Carlo precisions, which
    # the per-cell MC bands in the rendered table would then misstate.
    o_reps = sorted(old[old.eval_type.isin(types)].n_reps.unique())
    n_reps = sorted(new.n_reps.unique())
    if o_reps != n_reps:
        print(f"WARNING: n_reps differs -- replaced rows had {o_reps}, new rows have "
              f"{n_reps}. The table's MC bands assume one rep count.", file=sys.stderr)

    kept = old[~old.eval_type.isin(types)]
    out = pd.concat([kept, new], ignore_index=True)

    dup = out.duplicated(subset=KEY).sum()
    if dup:
        raise SystemExit(f"{dup} duplicate cell keys after splice -- the NEW file overlaps "
                         "rows the OLD file kept. Check --replace.")

    dropped = len(old) - len(kept)
    print(f"replaced eval types {types}: dropped {dropped} old rows, added {len(new)} new")
    print(f"  kept {len(kept)} rows for {sorted(kept.eval_type.unique())}")
    print(f"  wrote {len(out)} rows -> {a.out}")
    out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
