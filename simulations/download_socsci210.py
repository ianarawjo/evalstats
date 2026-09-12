"""Fetch the SocSci210 columns that cases/ci_unpaired.py needs.

SocSci210 (HuggingFace ``socratesft/SocSci210``) is 210 social-science
experiments reconstructed from NSF's TESS programme -- 2.9M responses from
400,491 participants. ci_unpaired uses it as its human-subjects real-data
source, because between-subjects designs are overwhelmingly an HCI and social
science practice and a recommendation aimed at that setting should be checked
against actual human-subjects experiments rather than only AI-eval corpora.

NOT REDISTRIBUTED WITH THIS REPO. The dataset card carries no license field,
so nothing from it is committed here and this script does not vendor it --
it downloads to ``simulations/out/socsci210/``, which is gitignored. Anyone
reproducing the real-data sweep runs this themselves. Papers should cite the
underlying TESS studies, which are individually archived and citable; this
corpus is a convenience packaging of them.

Why this is much smaller than the 1.45 GB the dataset card advertises: almost
all of that is the ``stimuli``, ``prompt`` and ``reasoning`` free-text columns,
none of which we need. Parquet is columnar and HuggingFace serves range
requests, so projecting to the five columns we do need is fetched directly
rather than downloaded and discarded.

Run:
    python -m simulations.download_socsci210
    python -m simulations.download_socsci210 --force        # re-fetch
    python -m simulations.download_socsci210 --full         # whole shards
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ID = "socratesft/SocSci210"
REPO_TYPE = "datasets"
DEFAULT_OUT = Path("simulations/out/socsci210")
SLIM_NAME = "socsci210_slim.parquet"

COLUMNS = ["study_id", "task_num", "condition_num", "participant", "response"]
"""Kept deliberately in sync with real_unpaired._SOCSCI_COLUMNS. Everything
else in the corpus is free text for LLM prompting and is irrelevant here."""


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def fetch_slim(out_dir: Path, force: bool) -> Path:
    """Column-projected fetch: read only COLUMNS from each remote shard."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out_path = out_dir / SLIM_NAME
    if out_path.exists() and not force:
        print(f"Already present: {out_path} ({_human(out_path.stat().st_size)})")
        print("Pass --force to re-fetch.")
        return out_path

    fs = HfFileSystem()
    pattern = f"{REPO_TYPE}/{REPO_ID}/**/*.parquet"
    shards = sorted(fs.glob(pattern))
    if not shards:
        raise SystemExit(f"No parquet shards found under {pattern!r}. "
                         f"Is the repo id still {REPO_ID!r}?")
    print(f"{len(shards)} shard(s) in {REPO_ID}; reading columns {COLUMNS}")

    tables, t0 = [], time.time()
    for i, shard in enumerate(shards, 1):
        with fs.open(shard, "rb") as fh:
            tables.append(pq.read_table(fh, columns=COLUMNS))
        rows = sum(t.num_rows for t in tables)
        print(f"  [{i}/{len(shards)}] {Path(shard).name}  cumulative rows: {rows:,}")
    table = pa.concat_tables(tables)

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")
    print(f"\nWrote {out_path} -- {table.num_rows:,} rows, "
          f"{_human(out_path.stat().st_size)}, {time.time()-t0:.0f}s")
    return out_path


def fetch_full(out_dir: Path, force: bool) -> Path:
    """Whole-file fallback via the hub client (downloads the text columns too)."""
    from huggingface_hub import snapshot_download

    print("Downloading FULL shards (~1.45 GB) -- only needed if you want the "
          "free-text columns; the sweep does not.")
    path = snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", local_dir=str(out_dir),
        force_download=force, allow_patterns=["*.parquet", "metadata/*"],
    )
    return Path(path)


def fetch_metadata(out_dir: Path) -> None:
    """The three small mapping JSONs, kept for provenance."""
    from huggingface_hub import hf_hub_download
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metadata/condition_mapping.json", "metadata/task_mapping.json",
                 "metadata/participant_mapping.json"):
        try:
            hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=name,
                            local_dir=str(out_dir))
            print(f"  fetched {name}")
        except Exception as exc:
            print(f"  (skipped {name}: {exc})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT),
                    help=f"Destination (default {str(DEFAULT_OUT)!r}; gitignored).")
    ap.add_argument("--force", action="store_true", help="Re-fetch even if present.")
    ap.add_argument("--full", action="store_true",
                    help="Download whole shards instead of projecting columns.")
    ap.add_argument("--skip-metadata", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    print(__doc__.split("Run:")[0].strip())
    print("=" * 72)
    try:
        if args.full:
            fetch_full(out_dir, args.force)
        else:
            fetch_slim(out_dir, args.force)
        if not args.skip_metadata:
            fetch_metadata(out_dir)
    except ImportError as exc:
        print(f"\nMissing dependency: {exc}\n  pip install huggingface_hub pyarrow")
        return 1
    except Exception as exc:
        print(f"\nDownload failed: {type(exc).__name__}: {exc}")
        return 1

    print("\nDone. The ci_unpaired real suite will now pick these up:")
    print("  python -m simulations.harness.cli ci_unpaired --data-source real "
          "--real-datasets socsci210")
    return 0


if __name__ == "__main__":
    sys.exit(main())
