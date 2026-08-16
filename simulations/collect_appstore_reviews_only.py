"""Standalone Apple App Store review collector -- fetches reviews only, no
judge scoring, and writes to its own untracked CSV that collect_judge_bias_
data.py never touches.

Why this exists as a separate script rather than reusing collect_judge_bias_
data.py directly: that script's `collect-data --types likert` writes to
simulations/out/judge_bias_appstore_items.csv, which the judge-bias
simulation harness (simulations/harness/scenarios/real_judge_bias.py,
tests/test_ppi_corrections.py, etc.) reads as shared, versioned-in-spirit
ground truth. Running it again to grow a paper example's dataset would
silently perturb every simulation that depends on that file. This script
clones just the App Store fetching logic (same RSS endpoint, same retry/
rate-limit handling) and writes somewhere else entirely.

Unlike the original's fetch_appstore_items (which pools ALL apps' reviews
together and truncates to one global --n-items), this loops per app and
keeps paging until EACH app individually reaches --target-per-app (or runs
out of pages/days-back window) -- the point is a usable N per group, not a
fixed total.

Apple's RSS feed only serves recent reviews and is rate-limited/flaky
per-call (see the original script's own notes) -- for a popular app you may
still fall short of --target-per-app in one run. Re-running is safe and
additive: existing item_ids are skipped, so each run only adds genuinely
new reviews. Run it again on a later day (or raise --days-back) to keep
accumulating.

Once you have enough reviews, score them with a judge YOURSELF (this
script does none of that) -- e.g. by adapting collect_judge_bias_data.py's
_build_appstore_prompt/_call_judge/_parse_appstore_response against this
file's judge_input column, or wiring this CSV into a fresh instance of
that script's own collect-judge-scores step against a separate --out-dir
so it still never writes into the shared judge_bias_appstore*.csv files.

Usage:
    # Default: the paper example's 4 apps (TikTok/FlipFlop, Google Maps/
    # Wavelength, Instagram/Snippet, Facebook/Bubblegum), target 300/app.
    python -m simulations.collect_appstore_reviews_only

    # Custom apps / target / lookback window:
    python -m simulations.collect_appstore_reviews_only \
        --app-ids 835599320 585027354 389801252 284882215 447188370 \
        --target-per-app 300 --days-back 90
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_OUT_PATH = "simulations/out/appstore_scenario_reviews.csv"
FIELDNAMES = ["item_id", "app_id", "human_label", "title", "text", "updated"]

# TikTok (FlipFlop), Google Maps (Wavelength), Instagram (Snippet), Facebook
# (Bubblegum) -- the four apps in the paper's FlipFlop worked example.
DEFAULT_APP_IDS = [835599320, 585027354, 389801252, 284882215]


def _progress(iterable, **kwargs):
    try:
        from tqdm import tqdm
    except ImportError:
        print("  (pip install tqdm to see a progress bar here)")
        return iterable
    return tqdm(iterable, **kwargs)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fetch_app_reviews(
    app_id: int, *, country: str, target: int, days_back: int, max_pages: int,
    already_have: set[str], empty_retries: int = 3, retry_sleep: float = 2.0,
) -> list[dict]:
    """Page through one app's RSS feed until `target` NEW reviews are
    collected (counting only ones not already in `already_have`), or the
    feed/page budget runs out. Mirrors collect_judge_bias_data.py's
    fetch_appstore_items retry/cutoff behavior exactly, just scoped to one
    app and one stopping condition (target reached) instead of a shared
    pool truncated to a single global n_items.
    """
    import requests

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    collected: list[dict] = []
    page = 1
    retries_left = empty_retries
    while page <= max_pages and len(collected) < target:
        url = (f"https://itunes.apple.com/{country}/rss/customerreviews/"
               f"page={page}/id={app_id}/sortBy=mostRecent/json")
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        except requests.RequestException as e:
            print(f"  {app_id} page {page}: request failed ({e}), stopping this app.")
            break
        if resp.status_code != 200:
            break
        entries = resp.json().get("feed", {}).get("entry", [])
        if not entries:
            # Same flakiness workaround as the original: only retry the
            # FIRST page, and only while this app hasn't yielded anything
            # yet -- a later empty page legitimately means end-of-feed.
            if len(collected) == 0 and retries_left > 0:
                retries_left -= 1
                time.sleep(retry_sleep)
                continue
            break
        for e in entries:
            rating = e.get("im:rating", {}).get("label")
            updated = e.get("updated", {}).get("label")
            rid = e.get("id", {}).get("label")
            if rating is None or updated is None or rid is None:
                continue
            try:
                ts = datetime.fromisoformat(updated)
            except ValueError:
                continue
            if ts < cutoff:
                continue
            item_id = f"appstore_{app_id}_{rid}"
            if item_id in already_have:
                continue
            collected.append({
                "item_id": item_id,
                "app_id": str(app_id),
                "human_label": str(int(rating)),
                "title": e.get("title", {}).get("label", ""),
                "text": e.get("content", {}).get("label", ""),
                "updated": updated,
            })
            if len(collected) >= target:
                break
        page += 1
        time.sleep(0.3)
    return collected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app-ids", type=int, nargs="+", default=DEFAULT_APP_IDS)
    ap.add_argument("--target-per-app", type=int, default=300)
    ap.add_argument("--country", default="us")
    ap.add_argument("--days-back", type=int, default=45,
                     help="Only keep reviews updated within this many days. Raise this if a "
                          "popular app still falls short of --target-per-app.")
    ap.add_argument("--max-pages", type=int, default=15, help="RSS pages (~50 reviews each) per app.")
    ap.add_argument("--empty-retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = ap.parse_args()

    out_path = Path(args.out)
    existing_rows = _read_csv(out_path)
    existing_ids_by_app: dict[str, set[str]] = {}
    for r in existing_rows:
        existing_ids_by_app.setdefault(r["app_id"], set()).add(r["item_id"])

    print(f"Output: {out_path} (gitignored; never touches simulations/out/judge_bias_appstore*.csv)")
    print(f"Target: {args.target_per_app} reviews/app across {len(args.app_ids)} app(s), "
          f"last {args.days_back} days")
    print()

    new_rows: list[dict] = []
    for app_id in _progress(args.app_ids, desc="apps", unit="app"):
        already = existing_ids_by_app.get(str(app_id), set())
        have_now = len(already)
        still_need = max(0, args.target_per_app - have_now)
        if still_need == 0:
            print(f"  app {app_id}: already have {have_now}/{args.target_per_app} -- skipping.")
            continue
        fetched = fetch_app_reviews(
            app_id, country=args.country, target=still_need, days_back=args.days_back,
            max_pages=args.max_pages, already_have=already,
            empty_retries=args.empty_retries, retry_sleep=args.retry_sleep,
        )
        new_rows.extend(fetched)
        total_now = have_now + len(fetched)
        status = "OK" if total_now >= args.target_per_app else "SHORT"
        print(f"  app {app_id}: {have_now} existing + {len(fetched)} new = {total_now}/{args.target_per_app} [{status}]")

    all_rows = existing_rows + new_rows
    _write_csv(out_path, all_rows)
    print()
    print(f"Wrote {len(all_rows)} total reviews ({len(new_rows)} new this run) to {out_path}")

    short_apps = []
    for app_id in args.app_ids:
        n = len(existing_ids_by_app.get(str(app_id), set())) + sum(
            1 for r in new_rows if r["app_id"] == str(app_id)
        )
        if n < args.target_per_app:
            short_apps.append((app_id, n))
    if short_apps:
        print()
        print("Still short of target for:")
        for app_id, n in short_apps:
            print(f"  app {app_id}: {n}/{args.target_per_app}")
        print("Apple's feed only serves recent reviews and is rate-limited per-call -- "
              "re-run this script later (it's additive and safe), or raise --days-back / "
              "--max-pages, to keep accumulating.")


if __name__ == "__main__":
    main()
