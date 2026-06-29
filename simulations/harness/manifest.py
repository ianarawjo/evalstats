"""Run-manifest writer for --official-tests runs.

Writes a single JSON index (manifest.json) summarizing every case run in an
`--official-tests` invocation: args used, output artifact paths, key metrics,
status, and wall-clock duration. This is the artifact intended to accompany
the paper's Supplementary Material.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .cases import CaseResult


def write_manifest(out_dir: str, results: list[CaseResult], args_list: list[argparse.Namespace]) -> str:
    """``args_list`` must be parallel to ``results`` (same length, same order)
    -- a plain list rather than a dict keyed by case_name, since a single
    case can now run more than once per invocation (--quick-test's synthetic
    + real data-source calls), which would collide on a case_name key."""
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    manifest_path = out_base / "manifest.json"

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": [
            {
                "case_name": r.case_name,
                "status": r.status,
                "duration_s": round(r.duration_s, 3),
                "output_paths": r.output_paths,
                "key_metrics": r.key_metrics,
                "error": r.error,
                "args": vars(case_args),
            }
            for r, case_args in zip(results, args_list)
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved manifest: {manifest_path}")
    return str(manifest_path)
