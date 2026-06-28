"""Centralized simulation harness CLI.

Usage:
  python -m simulations.harness.cli --list-cases
  python -m simulations.harness.cli ci_single --reps 50 --sizes 10 20
  python -m simulations.harness.cli ci_single --data-source dove
  python -m simulations.harness.cli --official-tests all
  python -m simulations.harness.cli --official-tests ci_single
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .cases import ci_single
from .manifest import write_manifest

CASES = {
    ci_single.CASE_NAME: ci_single,
}


def _case_summary(module) -> str:
    doc = module.__doc__ or ""
    lines = [ln.strip() for ln in doc.strip().splitlines() if ln.strip()]
    return lines[0] if lines else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m simulations.harness.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list-cases", action="store_true", help="List available cases and exit.")
    parser.add_argument(
        "--official-tests", default=None, metavar="all|case1,case2,...",
        help=(
            "Run the canonical official-test preset for the given case(s) "
            "(or 'all'), writing every artifact into one timestamped "
            "simulations/out/official_<timestamp>/ directory with a manifest.json index."
        ),
    )
    subparsers = parser.add_subparsers(dest="case", metavar="case")
    for name, module in CASES.items():
        sp = subparsers.add_parser(name, help=_case_summary(module))
        module.add_arguments(sp)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_cases:
        print("Available cases:")
        for name, module in CASES.items():
            print(f"  {name:<20} {_case_summary(module)}")
        return

    if args.official_tests:
        case_names = list(CASES) if args.official_tests == "all" else [c.strip() for c in args.official_tests.split(",")]
        unknown = [c for c in case_names if c not in CASES]
        if unknown:
            parser.error(f"Unknown case(s) in --official-tests: {unknown}. Available: {list(CASES)}")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        manifest_out_dir = f"simulations/out/official_{stamp}"
        results = []
        args_by_case: dict[str, argparse.Namespace] = {}

        for name in case_names:
            module = CASES[name]
            case_args = module.official_args()
            case_args.out_dir = manifest_out_dir
            case_args.plots_dir = str(Path(manifest_out_dir) / "plots")
            print(f"\n{'=' * 72}\nOFFICIAL TEST: {name}\n{'=' * 72}")
            result = module.run(case_args)
            results.append(result)
            args_by_case[name] = case_args
            if result.status != "ok":
                print(f"\n  Case {name} FAILED: {result.error}")

        write_manifest(manifest_out_dir, results, args_by_case)

        failed = [r for r in results if r.status != "ok"]
        if failed:
            print(f"\n{len(failed)} case(s) failed: {[r.case_name for r in failed]}")
            sys.exit(1)
        return

    if not args.case:
        parser.error("Specify a case (e.g. ci_single) or use --official-tests all. Use --list-cases to see options.")

    module = CASES[args.case]
    result = module.run(args)
    if result.status != "ok":
        print(f"\nCase {result.case_name} failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
