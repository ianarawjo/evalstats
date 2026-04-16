#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


OLD_NAME = "promptstats"
NEW_NAME = "evalstats"
OLD_ALIAS = "pstats"
NEW_ALIAS = "estats"

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".published",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".ruff_cache",
    ".vscode",
}

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".pdf",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".zip",
    ".gz",
    ".tar",
    ".whl",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".ico",
    ".icns",
}

TEXT_NAME_EXCLUDES = {
    "rename_promptstats_to_evalstats.py",
}

CONTENT_EXCLUDES = {
    "rename_promptstats_to_evalstats.py",
}

ALIAS_IMPORT_RE = re.compile(r"(?<!\S)import\s+promptstats\s+as\s+pstats\b")
NAME_RE = re.compile(r"\bpromptstats\b")
ALIAS_RE = re.compile(r"\bpstats\b")


@dataclass
class RenameStats:
    changed_files: list[Path] = field(default_factory=list)
    renamed_paths: list[tuple[Path, Path]] = field(default_factory=list)
    skipped_binary: list[Path] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename promptstats -> evalstats across the repository.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root to rewrite.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in place. Without this flag the script runs as a dry run.",
    )
    parser.add_argument(
        "--include-build",
        action="store_true",
        help="Also process the generated website/build directory.",
    )
    return parser.parse_args()


def should_skip_dir(path: Path, include_build: bool) -> bool:
    if path.name in SKIP_DIRS:
        return True
    if path.name.startswith(".pkgtest-venv"):
        return True
    if path.name == "dist":
        return True
    if path.name == "build":
        if include_build and path.parent.name == "website":
            return False
        return True
    return False


def iter_files(root: Path, include_build: bool) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(should_skip_dir(parent, include_build) for parent in path.parents if parent != root):
            continue
        if should_skip_dir(path, include_build):
            continue
        if path.is_file():
            yield path


def is_binary_bytes(raw: bytes) -> bool:
    if not raw:
        return False
    if b"\x00" in raw:
        return True
    sample = raw[:4096]
    text_bytes = sum(32 <= byte <= 126 or byte in {9, 10, 13, 12, 8} for byte in sample)
    return text_bytes / len(sample) < 0.85


def read_text_if_possible(path: Path) -> str | None:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return None
    raw = path.read_bytes()
    if is_binary_bytes(raw):
        return None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def rewrite_text(text: str) -> str:
    updated = ALIAS_IMPORT_RE.sub(f"import {NEW_NAME} as {NEW_ALIAS}", text)
    updated = NAME_RE.sub(NEW_NAME, updated)
    updated = ALIAS_RE.sub(NEW_ALIAS, updated)
    return updated


def rewrite_files(root: Path, include_build: bool, apply: bool, stats: RenameStats) -> None:
    for path in iter_files(root, include_build):
        if path.name in CONTENT_EXCLUDES:
            continue
        text = read_text_if_possible(path)
        if text is None:
            stats.skipped_binary.append(path)
            continue

        updated = rewrite_text(text)
        if updated == text:
            continue

        stats.changed_files.append(path)
        if apply:
            path.write_text(updated, encoding="utf-8")


def rename_paths(root: Path, include_build: bool, apply: bool, stats: RenameStats) -> None:
    candidates = []
    for path in root.rglob("*"):
        if any(should_skip_dir(parent, include_build) for parent in path.parents if parent != root):
            continue
        if should_skip_dir(path, include_build):
            continue
        if path.name in TEXT_NAME_EXCLUDES:
            continue
        if OLD_NAME not in path.name:
            continue
        candidates.append(path)

    for source in sorted(candidates, key=lambda item: (len(item.parts), item.as_posix()), reverse=True):
        target = source.with_name(source.name.replace(OLD_NAME, NEW_NAME))
        stats.renamed_paths.append((source, target))
        if apply:
            source.rename(target)


def summarize(root: Path, apply: bool, stats: RenameStats) -> None:
    mode = "APPLY" if apply else "DRY RUN"
    print(f"[{mode}] root={root}")
    print(f"changed files: {len(stats.changed_files)}")
    print(f"renamed paths: {len(stats.renamed_paths)}")
    for path in stats.changed_files[:20]:
        print(f"  file: {path.relative_to(root)}")
    if len(stats.changed_files) > 20:
        print(f"  ... {len(stats.changed_files) - 20} more files")
    for source, target in stats.renamed_paths[:20]:
        print(f"  move: {source.relative_to(root)} -> {target.relative_to(root)}")
    if len(stats.renamed_paths) > 20:
        print(f"  ... {len(stats.renamed_paths) - 20} more path renames")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    stats = RenameStats()
    rewrite_files(root=root, include_build=args.include_build, apply=args.apply, stats=stats)
    rename_paths(root=root, include_build=args.include_build, apply=args.apply, stats=stats)
    summarize(root=root, apply=args.apply, stats=stats)


if __name__ == "__main__":
    main()