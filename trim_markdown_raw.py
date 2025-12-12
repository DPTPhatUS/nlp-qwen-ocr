#!/usr/bin/env python3
"""Step 2: Trim the first N lines from each markdown_raw page."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def iter_markdown_raw_dirs(root: Path) -> Iterable[Path]:
    """Yield every markdown_raw directory under the given root."""
    for candidate in sorted(root.rglob("markdown_raw")):
        if candidate.is_dir():
            yield candidate


def iter_files(markdown_dir: Path, extensions: set[str] | None) -> Iterable[Path]:
    """Yield files in markdown_dir, optionally filtering by suffix."""
    for child in sorted(markdown_dir.iterdir()):
        if not child.is_file():
            continue
        if extensions and child.suffix.lower() not in extensions:
            continue
        yield child


def trim_file(path: Path, lines_to_remove: int, dry_run: bool) -> int:
    """Remove the first lines_to_remove lines from path and return lines stripped."""
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    if not lines:
        return 0

    trimmed = lines[lines_to_remove:]
    removed = min(lines_to_remove, len(lines))
    if dry_run:
        return removed

    path.write_text("".join(trimmed), encoding="utf-8")
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strip the first N lines from every file stored in any markdown_raw "
            "directory under the provided root (defaults to ./outputs)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Top-level directory that contains markdown_raw folders (default: outputs)",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=12,
        help="How many leading lines to remove from each file (default: 12)",
    )
    parser.add_argument(
        "--ext",
        dest="extensions",
        nargs="+",
        default=None,
        help=(
            "Optional list of file extensions to process, e.g. --ext .md .markdown. "
            "If omitted, every file inside markdown_raw directories is trimmed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the changes without rewriting any files",
    )
    return parser.parse_args()


def normalize_extensions(raw_exts: Iterable[str] | None) -> set[str] | None:
    if raw_exts is None:
        return None
    normalized: set[str] = set()
    for ext in raw_exts:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized or None


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()
    if not root.exists():
        raise SystemExit(f"Root directory {root} does not exist")

    extensions = normalize_extensions(args.extensions)
    total_files = 0
    total_removed = 0

    for markdown_dir in iter_markdown_raw_dirs(root):
        for file_path in iter_files(markdown_dir, extensions):
            removed = trim_file(file_path, args.lines, args.dry_run)
            if removed == 0:
                continue
            total_files += 1
            total_removed += removed
            rel_path = file_path.relative_to(root)
            prefix = "[dry-run] " if args.dry_run else ""
            print(f"{prefix}{rel_path}: removed {removed} lines")

    if total_files == 0:
        print("No files were updated.")
    else:
        action = "would remove" if args.dry_run else "removed"
        print(
            f"Processed {total_files} files and {action} {total_removed} lines in total."
        )


if __name__ == "__main__":
    main()
