#!/usr/bin/env python3
"""Count Markdown links/images within every markdown_raw directory."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

LINK_RE = re.compile(r"!?(?P<link>\[[^\]]*\]\([^\)]+\))")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse markdown_raw directories under a root folder, count all "
            "Markdown links (images + standard links), and write the per-file "
            "counts plus totals to a text report."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Top-level directory containing book folders with markdown_raw subdirs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("link_counts.txt"),
        help="Destination text file for the report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the report to stdout without writing the file",
    )
    return parser.parse_args()


def iter_markdown_files(root: Path) -> Iterable[Path]:
    pattern = "**/markdown_raw/*.md"
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            yield path


def extract_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [match.group(0) for match in LINK_RE.finditer(text)]


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()
    if not root.exists():
        raise SystemExit(f"Root directory {root} does not exist")

    lines: list[str] = []
    total_links = 0
    total_files = 0

    for md_file in iter_markdown_files(root):
        links = extract_links(md_file)
        count = len(links)
        total_files += 1
        total_links += count
        rel_path = md_file.relative_to(root)
        lines.append(f"{rel_path}: {count}")
        for link in links:
            lines.append(f"    - {link}")

    lines.append("")
    lines.append(f"Files scanned: {total_files}")
    lines.append(f"Total links: {total_links}")

    report = "\n".join(lines)
    if args.dry_run:
        print(report)
    else:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
