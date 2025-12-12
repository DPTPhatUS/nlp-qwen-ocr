#!/usr/bin/env python3
"""Find pages containing figures/images/charts/formulas and store the results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

DEFAULT_TARGET_TYPES = {"image", "figure", "chart", "formula"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan OCR JSON page data to locate pages containing media blocks "
            "such as figures, charts, or formulas and write the matches to a "
            "text report."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("docs"),
        help="Directory that holds the book folders (default: docs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("media_pages.txt"),
        help="Where to write the text report (default: media_pages.txt)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help=(
            "Optional list of block types to detect. Defaults to image, "
            "figure, chart, and formula."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matches to stdout without writing the report file",
    )
    return parser.parse_args()


def normalize_types(raw: Iterable[str] | None) -> set[str]:
    if not raw:
        return set(DEFAULT_TARGET_TYPES)
    normalized: set[str] = set()
    for value in raw:
        value = value.strip().lower()
        if value:
            normalized.add(value)
    return normalized


def iter_page_data(root: Path) -> Iterable[Path]:
    pattern = "**/pages_data/*.json"
    yield from sorted(root.glob(pattern))


def analyze_page(path: Path, target_types: set[str]) -> tuple[int | None, Counter[str]] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = data.get("content", [])
    counts: Counter[str] = Counter()
    for block in blocks:
        b_type = block.get("type")
        if b_type in target_types:
            counts[b_type] += 1
    if not counts:
        return None
    page_num = data.get("page_num")
    return page_num, counts


def format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()
    if not root.exists():
        raise SystemExit(f"Root directory {root} does not exist")

    target_types = normalize_types(args.types)
    matches: list[str] = []

    for page_json in iter_page_data(root):
        result = analyze_page(page_json, target_types)
        if not result:
            continue
        page_num, counts = result
        rel_path = page_json.relative_to(root)
        prefix = f"page {page_num}" if page_num is not None else "page ?"
        matches.append(f"{rel_path} ({prefix}): {format_counts(counts)}")

    if not args.dry_run:
        args.output.write_text("\n".join(matches) + ("\n" if matches else ""), encoding="utf-8")

    if matches:
        print(f"Found {len(matches)} matching pages containing {', '.join(sorted(target_types))}.")
        if args.dry_run:
            for line in matches:
                print(line)
        else:
            print(f"Report written to {args.output}")
    else:
        print("No matching pages found.")


if __name__ == "__main__":
    main()
