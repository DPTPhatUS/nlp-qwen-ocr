#!/usr/bin/env python3
"""Step 4: Insert </break> markers between heading 1 and heading 2 pairs in book markdown files."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

BREAK_TAG = "</break>"
HEADING_LINE = re.compile(r"^(?P<marker>#+) (?P<title>.+)$")
HeadingType = Literal["h1", "h2"]

VIETNAMESE_UPPER = "A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰÝỲỶỸỴ"
ROMAN_UPPER = "IVXLCDM"


@dataclass(frozen=True)
class BookRule:
    slug: str
    markdown_filename: str
    h1_patterns: Sequence[re.Pattern[str]]
    h2_patterns: Sequence[re.Pattern[str]]


BOOK_RULES: dict[str, BookRule] = {
    "book1": BookRule(
        slug="book1",
        markdown_filename="book1.md",
        h1_patterns=(re.compile(r"^\d+\.\s"),),
        h2_patterns=(re.compile(r"^\d+\.\d+\.\s"),),
    ),
    "book2": BookRule(
        slug="book2",
        markdown_filename="book2.md",
        h1_patterns=(re.compile(rf"^(?:[{ROMAN_UPPER}]+)\.\s"),),
        h2_patterns=(
            re.compile(
                rf"^(?![{ROMAN_UPPER}\s]+\.\s)[{VIETNAMESE_UPPER}]+(?:\s+[{VIETNAMESE_UPPER}]+)*\.\s"
            ),
        ),
    ),
    "book3": BookRule(
        slug="book3",
        markdown_filename="book3.md",
        h1_patterns=(re.compile(r"^(?:[IVXLCDM]+)\.\s"),),
        h2_patterns=(re.compile(r"^\d+\.\s"),),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert </break> between heading 1 and heading 2 pairs for the supported books.",
    )
    parser.add_argument(
        "--books",
        nargs="+",
        choices=sorted(BOOK_RULES.keys()),
        help="Subset of books to process (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without writing to disk.",
    )
    return parser.parse_args()


def detect_line_ending(lines: Sequence[str]) -> str:
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
    return "\n"


def classify_heading(line: str, rule: BookRule) -> HeadingType | None:
    stripped = line.rstrip("\r\n")
    match = HEADING_LINE.match(stripped)
    if not match:
        return None
    marker = match.group("marker")
    title = match.group("title").strip()
    for pattern in rule.h2_patterns:
        if pattern.match(title):
            return "h2"
    for pattern in rule.h1_patterns:
        if pattern.match(title):
            return "h1"
    return None


def inject_breaks(lines: Sequence[str], rule: BookRule, line_ending: str) -> tuple[list[str], int]:
    updated_lines: list[str] = []
    inserted = 0
    for line in lines:
        current_heading = classify_heading(line, rule)
        if current_heading is not None:
            if updated_lines and updated_lines[-1].rstrip("\r\n") != BREAK_TAG:
                updated_lines.append(f"{BREAK_TAG}{line_ending}")
                inserted += 1
        updated_lines.append(line)
    return updated_lines, inserted


def process_book(rule: BookRule, root: Path, dry_run: bool) -> None:
    markdown_path = root / "outputs" / rule.slug / "markdown" / rule.markdown_filename
    relative = markdown_path.relative_to(root)
    if not markdown_path.exists():
        print(f"[skip] {relative} (missing)")
        return
    lines = markdown_path.read_text(encoding="utf-8").splitlines(keepends=True)
    line_ending = detect_line_ending(lines)
    updated_lines, inserted = inject_breaks(lines, rule, line_ending)
    if inserted == 0:
        print(f"[ok] {relative} (no changes)")
        return
    if dry_run:
        print(f"[dry-run] {relative} (would insert {inserted} break tags)")
        return
    markdown_path.write_text("".join(updated_lines), encoding="utf-8")
    print(f"[updated] {relative} (inserted {inserted} break tags)")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    selected_books: Iterable[str] = args.books or BOOK_RULES.keys()
    for book in selected_books:
        process_book(BOOK_RULES[book], root, args.dry_run)


if __name__ == "__main__":
    main()
