#!/usr/bin/env python
"""Step 5: Convert aggregated Markdown into DOCX outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence

from nlp_qwen_ocr.docx_exporter import MarkdownDocxExporter

LOGGER = logging.getLogger("qwen_ocr.md2docx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Markdown books into DOCX")
    parser.add_argument("--output", default="outputs", type=Path, help="Root directory containing book markdown")
    parser.add_argument("--books", nargs="*", default=None, help="Subset of books to convert")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


class MarkdownToDocxPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.output_dir = args.output.resolve()
        self.books = self._discover_books(args.books)
        self.exporter = MarkdownDocxExporter()

    def _discover_books(self, requested: Optional[Sequence[str]]) -> List[str]:
        if requested:
            return list(requested)
        if not self.output_dir.exists():
            return []
        return sorted([d.name for d in self.output_dir.iterdir() if d.is_dir()])

    def run(self) -> None:
        if not self.books:
            LOGGER.warning("No books found to convert under %s", self.output_dir)
            return
        for book in self.books:
            LOGGER.info("Building DOCX for %s", book)
            self._process_book(book)

    def _process_book(self, book: str) -> None:
        markdown_dir = self.output_dir / book / "markdown"
        docx_dir = self.output_dir / book / "docx"
        book_md_path = markdown_dir / f"{book}.md"
        if not book_md_path.exists():
            LOGGER.warning("Missing aggregated Markdown at %s", book_md_path)
            return
        markdown_text = book_md_path.read_text(encoding="utf-8")
        docx_dir.mkdir(parents=True, exist_ok=True)
        docx_path = docx_dir / f"{book}.docx"
        self.exporter.convert(markdown_text, book_md_path, docx_path)
        LOGGER.info("Wrote %s", docx_path)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    pipeline = MarkdownToDocxPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
