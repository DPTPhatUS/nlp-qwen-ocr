#!/usr/bin/env python
"""Stage 2: Convert saved OCR JSON files into Markdown + cropped assets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from nlp_qwen_ocr.markdown_utils import build_page_markdown

LOGGER = logging.getLogger("qwen_ocr.json2md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform OCR JSON into Markdown")
    parser.add_argument("--source", default="docs", type=Path, help="Directory containing book folders")
    parser.add_argument("--json-root", default="outputs", type=Path, help="Root directory containing book/json outputs")
    parser.add_argument("--output", default="outputs", type=Path, help="Directory for Markdown + assets")
    parser.add_argument("--books", nargs="*", default=None, help="Subset of books to process")
    parser.add_argument("--start-page", type=int, default=1, help="Page number to start from (inclusive)")
    parser.add_argument("--max-pages", type=int, default=None, help="Number of pages to process after start page")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


class JsonToMarkdownPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.source_dir = args.source.resolve()
        self.json_root = args.json_root.resolve()
        self.output_dir = args.output.resolve()
        self.start_page = max(1, args.start_page)
        self.max_pages = args.max_pages
        self.books = self._discover_books(args.books)

    def _discover_books(self, requested: Optional[Sequence[str]]) -> List[str]:
        if requested:
            return list(requested)
        return sorted([d.name for d in self.source_dir.iterdir() if d.is_dir()])

    def run(self) -> None:
        for book in self.books:
            LOGGER.info("Rendering Markdown for %s", book)
            self._process_book(book)

    def _process_book(self, book: str) -> None:
        book_dir = self.source_dir / book
        page_meta_dir = book_dir / "pages_data"
        page_image_dir = book_dir / "pdf_images"
        crop_dir = book_dir / "crops"
        json_dir = self.json_root / book / "json"
        if not page_meta_dir.exists() or not page_image_dir.exists():
            LOGGER.warning("Book %s missing page metadata or pdf images", book)
            return
        if not json_dir.exists():
            LOGGER.warning("Book %s has no JSON directory at %s", book, json_dir)
            return
        markdown_dir = self.output_dir / book / "markdown"
        assets_root = self.output_dir / book / "assets"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        assets_root.mkdir(parents=True, exist_ok=True)

        aggregated: List[str] = []
        processed = 0
        page_files = sorted(page_meta_dir.glob("*.json"))
        for meta_path in page_files:
            page_payload = json.loads(meta_path.read_text())
            page_number = int(page_payload.get("page_num", 0))
            if page_number < self.start_page:
                continue
            if self.max_pages and processed >= self.max_pages:
                break
            page_json_path = json_dir / f"page_{page_number:04d}.json"
            if not page_json_path.exists():
                LOGGER.warning("Missing OCR JSON for %s", page_json_path)
                continue
            response_payload = json.loads(page_json_path.read_text())
            response = response_payload.get("response", response_payload)
            image_name = page_payload.get("source_image") or response_payload.get("source_image")
            if not image_name:
                LOGGER.warning("Cannot determine source_image for page %s", page_number)
                continue
            image_path = page_image_dir / image_name
            markdown_text, asset_count = build_page_markdown(
                response=response,
                page_number=page_number,
                image_path=image_path,
                markdown_dir=markdown_dir,
                assets_root=assets_root,
                crop_dir=crop_dir,
            )
            page_md_path = markdown_dir / f"page_{page_number:04d}.md"
            page_md_path.write_text(markdown_text)
            LOGGER.debug("Wrote %s (%d assets)", page_md_path, asset_count)
            aggregated.append(markdown_text)
            processed += 1
        if not aggregated:
            LOGGER.warning("Book %s produced no Markdown pages", book)
            return
        book_markdown = "\n\n".join(aggregated)
        book_md_path = markdown_dir / f"{book}.md"
        book_md_path.write_text(book_markdown)
        LOGGER.info("Wrote %s", book_md_path)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    pipeline = JsonToMarkdownPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
