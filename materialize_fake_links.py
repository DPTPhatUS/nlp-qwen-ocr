#!/usr/bin/env python
"""Stage 2: Turn FAKE Markdown links into real PNG crops + cleaned Markdown."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from nlp_qwen_ocr.markdown_utils import clamp_bbox_to_size

LOGGER = logging.getLogger("qwen_ocr.fake_assets")

FAKE_LINK_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<fake>FAKE_(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)(?:_[A-Za-z0-9_-]+)?\.png)\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw Markdown with FAKE links into Markdown + cropped PNG assets"
    )
    parser.add_argument("--source", default="docs", type=Path, help="Directory containing book folders")
    parser.add_argument(
        "--markdown-root",
        default="outputs",
        type=Path,
        help="Root directory containing book/markdown_raw pages from stage 1",
    )
    parser.add_argument("--output", default="outputs", type=Path, help="Directory for final Markdown + assets")
    parser.add_argument("--books", nargs="*", default=None, help="Subset of books to process")
    parser.add_argument("--start-page", type=int, default=1, help="Page number to start from (inclusive)")
    parser.add_argument("--max-pages", type=int, default=None, help="Number of pages to process after start page")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


class FakeLinkMaterializer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.source_dir = args.source.resolve()
        self.markdown_root = args.markdown_root.resolve()
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
            LOGGER.info("Materializing assets for %s", book)
            self._process_book(book)

    def _process_book(self, book: str) -> None:
        book_dir = self.source_dir / book
        page_meta_dir = book_dir / "pages_data"
        page_image_dir = book_dir / "pdf_images"
        raw_markdown_dir = self.markdown_root / book / "markdown_raw"
        if not page_meta_dir.exists() or not page_image_dir.exists():
            LOGGER.warning("Book %s missing metadata or source images", book)
            return
        if not raw_markdown_dir.exists():
            LOGGER.warning("Book %s has no markdown_raw directory at %s", book, raw_markdown_dir)
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
            raw_markdown_path = raw_markdown_dir / f"page_{page_number:04d}.md"
            if not raw_markdown_path.exists():
                LOGGER.warning("Missing raw markdown for page %s", page_number)
                continue
            image_name = page_payload.get("source_image")
            if not image_name:
                LOGGER.warning("page_data for %s has no source_image", meta_path.name)
                continue
            image_path = page_image_dir / image_name
            if not image_path.exists():
                LOGGER.warning("Missing source image %s for page %s", image_path, page_number)
                continue
            markdown_text = raw_markdown_path.read_text()
            page_markdown, asset_count = render_page_markdown(
                markdown_text=markdown_text,
                page_image=image_path,
                page_number=page_number,
                markdown_dir=markdown_dir,
                assets_root=assets_root,
            )
            page_md_path = markdown_dir / f"page_{page_number:04d}.md"
            page_md_path.write_text(page_markdown)
            LOGGER.debug("Wrote %s (%d assets)", page_md_path, asset_count)
            aggregated.append(page_markdown)
            processed += 1
        if not aggregated:
            LOGGER.warning("Book %s produced no Markdown pages", book)
            return
        book_markdown = "\n\n".join(aggregated)
        book_md_path = markdown_dir / f"{book}.md"
        book_md_path.write_text(book_markdown)
        LOGGER.info("Wrote %s", book_md_path)


def render_page_markdown(
    markdown_text: str,
    page_image: Path,
    page_number: int,
    markdown_dir: Path,
    assets_root: Path,
) -> Tuple[str, int]:
    if not FAKE_LINK_RE.search(markdown_text):
        return markdown_text.strip(), 0
    page_asset_dir = assets_root / f"page_{page_number:04d}"
    page_asset_dir.mkdir(parents=True, exist_ok=True)

    asset_cache: Dict[Tuple[int, int, int, int], Path] = {}
    with Image.open(page_image) as page_img:
        width, height = page_img.size
        counter = 1

        def replace(match: re.Match[str]) -> str:
            nonlocal counter
            bbox = (
                int(match.group("x1")),
                int(match.group("y1")),
                int(match.group("x2")),
                int(match.group("y2")),
            )
            clamped = clamp_bbox_to_size(bbox, width, height)
            if clamped is None:
                LOGGER.warning("Invalid bbox %s on page %s", bbox, page_number)
                return match.group(0)
            dest = asset_cache.get(clamped)
            if dest is None:
                dest = page_asset_dir / f"asset_{page_number:04d}_{counter:02d}.png"
                crop = page_img.crop(clamped)
                crop.save(dest)
                asset_cache[clamped] = dest
                counter += 1
            rel_path = os.path.relpath(dest, markdown_dir)
            alt = match.group("alt").strip() or dest.stem
            return f"![{alt}]({rel_path})"

        rendered = FAKE_LINK_RE.sub(replace, markdown_text)
    return rendered.strip(), len(asset_cache)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    pipeline = FakeLinkMaterializer(args)
    pipeline.run()


if __name__ == "__main__":
    main()
