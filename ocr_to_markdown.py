#!/usr/bin/env python
"""Stage 1: Run Qwen3-VL OCR and save Markdown drafts per page."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

from nlp_qwen_ocr.ocr_client import QwenOcrClient

LOGGER = logging.getLogger("qwen_ocr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL OCR and store Markdown outputs per page")
    parser.add_argument("--source", default="docs", type=Path, help="Directory containing book folders")
    parser.add_argument("--output", default="outputs", type=Path, help="Directory for OCR results")
    parser.add_argument(
        "--books",
        nargs="*",
        default=None,
        help="Names of book folders to process (default: all folders under --source)",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct", help="Model repository to load")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens to generate per page")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature > 0 enables sampling; 0 switches to greedy decoding",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype override",
    )
    parser.add_argument("--device-map", default="auto", help="transformers device_map setting")
    parser.add_argument("--min-pixels", type=int, default=512 * 512, help="Min pixels per side for processor")
    parser.add_argument("--max-pixels", type=int, default=2048 * 2048, help="Max pixels per side for processor")
    parser.add_argument("--start-page", type=int, default=1, help="Page number to start from (inclusive)")
    parser.add_argument("--max-pages", type=int, default=None, help="Number of pages to process after start page")
    parser.add_argument("--load-in-8bit", action="store_true", help="Enable bitsandbytes 8-bit loading")
    parser.add_argument("--gpu-mem-limit", type=float, default=14.0, help="Per-GPU memory budget (GiB)")
    parser.add_argument("--cpu-mem-limit", type=float, default=29.0, help="CPU memory budget (GiB)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


class MarkdownOcrPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.source_dir = args.source.resolve()
        self.output_dir = args.output.resolve()
        self.start_page = max(1, args.start_page)
        self.max_pages = args.max_pages
        self.model_id = args.model_id
        self.ocr = QwenOcrClient(
            model_id=args.model_id,
            device_map=args.device_map,
            torch_dtype=args.dtype,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            load_in_8bit=args.load_in_8bit,
            gpu_mem_limit=args.gpu_mem_limit,
            cpu_mem_limit=args.cpu_mem_limit,
        )
        self.books = self._discover_books(args.books)

    def _discover_books(self, requested: Optional[Sequence[str]]) -> List[str]:
        if requested:
            return list(requested)
        return sorted([d.name for d in self.source_dir.iterdir() if d.is_dir()])

    def run(self) -> None:
        for book in self.books:
            LOGGER.info("Processing %s", book)
            self._process_book(book)

    def _process_book(self, book: str) -> None:
        book_dir = self.source_dir / book
        page_meta_dir = book_dir / "pages_data"
        page_image_dir = book_dir / "pdf_images"
        if not page_meta_dir.exists() or not page_image_dir.exists():
            LOGGER.warning("Book %s missing required folders", book)
            return
        markdown_dir = (self.output_dir / book / "markdown_raw").resolve()
        markdown_dir.mkdir(parents=True, exist_ok=True)

        page_files = sorted(page_meta_dir.glob("*.json"))
        page_entries = []
        for meta_path in page_files:
            page_payload = json.loads(meta_path.read_text())
            page_number = int(page_payload.get("page_num", 0))
            if page_number < self.start_page:
                continue
            page_entries.append((page_number, meta_path, page_payload))
        page_entries.sort(key=lambda item: item[0])
        if self.max_pages is not None:
            page_entries = page_entries[: self.max_pages]
        total_targets = len(page_entries)
        if total_targets == 0:
            LOGGER.warning("Book %s has no pages to process in the selected range", book)
            return

        processed = 0
        for idx, (page_number, meta_path, page_payload) in enumerate(page_entries, start=1):
            image_name = page_payload.get("source_image")
            if not image_name:
                LOGGER.warning("Skipping %s with no source_image", meta_path.name)
                continue
            image_path = page_image_dir / image_name
            if not image_path.exists():
                LOGGER.warning("Missing image %s", image_path)
                continue
            LOGGER.info("[%d/%d] OCR page %04d", idx, total_targets, page_number)
            markdown_text = self.ocr.ocr_page(page_number, image_path).strip()
            output_path = markdown_dir / f"page_{page_number:04d}.md"
            if not markdown_text:
                LOGGER.warning("Received empty markdown for page %s", page_number)
            output_path.write_text(markdown_text)
            LOGGER.info("Saved %s", output_path)
            processed += 1
        if processed == 0:
            LOGGER.warning("Book %s produced no Markdown outputs", book)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    pipeline = MarkdownOcrPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
