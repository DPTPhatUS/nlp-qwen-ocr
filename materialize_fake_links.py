#!/usr/bin/env python
"""Stage 2: Turn FAKE Markdown links into real PNG crops + cleaned Markdown."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from nlp_qwen_ocr.markdown_utils import clamp_bbox_to_size, normalize_bbox

LOGGER = logging.getLogger("qwen_ocr.fake_assets")

FAKE_LINK_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<fake>FAKE_(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)(?:_(?P<label>[A-Za-z0-9_-]+))?\.png)\)"
)


@dataclass
class PageAsset:
    idx: int
    bbox: Tuple[int, int, int, int]
    img_path: Path
    text: str


class ClipMatcher:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.batch_size = max(1, batch_size)

    def best_match(
        self,
        description: str,
        assets: Sequence[PageAsset],
        cache: Dict[Path, torch.Tensor],
    ) -> Optional[Tuple[PageAsset, float]]:
        cleaned = description.strip()
        if not cleaned or not assets:
            return None
        text_features = self._encode_text(cleaned)
        image_paths = [asset.img_path for asset in assets]
        self._ensure_image_features(image_paths, cache)
        stacked = torch.stack([cache[path] for path in image_paths]).to(self.device)
        scores = torch.matmul(stacked, text_features.squeeze(0))
        if scores.numel() == 0:
            return None
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())
        return assets[best_idx], best_score

    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text])
        text_tokens = tokens.to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
        return F.normalize(features, dim=-1)

    def _ensure_image_features(self, paths: Sequence[Path], cache: Dict[Path, torch.Tensor]) -> None:
        missing = [path for path in paths if path not in cache]
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            if not chunk:
                continue
            batch_tensors: List[torch.Tensor] = []
            for path in chunk:
                with Image.open(path) as img:
                    batch_tensors.append(self.preprocess(img.convert("RGB")))
            batch = torch.stack(batch_tensors).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(batch)
            features = F.normalize(features, dim=-1)
            for path, feature in zip(chunk, features):
                cache[path] = feature.detach()


def _collect_page_assets(metadata: Optional[Dict[str, object]], crop_dir: Optional[Path]) -> List[PageAsset]:
    if not metadata or not crop_dir:
        return []
    content = metadata.get("content")
    if not isinstance(content, list):
        return []
    assets: List[PageAsset] = []
    for idx, item in enumerate(content):
        if not isinstance(item, dict):
            continue
        bbox = normalize_bbox(item.get("bbox"))
        img_rel = item.get("img_path")
        if not bbox or not img_rel:
            continue
        asset_path = crop_dir / img_rel
        if not asset_path.exists():
            continue
        assets.append(
            PageAsset(
                idx=idx,
                bbox=bbox,
                img_path=asset_path,
                text=str(item.get("text", "") or ""),
            )
        )
    return assets


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
    parser.add_argument(
        "--clip-model",
        default="xlm-roberta-base-ViT-B-32",
        help="open_clip model name for multilingual text-image matching",
    )
    parser.add_argument(
        "--clip-pretrained",
        default="laion5b_s13b_b90k",
        help="Pretrained checkpoint identifier for the CLIP model",
    )
    parser.add_argument(
        "--clip-device",
        default=None,
        help="Device to run CLIP on (e.g. cuda, cuda:0, cpu). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=8,
        help="Number of page crops to encode per batch when building CLIP embeddings",
    )
    return parser.parse_args()


class FakeLinkMaterializer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.source_dir = args.source.resolve()
        self.markdown_root = args.markdown_root.resolve()
        self.output_dir = args.output.resolve()
        self.start_page = max(1, args.start_page)
        self.max_pages = args.max_pages
        self.books = self._discover_books(args.books)
        self.clip_matcher = ClipMatcher(
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.clip_device,
            batch_size=args.clip_batch_size,
        )

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
        crop_dir = book_dir / "crops"
        crop_dir = crop_dir if crop_dir.exists() else None
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
                page_metadata=page_payload,
                crop_dir=crop_dir,
                clip_matcher=self.clip_matcher,
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
    page_metadata: Optional[Dict[str, object]],
    crop_dir: Optional[Path],
    clip_matcher: Optional[ClipMatcher],
) -> Tuple[str, int]:
    if not FAKE_LINK_RE.search(markdown_text):
        return markdown_text.strip(), 0
    page_asset_dir = assets_root / f"page_{page_number:04d}"
    page_asset_dir.mkdir(parents=True, exist_ok=True)
    page_assets = _collect_page_assets(page_metadata, crop_dir)
    used_assets: Set[int] = set()
    asset_cache: Dict[str, Path] = {}
    clip_cache: Dict[Path, torch.Tensor] = {}
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
            alt_text = match.group("alt").strip()
            clip_choice: Optional[Tuple[PageAsset, float]] = None
            if clip_matcher and alt_text:
                available = [asset for asset in page_assets if asset.idx not in used_assets]
                clip_choice = clip_matcher.best_match(alt_text, available, clip_cache)
            if clip_choice:
                candidate, similarity = clip_choice
                cache_key = f"meta:{candidate.idx}"
                dest = asset_cache.get(cache_key)
                if dest is None:
                    dest = page_asset_dir / f"asset_{page_number:04d}_{counter:02d}.png"
                    try:
                        shutil.copyfile(candidate.img_path, dest)
                    except OSError as exc:
                        LOGGER.warning(
                            "Failed to copy asset %s for page %s: %s",
                            candidate.img_path,
                            page_number,
                            exc,
                        )
                        dest = None
                    else:
                        asset_cache[cache_key] = dest
                        counter += 1
                if dest is not None:
                    used_assets.add(candidate.idx)
                    rel_path = os.path.relpath(dest, markdown_dir)
                    display_alt = alt_text or candidate.text.strip() or dest.stem
                    LOGGER.debug(
                        "CLIP matched %s to '%s' on page %s (score %.3f)",
                        candidate.img_path.name,
                        alt_text,
                        page_number,
                        similarity,
                    )
                    return f"![{display_alt}]({rel_path})"
            clamped = clamp_bbox_to_size(bbox, width, height)
            if clamped is None:
                LOGGER.warning("Invalid bbox %s on page %s", bbox, page_number)
                return match.group(0)
            cache_key = f"bbox:{clamped}"
            dest = asset_cache.get(cache_key)
            if dest is None:
                dest = page_asset_dir / f"asset_{page_number:04d}_{counter:02d}.png"
                crop = page_img.crop(clamped)
                crop.save(dest)
                crop.close()
                asset_cache[cache_key] = dest
                counter += 1
            rel_path = os.path.relpath(dest, markdown_dir)
            fallback_alt = alt_text or dest.stem
            return f"![{fallback_alt}]({rel_path})"

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
