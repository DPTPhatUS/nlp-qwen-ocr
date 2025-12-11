"""Utilities for turning OCR JSON into Markdown and asset crops."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

ASSET_PLACEHOLDER_RE = re.compile(r"\{\{ASSET:([A-Za-z0-9_-]+)\}\}")


@dataclass
class AssetSpec:
    asset_id: str
    caption: Optional[str]
    bbox: Optional[Sequence[float]]
    asset_type: Optional[str]


def build_page_markdown(
    response: Dict[str, object],
    page_number: int,
    image_path: Path,
    markdown_dir: Path,
    assets_root: Path,
    crop_dir: Path,
) -> Tuple[str, int]:
    """Convert a single OCR JSON payload into Markdown text."""

    markdown_body = str(response.get("markdown", "")).strip()
    raw_assets = response.get("assets", []) or []
    page_asset_dir = assets_root / f"page_{page_number:04d}"
    page_asset_dir.mkdir(parents=True, exist_ok=True)
    rendered_assets = materialize_assets(
        raw_assets=raw_assets,
        page_image=image_path,
        crop_dir=crop_dir,
        dest_dir=page_asset_dir,
    )
    markdown_with_assets = inject_assets(markdown_body, rendered_assets, markdown_dir)
    return markdown_with_assets.strip(), len(rendered_assets)


def materialize_assets(
    raw_assets: Iterable[Dict[str, object]],
    page_image: Path,
    crop_dir: Path,
    dest_dir: Path,
) -> Dict[str, Path]:
    image = Image.open(page_image)
    width, height = image.size
    rendered: Dict[str, Path] = {}
    try:
        for asset_dict in raw_assets:
            spec = AssetSpec(
                asset_id=str(asset_dict.get("id") or asset_dict.get("asset_id") or f"asset-{len(rendered)+1}"),
                caption=asset_dict.get("caption"),
                bbox=asset_dict.get("bbox"),
                asset_type=asset_dict.get("type"),
            )
            asset_path = dest_dir / f"asset_{slugify(spec.asset_id)}.png"
            bbox = normalize_bbox(spec.bbox)
            crop_img = None
            if bbox:
                clamped = clamp_bbox_to_size(bbox, width, height)
                if clamped:
                    crop_img = image.crop(clamped)
            elif ref := asset_dict.get("img_path"):
                ref_path = crop_dir / ref
                if ref_path.exists():
                    with Image.open(ref_path) as ref_img:
                        crop_img = ref_img.copy()
            if crop_img is None:
                continue
            crop_img.save(asset_path)
            crop_img.close()
            rendered[spec.asset_id] = asset_path
        return rendered
    finally:
        image.close()


def inject_assets(markdown: str, assets: Dict[str, Path], markdown_dir: Path) -> str:
    if not assets:
        return markdown

    used: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        asset_id = match.group(1)
        if asset_id not in assets:
            return match.group(0)
        asset_path = assets[asset_id]
        rel_path = os.path.relpath(asset_path, markdown_dir)
        used.add(asset_id)
        alt = asset_id.replace("-", " ")
        return f"![{alt}]({rel_path})"

    rendered_markdown = ASSET_PLACEHOLDER_RE.sub(replace, markdown)
    leftover = [aid for aid in assets if aid not in used]
    if not leftover:
        return rendered_markdown
    lines: List[str] = [rendered_markdown.rstrip(), "", "### Hình ảnh bổ sung"]
    for aid in leftover:
        rel_path = os.path.relpath(assets[aid], markdown_dir)
        lines.append(f"![{aid}]({rel_path})")
    return "\n".join(lines)


def normalize_bbox(bbox: Optional[Sequence[float]]) -> Optional[Tuple[int, int, int, int]]:
    if not bbox:
        return None
    if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        x1, y1, x2, y2 = bbox
    elif isinstance(bbox[0], (list, tuple)):
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None
    return int(max(0, min(x1, x2))), int(max(0, min(y1, y2))), int(max(x1, x2)), int(max(y1, y2))


def clamp_bbox_to_size(bbox: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    left, top, right, bottom = bbox
    left = max(0, min(left, width))
    right = max(0, min(right, width))
    top = max(0, min(top, height))
    bottom = max(0, min(bottom, height))
    if right - left < 2 or bottom - top < 2:
        return None
    return int(left), int(top), int(right), int(bottom)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower() or "asset"
