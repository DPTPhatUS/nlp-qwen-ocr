"""Markdown to DOCX conversion helpers."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.shared import Inches

IMAGE_LINE_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)$")
IMAGE_INLINE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET_RE = re.compile(r"^[-*]\s+(.*)$")
NUMBERED_RE = re.compile(r"^(\d+)\.\s+(.*)$")


class MarkdownDocxExporter:
    """Minimal Markdown -> DOCX renderer (headings, lists, figures)."""

    def convert(self, markdown_text: str, markdown_path: Path, docx_path: Path) -> None:
        doc = Document()
        parent = markdown_path.parent
        for line in markdown_text.splitlines():
            stripped = line.rstrip()
            if not stripped:
                doc.add_paragraph("")
                continue
            if match := HEADING_RE.match(stripped):
                level = min(len(match.group(1)), 4)
                doc.add_heading(match.group(2).strip(), level=level)
                continue
            if match := BULLET_RE.match(stripped):
                doc.add_paragraph(match.group(1).strip(), style="List Bullet")
                continue
            if match := NUMBERED_RE.match(stripped):
                doc.add_paragraph(match.group(2).strip(), style="List Number")
                continue
            if match := IMAGE_LINE_RE.match(stripped):
                self._insert_image(doc, parent / match.group(2).strip(), match.group(1))
                continue
            if IMAGE_INLINE_RE.search(stripped):
                text_only = IMAGE_INLINE_RE.sub(lambda m: f"{m.group(1)} (xem hình {m.group(2)})", stripped)
                doc.add_paragraph(text_only)
                for img_match in IMAGE_INLINE_RE.finditer(stripped):
                    self._insert_image(doc, parent / img_match.group(2).strip(), img_match.group(1))
                continue
            doc.add_paragraph(stripped)
        doc.save(docx_path)

    def _insert_image(self, document: Document, image_path: Path, caption: str) -> None:
        try:
            paragraph = document.add_paragraph(caption or "Hình")
            paragraph.alignment = 1
            document.add_picture(str(image_path), width=Inches(5.5))
        except Exception as exc:  # noqa: BLE001
            paragraph = document.add_paragraph(f"[Không thể chèn hình {image_path.name}: {exc}]")
            paragraph.alignment = 1
