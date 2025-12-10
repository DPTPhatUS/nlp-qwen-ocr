"""End-to-end OCR pipeline powered by Qwen3-VL-8B."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from docx import Document
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

try:  # transformers exposes a specific Qwen VL head in newer builds
    from transformers import (  # type: ignore
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration as QwenVisionModel,
    )
except ImportError:  # pragma: no cover - fallback for older wheels
    from transformers import AutoModelForVision2Seq as QwenVisionModel  # type: ignore
    from transformers import AutoProcessor  # type: ignore


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

PAGE_PROMPT = (
    "Bạn là chuyên gia OCR cho sách y học cổ truyền Việt Nam. Phân tích bố cục từng trang, "
    "tách các khối (block) theo tiêu đề, đoạn văn, bảng và ghi chú. Trả về Markdown với dạng: "
    "## Page <số trang> rồi tới ### Block <chỉ số>. Giữ nguyên tiếng Việt, chèn bảng ở dạng Markdown, "
    "sử dụng danh sách đánh số nếu cần và bổ sung metadata như loại bài thuốc, thành phần, liều lượng."
)


@dataclass(slots=True)
class OCRConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_new_tokens: int = 1024
    temperature: float = 0.1
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None


class QwenBookOCR:
    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        processor_kwargs = {}
        if config.min_pixels is not None:
            processor_kwargs["min_pixels"] = config.min_pixels
        if config.max_pixels is not None:
            processor_kwargs["max_pixels"] = config.max_pixels
        self.processor = AutoProcessor.from_pretrained(config.model_id, **processor_kwargs)
        model_kwargs = {"torch_dtype": dtype}
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"
        self.model = QwenVisionModel.from_pretrained(config.model_id, **model_kwargs)

    def ocr_page(self, image_path: Path) -> str:
        file_uri = image_path.resolve().as_uri()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": file_uri},
                    {"type": "text", "text": PAGE_PROMPT},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
            )
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated, strict=True)
        ]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0].strip()


def markdown_to_docx(markdown_text: str, output_path: Path) -> None:
    doc = Document()
    in_code = False
    code_lines: List[str] = []
    for line in markdown_text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                doc.add_paragraph("\n".join(code_lines), style="Intense Quote")
                in_code = False
            continue
        if in_code:
            code_lines.append(stripped)
            continue
        if not stripped:
            doc.add_paragraph("")
            continue
        if stripped.startswith("#"):
            level = min(len(stripped) - len(stripped.lstrip("#")), 3)
            heading_text = stripped[level:].strip()
            doc.add_heading(heading_text or "Untitled", level=level)
            continue
        if stripped.startswith(('-', '*', '+')):
            doc.add_paragraph(stripped.lstrip("-*+ "), style="List Bullet")
            continue
        if stripped[0].isdigit() and stripped.split(".", 1)[0].isdigit():
            doc.add_paragraph(stripped, style="List Number")
            continue
        if stripped.startswith(">"):
            doc.add_paragraph(stripped.lstrip("> "), style="Intense Quote")
            continue
        doc.add_paragraph(stripped)
    doc.save(output_path)


class BookPipeline:
    def __init__(self, source_root: Path, output_root: Path, ocr_model: QwenBookOCR) -> None:
        self.source_root = source_root
        self.output_root = output_root
        self.ocr_model = ocr_model

    def run(self, selected_books: Optional[Iterable[str]] = None) -> None:
        if not selected_books:
            book_dirs = sorted(p for p in self.source_root.iterdir() if p.is_dir())
        else:
            selected = {name.lower() for name in selected_books}
            book_dirs = [
                p
                for p in sorted(self.source_root.iterdir())
                if p.is_dir() and p.name.lower() in selected
            ]
        if not book_dirs:
            logging.warning("No book directories found under %s", self.source_root)
            return
        for book_dir in book_dirs:
            self._process_book(book_dir)

    def _process_book(self, book_dir: Path) -> None:
        page_paths = sorted(
            (p for p in book_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS),
            key=lambda p: p.name,
        )
        if not page_paths:
            logging.warning("No supported page files inside %s", book_dir)
            return
        markdown_dir = self.output_root / book_dir.name / "markdown"
        docx_dir = self.output_root / book_dir.name / "docx"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        docx_dir.mkdir(parents=True, exist_ok=True)
        aggregated: List[str] = []
        for idx, page_path in enumerate(
            tqdm(page_paths, desc=f"Processing {book_dir.name}")
        ):
            try:
                page_markdown = self.ocr_model.ocr_page(page_path)
            except Exception as exc:  # pragma: no cover - surface inference failures
                logging.exception("Failed to OCR %s: %s", page_path, exc)
                continue
            page_header = f"## Page {idx + 1}: {page_path.stem}"
            normalized = f"{page_header}\n\n{page_markdown.strip()}"
            (markdown_dir / f"{page_path.stem}.md").write_text(normalized, encoding="utf-8")
            aggregated.append(normalized)
        if not aggregated:
            logging.warning("Skipping Markdown to DOCX conversion for %s (empty output)", book_dir)
            return
        book_markdown = "\n\n".join(aggregated)
        book_markdown_path = markdown_dir / f"{book_dir.name}.md"
        book_markdown_path.write_text(book_markdown, encoding="utf-8")
        docx_path = docx_dir / f"{book_dir.name}.docx"
        markdown_to_docx(book_markdown, docx_path)
        logging.info("Finished %s -> %s", book_dir.name, docx_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR Vietnamese medicine books with Qwen3-VL-8B")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("docs"),
        help="Directory that contains book folders (default: docs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory where Markdown and DOCX results will be stored",
    )
    parser.add_argument(
        "--books",
        nargs="*",
        help="Optional subset of book folder names to process (e.g. book1)",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Hugging Face model repo to use",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens generated per page",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        help="Optional minimum pixel budget for the processor",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        help="Optional maximum pixel budget for the processor",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = OCRConfig(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    ocr_model = QwenBookOCR(config)
    pipeline = BookPipeline(args.source, args.output, ocr_model)
    pipeline.run(args.books)


if __name__ == "__main__":
    main()
