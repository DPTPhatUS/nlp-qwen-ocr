"""Model loading and inference helpers for Qwen OCR."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

LOGGER = logging.getLogger(__name__)

PAGE_PROMPT = (
    "Bạn là trợ lý OCR tiếng Việt. Hãy đọc toàn bộ hình trang sách và tạo bản phục dựng Markdown "
    "bảo tồn tiêu đề, chú thích, bảng và công thức. Để giữ nguyên vị trí biểu đồ/hình ảnh, hãy trả về JSON với cấu trúc:\n"
    "{\n"
    '  "markdown": "...Markdown sử dụng placeholder {{ASSET:<id>}} cho từng hình...",\n'
    '  "assets": [\n'
    "    {\n"
    '      "id": "fig-1",\n'
    '      "caption": "Chú thích ngắn gọn",\n'
    '      "type": "figure|chart|formula",\n'
    '      "bbox": [x1, y1, x2, y2]\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "- Markdown phải ở đúng trình tự đọc từ trái sang phải, trên xuống dưới.\n"
    "- Giữ nguyên dấu tiếng Việt và định dạng (## cấp 2 cho tiêu đề cấp trang, ### cho khối con, danh sách với - hoặc 1.).\n"
    "- Nếu không tìm thấy tài liệu, trả về markdown rỗng và assets rỗng.\n"
    "- Không được thêm lời giải thích bên ngoài JSON."
)

INVALID_ESCAPE_RE = re.compile(r"\\(?![\"\\/bfnrtu])")


class QwenOcrClient:
    """Wrapper around AutoProcessor + Qwen3-VL model."""

    def __init__(
        self,
        model_id: str,
        device_map: str,
        torch_dtype: str,
        min_pixels: int,
        max_pixels: int,
        max_new_tokens: int,
        temperature: float,
        attn_impl: str,
        load_in_8bit: bool,
        gpu_mem_limit: float,
        cpu_mem_limit: float,
    ) -> None:
        dtype = self._resolve_dtype(torch_dtype)
        if dtype == "auto":
            dtype = self._default_dtype()
        LOGGER.info("Loading processor %s", model_id)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        LOGGER.info("Loading model %s", model_id)
        model_kwargs: Dict[str, object] = {"trust_remote_code": True}
        if attn_impl and attn_impl != "auto":
            model_kwargs["attn_implementation"] = attn_impl
        if load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except ImportError as exc:  # noqa: F401
                raise RuntimeError(
                    "bitsandbytes is required for --load-in-8bit; install it via pip install bitsandbytes",
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = dtype
        device_map_value = device_map or "auto"
        model_kwargs["device_map"] = device_map_value
        if device_map_value == "auto":
            max_memory = self._build_max_memory_dict(gpu_mem_limit, cpu_mem_limit)
            if max_memory:
                model_kwargs["max_memory"] = max_memory
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = max(0.0, float(temperature))
        self.do_sample = self.temperature > 0

    def _resolve_dtype(self, name: str):
        if name == "auto":
            return "auto"
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping[name]

    def _default_dtype(self):
        return torch.float16 if torch.cuda.is_available() else torch.float32

    def _build_max_memory_dict(self, gpu_limit: float, cpu_limit: float) -> Optional[Dict[object, str]]:
        if not torch.cuda.is_available():
            return None
        limits: Dict[object, str] = {}
        for idx in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            cap = min(max(total_gb - 1, 2.0), gpu_limit) if gpu_limit else total_gb - 1
            limits[idx] = f"{cap:.2f}GiB"
        if cpu_limit:
            limits["cpu"] = f"{cpu_limit:.0f}GiB"
        return limits

    def ocr_page(self, page_num: int, image_path: Path) -> Dict[str, object]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": PAGE_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Trang số {page_num:04d}. Hãy trả về JSON theo yêu cầu."},
                    {"type": "image", "image": str(image_path)},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        vision_inputs, _ = process_vision_info(messages)
        model_inputs = self.processor(
            text=[prompt],
            images=vision_inputs,
            return_tensors="pt",
        ).to(self.model.device)
        gen_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
        generated = self.model.generate(**model_inputs, **gen_kwargs)
        output = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        payload = extract_json_block(output)
        if payload is None:
            LOGGER.warning("Falling back to raw markdown because JSON parse failed")
            return {"markdown": output.strip(), "assets": []}
        return payload


def extract_json_block(text_blob: str) -> Optional[Dict[str, object]]:
    text_blob = text_blob.strip()
    candidates = []
    if "```" in text_blob:
        for block in re.findall(r"```(?:json)?\s*(.*?)```", text_blob, flags=re.S):
            candidates.append(block.strip())
    candidates.append(text_blob)
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        parsed = try_parse_json(candidate)
        if isinstance(parsed, dict):
            return parsed
        idx = candidate.find("{")
        while idx != -1:
            snippet = candidate[idx:]
            parsed = try_parse_json(snippet)
            if isinstance(parsed, dict):
                return parsed
            idx = candidate.find("{", idx + 1)
    return None


def try_parse_json(candidate: str) -> Optional[Dict[str, object]]:
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        sanitized = INVALID_ESCAPE_RE.sub(lambda m: "\\\\" + m.group(0)[1:], candidate)
        if sanitized != candidate:
            try:
                parsed = json.loads(sanitized)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None
