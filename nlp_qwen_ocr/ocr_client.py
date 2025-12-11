"""Model loading and inference helpers for Qwen OCR."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

LOGGER = logging.getLogger(__name__)

PAGE_PROMPT = (
    "Bạn là trợ lý OCR tiếng Việt. Hãy đọc toàn bộ hình trang sách và trả về DUY NHẤT nội dung Markdown.\n"
    "- Giữ nguyên cấu trúc tiêu đề, đoạn, danh sách, bảng và công thức (# cho tiêu đề trang, ## cho mục con).\n"
    "- Với mỗi hình/biểu đồ/ảnh/chú thích hay công thức độc lập, chèn ngay tại vị trí đó cú pháp Markdown: "
    "![mô tả ngắn](FAKE_x1_y1_x2_y2_loai.png).\n"
    "  * x1,y1,x2,y2 là toạ độ pixel (trên ảnh gốc) theo hệ toạ độ góc trái trên, x1<x2, y1<y2.\n"
    "  * 'loai' là figure | chart | formula | photo, chỉ dùng chữ thường và không dấu.\n"
    "  * Không thêm đường dẫn hay thư mục trước FAKE_...\n"
    "- Nếu không có hình/biểu đồ/công thức thì không chèn FAKE link nào.\n"
    "- Không giải thích thêm, không bọc trong ```json``` hay văn bản khác."
)


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

    def ocr_page(self, page_num: int, image_path: Path) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": PAGE_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Trang số {page_num:04d}. Trả về Markdown theo yêu cầu."},
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
        return output.strip()
