# Qwen OCR Pipeline

Pipeline for turning scanned Vietnamese traditional medicine books into Markdown and DOCX using Qwen3-VL-8B.

## Why Qwen3-VL-8B?

Qwen VL models ship native layout-aware OCR via the `process_vision_info` helper and chat templates described in the [official Hugging Face quickstart](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). The same interface applies to Qwen3-VL-8B-Instruct, so we can reuse the documented `AutoProcessor` + `generate` workflow and simply swap the checkpoint id.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Make sure the machine has enough VRAM (24 GB+ works well for FP16) or enable CPU inference with smaller batches.

Key runtime dependencies bundled in `pyproject.toml`:

- `accelerate` enables `device_map="auto"` placement across GPUs/CPUs.
- `safetensors` ensures faster, secure checkpoint loading from Hugging Face.
- `sentencepiece` and `huggingface-hub` cover tokenizer + model download needs for Qwen3-VL.
- `torchvision` and `wrapt` are needed because `qwen-vl-utils` taps torchvision transforms inside Kaggle/colab runners.
- `bitsandbytes` is optional; install it if you plan to pass `--load-in-8bit` for tighter VRAM budgets.

## Running OCR

```bash
python main.py \
  --source docs \
  --output outputs \
  --books book1 \
  --model-id Qwen/Qwen3-VL-8B-Instruct
```

- Omit `--books` to process every book folder inside `docs/`.
- Use `--min-pixels` / `--max-pixels` to trade off speed vs fidelity, mirroring the guidance from the Qwen docs.
- Every page gets its own Markdown file as well as a stitched `book.md` and DOCX so you can keep Kaggle submissions aligned with the original pipeline layout.

### Helpful CLI flags

- `--books book3 book1` lets you run a single title (or a short list) during Kaggle submissions.
- `--max-pages 5` is handy for smoke tests without burning GPU quota.
- `--device-map cuda:0 --dtype float16` pins inference to a known GPU / precision, while `--temperature` tunes creativity.
- `--model-id Qwen/Qwen3-VL-8B-Instruct` can be swapped for another compatible checkpoint (quantized or fine-tuned) without touching the code.
- Kaggle's dual T4s run smoothly with the defaults: `--attn-impl flash_attention_2 --gpu-mem-limit 14 --load-in-8bit` if you need extra headroom; the script auto-distributes weights across both GPUs via `device_map="auto"` and `max_memory` guards.

## Output Structure

```
outputs/
	book1/
		markdown/
			page_001.md
			book1.md
		assets/
			page_001/
				asset_fig-1.png
		docx/
			book1.docx
```

Each page-level Markdown preserves the block layout (`## Page n` + `### Block k`). When the model tags a figure/chart/formula using `{{ASSET:...}}`, the pipeline crops the bounding box from the scanned page, saves it under `assets/`, and rewrites the placeholder as a Markdown image pointing at the cropped PNG. The aggregated Markdown is then rendered to DOCX via `python-docx`, which now embeds those figures directly.

## Figures, charts, formulas

- Qwen3-VL-8B is prompted to return JSON with a `markdown` string plus an `assets` array describing each figure/chart/formula along with absolute pixel bounding boxes.
- Bounding boxes are clamped to the page size and cropped with Pillow; filenames are slugged per asset id to keep per-page folders clean.
- Placeholders such as `{{ASSET:fig-1}}` are automatically replaced by `![caption](../assets/page/asset_fig-1.png)` so the Markdown for that page links the correct figure inline.
- The DOCX converter scans those Markdown image links and inserts the underlying PNG into the Word output, so figures travel with the text end-to-end.

## Tips

- For Kaggle you can gate the runtime by asking for a single book per run (`--books book3`).
- The prompt in `PAGE_PROMPT` is localized to Vietnamese; adjust it if you need bilingual output or to request additional metadata from Qwen (e.g., extra JSON fields per block).
- If you need faster inference, quantized checkpoints from the same Hugging Face repo also satisfy the API contract (update `--model-id`).
