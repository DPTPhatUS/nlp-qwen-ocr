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

## Stage 1 – OCR to Markdown (raw)

```bash
python ocr_to_markdown.py \
	--source docs \
	--output outputs \
	--books book1 \
	--model-id Qwen/Qwen3-VL-8B-Instruct
```

- Omit `--books` to process every book folder inside `docs/`.
- Use `--min-pixels` / `--max-pixels` to trade off speed vs fidelity, mirroring the guidance from the Qwen docs.
- The script now emits per-page Markdown drafts under `outputs/<book>/markdown_raw/page_XXXX.md`, already containing inline `FAKE_x1_y1_x2_y2_type.png` links where the model detected figures/charts/formulas. No intermediate JSON is written anymore, so reruns are lighter.

## Stage 2 – FAKE links to Markdown + assets

```bash
python materialize_fake_links.py \
	--source docs \
	--markdown-root outputs \
	--output outputs \
	--books book1
```

- Scans `outputs/<book>/markdown_raw/` files, detects every `FAKE_x1_y1_x2_y2_type.png` link, crops the corresponding bounding box from the original page image, and rewrites the Markdown link to the real PNG path in `outputs/<book>/assets/page_xxxx/`.
- Writes cleaned per-page Markdown plus a `book.md` glue file under `outputs/<book>/markdown/`, preserving the ability to hand-edit before exporting to DOCX.
- Supports `--start-page` / `--max-pages` just like stage 1 so you can iterate on a subset without invoking the model again.

## Stage 3 – Markdown to DOCX

```bash
python markdown_to_docx.py --output outputs --books book1
```

- Converts each `outputs/<book>/markdown/<book>.md` into `outputs/<book>/docx/<book>.docx` using `python-docx`.
- Safe to rerun whenever you tweak Markdown by hand.

### Helpful CLI flags

- `--books book3 book1` lets you run a single title (or a short list) during Kaggle submissions.
- `--max-pages 5` is handy for smoke tests without burning GPU quota.
- `--start-page 120 --max-pages 10` jumps into the middle of a book and processes exactly 10 pages from that point onward.
- `--device-map cuda:0 --dtype float16` pins inference to a known GPU / precision, while `--temperature` tunes creativity.
- `--model-id Qwen/Qwen3-VL-8B-Instruct` can be swapped for another compatible checkpoint (quantized or fine-tuned) without touching the code.
- Kaggle's dual T4s run smoothly with the defaults: `--attn-impl flash_attention_2 --gpu-mem-limit 14 --load-in-8bit` if you need extra headroom; the script auto-distributes weights across both GPUs via `device_map="auto"` and `max_memory` guards.

## Output Structure

```
outputs/
	book1/
		markdown_raw/
			page_0001.md
		markdown/
			page_0001.md
			book1.md
		assets/
			page_001/
				asset_0001_01.png
		docx/
			book1.docx
```

Each stage writes to its own subfolder so you can resume anywhere in the pipeline. Stage 1 leaves Markdown drafts (with FAKE links) under `markdown_raw/`, while stage 2 materializes those links into real PNGs and final Markdown that downstream tools can consume.

## Figures, charts, formulas

- Qwen3-VL-8B is now prompted to emit Markdown directly. Every detected non-text object becomes `![caption](FAKE_x1_y1_x2_y2_type.png)` so bounding boxes stay encoded inside the filename.
- Stage 2 replays those coordinates against the original page image, clamps them safely, saves the crops to `assets/page_xxxx/asset_XXXX_YY.png`, and rewrites the Markdown links to point to the real PNGs.
- Because the final Markdown already references physical images, the DOCX converter keeps the figures inline without any extra metadata juggling.

## Tips

- For Kaggle you can gate the runtime by asking for a single book per run (`--books book3`).
- The prompt in `PAGE_PROMPT` is localized to Vietnamese; adjust it if you need bilingual output or to request additional structured metadata from Qwen (e.g., explicit table schemas or glossary notes).
- If you need faster inference, quantized checkpoints from the same Hugging Face repo also satisfy the API contract (update `--model-id`).
