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

## Stage 1 – OCR to JSON

```bash
python main.py \
	--source docs \
	--output outputs \
	--books book1 \
	--model-id Qwen/Qwen3-VL-8B-Instruct
```

- Omit `--books` to process every book folder inside `docs/`.
- Use `--min-pixels` / `--max-pixels` to trade off speed vs fidelity, mirroring the guidance from the Qwen docs.
- The script now stops after saving Qwen’s raw JSON response per page to `outputs/<book>/json/page_XXXX.json` so you can rerun later stages without re-OCRing.

## Stage 2 – JSON to Markdown

```bash
python json_to_markdown.py \
	--source docs \
	--json-root outputs \
	--output outputs \
	--books book1
```

- Reads `outputs/<book>/json/` artifacts, crops any referenced assets, and writes Markdown files under `outputs/<book>/markdown/` plus a stitched `book.md`.
- Supports `--start-page` / `--max-pages` as well, so you can iterate on a subset without invoking the model.

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
		json/
			page_0001.json
		markdown/
			page_001.md
			book1.md
		assets/
			page_001/
				asset_fig-1.png
		docx/
			book1.docx
```

Each stage writes to its own subfolder so you can resume anywhere in the pipeline. Page-level Markdown preserves the block layout and still rewrites any `{{ASSET:...}}` placeholder after the JSON stage.

## Figures, charts, formulas

- Qwen3-VL-8B is prompted to return JSON with a `markdown` string plus an `assets` array describing each figure/chart/formula along with absolute pixel bounding boxes.
- Bounding boxes are clamped to the page size and cropped with Pillow; filenames are slugged per asset id to keep per-page folders clean.
- Placeholders such as `{{ASSET:fig-1}}` are automatically replaced by `![caption](../assets/page/asset_fig-1.png)` so the Markdown for that page links the correct figure inline.
- The DOCX converter scans those Markdown image links and inserts the underlying PNG into the Word output, so figures travel with the text end-to-end.

## Tips

- For Kaggle you can gate the runtime by asking for a single book per run (`--books book3`).
- The prompt in `PAGE_PROMPT` is localized to Vietnamese; adjust it if you need bilingual output or to request additional metadata from Qwen (e.g., extra JSON fields per block).
- If you need faster inference, quantized checkpoints from the same Hugging Face repo also satisfy the API contract (update `--model-id`).
