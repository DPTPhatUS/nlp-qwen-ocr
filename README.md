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

## Workflow overview

1. `ocr_to_markdown.py` — run Qwen3-VL-8B over every scanned page to get raw Markdown with inline FAKE links.
2. `trim_markdown_raw.py` — strip noisy page headers/footers from each raw Markdown file so downstream diffs stay clean.
3. `materialize_fake_links.py` — crop figures from the source scans and swap FAKE placeholders for real asset paths.
4. `insert_breaks.py` — add `</break>` tags between heading tiers so DOCX exports can preserve intentional section breaks.
5. `markdown_to_docx.py` — stitch the cleaned Markdown into DOCX deliverables per book.

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

## Stage 2 – Trim raw Markdown headers

```bash
python trim_markdown_raw.py \
	--root outputs \
	--lines 12 \
	--ext .md
```

- Removes the first `N` lines (defaults to 12) from every file inside `outputs/<book>/markdown_raw/` so boilerplate scanner headers and duplicated prompts disappear before diffing.
- Accepts `--ext` filters (for example `--ext .md`) and `--dry-run` so you can preview which files would change when tuning the safe trim length.
- Run this immediately after OCR; it is idempotent as long as you keep `N` consistent.

## Stage 3 – FAKE links to Markdown + assets

```bash
python materialize_fake_links.py \
	--source docs \
	--markdown-root outputs \
	--output outputs \
	--books book1
```

- Scans the trimmed `outputs/<book>/markdown_raw/` files, detects every `FAKE_x1_y1_x2_y2_type.png` link, crops the corresponding bounding box from the original page image, and rewrites the Markdown link to the real PNG path in `outputs/<book>/assets/page_xxxx/`.
- Writes cleaned per-page Markdown plus a `book.md` glue file under `outputs/<book>/markdown/`, preserving the ability to hand-edit before exporting to DOCX.
- Supports `--start-page` / `--max-pages` just like stage 1 so you can iterate on a subset without invoking the model again.

## Stage 4 – Insert break tags

```bash
python insert_breaks.py --books book1 book3
```

- Looks at each `outputs/<book>/markdown/<book>.md`, classifies headings via regex rules per title, and inserts a standalone `</break>` line before each `h1`/`h2` run.
- Ensures page/section breaks survive when flowing into DOCX; the converter turns `</break>` into paragraph breaks downstream.
- Supports `--dry-run` so you can confirm how many tags would be added before rewriting files.

## Stage 5 – Markdown to DOCX

```bash
python markdown_to_docx.py --output outputs --books book1
```

- Converts each fully-processed `outputs/<book>/markdown/<book>.md` into `outputs/<book>/docx/<book>.docx` using `python-docx`.
- Safe to rerun whenever you tweak Markdown by hand.

### Helpful CLI flags

- `--books book3 book1` lets you run a single title (or a short list) during Kaggle submissions.
- `--max-pages 5` is handy for smoke tests without burning GPU quota.
- `--start-page 120 --max-pages 10` jumps into the middle of a book and processes exactly 10 pages from that point onward.
- `--device-map cuda:0 --dtype float16` pins inference to a known GPU / precision, while `--temperature` tunes creativity.
- `--model-id Qwen/Qwen3-VL-8B-Instruct` can be swapped for another compatible checkpoint (quantized or fine-tuned) without touching the code.
- Kaggle's dual T4s run smoothly with the defaults: `--gpu-mem-limit 14 --load-in-8bit` if you need extra headroom; the script auto-distributes weights across both GPUs via `device_map="auto"` and `max_memory` guards.

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

Each stage writes to its own subfolder so you can resume anywhere in the pipeline. Stage 1 leaves Markdown drafts (with FAKE links) under `markdown_raw/`, Stage 3 materializes those links into real PNGs and final Markdown, and Stage 4 edits the per-book glue file in place so DOCX exports inherit the break markers.

## Figures, charts, formulas

- Qwen3-VL-8B is now prompted to emit Markdown directly. Every detected non-text object becomes `![caption](FAKE_x1_y1_x2_y2_type.png)` so bounding boxes stay encoded inside the filename.
- Stage 3 replays those coordinates against the original page image, clamps them safely, saves the crops to `assets/page_xxxx/asset_XXXX_YY.png`, and rewrites the Markdown links to point to the real PNGs.
- Because the final Markdown already references physical images, the DOCX converter keeps the figures inline without any extra metadata juggling.

## Tips

- For Kaggle you can gate the runtime by asking for a single book per run (`--books book3`).
- The prompt in `PAGE_PROMPT` is localized to Vietnamese; adjust it if you need bilingual output or to request additional structured metadata from Qwen (e.g., explicit table schemas or glossary notes).
- If you need faster inference, quantized checkpoints from the same Hugging Face repo also satisfy the API contract (update `--model-id`).
