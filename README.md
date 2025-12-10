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
- Per-page Markdown lands in `outputs/<book>/markdown`. The merged Markdown and Docx live beside them so you can keep Kaggle submissions aligned with the original pipeline layout.

## Output Structure

```
outputs/
	book1/
		markdown/
			page_001.md
			book1.md
		docx/
			book1.docx
```

Each page level file preserves the block layout (`## Page n` + `### Block k`) while the aggregated Markdown is used to render the final Docx via `python-docx`.

## Tips

- For Kaggle you can gate the runtime by asking for a single book per run (`--books book3`).
- The prompt in `PAGE_PROMPT` is localized to Vietnamese; adjust it if you need bilingual output or extra JSON metadata.
- If you need faster inference, quantized checkpoints from the same Hugging Face repo also satisfy the API contract (update `--model-id`).
