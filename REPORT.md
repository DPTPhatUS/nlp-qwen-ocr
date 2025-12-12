# OCR Processing Report (12 Dec 2025)

## Processing Summary
We finished the OCR-to-Markdown flow for the currently scoped books (book1 and book3) using the five-stage toolchain documented in [README.md](README.md). Stages 1–3 have been executed end-to-end, yielding cleaned Markdown plus linked page assets; stages 4–5 are staged but still pending so that layout-specific break rules and DOCX packaging can be tuned once the editorial review wraps. The run produced 622 Markdown pages (259 for book1, 363 for book3) sourced from 622 scanned page JSON descriptors inside `docs/`.

## Workflow & Key Steps
1. **Stage 1 – OCR to Markdown** ([ocr_to_markdown.py](ocr_to_markdown.py))
   - Loaded Qwen/Qwen3-VL-8B-Instruct through `QwenOcrClient` with layout-aware prompts so each PDF page (JSON + image) yielded a Markdown draft under `outputs/<book>/markdown_raw`.
   - Processed page ranges 1..259 for book1 and 1..363 for book3; book2 is queued but not yet batched through the model.
2. **Stage 2 – Trim Scanner Prompts** ([trim_markdown_raw.py](trim_markdown_raw.py))
   - Removed the first 12 boilerplate lines from every raw Markdown file and kept untouched backups in `outputs_backup/` for auditing.
3. **Stage 3 – Materialize FAKE Links** ([materialize_fake_links.py](materialize_fake_links.py))
   - Replayed each FAKE bounding box against the `docs/<book>/crops` images, matched descriptions to crops via open_clip's `xlm-roberta-base-ViT-B-32` encoder, and rewrote the Markdown to point at `outputs/<book>/assets/page_xxxx/*.png`.
   - Aggregated page-level Markdown into per-book glue files (`outputs/book*/markdown/book*.md`).
4. **Stage 4 – Insert Break Tags** ([insert_breaks.py](insert_breaks.py))
   - Not yet executed; rules exist for all three books to drop `</break>` before heading transitions so DOCX page breaks survive.
5. **Stage 5 – Markdown to DOCX** ([markdown_to_docx.py](markdown_to_docx.py))
   - Pending run. `MarkdownDocxExporter` in [nlp_qwen_ocr/docx_exporter.py](nlp_qwen_ocr/docx_exporter.py) converts the cleaned Markdown (with references to local PNG assets) into paginated DOCX deliverables.

## Tools & Algorithms
- **Model inference**: Hugging Face `transformers` + `accelerate` to stream Qwen/Qwen3-VL-8B-Instruct with `process_vision_info`, clipping pixels between 512² and 2048² per page, greedy decoding (`temperature=0`).
- **Pre/Post processing**: Pure Python scripts with `pathlib`, `json`, and regex-heavy parsing to normalize headings, enforce naming, and keep runs idempotent.
- **Image-to-text matching**: `open_clip` multilingual encoder (`xlm-roberta-base-ViT-B-32/laion5b_s13b_b90k`) to score FAKE link descriptions against available crops before copying assets into the `outputs/<book>/assets` tree.
- **Document packaging**: `python-docx` (via `MarkdownDocxExporter`) ready to consume the break-tagged Markdown once Stage 4 runs.

## Data Volume & Coverage
| Book  | JSON pages in `docs/` | Raw Markdown pages | Final Markdown pages | PNG assets copied |
|-------|-----------------------|--------------------|----------------------|-------------------|
| book1 | 259                   | 259                | 259                  | 1 |
| book2 | 686                   | 686  | 686                    | 0 *(not run yet)* |
| book3 | 363                   | 363                | 363                  | 33 |

Totals: 1,308 source pages inventoried, 622 pages OCR'd/trimmed/cleaned, and 33 page-level illustrations materialized into the assets tree.

## Sample Transformations
1. **Header trimming (Stage 2)**
   - *Before* – raw model transcript preserved the full system/user prompt envelope in [outputs_backup/book1/markdown_raw/page_0001.md](outputs_backup/book1/markdown_raw/page_0001.md#L1-L22):

```
system
Bạn là trợ lý OCR tiếng Việt. Hãy đọc toàn bộ hình trang sách và trả về DUY NHẤT nội dung Markdown.
...
assistant
# BỆNH HỌC VÀ ĐIỀU TRỊ
Đông Y
```

   - *After* – the trimmed version in [outputs/book1/markdown_raw/page_0001.md](outputs/book1/markdown_raw/page_0001.md#L1-L18) drops the injected prompt, leaving only book content for downstream diffs:

```
# BỆNH HỌC VÀ ĐIỀU TRỊ
Đông Y

SÁCH ĐÀO TẠO BÁC SĨ Y HỌC CỔ TRUYỀN
...
BỆNH HỌC VÀ ĐIỀU TRỊ ĐÔNG Y
```

2. **FAKE link materialization (Stage 3)**
   - *Before* – Stage 1 emitted an inline figure placeholder for the Taiji diagram in [outputs/book3/markdown_raw/page_0014.md](outputs/book3/markdown_raw/page_0014.md#L1-L14):

```
![hình đồ thái cực](FAKE_587_687_827_877_figure.png)
```

   - *After* – Stage 3 cropped the bounding box, stored it under `assets/page_0014/`, and rewrote the Markdown reference in [outputs/book3/markdown/page_0014.md](outputs/book3/markdown/page_0014.md#L1-L14):

```
[../assets/page_0014/asset_0014_01.png](../assets/page_0014/asset_0014_01.png)
```

## Next Actions
- Run [insert_breaks.py](insert_breaks.py) once heading rules are validated so DOCX pagination honors the book-specific hierarchy.
- Convert the validated Markdown into DOCX files with [markdown_to_docx.py](markdown_to_docx.py), then archive hashes of both Markdown + DOCX deliverables for reproducibility.
- Schedule the remaining 686 pages of book2 through the same three stages so the corpus is consistent across all titles.
