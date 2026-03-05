# CLAUDE.md

## Project Overview

**spec_regen** is a document processing pipeline that converts technical specification PDFs (via OCR JSON output) into formatted Word (.docx) documents. It automates heading detection, table reconstruction, figure extraction, and document structure — reducing what was otherwise an entirely manual re-typing process.

## Architecture

All application code lives in `app/`. Tests live in `tests/`. The pipeline is orchestrated by `simple_pipeline.py`.

### Pipeline Steps (in order)

| Step | Script | Description |
|------|--------|-------------|
| 1. classify | `classify_agent.py` | Identifies document type and content start page using LLM |
| 2. title | `title_agent.py` | Extracts title page metadata from page 1 |
| 3. structure | `section_processor.py` | Detects and organizes headings/sections from OCR text |
| 4. ml_filter | `section_classifier.py` | XGBoost model filters out false-positive section detections |
| 5. yolo | `yolo_asset_extractor.py` | DocLayout-YOLO detects and crops figures/tables from page images |
| 6. assets | `asset_processor.py` | Integrates extracted figure/table images into the element list |
| 7. tables | `table_iris_processor.py` | Matches IRIS table deliverables (Excel/JSON) to table elements |
| 8. write | `docx_writer.py` | Writes final .docx from structured elements |
| 9. validate | `validation_agent.py` | Validates TOC coverage and section precision |

### Key Modules

- `simple_pipeline.py` — Main pipeline entry point. Run from project root: `uv run py -i .\app\simple_pipeline.py`
- `asset_processor.py` — Integrates YOLO assets into element stream. Contains `resolve_asset_directory()` for fuzzy stem matching (spaces ↔ underscores).
- `table_iris_processor.py` — **Replaced `table_processor_agent.py`**. No LLM dependency. Reads pre-built Excel files from IRIS deliverable. Falls back to direct IRIS JSON conversion. Extracts table captions from IRIS metadata.
- `docx_writer.py` — Renders elements to Word. Routes Excel-format tables through `add_excel_table_to_docx` (preserves column widths). Wide tables (>7 cols) get landscape pages via `add_isolated_landscape_table`.
- `complex_table_schema.py` — Rich table renderer supporting merged cells, alignment, shading. Used as fallback when table_data is in complex schema format (not Excel format).
- `excel_reader.py` — Standalone Excel→Word converter for testing. The pipeline uses the same logic internally.
- `reconstruct_to_df.py` — IRIS tool that converts IRIS JSON → Excel. Lives in `app/`. Has internal dependency issues (`header.utils`) that prevent direct import; was previously called as subprocess but is now unnecessary since IRIS deliverables include pre-built Excel files.
- `utils.py` — Shared utilities including `load_json_with_recovery`

## Coding Standards

1. **Fail Fast:** Do not use defensive try-except blocks. Let errors crash the program so they are visible. API or external calls are an exception to this rule.
2. **Scoped & Pure:** Avoid global state. Use dependency injection. Write small, testable, pure functions.
3. **Use Design Patterns:** Medallion architecture is great for data and machine learning workflows.

## Running the Pipeline

Run from the **project root** (`C:\spec_regen>`):

```bash
# Full pipeline
uv run py -i .\app\simple_pipeline.py

# Run a specific step only
uv run py -i .\app\simple_pipeline.py --step tables
uv run py -i .\app\simple_pipeline.py --step write

# Force re-run YOLO even if exports exist
uv run py -i .\app\simple_pipeline.py --force-yolo

# Override IRIS table deliverable location
uv run py -i .\app\simple_pipeline.py --table-jsons-dir path/to/table_jsons

# Adjust YOLO settings
uv run py -i .\app\simple_pipeline.py --yolo-conf 0.35 --yolo-device cuda:0
```

### LLM Provider

Configured at the top of `simple_pipeline.py` via `LLM_CONFIG`. Two options:

- **Ollama (local)** — default, uses `gemma3:27b` at `http://localhost:11434`
- **Mission Assist (remote)** — uncomment the second block for remote inference

LLM is only used in steps 1 (classify) and 2 (title). Tables no longer use LLM.

## Directory Structure Expected at Runtime

```
C:\spec_regen\                # Project root — run commands from here
├── app/                      # All source code
│   ├── simple_pipeline.py
│   ├── asset_processor.py
│   ├── table_iris_processor.py
│   ├── docx_writer.py
│   ├── complex_table_schema.py
│   ├── excel_reader.py
│   ├── reconstruct_to_df.py
│   └── ...
├── iris_ocr/
│   └── CM_Spec_OCR_and_figtab_output/
│       ├── raw_data_advanced/    # Input: raw OCR JSON files (one per document)
│       └── table_jsons/          # IRIS table deliverable (see below)
├── docs_images/                  # Input: page images for YOLO (named {doc}_page{N}.png)
├── yolo_exports/                 # Output: YOLO-extracted figure/table crops
│   └── {doc_name}/              # One subfolder per document
│       ├── {doc}_tab_p045_003.png
│       ├── {doc}_tab_p045_003.json  # Asset metadata
│       └── ...
├── results_simple/               # Output: intermediate JSONs and final .docx files
│   ├── {doc}_classification.json
│   ├── {doc}_title.json
│   ├── {doc}_organized.json
│   ├── {doc}_ml_filtered.json
│   ├── {doc}_with_assets.json
│   ├── {doc}_with_tables.json
│   └── {doc}.docx
└── tests/
    └── test_data/
```

### IRIS Table Deliverable Structure

The IRIS deliverable lives at `iris_ocr/CM_Spec_OCR_and_figtab_output/table_jsons/` and follows this layout:

```
table_jsons/
├── {doc_name}/
│   ├── excel/
│   │   ├── {doc}_page045_tab_layout_0.xlsx    # Pre-built Excel tables (primary)
│   │   ├── {doc}_page045_tab_caption_1.xlsx
│   │   └── ...
│   └── table_jsons/
│       ├── {doc}_page045_tab_layout_0.json    # IRIS metadata (fallback + captions)
│       ├── {doc}_page045_tab_caption_1.json
│       └── ...
```

### Naming Convention Mapping

YOLO crops and IRIS files use different naming. The pipeline bridges them via canonical key extraction `(doc, page, table_id)`:

| Source | Pattern | Example |
|--------|---------|---------|
| YOLO crops | `{doc}_tab_p{page}_{id}` | `file_name_tab_p045_003.png` |
| IRIS Excel | `{doc}_page{page}_tab_layout_{id}` | `file_name_page045_tab_layout_0.xlsx` |
| IRIS Excel | `{doc}_page{page}_tab_caption_{id}` | `file_name_page045_tab_caption_1.xlsx` |
| IRIS JSON | Same as Excel but `.json` | `file_name_page045_tab_layout_0.json` |

**Known issue:** YOLO and IRIS may use different table ID numbering (YOLO `_003` vs IRIS `_0`). If tables aren't matching, check the canonical keys in log output.

### Space/Underscore Directory Name Mismatch

The YOLO extractor sanitizes directory names (spaces → underscores). `asset_processor.resolve_asset_directory()` handles this by fuzzy-matching directory names. `simple_pipeline.py` resolves the correct YOLO subdirectory name once per document into `figures_stem` and passes it to all downstream steps.

## Table Processing Pipeline (Step 7)

`table_iris_processor.py` handles table rendering. It does NOT use LLM. Flow:

1. Builds two indexes by scanning `table_jsons_dir`:
   - Excel index: `*/excel/*.xlsx` (primary)
   - JSON index: `*/table_jsons/*.json` (fallback + caption source)
2. For each table element in the stream:
   - Extracts `caption_text` from IRIS JSON metadata (`table_title` or `context.title.title`)
   - Looks up pre-built Excel by canonical key match
   - If Excel found: reads with openpyxl → `{"columns": [{"width": ...}], "rows": [[...]]}` format
   - If no Excel: reads IRIS JSON → `complex_table_schema` rich format
   - If neither: element keeps its YOLO crop image for fallback rendering
3. Sets `_render_landscape = True` for tables with >7 columns

### Table Rendering in docx_writer.py

The `create_docx_from_elements` function routes tables based on format:

- **Excel format** (columns have `"width"` key): `add_excel_table_to_docx()` — scales column widths proportionally to fill full page width (6.5" portrait, 9" landscape)
- **Complex schema format** (columns have `"name"` key or are strings): `add_docx_table_from_data()` → `complex_table_schema.add_complex_table()`
- **Landscape tables** (`_render_landscape` flag): `add_isolated_landscape_table()` — creates landscape section, renders table + caption, then reverts to portrait
- **Image fallback** (no `table_data`): renders the YOLO crop image with `doc.add_picture()`

## Config Reference

`simple_pipeline.py` `Config` class:

```python
class Config:
    step = "all"
    raw_ocr_dir = "iris_ocr/CM_Spec_OCR_and_figtab_output/raw_data_advanced"
    images_dir = "docs_images"
    yolo_exports_dir = "yolo_exports"
    results_dir = "results_simple"
    table_jsons_dir = "iris_ocr/CM_Spec_OCR_and_figtab_output/table_jsons"
    ml_threshold = 0.5
    use_yolo = True
    yolo_confidence = 0.25
    yolo_device = 'cpu'
    skip_yolo_if_exists = True
```

## Running Tests

Tests live in `tests/` and are run from the **project root**:

```bash
python -m tests.test_ocr_to_table
python -m tests.test_complex_tables
python tests/diagnose_pipeline.py
python tests/diagnose_page.py
```

## Dependencies

Core Python packages:

```
python-docx
openpyxl
xgboost
scikit-learn
pandas
numpy
joblib
spacy (en_core_web_sm)
doclayout-yolo (or ultralytics)
```

spaCy model: `python -m spacy download en_core_web_sm`

## Important Notes

- **No `requirements.txt` exists yet** — dependencies must be installed manually.
- The `legacy/` subdirectory contains older algorithm versions; these are not part of the active pipeline.
- `Config` class in `simple_pipeline.py` is the primary configuration point — edit it directly rather than using CLI args for persistent changes.
- YOLO extraction is skipped by default if `yolo_exports/` already contains output. Use `--force-yolo` to re-run.
- LLM calls (classify, title) are the slowest steps and require a running Ollama instance or network access to the remote endpoint. Tables no longer require LLM.
- `table_processor_agent.py` is **deprecated** — replaced by `table_iris_processor.py`.
- `reconstruct_to_df.py` has an import dependency on `header.utils` that can be commented out. It is no longer called by the pipeline since IRIS deliverables now include pre-built Excel files.