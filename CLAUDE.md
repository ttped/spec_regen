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
| 7. tables | `table_processor_agent.py` | Reconstructs table structure from YOLO crops using OCR + LLM |
| 8. write | `docx_writer.py` | Writes final .docx from structured elements |
| 9. validate | `validation_agent.py` | Validates TOC coverage and section precision |

### Key Modules

- `simple_pipeline.py` — Main pipeline entry point (run this)
- `utils.py` — Shared utilities including `load_json_with_recovery`
- `ocr_to_table.py` + `complex_table_schema.py` — OCR grid → Word table conversion
- `consolidate_columns.py` — Cleans up empty columns in OCR table grids
- `feature_extractor.py` — Extracts features for ML training from structured documents
- `prepare_yolo_dataset.py` / `yolo_benchmark.py` — YOLO model tooling

## Running the Pipeline

All commands should be run from the `app/` directory:

```bash
cd app/

# Full pipeline (YOLO enabled by default)
python simple_pipeline.py

# Run a specific step only
python simple_pipeline.py --step classify
python simple_pipeline.py --step tables

# Force re-run YOLO even if exports exist
python simple_pipeline.py --force-yolo

# Adjust YOLO settings
python simple_pipeline.py --yolo-conf 0.35 --yolo-device cuda:0
```

### LLM Provider

Configured at the top of `simple_pipeline.py` via `LLM_CONFIG`. Two options:

- **Ollama (local)** — default, uses `gemma3:27b` at `http://localhost:11434`
- **Mission Assist (remote)** — uncomment the second block for remote inference

## Directory Structure Expected at Runtime

```
app/                          # All source code
iris_ocr/
  CM_Spec_OCR_and_figtab_output/
    raw_data_advanced/        # Input: raw OCR JSON files (one per document)
docs_images/                  # Input: page images for YOLO (named {doc}_page{N}.png)
yolo_exports/                 # Output: YOLO-extracted figure/table crops
results_simple/               # Output: intermediate JSONs and final .docx files
tests/
  test_data/                  # Test fixtures
```

## Running Tests

Tests live in `tests/` and are run from the **project root**:

```bash
# Table pipeline test (generates a .docx in tests/test_outputs/)
python -m tests.test_ocr_to_table

# Complex table rendering test
python -m tests.test_complex_tables

# Diagnostic scripts (not automated tests, run manually)
python tests/diagnose_pipeline.py
python tests/diagnose_page.py
```

Tests require `tests/test_data/table_json_example.json` to exist.

## ML Classifier Workflow

The section classifier (`section_classifier.py`) uses XGBoost to remove false-positive section detections. It requires labeled training data.

```bash
# 1. Extract features after running the structure step
python app/feature_extractor.py --results_dir results_simple

# 2. Open results_simple/training_features.csv and add _label column
#    1 = valid section, 0 = false positive

# 3. Train + evaluate
python app/section_classifier.py train -f results_simple/training_features.csv -o section_model.joblib
python app/section_classifier.py evaluate -f results_simple/training_features.csv
```

See `app/ML_CLASSIFIER_README.md` for full details.

## Dependencies

Core Python packages (install via pip):

```
python-docx
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
- LLM calls (classify, title, table OCR) are the slowest steps and require a running Ollama instance or network access to the remote endpoint.
