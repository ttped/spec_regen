"""
simple_pipeline_yolo.py - Document processing pipeline with YOLO-based asset extraction.

This is an enhanced version of simple_pipeline.py that adds automatic figure/table
detection using DocLayout-YOLO. The YOLO step replaces the need for manually
exported figures.

Steps: 
    classify -> title -> structure -> ml_filter -> [yolo_extract] -> assets -> tables -> write -> validate

New step:
    yolo_extract: Runs YOLO on page images to detect and crop figures/tables

Usage:
    python simple_pipeline_yolo.py                    # Run full pipeline with YOLO
    python simple_pipeline_yolo.py --skip-yolo        # Skip YOLO (use manual exports)
    python simple_pipeline_yolo.py --step yolo        # Run only YOLO extraction
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional


def _load_env(env_path: Path) -> None:
    """Load key=value pairs from a .env file into os.environ (no-op if missing)."""
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip())

_load_env(Path(__file__).resolve().parent.parent / ".env")


# Direct imports (existing)
from classify_agent import run_classification_on_file, load_content_start_page, load_classification_result
from title_agent import extract_title_page_info
from asset_processor import run_asset_integration
from table_iris_processor import run_iris_table_processing
from docx_writer import run_docx_creation
from section_processor import run_section_processing_on_file
from section_classifier import train_and_predict
from validation_agent import run_validation_on_file
from utils import load_json_with_recovery


from yolo_asset_extractor import run_yolo_extraction, get_yolo_exports_dir


# =============================================================================
# CONFIG  (values come from .env at project root — no fallbacks)
# =============================================================================

def _require_env(key: str) -> str:
    """Return an environment variable or raise with a clear message."""
    val = os.environ.get(key)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {key}  (check your .env file)")
    return val

DEFAULT_RAW_OCR_DIR      = _require_env("RAW_OCR_DIR")
DEFAULT_FIGURES_DIR      = os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "exports")  # Manual exports (unused)
DEFAULT_IMAGES_DIR       = _require_env("IMAGES_DIR")
DEFAULT_YOLO_EXPORTS_DIR = _require_env("YOLO_EXPORTS_DIR")
DEFAULT_TABLE_JSONS_DIR  = _require_env("TABLE_JSONS_DIR")
DEFAULT_RESULTS_DIR      = _require_env("RESULTS_DIR")

LLM_CONFIG = {
    "provider":   _require_env("LLM_PROVIDER"),
    "model_name": _require_env("LLM_MODEL"),
    "base_url":   _require_env("LLM_BASE_URL"),
    "api_key":    os.environ.get("LLM_API_KEY") or None,
}


def has_table_crops(yolo_exports_dir: str, doc_stem: str) -> bool:
    """Check if YOLO exported any table crops for this document."""
    from asset_processor import resolve_asset_directory
    doc_dir = resolve_asset_directory(yolo_exports_dir, doc_stem)
    if not doc_dir or not os.path.isdir(doc_dir):
        return False
    for fname in os.listdir(doc_dir):
        if not fname.endswith('.json'):
            continue
        meta_path = os.path.join(doc_dir, fname)
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('asset_type') in ('table', 'tab', 'table_layout', 'tab_layout'):
            return True
    return False


def iris_ready_for_doc(table_jsons_dir: str, doc_stem: str) -> bool:
    """Check if IRIS table OCR deliverable exists for this document."""
    import re
    if not os.path.isdir(table_jsons_dir):
        return False
    doc_norm = re.sub(r'[\s_-]+', '', doc_stem.lower())
    for entry in os.listdir(table_jsons_dir):
        entry_path = os.path.join(table_jsons_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        entry_norm = re.sub(r'[\s_-]+', '', entry.lower())
        if entry_norm == doc_norm:
            excel_dir = os.path.join(entry_path, 'excel')
            json_dir = os.path.join(entry_path, 'table_jsons')
            has_excel = os.path.isdir(excel_dir) and any(f.endswith('.xlsx') for f in os.listdir(excel_dir))
            has_json = os.path.isdir(json_dir) and any(f.endswith('.json') for f in os.listdir(json_dir))
            return has_excel or has_json
    return False


def get_document_stems(input_dir: str) -> List[str]:
    stems = set()
    try:
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                stems.add(os.path.splitext(filename)[0])
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found.")
        return []
    return sorted(list(stems))


def get_document_stems_from_images(images_dir: str) -> List[str]:
    """Get document stems from page images directory."""
    import re
    stems = set()
    
    if not os.path.exists(images_dir):
        return []
    
    for filename in os.listdir(images_dir):
        match = re.match(r'^(.+)_page\d+\.[a-zA-Z]+$', filename)
        if match:
            stems.add(match.group(1))
    
    return sorted(list(stems))


def extract_title_from_first_page(raw_input_path: str, output_path: str):
    """Extract title page information from page 1."""
    print(f"  Loading: {raw_input_path}")
    
    data = load_json_with_recovery(raw_input_path)

    page_text = ""
    if isinstance(data, list) and len(data) > 0:
        data.sort(key=lambda x: int(x.get('page_num', 0) or x.get('page_Id', 0) or 0))
        first_item = data[0]
        page_dict = first_item.get('page_dict', first_item)
        text_list = page_dict.get('text', [])
        if isinstance(text_list, list):
            page_text = " ".join([str(t) for t in text_list])
        else:
            page_text = str(text_list)
    elif isinstance(data, dict):
        key = "1" if "1" in data else next(iter(data), None)
        if key:
            page_obj = data[key]
            page_dict = page_obj.get('page_dict', page_obj) if isinstance(page_obj, dict) else page_obj
            text_list = page_dict.get('text', [])
            if isinstance(text_list, list):
                page_text = " ".join([str(t) for t in text_list])
            else:
                page_text = str(text_list)

    info = None
    if page_text and page_text.strip():
        try:
            info = extract_title_page_info(
                page_text, 
                LLM_CONFIG['model_name'], 
                LLM_CONFIG['base_url'], 
                LLM_CONFIG['api_key'], 
                LLM_CONFIG['provider']
            )
        except Exception as e:
            # LLM calls are external/flaky — warn but don't crash the pipeline
            print(f"  [WARNING] Title extraction LLM call failed: {e}")
            print(f"            Document will be created without title page.")
            info = None
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([info] if info else [], f, indent=4)


def reclassify_elements(elements: List[Dict], sections_to_keep: Set[Tuple[str, int]]) -> Tuple[List[Dict], int, int]:
    result = []
    kept_as_section = 0
    reclassified = 0
    
    for elem in elements:
        if elem.get('type') != 'section':
            result.append(elem)
        else:
            sec_num = elem.get('section_number', '')
            page = elem.get('page_number', 0)
            
            if (sec_num, page) in sections_to_keep:
                result.append(elem)
                kept_as_section += 1
            else:
                topic = elem.get('topic', '')
                header_text = f"{sec_num} {topic}".strip()
                body_content = elem.get('content', '')
                
                if header_text and body_content:
                    full_content = f"{header_text}\n\n{body_content}"
                elif header_text:
                    full_content = header_text
                else:
                    full_content = body_content

                reclassified_elem = {
                    'type': 'unassigned_text_block',
                    'page_number': elem.get('page_number'),
                    'content': full_content,
                    'bbox': elem.get('bbox'),
                    '_original_type': 'section',
                    '_reclassified_by': 'ml_filter'
                }
                result.append(reclassified_elem)
                reclassified += 1
    
    return result, kept_as_section, reclassified


class Config:
    """Pipeline configuration - edit these values as needed."""
    # Pipeline control
    step = "all"  # Options: "classify", "title", "structure", "ml_filter", "yolo", "assets", "tables", "write", "validate", "all"
    
    # Directories
    raw_ocr_dir = DEFAULT_RAW_OCR_DIR
    #figures_dir = DEFAULT_FIGURES_DIR      # Manual exports (fallback)
    images_dir = DEFAULT_IMAGES_DIR        # Page images for YOLO
    yolo_exports_dir = DEFAULT_YOLO_EXPORTS_DIR  # YOLO output
    results_dir = DEFAULT_RESULTS_DIR
    
    # ML settings
    ml_threshold = 0.5
    
    # IRIS table OCR
    table_jsons_dir = DEFAULT_TABLE_JSONS_DIR
    
    # YOLO settings
    use_yolo = True           # Use YOLO for asset extraction
    yolo_confidence = float(_require_env("YOLO_CONFIDENCE"))
    yolo_device = _require_env("YOLO_DEVICE")
    skip_yolo_if_exists = True  # Skip YOLO if exports already exist


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Document processing pipeline with YOLO integration")
    parser.add_argument('--step', type=str, default='all',
                       help="Step to run: classify, title, structure, ml_filter, yolo, assets, tables, write, validate, all")
    parser.add_argument('--yolo-conf', type=float, default=None,
                       help="YOLO confidence threshold (overrides .env)")
    parser.add_argument('--yolo-device', type=str, default=None,
                       help="YOLO device: cpu, cuda:0, or mps (overrides .env)")
    parser.add_argument('--force', action='store_true',
                       help="Force re-run of classify/title even if outputs exist")
    parser.add_argument('--force-yolo', action='store_true',
                       help="Force re-run YOLO even if exports exist")
    parser.add_argument('--table-jsons-dir', type=str, default=None,
                       help="Directory containing IRIS table OCR JSONs (overrides .env)")

    cli_args = parser.parse_args()

    # Apply CLI overrides to Config
    Config.step = cli_args.step
    if cli_args.yolo_conf is not None:
        Config.yolo_confidence = cli_args.yolo_conf
    if cli_args.yolo_device is not None:
        Config.yolo_device = cli_args.yolo_device
    Config.skip_yolo_if_exists = not cli_args.force_yolo
    if cli_args.table_jsons_dir:
        Config.table_jsons_dir = cli_args.table_jsons_dir

    args = Config()

    # --- Startup diagnostics ---
    print("=" * 60)
    print("PIPELINE CONFIGURATION")
    print("=" * 60)
    _dirs = {
        "raw_ocr_dir":      args.raw_ocr_dir,
        "images_dir":       args.images_dir,
        "yolo_exports_dir": args.yolo_exports_dir,
        "results_dir":      args.results_dir,
        "table_jsons_dir":  args.table_jsons_dir,
    }
    for name, path in _dirs.items():
        resolved = os.path.abspath(path)
        exists = os.path.isdir(resolved)
        status = "OK" if exists else "MISSING"
        print(f"  {name:20s} = {path}")
        print(f"  {'':20s}   -> {resolved}  [{status}]")
    print(f"  {'yolo_confidence':20s} = {args.yolo_confidence}")
    print(f"  {'yolo_device':20s} = {args.yolo_device}")
    print()

    os.makedirs(args.results_dir, exist_ok=True)

    # Determine which figures directory to use (Always YOLO)
    figures_dir = args.yolo_exports_dir
    print(f"[Mode] Using YOLO-extracted assets from: {figures_dir}")
    print()

    # ==========================================================================
    # YOLO EXTRACTION (before processing documents)
    # ==========================================================================
    if args.step in ["yolo", "all"]:
        print("=" * 60)
        print("YOLO ASSET EXTRACTION")
        print("=" * 60)

        # Check if we should skip
        if args.skip_yolo_if_exists and os.path.exists(args.yolo_exports_dir):
            existing_docs = [d for d in os.listdir(args.yolo_exports_dir)
                          if os.path.isdir(os.path.join(args.yolo_exports_dir, d))]
            if existing_docs:
                print(f"  [Skip] YOLO exports already exist for {len(existing_docs)} documents")
                print(f"  Set skip_yolo_if_exists=False to re-run extraction")
                print()
            else:
                run_yolo_extraction(
                    images_dir=Path(args.images_dir),
                    output_dir=Path(args.yolo_exports_dir),
                    confidence_threshold=args.yolo_confidence,
                    device=args.yolo_device,
                    raw_ocr_dir=args.raw_ocr_dir
                )
        else:
            run_yolo_extraction(
                images_dir=Path(args.images_dir),
                output_dir=Path(args.yolo_exports_dir),
                confidence_threshold=args.yolo_confidence,
                device=args.yolo_device,
                raw_ocr_dir=args.raw_ocr_dir
            )
        print()

    # ==========================================================================
    # DOCUMENT LIST
    # ==========================================================================
    doc_stems = get_document_stems(args.raw_ocr_dir)
    print(f"Found {len(doc_stems)} documents in {args.raw_ocr_dir}\n")

    # ==========================================================================
    # ML TRAINING
    # ==========================================================================
    sections_to_keep_by_doc = {}
    ml_metrics = None

    if args.step in ["ml_filter", "all"]:
        print("=" * 60)
        print("ML TRAINING & PREDICTION")
        print("=" * 60)

        features_path = os.path.join(args.results_dir, "training_features.csv")

        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"training_features.csv not found at {features_path}\n"
                f"Run: python feature_extractor.py --results_dir {args.results_dir}"
            )

        sections_to_keep_by_doc, ml_metrics = train_and_predict(features_path, args.ml_threshold)
        print()

    # ==========================================================================
    # PROCESS DOCUMENTS
    # ==========================================================================
    validation_results = []

    total_docs = len(doc_stems)
    for doc_idx, stem in enumerate(doc_stems, 1):
        pct = doc_idx / total_docs * 100
        print(f"{'=' * 60}")
        print(f"[{doc_idx}/{total_docs} ({pct:.0f}%)] {stem}")
        print(f"{'=' * 60}")

        # Resolve the actual YOLO exports subdirectory for this stem.
        # Handles space/underscore naming mismatches between OCR filenames and YOLO dir names.
        from asset_processor import resolve_asset_directory
        doc_asset_dir = resolve_asset_directory(figures_dir, stem)
        if doc_asset_dir:
            figures_stem = os.path.basename(doc_asset_dir)
        else:
            figures_stem = stem

        raw_input = os.path.join(args.raw_ocr_dir, f"{stem}.json")
        classify_out = os.path.join(args.results_dir, f"{stem}_classification.json")
        title_out = os.path.join(args.results_dir, f"{stem}_title.json")
        organized_out = os.path.join(args.results_dir, f"{stem}_organized.json")
        ml_filtered_out = os.path.join(args.results_dir, f"{stem}_ml_filtered.json")
        with_assets_out = os.path.join(args.results_dir, f"{stem}_with_assets.json")
        tables_out = os.path.join(args.results_dir, f"{stem}_with_tables.json")
        final_docx = os.path.join(args.results_dir, f"{stem}.docx")

        # STEP 1: CLASSIFY
        if args.step in ["classify", "all"]:
            print("[1: Classify]")
            if os.path.exists(classify_out) and not args.force:
                print(f"  [Skip] Already done")
            else:
                run_classification_on_file(raw_input, classify_out, LLM_CONFIG, max_workers=4)
            result = load_classification_result(classify_out)
            if result and result.get('is_stub'):
                print("  [Info] Document is a stub, skipping remaining steps.")
                continue

        # STEP 2: TITLE
        if args.step in ["title", "all"]:
            print("[2: Title]")
            if os.path.exists(title_out) and not args.force:
                print(f"  [Skip] Already done")
            else:
                if not os.path.exists(classify_out):
                    print(f"  [FAIL] Classification output not found: {classify_out}")
                    print(f"         Step 1 (classify) must complete successfully first.")
                    print(f"         Skipping remaining steps for {stem}.")
                    continue

                result = load_classification_result(classify_out)
                doc_type = result.get('document_type', 'unknown') if result else 'unknown'

                if doc_type == 'no_title':
                    with open(title_out, 'w') as f: json.dump([], f)
                else:
                    extract_title_from_first_page(raw_input, title_out)

        # STEP 3: STRUCTURE
        if args.step in ["structure", "all"]:
            print("[3: Structure]")
            if args.step == "all" and os.path.exists(organized_out):
                print(f"  [Skip] Already done")
            else:
                if not os.path.exists(classify_out):
                    print(f"  [FAIL] Classification output not found: {classify_out}")
                    print(f"         Step 1 (classify) must complete successfully first.")
                    print(f"         Skipping remaining steps for {stem}.")
                    continue
                content_start = load_content_start_page(classify_out, default=1)
                run_section_processing_on_file(
                    raw_input,
                    organized_out,
                    content_start_page=content_start,
                    yolo_exports_dir=figures_dir,  # Pass YOLO exports for overlap filtering
                    doc_stem=figures_stem
                )
                if not os.path.exists(organized_out):
                    print(f"  [FAIL] Structure processing produced no output: {organized_out}")
                    print(f"         Skipping remaining steps for {stem}.")
                    continue

        # STEP 4: ML FILTER
        if args.step in ["ml_filter", "all"]:
            print("[4: ML Reclassify]")
            if args.step == "all" and os.path.exists(ml_filtered_out):
                print(f"  [Skip] Already done")
            else:
                if not os.path.exists(organized_out):
                    print(f"  [FAIL] Organized output not found: {organized_out}")
                    print(f"         Step 3 (structure) must complete successfully first.")
                    print(f"         Skipping remaining steps for {stem}.")
                    continue

                if stem not in sections_to_keep_by_doc:
                    shutil.copy(organized_out, ml_filtered_out)
                else:
                    sections_to_keep = sections_to_keep_by_doc[stem]
                    with open(organized_out, 'r') as f: data = json.load(f)

                    elements = data.get('elements', []) if isinstance(data, dict) else data
                    page_metadata = data.get('page_metadata', {}) if isinstance(data, dict) else {}

                    reclassified_elements, kept, converted = reclassify_elements(elements, sections_to_keep)
                    print(f"  Sections kept: {kept}, converted to text blocks: {converted}")

                    with open(ml_filtered_out, 'w') as f:
                        json.dump({'page_metadata': page_metadata, 'elements': reclassified_elements}, f, indent=4)

        # STEP 5: ASSETS (using YOLO exports or manual exports)
        if args.step in ["assets", "all"]:
            print("[5: Assets]")
            if args.step == "all" and os.path.exists(with_assets_out):
                print(f"  [Skip] Already done")
            else:
                input_file = ml_filtered_out if os.path.exists(ml_filtered_out) else organized_out
                if not os.path.exists(input_file):
                    print(f"  [FAIL] Asset input not found: {input_file}")
                    print(f"         Expected from step 4 (ml_filter) or step 3 (structure).")
                    print(f"         Skipping remaining steps for {stem}.")
                    continue
                run_asset_integration(input_file, with_assets_out, figures_dir, figures_stem, positioning_mode="bbox")
                if not os.path.exists(with_assets_out):
                    print(f"  [FAIL] Asset integration produced no output: {with_assets_out}")
                    print(f"         Skipping remaining steps for {stem}.")
                    continue

        # STEP 6: TABLES (IRIS table OCR)
        if args.step in ["tables", "all"]:
            print("[6: Tables (IRIS)]")
            if args.step == "all" and os.path.exists(tables_out):
                print(f"  [Skip] Already done")
            else:
                if not os.path.exists(with_assets_out):
                    print(f"  [FAIL] Tables input not found: {with_assets_out}")
                    print(f"         Step 5 (assets) must complete successfully first.")
                    print(f"         Skipping remaining steps for {stem}.")
                    continue
                # Gate: if doc has table crops, IRIS table OCR must be present
                if has_table_crops(figures_dir, figures_stem):
                    if not iris_ready_for_doc(args.table_jsons_dir, figures_stem):
                        print(f"  [WAIT] Document has table crops but IRIS table OCR not yet received.")
                        print(f"         Deferring tables/write/validate for {stem}.")
                        continue
                run_iris_table_processing(
                    with_assets_out,
                    tables_out,
                    args.table_jsons_dir,
                    figures_stem,
                )

        # STEP 7: WRITE
        if args.step in ["write", "all"]:
            print("[7: Write]")
            if args.step == "all" and os.path.exists(final_docx):
                print(f"  [Skip] Already done")
            else:
                if os.path.exists(tables_out):
                    input_file = tables_out
                elif has_table_crops(figures_dir, figures_stem):
                    # Doc has tables but tables_out is missing — don't write a broken docx
                    print(f"  [WAIT] Tables output not found and document has table crops.")
                    print(f"         Deferring write for {stem} until table processing completes.")
                    continue
                elif os.path.exists(with_assets_out):
                    # No table crops — safe to write directly from assets
                    input_file = with_assets_out
                else:
                    print(f"  [FAIL] No input available for write step.")
                    print(f"         Neither {tables_out} nor {with_assets_out} exist.")
                    print(f"         Steps 5-6 must complete successfully first.")
                    continue
                run_docx_creation(input_file, final_docx, figures_dir, figures_stem, title_out)

        # STEP 8: VALIDATE
        if args.step in ["validate", "all"]:
            print("[8: Validate]")
            if not os.path.exists(final_docx):
                print(f"  [FAIL] Cannot validate - DOCX not found: {final_docx}")
                continue
            try:
                result = run_validation_on_file(stem, args.raw_ocr_dir, args.results_dir)
                if result:
                    validation_results.append({
                        'document': stem,
                        'has_toc': result.has_toc,
                        'toc_coverage': result.toc_coverage,
                        'precision': result.precision,
                    })
            except Exception as e:
                print(f"  [WARNING] Validation failed: {e}")
        print()

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================

    if validation_results:
        print("=" * 60)
        print("TOC VALIDATION SUMMARY")
        print("=" * 60)
        docs_with_toc = [r for r in validation_results if r['has_toc']]
        if docs_with_toc:
            avg_cov = sum(r['toc_coverage'] for r in docs_with_toc) / len(docs_with_toc)
            avg_prec = sum(r['precision'] for r in docs_with_toc) / len(docs_with_toc)
            print(f"Documents with TOC: {len(docs_with_toc)}")
            print(f"Avg TOC Match:     {avg_cov:.1f}%")
            print(f"Avg Precision:     {avg_prec:.1f}%")

    if ml_metrics:
        val = ml_metrics.get('val', {})
        train = ml_metrics.get('train', {})

        print("\n" + "=" * 60)
        print("ML MODEL PERFORMANCE")
        print("=" * 60)

        print("1. VALIDATION (80/20 SPLIT)")
        print(f"   Accuracy:  {val.get('accuracy', 0):.1%}")
        print(f"   Precision: {val.get('precision', 0):.3f}")
        print(f"   Recall:    {val.get('recall', 0):.3f}")
        print(f"   F1 Score:  {val.get('f1', 0):.3f}")

        if 'confusion' in val:
            tn, fp, fn, tp = val['confusion'].ravel()
            print(f"   Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        print("\n2. FINAL MODEL (100% TRAINING DATA)")
        print(f"   Samples:   {train.get('total_samples', 0)}")
        print(f"   Precision: {train.get('precision', 0):.3f}")
        print(f"   Recall:    {train.get('recall', 0):.3f}")
        print(f"   F1 Score:  {train.get('f1', 0):.3f}")

    print("\nDone!")