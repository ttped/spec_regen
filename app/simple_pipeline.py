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

# Direct imports (existing)
from classify_agent import run_classification_on_file, load_content_start_page, load_classification_result
from title_agent import extract_title_page_info
from asset_processor import run_asset_integration
from table_processor_agent import run_table_processing_on_file
from docx_writer import run_docx_creation
from section_processor import run_section_processing_on_file
from section_classifier import train_and_predict
from validation_agent import run_validation_on_file
from utils import load_json_with_recovery


from yolo_asset_extractor import run_yolo_extraction, get_yolo_exports_dir


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_RAW_OCR_DIR = os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced")
DEFAULT_FIGURES_DIR = os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "exports")  # Manual exports
DEFAULT_IMAGES_DIR = "docs_images"  # Page images for YOLO
DEFAULT_YOLO_EXPORTS_DIR = "yolo_exports"  # YOLO-extracted assets
DEFAULT_RESULTS_DIR = "results_simple"

## --- LLM Provider Toggle ---
## Uncomment ONE of the following configs:

# Option A: Ollama (local)
LLM_CONFIG = {
    "provider": "ollama",
    "model_name": "gemma3:27b",
    "base_url": "http://localhost:11434",
    "api_key": None
}

# Option B: Mission Assist (remote)
# LLM_CONFIG = {
#     "provider": "mission_assist",
#     "model_name": "gemma3", 
#     "base_url": "http://devmissionassist.api.us.baesystems.com",
#     "api_key": "aTOIT9hJM3DBYMQbEY"
# }


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
    no_table_ocr = True
    
    # YOLO settings
    use_yolo = True           # Use YOLO for asset extraction
    yolo_confidence = 0.25    # Confidence threshold
    yolo_device = 'cpu'       # 'cpu', 'cuda:0', or 'mps'
    skip_yolo_if_exists = True  # Skip YOLO if exports already exist


def main():
    args = Config()
    
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

    for stem in doc_stems:
        print(f"{'=' * 60}")
        print(f"{stem}")
        print(f"{'=' * 60}")
        
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
            run_classification_on_file(raw_input, classify_out, LLM_CONFIG, max_workers=4)
            result = load_classification_result(classify_out)
            if result and result.get('is_stub'):
                print("  [Info] Document is a stub, skipping remaining steps.")
                continue
                
        # STEP 2: TITLE
        if args.step in ["title", "all"]:
            print("[2: Title]")
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
                doc_stem=stem
            )
            if not os.path.exists(organized_out):
                print(f"  [FAIL] Structure processing produced no output: {organized_out}")
                print(f"         Skipping remaining steps for {stem}.")
                continue

        # STEP 4: ML FILTER
        if args.step in ["ml_filter", "all"]:
            print("[4: ML Reclassify]")
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
            input_file = ml_filtered_out if os.path.exists(ml_filtered_out) else organized_out
            if not os.path.exists(input_file):
                print(f"  [FAIL] Asset input not found: {input_file}")
                print(f"         Expected from step 4 (ml_filter) or step 3 (structure).")
                print(f"         Skipping remaining steps for {stem}.")
                continue
            run_asset_integration(input_file, with_assets_out, figures_dir, stem, positioning_mode="bbox")
            if not os.path.exists(with_assets_out):
                print(f"  [FAIL] Asset integration produced no output: {with_assets_out}")
                print(f"         Skipping remaining steps for {stem}.")
                continue

        # STEP 6: TABLES
        if args.step in ["tables", "all"]:
            print("[6: Tables]")
            if not os.path.exists(with_assets_out):
                print(f"  [FAIL] Tables input not found: {with_assets_out}")
                print(f"         Step 5 (assets) must complete successfully first.")
                print(f"         Skipping remaining steps for {stem}.")
                continue
            run_table_processing_on_file(with_assets_out, tables_out, figures_dir, stem, LLM_CONFIG, use_llm_ocr=not args.no_table_ocr)

        # STEP 7: WRITE
        if args.step in ["write", "all"]:
            print("[7: Write]")
            # Prefer tables_out (has both assets + tables), fall back to with_assets_out
            # (has assets but no table processing). Never silently skip assets.
            if os.path.exists(tables_out):
                input_file = tables_out
            elif os.path.exists(with_assets_out):
                print(f"  [WARNING] Tables output not found, writing from asset output instead.")
                print(f"            Table processing may have failed for {stem}.")
                input_file = with_assets_out
            else:
                print(f"  [FAIL] No input available for write step.")
                print(f"         Neither {tables_out} nor {with_assets_out} exist.")
                print(f"         Steps 5-6 must complete successfully first.")
                continue
            run_docx_creation(input_file, final_docx, figures_dir, stem, title_out)

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


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Document processing pipeline with YOLO integration")
    parser.add_argument('--step', type=str, default='all',
                       help="Step to run: classify, title, structure, ml_filter, yolo, assets, tables, write, validate, all")
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                       help="YOLO confidence threshold")
    parser.add_argument('--yolo-device', type=str, default='cpu',
                       help="YOLO device: cpu, cuda:0, or mps")
    parser.add_argument('--force-yolo', action='store_true',
                       help="Force re-run YOLO even if exports exist")
    
    args = parser.parse_args()
    
    # Apply CLI args to config
    Config.step = args.step
    Config.yolo_confidence = args.yolo_conf
    Config.yolo_device = args.yolo_device
    Config.skip_yolo_if_exists = not args.force_yolo
    
    main()