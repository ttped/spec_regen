"""
simple_pipeline.py - Document processing pipeline with ML classification.

Pipeline Steps:
1. classify  - Determine where content starts (skip ToC)
2. title     - Extract title page information  
3. structure - Parse sections from content pages
4. ml_filter - Train ML model, predict, filter false positives
5. assets    - Integrate figures/tables
6. tables    - OCR table images (or skip with --no-table-ocr)
7. write     - Generate final DOCX
8. validate  - Compare extracted sections against TOC

Usage:
    python simple_pipeline.py --step all
    python simple_pipeline.py --step all --ml-threshold 0.6
"""

import os
import argparse
import json
from json import JSONDecodeError
from typing import List, Dict, Set, Tuple

# Import pipeline components
from classify_agent import run_classification_on_file, load_content_start_page, load_classification_result
from title_agent import extract_title_page_info
from asset_processor import run_asset_integration
from table_processor_agent import run_table_processing_on_file
from docx_writer import run_docx_creation
from section_processor import run_section_processing_on_file
from section_classifier import train_and_predict
from validation_agent import run_validation_on_file


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_RAW_OCR_DIR = os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced")
DEFAULT_FIGURES_DIR = os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "exports")
DEFAULT_RESULTS_DIR = "results_simple"

LLM_CONFIG = {
    "provider": "mission_assist",
    "model_name": "gemma3", 
    "base_url": "http://devmissionassist.api.us.baesystems.com",
    "api_key": "aTOIT9hJM3DBYMQbEY"
}


def get_document_stems(input_dir: str) -> List[str]:
    """Find all unique document stems in a directory."""
    stems = set()
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            stems.add(os.path.splitext(filename)[0])
    return sorted(list(stems))


def extract_title_from_first_page(raw_input_path: str, output_path: str):
    """Extract title page information from page 1."""
    with open(raw_input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    page_text = ""
    if isinstance(data, list) and len(data) > 0:
        data.sort(key=lambda x: int(x.get('page_num', 0) or x.get('page_Id', 0) or 0))
        page_dict = data[0].get('page_dict', data[0])
        page_text = " ".join([str(t) for t in page_dict.get('text', [])])
    elif isinstance(data, dict):
        key = "1" if "1" in data else next(iter(data), None)
        if key:
            page_obj = data[key]
            page_dict = page_obj.get('page_dict', page_obj) if isinstance(page_obj, dict) else page_obj
            if isinstance(page_dict, dict):
                page_text = " ".join([str(t) for t in page_dict.get('text', [])])

    info = None
    if page_text:
        try:
            info = extract_title_page_info(
                page_text, 
                LLM_CONFIG['model_name'], 
                LLM_CONFIG['base_url'], 
                LLM_CONFIG['api_key'], 
                LLM_CONFIG['provider']
            )
        except Exception as e:
            print(f"    [Error] Title extraction failed: {e}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([info] if info else [], f, indent=4)


def filter_elements(
    elements: List[Dict], 
    sections_to_keep: Set[Tuple[str, int]]
) -> Tuple[List[Dict], int, int]:
    """Filter elements, keeping only sections in sections_to_keep."""
    filtered = []
    kept = 0
    removed = 0
    
    for elem in elements:
        if elem.get('type') != 'section':
            filtered.append(elem)
        else:
            sec_num = elem.get('section_number', '')
            page = elem.get('page_number', 0)
            
            if (sec_num, page) in sections_to_keep:
                filtered.append(elem)
                kept += 1
            else:
                removed += 1
    
    return filtered, kept, removed


def main():
    parser = argparse.ArgumentParser(description="Document processing pipeline")
    parser.add_argument("--step", type=str, default="all", 
        choices=["classify", "title", "structure", "ml_filter", "assets", "tables", "write", "validate", "all"])
    parser.add_argument("--raw_ocr_dir", type=str, default=DEFAULT_RAW_OCR_DIR)
    parser.add_argument("--figures_dir", type=str, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--ml-threshold", type=float, default=0.5)
    parser.add_argument("--no-table-ocr", action="store_true")
    
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    doc_stems = get_document_stems(args.raw_ocr_dir)
    print(f"Found {len(doc_stems)} documents in {args.raw_ocr_dir}\n")

    # ==========================================================================
    # ML TRAINING (once, before processing documents)
    # ==========================================================================
    sections_to_keep_by_doc = {}
    
    if args.step in ["ml_filter", "all"]:
        print("="*60)
        print("STEP 4: ML TRAINING AND PREDICTION")
        print("="*60)
        
        features_path = os.path.join(args.results_dir, "training_features.csv")
        print(f"Features file: {features_path}")
        print(f"Exists: {os.path.exists(features_path)}")
        
        if not os.path.exists(features_path):
            print("\n*** ERROR: training_features.csv not found! ***")
            print("Run feature_extractor.py first:")
            print(f"  python feature_extractor.py --results_dir {args.results_dir}")
            return
        
        # This will raise exceptions on failure - no silent fallbacks
        sections_to_keep_by_doc = train_and_predict(features_path, args.ml_threshold)
        
        print(f"\n  Documents with predictions: {list(sections_to_keep_by_doc.keys())[:5]}...")
        print()

    # ==========================================================================
    # PROCESS EACH DOCUMENT
    # ==========================================================================
    validation_results = []

    for stem in doc_stems:
        print(f"{'='*60}")
        print(f"Processing: {stem}")
        print(f"{'='*60}")
        
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
            print("\n[Step 1: Classify]")
            run_classification_on_file(raw_input, classify_out, LLM_CONFIG, max_workers=4)
            
            result = load_classification_result(classify_out)
            if result and result.get('is_stub'):
                print(f"  STUB DOCUMENT - Skipping")
                continue

        # STEP 2: TITLE
        if args.step in ["title", "all"]:
            print("\n[Step 2: Title]")
            result = load_classification_result(classify_out)
            doc_type = result.get('document_type', 'unknown') if result else 'unknown'
            
            if doc_type == 'no_title':
                with open(title_out, 'w') as f:
                    json.dump([], f)
            else:
                extract_title_from_first_page(raw_input, title_out)

        # STEP 3: STRUCTURE
        if args.step in ["structure", "all"]:
            print("\n[Step 3: Structure]")
            content_start = load_content_start_page(classify_out, default=1)
            run_section_processing_on_file(
                raw_input, organized_out, 
                content_start_page=content_start,
                header_top_threshold=600,
                footer_top_threshold=6100
            )

        # STEP 4: ML FILTER
        if args.step in ["ml_filter", "all"]:
            print("\n[Step 4: ML Filter]")
            
            # Debug: check if stem is in predictions
            print(f"  Looking for '{stem}' in predictions...")
            print(f"  Available doc names: {list(sections_to_keep_by_doc.keys())[:5]}...")
            
            if stem not in sections_to_keep_by_doc:
                print(f"  *** WARNING: '{stem}' not found in predictions! ***")
                print(f"  Copying without filtering...")
                import shutil
                shutil.copy(organized_out, ml_filtered_out)
            else:
                sections_to_keep = sections_to_keep_by_doc[stem]
                print(f"  Sections to keep for this doc: {len(sections_to_keep)}")
                
                with open(organized_out, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and 'elements' in data:
                    elements = data['elements']
                    page_metadata = data.get('page_metadata', {})
                else:
                    elements = data if isinstance(data, list) else []
                    page_metadata = {}
                
                filtered, kept, removed = filter_elements(elements, sections_to_keep)
                print(f"  Filtered: kept {kept}, removed {removed}")
                
                with open(ml_filtered_out, 'w') as f:
                    json.dump({'page_metadata': page_metadata, 'elements': filtered}, f, indent=4)

        # STEP 5: ASSETS
        if args.step in ["assets", "all"]:
            print("\n[Step 5: Assets]")
            input_file = ml_filtered_out if os.path.exists(ml_filtered_out) else organized_out
            run_asset_integration(input_file, with_assets_out, args.figures_dir, stem, positioning_mode="bbox")

        # STEP 6: TABLES
        if args.step in ["tables", "all"]:
            print("\n[Step 6: Tables]")
            for candidate in [with_assets_out, ml_filtered_out, organized_out]:
                if os.path.exists(candidate):
                    input_file = candidate
                    break
            run_table_processing_on_file(
                input_file, tables_out, args.figures_dir, stem, 
                LLM_CONFIG, use_llm_ocr=not args.no_table_ocr
            )

        # STEP 7: WRITE DOCX
        if args.step in ["write", "all"]:
            print("\n[Step 7: Write DOCX]")
            for candidate in [tables_out, with_assets_out, ml_filtered_out, organized_out]:
                if os.path.exists(candidate):
                    input_file = candidate
                    break
            run_docx_creation(input_file, final_docx, args.figures_dir, stem, title_out)

        # STEP 8: VALIDATE
        if args.step in ["validate", "all"]:
            print("\n[Step 8: Validate]")
            result = run_validation_on_file(stem, args.raw_ocr_dir, args.results_dir)
            if result:
                validation_results.append({
                    'document': stem,
                    'has_toc': result.has_toc,
                    'toc_coverage': result.toc_coverage,
                    'precision': result.precision,
                })

        print()

    # ==========================================================================
    # VALIDATION SUMMARY
    # ==========================================================================
    if validation_results:
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        docs_with_toc = [r for r in validation_results if r['has_toc']]
        
        if docs_with_toc:
            avg_cov = sum(r['toc_coverage'] for r in docs_with_toc) / len(docs_with_toc)
            avg_prec = sum(r['precision'] for r in docs_with_toc) / len(docs_with_toc)
            
            print(f"\nDocuments with TOC: {len(docs_with_toc)}")
            print(f"Avg Coverage:  {avg_cov:.1f}%")
            print(f"Avg Precision: {avg_prec:.1f}%")
            
            for doc in docs_with_toc:
                print(f"  {doc['document']}: Cov={doc['toc_coverage']:.0f}%, Prec={doc['precision']:.0f}%")

    print("\nDone!")


if __name__ == '__main__':
    main()