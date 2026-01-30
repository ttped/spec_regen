"""
simple_pipeline.py - Document processing pipeline with ML classification.

Steps: classify -> title -> structure -> ml_filter -> assets -> tables -> write -> validate
"""

import os
import json
import numpy as np
from typing import List, Dict, Set, Tuple

# Direct imports
from classify_agent import run_classification_on_file, load_content_start_page, load_classification_result
from title_agent import extract_title_page_info
from asset_processor import run_asset_integration
from table_processor_agent import run_table_processing_on_file
from docx_writer import run_docx_creation
from section_processor import run_section_processing_on_file
from section_classifier import train_and_predict
from validation_agent import run_validation_on_file
from utils import load_json_with_recovery


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
    stems = set()
    try:
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                stems.add(os.path.splitext(filename)[0])
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found.")
        return []
    return sorted(list(stems))


def extract_title_from_first_page(raw_input_path: str, output_path: str):
    """Extract title page information from page 1."""
    print(f"  Loading: {raw_input_path}")
    
    # 1. Safe Load
    try:
        data = load_json_with_recovery(raw_input_path)
    except Exception as e:
        print(f"  [Error] Failed to load raw OCR JSON: {e}")
        # Create empty output to prevent downstream failures
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return

    # 2. Extract Text
    page_text = ""
    try:
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
    except Exception as e:
        print(f"  [Warning] Error parsing page text structure: {e}")

    # 3. LLM Extraction
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
            print(f"  [Warning] Title extraction LLM failure: {e}")
            info = None
    
    # 4. Safe Write
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
    step = "all"  # Options: "classify", "title", "structure", "ml_filter", "assets", "tables", "write", "validate", "all"
    raw_ocr_dir = DEFAULT_RAW_OCR_DIR
    figures_dir = DEFAULT_FIGURES_DIR
    results_dir = DEFAULT_RESULTS_DIR
    ml_threshold = 0.5
    no_table_ocr = True  # Disabled - too slow and not effective yet


def main():
    # Configuration - edit Config class above or override here
    args = Config()
    
    os.makedirs(args.results_dir, exist_ok=True)

    doc_stems = get_document_stems(args.raw_ocr_dir)
    print(f"Found {len(doc_stems)} documents in {args.raw_ocr_dir}\n")

    # ==========================================================================
    # ML TRAINING (once, on full CSV)
    # ==========================================================================
    sections_to_keep_by_doc = {}
    ml_metrics = None
    
    if args.step in ["ml_filter", "all"]:
        print("="*60)
        print("ML TRAINING & PREDICTION")
        print("="*60)
        
        features_path = os.path.join(args.results_dir, "training_features.csv")
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"training_features.csv not found at {features_path}\n"
                f"Run: python feature_extractor.py --results_dir {args.results_dir}"
            )
        
        # No try/except - let ML failures crash loudly so we see what's wrong
        sections_to_keep_by_doc, ml_metrics = train_and_predict(features_path, args.ml_threshold)
        print()

    # ==========================================================================
    # PROCESS DOCUMENTS
    # ==========================================================================
    validation_results = []

    for stem in doc_stems:
        print(f"{'='*60}")
        print(f"{stem}")
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
            print("[1: Classify]")
            try:
                run_classification_on_file(raw_input, classify_out, LLM_CONFIG, max_workers=4)
                result = load_classification_result(classify_out)
                if result and result.get('is_stub'):
                    print("  [Info] Document is a stub, skipping remaining steps.")
                    continue
            except Exception as e:
                print(f"  [Error] Classification failed: {e}")
                # We attempt to continue, but subsequent steps might fail if classify_out is missing
                
        # STEP 2: TITLE
        if args.step in ["title", "all"]:
            print("[2: Title]")
            # Robust loading of classification result
            try:
                result = load_classification_result(classify_out)
            except Exception as e:
                print(f"  [Warning] Could not load classification for title step: {e}")
                result = None
            
            doc_type = result.get('document_type', 'unknown') if result else 'unknown'
            
            if doc_type == 'no_title':
                with open(title_out, 'w') as f: json.dump([], f)
            else:
                # This function now has internal try/catch for JSON errors
                extract_title_from_first_page(raw_input, title_out)

        # STEP 3: STRUCTURE
        if args.step in ["structure", "all"]:
            print("[3: Structure]")
            try:
                content_start = load_content_start_page(classify_out, default=1)
                run_section_processing_on_file(raw_input, organized_out, content_start_page=content_start)
            except Exception as e:
                print(f"  [Error] Structure processing failed: {e}")

        # STEP 4: ML FILTER
        if args.step in ["ml_filter", "all"]:
            print("[4: ML Reclassify]")
            if not os.path.exists(organized_out):
                print(f"  [Skip] Organized output not found.")
            else:
                if stem not in sections_to_keep_by_doc:
                    # If ML failed or this doc wasn't in training set, preserve original
                    import shutil
                    shutil.copy(organized_out, ml_filtered_out)
                else:
                    try:
                        sections_to_keep = sections_to_keep_by_doc[stem]
                        with open(organized_out, 'r') as f: data = json.load(f)
                        
                        elements = data.get('elements', []) if isinstance(data, dict) else data
                        page_metadata = data.get('page_metadata', {}) if isinstance(data, dict) else {}
                        
                        reclassified_elements, kept, converted = reclassify_elements(elements, sections_to_keep)
                        print(f"  Sections kept: {kept}, converted to text blocks: {converted}")
                        
                        with open(ml_filtered_out, 'w') as f:
                            json.dump({'page_metadata': page_metadata, 'elements': reclassified_elements}, f, indent=4)
                    except Exception as e:
                        print(f"  [Error] ML Reclassify failed: {e}")
                        import shutil
                        shutil.copy(organized_out, ml_filtered_out)

        # STEP 5: ASSETS
        if args.step in ["assets", "all"]:
            print("[5: Assets]")
            input_file = ml_filtered_out if os.path.exists(ml_filtered_out) else organized_out
            if os.path.exists(input_file):
                run_asset_integration(input_file, with_assets_out, args.figures_dir, stem, positioning_mode="bbox")
            else:
                print(f"  [Skip] Input file for assets not found: {input_file}")

        # STEP 6: TABLES
        if args.step in ["tables", "all"]:
            print("[6: Tables]")
            input_file = with_assets_out if os.path.exists(with_assets_out) else ml_filtered_out
            if os.path.exists(input_file):
                run_table_processing_on_file(input_file, tables_out, args.figures_dir, stem, LLM_CONFIG, use_llm_ocr=not args.no_table_ocr)
            else:
                print(f"  [Skip] Input file for tables not found: {input_file}")

        # STEP 7: WRITE
        if args.step in ["write", "all"]:
            print("[7: Write]")
            input_file = tables_out if os.path.exists(tables_out) else with_assets_out
            if os.path.exists(input_file):
                run_docx_creation(input_file, final_docx, args.figures_dir, stem, title_out)
            else:
                print(f"  [Skip] Input file for write not found: {input_file}")

        # STEP 8: VALIDATE
        if args.step in ["validate", "all"]:
            print("[8: Validate]")
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
                print(f"  [Warning] Validation failed: {e}")
        print()

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    
    # 1. TOC Validation Summary
    if validation_results:
        print("="*60)
        print("TOC VALIDATION SUMMARY")
        print("="*60)
        docs_with_toc = [r for r in validation_results if r['has_toc']]
        if docs_with_toc:
            avg_cov = sum(r['toc_coverage'] for r in docs_with_toc) / len(docs_with_toc)
            avg_prec = sum(r['precision'] for r in docs_with_toc) / len(docs_with_toc)
            print(f"Documents with TOC: {len(docs_with_toc)}")
            print(f"Avg TOC Match:     {avg_cov:.1f}%")
            print(f"Avg Precision:     {avg_prec:.1f}%")

    # 2. ML Performance Summary
    if ml_metrics:
        val = ml_metrics.get('val', {})
        train = ml_metrics.get('train', {})
        
        print("\n" + "="*60)
        print("ML MODEL PERFORMANCE")
        print("="*60)
        
        print("1. VALIDATION (80/20 SPLIT)")
        print(f"   (How well the model generalizes to unseen data)")
        print(f"   Accuracy:  {val.get('accuracy', 0):.1%}")
        print(f"   Precision: {val.get('precision', 0):.3f}")
        print(f"   Recall:    {val.get('recall', 0):.3f}")
        print(f"   F1 Score:  {val.get('f1', 0):.3f}")
        
        if 'confusion' in val:
            tn, fp, fn, tp = val['confusion'].ravel()
            print(f"   Confusion: TP={tp} (Valid kept), FP={fp} (Junk kept)")
            print(f"              TN={tn} (Junk removed), FN={fn} (Valid removed)")

        print("\n2. FINAL MODEL (100% TRAINING DATA)")
        print(f"   (The model actually used for processing {train.get('total_samples', 0)} labeled rows)")
        print(f"   Precision: {train.get('precision', 0):.3f}")
        print(f"   Recall:    {train.get('recall', 0):.3f}")
        print(f"   F1 Score:  {train.get('f1', 0):.3f}")
        
    print("\nDone!")


if __name__ == '__main__':
    main()