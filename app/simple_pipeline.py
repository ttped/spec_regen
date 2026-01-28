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
    python simple_pipeline.py --step all --no-table-ocr
"""

import os
import argparse
import json
from json import JSONDecodeError
from typing import List, Dict, Set, Tuple

# Import pipeline components
try:
    from classify_agent import run_classification_on_file, load_content_start_page, load_classification_result
    from title_agent import extract_title_page_info
    from asset_processor import run_asset_integration
    from table_processor_agent import run_table_processing_on_file
    from docx_writer import run_docx_creation
    from section_processor import run_section_processing_on_file
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure all agent modules are in the same directory or Python path.")
    exit(1)

# ML classifier
try:
    from section_classifier import (
        load_training_data, 
        train_classifier, 
        predict_all,
        get_sections_to_keep
    )
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"[Warning] section_classifier not available: {e}")

# Feature extraction
try:
    from feature_extractor import process_directory as extract_features
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False

# Validation
try:
    from validation_agent import run_validation_on_file
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


# =============================================================================
# DEFAULT PATHS
# =============================================================================

DEFAULT_RAW_OCR_DIR = os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced")
DEFAULT_FIGURES_DIR = os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "exports")
DEFAULT_RESULTS_DIR = "results_simple"


def get_document_stems(input_dir: str) -> List[str]:
    """Find all unique document stems in a directory."""
    stems = set()
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist.")
        return []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            stems.add(os.path.splitext(filename)[0])
    return sorted(list(stems))


def extract_title_from_first_page(raw_input_path: str, output_path: str, llm_config: dict):
    """Extract title page information from page 1."""
    if not os.path.exists(raw_input_path):
        with open(output_path, 'w') as f:
            json.dump([], f)
        return
    
    try:
        with open(raw_input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except JSONDecodeError:
        with open(output_path, 'w') as f:
            json.dump([], f)
        return
    
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

    if not page_text:
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

    info = None
    try:
        info = extract_title_page_info(
            page_text, 
            llm_config['model_name'], 
            llm_config['base_url'], 
            llm_config['api_key'], 
            llm_config['provider']
        )
    except Exception as e:
        print(f"    [Error] Title extraction failed: {e}")
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([info] if info else [], f, indent=4)


def train_and_predict(results_dir: str, threshold: float = 0.5) -> Dict[str, Set[Tuple[str, int]]]:
    """
    Train ML model and get predictions for all documents.
    
    Returns:
        Dict mapping doc_name -> set of (section_number, page) tuples to KEEP
    """
    if not ML_AVAILABLE or not FEATURES_AVAILABLE:
        print("  [Warning] ML or feature extraction not available")
        return {}
    
    features_path = os.path.join(results_dir, "training_features.csv")
    
    if not os.path.exists(features_path):
        print(f"  [Warning] No features file at {features_path}")
        print("  Run feature_extractor.py first, then label some rows.")
        return {}
    
    # Load features
    print("  Loading features...")
    df = load_training_data(features_path, verbose=False)
    
    # Check for labeled data
    labeled = df[df['_label'].notna()]
    labeled = labeled[labeled['_label'].apply(lambda x: float(x) >= 0 if str(x).replace('.','').replace('-','').isdigit() else False)]
    
    if len(labeled) < 20:
        print(f"  [Warning] Only {len(labeled)} labeled samples (need 20+)")
        print("  Label more rows in training_features.csv")
        return {}
    
    pos_count = (labeled['_label'].astype(float) == 1).sum()
    neg_count = (labeled['_label'].astype(float) == 0).sum()
    print(f"  Found {len(labeled)} labeled samples ({pos_count} valid, {neg_count} false positives)")
    
    # Train model
    print("  Training model...")
    try:
        result = train_classifier(df, test_size=0.2, verbose=True)
    except Exception as e:
        print(f"  [Error] Training failed: {e}")
        return {}
    
    # Predict on all data
    print(f"\n  Predicting on all {len(df)} rows (threshold={threshold})...")
    df_pred = predict_all(df, result.model, result.feature_columns, threshold=threshold, verbose=True)
    
    # Build dict of sections to keep per document
    sections_to_keep = {}
    for doc_name in df_pred['_doc_name'].unique():
        doc_sections = get_sections_to_keep(df_pred, doc_name)
        sections_to_keep[doc_name] = set(doc_sections)
    
    return sections_to_keep


def filter_elements_by_predictions(
    elements: List[Dict], 
    sections_to_keep: Set[Tuple[str, int]]
) -> List[Dict]:
    """
    Filter elements list, keeping only sections that are in sections_to_keep.
    Non-section elements (text blocks, figures, tables) are always kept.
    """
    filtered = []
    removed_count = 0
    
    for elem in elements:
        if elem.get('type') != 'section':
            # Keep all non-section elements
            filtered.append(elem)
        else:
            # Check if this section should be kept
            sec_num = elem.get('section_number', '')
            page = elem.get('page_number', 0)
            
            if (sec_num, page) in sections_to_keep:
                filtered.append(elem)
            else:
                removed_count += 1
    
    return filtered, removed_count


def main():
    parser = argparse.ArgumentParser(
        description="Document processing pipeline with ML classification"
    )
    parser.add_argument(
        "--step", 
        type=str, 
        default="all", 
        choices=["classify", "title", "structure", "ml_filter", "assets", "tables", "write", "validate", "all"],
        help="Pipeline step to execute (default: all)"
    )
    parser.add_argument(
        "--raw_ocr_dir", 
        type=str, 
        default=DEFAULT_RAW_OCR_DIR,
        help="Directory containing raw OCR JSON files"
    )
    parser.add_argument(
        "--figures_dir", 
        type=str, 
        default=DEFAULT_FIGURES_DIR,
        help="Directory containing exported figures/tables"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default=DEFAULT_RESULTS_DIR,
        help="Directory for output files"
    )
    parser.add_argument(
        "--ml-threshold",
        type=float,
        default=0.5,
        help="ML classification threshold (default: 0.5). Lower keeps more sections."
    )
    parser.add_argument(
        "--no-table-ocr",
        action="store_true",
        help="Skip LLM OCR for tables."
    )
    
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    llm_config = {
        "provider": "mission_assist",
        "model_name": "gemma3", 
        "base_url": "http://devmissionassist.api.us.baesystems.com",
        "api_key": "aTOIT9hJM3DBYMQbEY"
    }

    print(f"Looking for documents in: {args.raw_ocr_dir}")
    doc_stems = get_document_stems(args.raw_ocr_dir)
    print(f"Found {len(doc_stems)} document(s).\n")

    # Train ML model once at the start and get predictions for all docs
    sections_to_keep_by_doc = {}
    
    if args.step in ["ml_filter", "all"]:
        print("="*60)
        print("ML TRAINING AND PREDICTION")
        print("="*60)
        sections_to_keep_by_doc = train_and_predict(args.results_dir, args.ml_threshold)
        
        if not sections_to_keep_by_doc:
            print("\n[Note] ML filtering will be skipped (no predictions available)")
        else:
            total_keep = sum(len(v) for v in sections_to_keep_by_doc.values())
            print(f"\n  Total sections to keep across all docs: {total_keep}")
        print()

    # Track validation results
    validation_results = []

    for stem in doc_stems:
        print(f"{'='*60}")
        print(f"Processing: {stem}")
        print(f"{'='*60}")
        
        # Define file paths
        raw_input = os.path.join(args.raw_ocr_dir, f"{stem}.json")
        classify_out = os.path.join(args.results_dir, f"{stem}_classification.json")
        title_out = os.path.join(args.results_dir, f"{stem}_title.json")
        organized_out = os.path.join(args.results_dir, f"{stem}_organized.json")
        ml_filtered_out = os.path.join(args.results_dir, f"{stem}_ml_filtered.json")
        with_assets_out = os.path.join(args.results_dir, f"{stem}_with_assets.json")
        tables_out = os.path.join(args.results_dir, f"{stem}_with_tables.json")
        final_docx = os.path.join(args.results_dir, f"{stem}.docx")

        # ========== STEP 1: CLASSIFY ==========
        if args.step in ["classify", "all"]:
            print("\n[Step 1: Classify]")
            run_classification_on_file(raw_input, classify_out, llm_config, max_workers=4)
            
            classification_result = load_classification_result(classify_out)
            if classification_result and classification_result.get('is_stub'):
                print(f"  - STUB DOCUMENT - Skipping")
                continue

        # ========== STEP 2: TITLE ==========
        if args.step in ["title", "all"]:
            print("\n[Step 2: Title Extraction]")
            
            classification_result = load_classification_result(classify_out)
            doc_type = classification_result.get('document_type', 'unknown') if classification_result else 'unknown'
            
            if doc_type == 'no_title':
                print(f"  - No title page")
                with open(title_out, 'w') as f:
                    json.dump([], f)
            else:
                extract_title_from_first_page(raw_input, title_out, llm_config)

        # ========== STEP 3: STRUCTURE ==========
        if args.step in ["structure", "all"]:
            print("\n[Step 3: Structure/Section Detection]")
            
            content_start_page = load_content_start_page(classify_out, default=1)
            print(f"  - Content starts at page: {content_start_page}")
            
            run_section_processing_on_file(
                raw_input, 
                organized_out, 
                content_start_page=content_start_page,
                header_top_threshold=600,
                footer_top_threshold=6100
            )

        # ========== STEP 4: ML FILTER ==========
        if args.step in ["ml_filter", "all"]:
            print("\n[Step 4: ML Section Filtering]")
            
            if stem not in sections_to_keep_by_doc:
                print("  - No ML predictions, copying without filtering")
                if os.path.exists(organized_out):
                    import shutil
                    shutil.copy(organized_out, ml_filtered_out)
            elif os.path.exists(organized_out):
                # Load organized data
                with open(organized_out, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and 'elements' in data:
                    elements = data.get('elements', [])
                    page_metadata = data.get('page_metadata', {})
                else:
                    elements = data if isinstance(data, list) else []
                    page_metadata = {}
                
                sections_before = sum(1 for e in elements if e.get('type') == 'section')
                
                # Filter using predictions
                filtered_elements, removed_count = filter_elements_by_predictions(
                    elements, 
                    sections_to_keep_by_doc[stem]
                )
                
                sections_after = sum(1 for e in filtered_elements if e.get('type') == 'section')
                print(f"  - Sections: {sections_before} -> {sections_after} (removed {removed_count})")
                
                # Save filtered
                output_data = {
                    'page_metadata': page_metadata,
                    'elements': filtered_elements
                }
                with open(ml_filtered_out, 'w') as f:
                    json.dump(output_data, f, indent=4)
            else:
                print(f"  - [Skip] Missing: {organized_out}")

        # ========== STEP 5: ASSETS ==========
        if args.step in ["assets", "all"]:
            print("\n[Step 5: Asset Integration]")
            
            input_for_assets = ml_filtered_out if os.path.exists(ml_filtered_out) else organized_out
            
            if os.path.exists(input_for_assets):
                run_asset_integration(
                    input_for_assets,
                    with_assets_out,
                    args.figures_dir,
                    stem,
                    positioning_mode="bbox"
                )
            else:
                print(f"  - [Skip] Missing input file")

        # ========== STEP 6: TABLES ==========
        if args.step in ["tables", "all"]:
            print("\n[Step 6: Table OCR]")
            
            if args.no_table_ocr:
                print("  - Skipped (--no-table-ocr)")
            
            for candidate in [with_assets_out, ml_filtered_out, organized_out]:
                if os.path.exists(candidate):
                    input_for_tables = candidate
                    break
            else:
                input_for_tables = None
            
            if input_for_tables:
                run_table_processing_on_file(
                    input_for_tables, 
                    tables_out, 
                    args.figures_dir, 
                    stem, 
                    llm_config,
                    use_llm_ocr=not args.no_table_ocr
                )

        # ========== STEP 7: WRITE DOCX ==========
        if args.step in ["write", "all"]:
            print("\n[Step 7: Write DOCX]")
            
            for candidate in [tables_out, with_assets_out, ml_filtered_out, organized_out]:
                if os.path.exists(candidate):
                    input_for_docx = candidate
                    break
            else:
                print("  - [Error] No input file")
                continue
            
            run_docx_creation(
                input_for_docx, 
                final_docx, 
                args.figures_dir, 
                stem, 
                title_out
            )

        # ========== STEP 8: VALIDATE ==========
        if args.step in ["validate", "all"]:
            print("\n[Step 8: Validation]")
            
            if VALIDATION_AVAILABLE:
                result = run_validation_on_file(stem, args.raw_ocr_dir, args.results_dir)
                if result:
                    validation_results.append({
                        'document': stem,
                        'has_toc': result.has_toc,
                        'toc_coverage': result.toc_coverage,
                        'precision': result.precision,
                        'toc_count': len(result.toc_sections),
                        'output_count': len(result.output_sections),
                        'missing_count': len(result.in_toc_not_output),
                        'extra_count': len(result.in_output_not_toc)
                    })

        print()

    # ========== VALIDATION SUMMARY ==========
    if validation_results:
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        docs_with_toc = [r for r in validation_results if r['has_toc']]
        docs_without_toc = [r for r in validation_results if not r['has_toc']]
        
        print(f"\nDocuments: {len(validation_results)} total")
        
        if docs_with_toc:
            avg_coverage = sum(r['toc_coverage'] for r in docs_with_toc) / len(docs_with_toc)
            avg_precision = sum(r['precision'] for r in docs_with_toc) / len(docs_with_toc)
            
            print(f"\nDocuments WITH TOC ({len(docs_with_toc)}):")
            print(f"  Avg TOC Coverage: {avg_coverage:.1f}%")
            print(f"  Avg Precision:    {avg_precision:.1f}%")
            
            for doc in docs_with_toc:
                status = "OK" if doc['toc_coverage'] >= 90 and doc['precision'] >= 90 else "**"
                print(f"  {status} {doc['document']}: Cov={doc['toc_coverage']:.0f}%, Prec={doc['precision']:.0f}%")
        
        if docs_without_toc:
            print(f"\nDocuments WITHOUT TOC ({len(docs_without_toc)}):")
            for doc in docs_without_toc:
                print(f"  {doc['document']}: {doc['output_count']} sections")
        
        # Save summary
        summary_path = os.path.join(args.results_dir, "_validation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'ml_threshold': args.ml_threshold,
                'average_toc_coverage': round(avg_coverage, 2) if docs_with_toc else None,
                'average_precision': round(avg_precision, 2) if docs_with_toc else None,
                'documents': validation_results
            }, f, indent=2)

    print("\nPipeline complete!")


if __name__ == '__main__':
    main()