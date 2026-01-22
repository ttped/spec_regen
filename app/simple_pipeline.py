"""
simple_pipeline.py - Streamlined document processing pipeline.

Pipeline Steps:
1. classify  - Determine where content starts (skip ToC)
2. title     - Extract title page information  
3. structure - Parse sections from content pages (preserving original page numbers)
4. repair    - Validate section numbering and demote false positives (lists, tables)
5. assets    - Integrate figures/tables at correct positions based on bbox
6. tables    - OCR any table images into structured data (or skip with --no-table-ocr)
7. write     - Generate final DOCX
8. validate  - Compare extracted sections against TOC (optional, runs with --validate)

Usage:
    python simple_pipeline.py --step all
    python simple_pipeline.py --step classify
    python simple_pipeline.py --step structure --header-threshold 600
    python simple_pipeline.py --step repair --repair-confidence 0.7
    python simple_pipeline.py --step all --no-table-ocr  # Skip LLM table OCR, use images
    python simple_pipeline.py --step all --validate      # Run validation at end
"""

import os
import argparse
import json
from json import JSONDecodeError
from typing import List, Optional

# Import pipeline components
try:
    from classify_agent import run_classification_on_file, load_content_start_page, load_classification_result
    from title_agent import extract_title_page_info
    from section_repair_agent import run_section_repair
    from asset_processor import run_asset_integration
    from table_processor_agent import run_table_processing_on_file
    from docx_writer import run_docx_creation
    from section_processor import run_section_processing_on_file
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure all agent modules are in the same directory or Python path.")
    exit(1)

# Validation is optional - don't fail if not present
try:
    from validation_agent import run_validation_on_file
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


def get_document_stems(input_dir: str) -> List[str]:
    """Find all unique document stems (JSON files without extension) in a directory."""
    stems = set()
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist.")
        return []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            stems.add(os.path.splitext(filename)[0])
    return sorted(list(stems))


def extract_title_from_first_page(raw_input_path: str, output_path: str, llm_config: dict):
    """
    Extract title page information from page 1 of the raw OCR.
    Uses flattened text only for LLM processing.
    """
    if not os.path.exists(raw_input_path):
        print(f"    [Warning] Raw file not found: {raw_input_path}")
        with open(output_path, 'w') as f:
            json.dump([], f)
        return
    
    # 1. Load Raw Data (Defensive)
    try:
        with open(raw_input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except JSONDecodeError as e:
        print(f"    [Error] Failed to parse raw OCR file: {e}")
        with open(output_path, 'w') as f:
            json.dump([], f)
        return
    
    page_text = ""
    
    # Handle List structure
    if isinstance(data, list) and len(data) > 0:
        # Sort by page number to ensure we get page 1
        data.sort(key=lambda x: int(x.get('page_num', 0) or x.get('page_Id', 0) or 0))
        page_dict = data[0].get('page_dict', data[0])
        page_text = " ".join([str(t) for t in page_dict.get('text', [])])
    
    # Handle Dict structure
    elif isinstance(data, dict):
        # Look for key "1" first, then fall back to first key
        key = "1" if "1" in data else next(iter(data), None)
        if key:
            page_obj = data[key]
            page_dict = page_obj.get('page_dict', page_obj) if isinstance(page_obj, dict) else page_obj
            if isinstance(page_dict, dict):
                page_text = " ".join([str(t) for t in page_dict.get('text', [])])

    if not page_text:
        print("    [Warning] Could not extract text from page 1")
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

    print("    Extracting title info from Page 1...")
    
    # 2. Call Title Agent (Defensive)
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
        print(f"    [Error] Title extraction failed (LLM or JSON error): {e}")
        # Continue without title info rather than crashing pipeline
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([info] if info else [], f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Simple document processing pipeline")
    parser.add_argument(
        "--step", 
        type=str, 
        default="all", 
        choices=["classify", "title", "structure", "repair", "assets", "tables", "write", "validate", "all"],
        help="Pipeline step to execute"
    )
    parser.add_argument(
        "--raw_ocr_dir", 
        type=str, 
        default=os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced"),
        help="Directory containing raw OCR JSON files"
    )
    parser.add_argument(
        "--figures_dir", 
        type=str, 
        default=os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "exports"),
        help="Directory containing exported figures/tables"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results_simple",
        help="Directory for output files"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers for LLM tasks"
    )
    parser.add_argument(
        "--asset-positioning",
        type=str,
        default="bbox",
        choices=["bbox", "page_end"],
        help="How to position figures/tables: 'bbox' for precise positioning, 'page_end' for simpler approach"
    )
    parser.add_argument(
        "--header-threshold",
        type=int,
        default=600,
        help="Vertical (top) coordinate threshold for header filtering. Text above this value (less than) is dropped. Set to 0 to disable."
    )
    parser.add_argument(
        "--footer-threshold",
        type=int,
        default=6100,
        help="Vertical (top) coordinate threshold for footer filtering. Text below this value (greater than) is dropped. Set to 0 to disable."
    )
    parser.add_argument(
        "--repair-confidence",
        type=float,
        default=0.7,
        help="Confidence threshold for section repair. Only violations with confidence >= this value are fixed. (default: 0.7)"
    )
    parser.add_argument(
        "--skip-repair",
        action="store_true",
        help="Skip the section repair step (useful for debugging)"
    )
    parser.add_argument(
        "--no-table-ocr",
        action="store_true",
        help="Skip LLM OCR for tables. Tables will be rendered as images in the final document. "
             "This still preserves Table captions (separate from Figure captions) for Table of Tables support."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation step to compare extracted sections against TOC"
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

    # Track validation results for summary
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
        repaired_out = os.path.join(args.results_dir, f"{stem}_repaired.json")
        with_assets_out = os.path.join(args.results_dir, f"{stem}_with_assets.json")
        tables_out = os.path.join(args.results_dir, f"{stem}_with_tables.json")
        final_docx = os.path.join(args.results_dir, f"{stem}.docx")
        validation_out = os.path.join(args.results_dir, f"{stem}_validation.json")

        # ========== STEP 1: CLASSIFY ==========
        if args.step in ["classify", "all"]:
            print("\n[Step 1: Classify]")
            run_classification_on_file(raw_input, classify_out, llm_config, max_workers=args.max_workers)
            
            # Check for stub documents
            classification_result = load_classification_result(classify_out)
            if classification_result and classification_result.get('is_stub'):
                print(f"  - STUB DOCUMENT: Redirects to '{classification_result.get('stub_redirect')}'")
                print(f"  - Skipping remaining steps for this document.")
                continue

        # ========== STEP 2: TITLE ==========
        if args.step in ["title", "all"]:
            print("\n[Step 2: Title Extraction]")
            
            # Check document type - skip title extraction if no title page
            classification_result = load_classification_result(classify_out)
            doc_type = classification_result.get('document_type', 'unknown') if classification_result else 'unknown'
            
            if doc_type == 'no_title':
                print(f"  - Document has no title page (content starts on page 1). Skipping title extraction.")
                # Create empty title file
                os.makedirs(os.path.dirname(title_out) or '.', exist_ok=True)
                with open(title_out, 'w') as f:
                    json.dump([], f)
            else:
                extract_title_from_first_page(raw_input, title_out, llm_config)

        # ========== STEP 3: STRUCTURE ==========
        if args.step in ["structure", "all"]:
            print("\n[Step 3: Structure/Section Detection]")
            
            # Load content start page from classification result
            content_start_page = load_content_start_page(classify_out, default=1)
            print(f"  - Using content start page: {content_start_page}")
            
            # Process sections, skipping ToC pages
            run_section_processing_on_file(
                raw_input, 
                organized_out, 
                content_start_page=content_start_page,
                header_top_threshold=args.header_threshold,
                footer_top_threshold=args.footer_threshold
            )

        # ========== STEP 4: REPAIR ==========
        if args.step in ["repair", "all"] and not args.skip_repair:
            print("\n[Step 4: Section Repair]")
            if os.path.exists(organized_out):
                run_section_repair(
                    organized_out,
                    repaired_out,
                    confidence_threshold=args.repair_confidence,
                    classification_path=classify_out,
                    raw_ocr_path=raw_input
                )
            else:
                print(f"  [Skipping] Missing organized file: {organized_out}")
        elif args.skip_repair and args.step == "all":
            print("\n[Step 4: Section Repair] SKIPPED (--skip-repair flag)")
            # Copy organized to repaired so downstream steps work
            if os.path.exists(organized_out):
                import shutil
                shutil.copy(organized_out, repaired_out)
                print(f"  - Copied {organized_out} to {repaired_out}")

        # ========== STEP 5: ASSETS ==========
        if args.step in ["assets", "all"]:
            print("\n[Step 5: Asset Integration (Figures/Tables)]")
            # Prefer repaired file, fall back to organized
            if os.path.exists(repaired_out):
                input_for_assets = repaired_out
            elif os.path.exists(organized_out):
                input_for_assets = organized_out
                print("  [Note] Using organized file (no repair step)")
            else:
                print(f"  [Skipping] Missing input file")
                input_for_assets = None
            
            if input_for_assets:
                run_asset_integration(
                    input_for_assets,
                    with_assets_out,
                    args.figures_dir,
                    stem,
                    positioning_mode=args.asset_positioning
                )

        # ========== STEP 6: TABLES ==========
        if args.step in ["tables", "all"]:
            print("\n[Step 6: Table OCR Processing]")
            
            if args.no_table_ocr:
                print("  - LLM table OCR disabled (--no-table-ocr). Tables will render as images.")
            
            # Prefer file with assets, fall back through the chain
            if os.path.exists(with_assets_out):
                input_for_tables = with_assets_out
            elif os.path.exists(repaired_out):
                input_for_tables = repaired_out
                print("  [Note] Using repaired file (no asset integration)")
            elif os.path.exists(organized_out):
                input_for_tables = organized_out
                print("  [Note] Using organized file (no repair or asset integration)")
            else:
                print(f"  [Skipping] No input file available")
                input_for_tables = None
            
            if input_for_tables:
                run_table_processing_on_file(
                    input_for_tables, 
                    tables_out, 
                    args.figures_dir, 
                    stem, 
                    llm_config,
                    use_llm_ocr=not args.no_table_ocr  # Pass the flag
                )

        # ========== STEP 7: WRITE DOCX ==========
        if args.step in ["write", "all"]:
            print("\n[Step 7: Write DOCX]")
            
            # Prefer most processed file, with fallbacks
            if os.path.exists(tables_out):
                input_for_docx = tables_out
            elif os.path.exists(with_assets_out):
                input_for_docx = with_assets_out
                print("  [Note] Using file with assets (no table OCR)")
            elif os.path.exists(repaired_out):
                input_for_docx = repaired_out
                print("  [Note] Using repaired file (no assets or table OCR)")
            elif os.path.exists(organized_out):
                input_for_docx = organized_out
                print("  [Note] Using organized file (no repair, assets, or table OCR)")
            else:
                print("  [Error] No input file for DOCX creation")
                continue
            
            run_docx_creation(
                input_for_docx, 
                final_docx, 
                args.figures_dir, 
                stem, 
                title_out
            )

        # ========== STEP 8: VALIDATE ==========
        if args.step in ["validate", "all"] and args.validate:
            print("\n[Step 8: Validation]")
            
            if not VALIDATION_AVAILABLE:
                print("  [Warning] validation_agent.py not found. Skipping validation.")
            else:
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

        print()  # Blank line between documents

    # ========== VALIDATION SUMMARY ==========
    if validation_results:
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        # Separate documents with and without TOC
        docs_with_toc = [r for r in validation_results if r['has_toc']]
        docs_without_toc = [r for r in validation_results if not r['has_toc']]
        
        print(f"\nTotal documents validated: {len(validation_results)}")
        print(f"  - With TOC: {len(docs_with_toc)}")
        print(f"  - Without TOC: {len(docs_without_toc)}")
        
        # Calculate stats only for documents WITH a TOC
        if docs_with_toc:
            avg_coverage = sum(r['toc_coverage'] for r in docs_with_toc) / len(docs_with_toc)
            avg_precision = sum(r['precision'] for r in docs_with_toc) / len(docs_with_toc)
            
            print(f"\n--- Documents WITH TOC ---")
            print(f"Average TOC Coverage: {avg_coverage:.1f}%")
            print(f"Average Precision: {avg_precision:.1f}%")
            
            # Flag documents with issues
            problem_docs = [r for r in docs_with_toc if r['toc_coverage'] < 90 or r['precision'] < 90]
            if problem_docs:
                print(f"\nDocuments with potential issues (<90% coverage or precision):")
                for doc in problem_docs:
                    print(f"  - {doc['document']}: Coverage={doc['toc_coverage']:.1f}%, Precision={doc['precision']:.1f}%, Missing={doc['missing_count']}, Extra={doc['extra_count']}")
            else:
                print("\nAll documents with TOC have good coverage and precision!")
        
        # Report on documents without TOC
        if docs_without_toc:
            print(f"\n--- Documents WITHOUT TOC (not included in stats) ---")
            for doc in docs_without_toc:
                print(f"  - {doc['document']}: {doc['output_count']} sections extracted")
        
        # Write summary file
        summary_path = os.path.join(args.results_dir, "_validation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_documents': len(validation_results),
                'documents_with_toc': len(docs_with_toc),
                'documents_without_toc': len(docs_without_toc),
                'average_toc_coverage': round(avg_coverage, 2) if docs_with_toc else None,
                'average_precision': round(avg_precision, 2) if docs_with_toc else None,
                'documents': validation_results
            }, f, indent=2)
        print(f"\nValidation summary saved to: {summary_path}")

    print("\nPipeline complete!")


if __name__ == '__main__':
    main()