import os
import argparse
import json
import shutil
from typing import List

# Import the core functions from our agent scripts
from classify_agent import run_classification_on_file
from title_agent import run_title_extraction_on_file
from organize_agent import run_organization_on_file
from deduplicate_agent import run_deduplication_on_file
from repair_agent import run_repair_on_file
from polish_agent import run_polish_on_file
from docx_writer import run_docx_creation
from table_processor_agent import run_table_processing_on_file

def get_document_stems(input_dir: str) -> List[str]:
    """
    Finds all unique document stems (filenames without extension) in a directory.
    Includes enhanced debugging to diagnose file discovery issues.
    """
    stems = set()
    
    # --- START DEBUGGING ---
    print(f"\n[DEBUG] Inside get_document_stems...")
    print(f"[DEBUG] Checking for documents in directory: '{os.path.abspath(input_dir)}'")
    
    if not os.path.exists(input_dir):
        print(f"[DEBUG] ERROR: The directory does not exist!")
        return []
        
    try:
        all_files = os.listdir(input_dir)
        print(f"[DEBUG] Found {len(all_files)} total files/folders in the directory.")
        if len(all_files) < 15: # Print file list only if it's reasonably short
             print(f"[DEBUG] File list: {all_files}")
    except Exception as e:
        print(f"[DEBUG] ERROR: Could not list files in the directory. Reason: {e}")
        return []
    # --- END DEBUGGING ---

    for filename in all_files:
        if filename.endswith(".json"):
            stems.add(os.path.splitext(filename)[0])
            
    if not stems:
        print("[DEBUG] No files ending with '.json' were found in the listed files.")
        
    return sorted(list(stems))

if __name__ == '__main__':
    START = 1
    STOP = START + 1# + 9999

    parser = argparse.ArgumentParser(description="Run the document processing pipeline in stages.")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["classify", "title", "organize", "deduplicate", "repair", "polish", "tables", "write", "all"],
        help="The pipeline step to execute."
    )
    parser.add_argument(
        "--raw_ocr_dir",
        type=str,
        default=os.path.join("iris_ocr", "new_baseline"),#os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data"),
        help="Directory containing the raw OCR JSON files."
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default=os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "exports"),
        help="Directory containing the exported figures and their metadata."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save intermediate and final results."
    )
    parser.add_argument(
        "--disable-deduplication",
        action="store_true",
        help="Disable the deduplication and content merging step."
    )
    parser.add_argument(
        "--disable-repair",
        action="store_true",
        help="Disable the section number cleaning and typo repair step."
    )
    parser.add_argument(
        "--polish",
        action="store_true",
        help="Enable the final polishing agent to clean up section content."
    )
    parser.add_argument(
        "--pre-clean-pages",
        action="store_true",
        help="Enable the pre-cleaning step to remove headers/footers from each page."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel threads for LLM-heavy tasks."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=950,
        help="Character size of each chunk for the organization step."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=75,
        help="Character overlap between chunks for the organization step."
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    llm_config = {
        "provider": "mission_assist",
        "model_name": "gptoss", # gpt-oss or gemma3
        "base_url": "http://devmissionassist.api.us.baesystems.com",
        "api_key": "aTOIT9hJM3DBYMQbEY"
    }

    doc_stems = get_document_stems(args.raw_ocr_dir)
    print(f"Found {len(doc_stems)} documents to process.")
    print(f"Deduplication Disabled: {args.disable_deduplication}")
    print(f"Repair Disabled: {args.disable_repair}")
    print(f"Final Polish Enabled: {args.polish}")
    print(f"Max Parallel Workers: {args.max_workers}")
    print(f"Organization Chunk Size: {args.chunk_size}, Overlap: {args.overlap}")

    if args.step in ["classify", "all"]:
        print("\n--- RUNNING CLASSIFICATION STEP ---")
        for stem in doc_stems[START:STOP]:
            input_path = os.path.join(args.raw_ocr_dir, f"{stem}.json")
            output_path = os.path.join(args.results_dir, f"{stem}_classified.json")
            print(f"Classifying: {stem}")
            run_classification_on_file(input_path, output_path, llm_config, max_workers=args.max_workers)

    if args.step in ["title", "all"]:
        print("\n--- RUNNING TITLE EXTRACTION STEP ---")
        for stem in doc_stems[START:STOP]:
            input_path = os.path.join(args.results_dir, f"{stem}_classified.json")
            output_path = os.path.join(args.results_dir, f"{stem}_title_data.json")
            print(f"Extracting title info for: {stem}")
            run_title_extraction_on_file(input_path, output_path, llm_config)

    if args.step in ["organize", "all"]:
        print("\n--- RUNNING ORGANIZATION (RAW EXTRACTION) STEP ---")
        for stem in doc_stems[START:STOP]:
            input_path = os.path.join(args.results_dir, f"{stem}_classified.json")
            output_path = os.path.join(args.results_dir, f"{stem}_organized_raw.json")
            print(f"Organizing (raw extraction): {stem}")
            
            run_organization_on_file(
                input_path, 
                output_path, 
                llm_config,
                figures_base_path=args.figures_dir,
                doc_stem=stem,
                pre_clean_pages=args.pre_clean_pages,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                max_workers=args.max_workers
            )

    if args.step in ["deduplicate", "all"]:
        if not args.disable_deduplication:
            print("\n--- RUNNING DEDUPLICATION & MERGE STEP ---")
            for stem in doc_stems[START:STOP]:
                input_path = os.path.join(args.results_dir, f"{stem}_organized_raw.json")
                output_path = os.path.join(args.results_dir, f"{stem}_organized_with_figures.json")
                print(f"Deduplicating and merging for: {stem}")
                run_deduplication_on_file(
                    input_path,
                    output_path
                )
        else:
            print("\n--- SKIPPING DEDUPLICATION & MERGE STEP ---")
            for stem in doc_stems[START:STOP]:
                input_path = os.path.join(args.results_dir, f"{stem}_organized_raw.json")
                output_path = os.path.join(args.results_dir, f"{stem}_organized_with_figures.json")
                if os.path.exists(input_path):
                    print(f"Copying raw organized file for: {stem}")
                    shutil.copy(input_path, output_path)
                else:
                    print(f"Warning: Input file {input_path} not found. Cannot create fallback for {stem}.")


    if args.step in ["repair", "all"]:
        if not args.disable_repair:
            print("\n--- RUNNING REPAIR STEP ---")
            for stem in doc_stems[START:STOP]:
                input_path = os.path.join(args.results_dir, f"{stem}_organized_with_figures.json")
                output_path = os.path.join(args.results_dir, f"{stem}_repaired.json")
                print(f"Repairing: {stem}")
                run_repair_on_file(
                    input_path, output_path, llm_config, 
                    repair_typos=False, # Typo repair is now part of this flag
                    max_workers=args.max_workers
                )
        else:
            print("\n--- SKIPPING REPAIR STEP ---")
            for stem in doc_stems[START:STOP]:
                input_path = os.path.join(args.results_dir, f"{stem}_organized_with_figures.json")
                output_path = os.path.join(args.results_dir, f"{stem}_repaired.json")
                if os.path.exists(input_path):
                    print(f"Copying file to bypass repair for: {stem}")
                    shutil.copy(input_path, output_path)
                else:
                    print(f"Warning: Input file {input_path} not found. Cannot create fallback for {stem}.")


    if args.step in ["polish", "all"] and args.polish:
        print("\n--- RUNNING POLISH STEP ---")
        for stem in doc_stems[START:STOP]:
            input_path = os.path.join(args.results_dir, f"{stem}_repaired.json")
            output_path = os.path.join(args.results_dir, f"{stem}_polished.json")
            print(f"Polishing: {stem}")
            run_polish_on_file(input_path, output_path, llm_config, max_workers=args.max_workers)


    if args.step in ["tables", "all"]:
        print("\n--- RUNNING TABLE PROCESSING STEP ---")
        for stem in doc_stems[START:STOP]:
            # Determine the correct input file (polished or repaired)
            polished_path = os.path.join(args.results_dir, f"{stem}_polished.json")
            repaired_path = os.path.join(args.results_dir, f"{stem}_repaired.json")
            
            if args.polish and os.path.exists(polished_path):
                input_path = polished_path
            else:
                input_path = repaired_path

            output_path = os.path.join(args.results_dir, f"{stem}_with_tables.json")
            print(f"Processing tables for: {stem}")
            run_table_processing_on_file(input_path, output_path, args.figures_dir, stem, llm_config)

    if args.step in ["write", "all"]:
        print("\n--- RUNNING DOCX WRITER STEP ---")
        for stem in doc_stems[START:STOP]:
            # Define paths for all possible input files
            with_tables_path = os.path.join(args.results_dir, f"{stem}_with_tables.json")
            polished_path = os.path.join(args.results_dir, f"{stem}_polished.json")
            repaired_path = os.path.join(args.results_dir, f"{stem}_repaired.json")
            organized_path = os.path.join(args.results_dir, f"{stem}_organized_with_figures.json")


            # Prioritize the file with the most processing
            if os.path.exists(with_tables_path):
                input_path = with_tables_path
                print(f"Writing DOCX for '{stem}' from TABLE-PROCESSED data.")
            elif args.polish and os.path.exists(polished_path):
                input_path = polished_path
                print(f"Writing DOCX for '{stem}' from POLISHED data (table step skipped).")
            elif os.path.exists(repaired_path):
                input_path = repaired_path
                print(f"Writing DOCX for '{stem}' from REPAIRED data (polish/table steps skipped).")
            else:
                input_path = organized_path # Fallback to the direct output of organize
                print(f"Writing DOCX for '{stem}' from ORGANIZED data (dedupe/repair/polish/table steps skipped).")

            
            title_path = os.path.join(args.results_dir, f"{stem}_title_data.json")
            output_path = os.path.join(args.results_dir, f"{stem}_final.docx")
            
            # Call docx creation without the llm_config
            run_docx_creation(input_path, output_path, args.figures_dir, stem, title_path)

    print("\nPipeline execution finished.")