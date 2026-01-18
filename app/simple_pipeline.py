import os
import argparse
import json
import logging
from typing import List

# --- IMPORTS ---
# 1. Agents we are keeping
from title_agent import extract_title_page_info
from table_processor_agent import run_table_processing_on_file
from docx_writer import run_docx_creation

# 2. The new Section Processor (Make sure section_processor.py is in this folder!)
from section_processor import run_section_processing_on_file

def get_document_stems(input_dir: str) -> List[str]:
    """Finds all unique document stems (filenames without extension)."""
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
    Simplified title extraction: loads the raw JSON, finds Page 1 text, 
    and sends it to the Title Agent.
    """
    if not os.path.exists(raw_input_path):
        return

    with open(raw_input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Locate page 1 text robustly
    page_text = ""
    
    # Case 1: List of pages
    if isinstance(data, list) and len(data) > 0:
        data.sort(key=lambda x: int(x.get('page_num', 0) or x.get('page_Id', 0)))
        page_dict = data[0].get('page_dict', data[0])
        text_list = page_dict.get('text', [])
        page_text = " ".join([str(t) for t in text_list])
        
    # Case 2: Dict of pages (e.g. {"1": {...}, "2": {...}})
    elif isinstance(data, dict):
        # Try to find key '1', or just take the first one available
        first_key = next(iter(data))
        if '1' in data:
            page_obj = data['1']
        else:
            page_obj = data[first_key]
            
        page_dict = page_obj.get('page_dict', page_obj)
        text_list = page_dict.get('text', [])
        page_text = " ".join([str(t) for t in text_list])

    if not page_text:
        print(f"Warning: Could not extract text from Page 1 of {raw_input_path}")
        with open(output_path, 'w') as f: json.dump([], f)
        return

    print("  - Extracting title info from Page 1...")
    info = extract_title_page_info(
        page_text, 
        llm_config['model_name'], 
        llm_config['base_url'], 
        llm_config['api_key'], 
        llm_config['provider']
    )
    
    # Wrap in list as expected by docx writer
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([info], f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the simplified document processing pipeline.")
    parser.add_argument("--step", type=str, default="all", choices=["title", "structure", "tables", "write", "all"])
    # Adjust this path to where your RAW OCR files live
    parser.add_argument("--raw_ocr_dir", type=str, default=os.path.join("iris_ocr", "new_baseline"))
    # Adjust this path to where your figure exports live
    parser.add_argument("--figures_dir", type=str, default=os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "exports"))
    parser.add_argument("--results_dir", type=str, default="results_simple")
    
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    llm_config = {
        "provider": "mission_assist",
        "model_name": "gemma3", 
        "base_url": "http://devmissionassist.api.us.baesystems.com",
        "api_key": "aTOIT9hJM3DBYMQbEY"
    }

    doc_stems = get_document_stems(args.raw_ocr_dir)
    print(f"Found {len(doc_stems)} documents to process in '{args.raw_ocr_dir}'")

    for stem in doc_stems:
        print(f"\n=== Processing: {stem} ===")
        
        raw_input = os.path.join(args.raw_ocr_dir, f"{stem}.json")
        title_output = os.path.join(args.results_dir, f"{stem}_title.json")
        organized_output = os.path.join(args.results_dir, f"{stem}_organized.json")
        tables_output = os.path.join(args.results_dir, f"{stem}_with_tables.json")
        final_docx = os.path.join(args.results_dir, f"{stem}.docx")

        # 1. Title Extraction
        if args.step in ["title", "all"]:
            extract_title_from_first_page(raw_input, title_output, llm_config)

        # 2. Algorithmic Structure (Uses section_processor.py)
        if args.step in ["structure", "all"]:
            print("  - Running Algorithmic Section Detection...")
            run_section_processing_on_file(raw_input, organized_output)

        # 3. Table Processing
        if args.step in ["tables", "all"]:
            print("  - Running Table Processing...")
            # If structure step was skipped, ensure we have input
            input_for_tables = organized_output
            if not os.path.exists(input_for_tables):
                print(f"    [Skipping] Missing organized file: {input_for_tables}")
            else:
                run_table_processing_on_file(input_for_tables, tables_output, args.figures_dir, stem, llm_config)

        # 4. Write DOCX
        if args.step in ["write", "all"]:
            print("  - Writing DOCX...")
            # Prefer table output, fall back to organized output
            input_for_docx = tables_output if os.path.exists(tables_output) else organized_output
            
            if not os.path.exists(input_for_docx):
                print(f"    [Error] No suitable input file found for DOCX writing.")
            else:
                run_docx_creation(input_for_docx, final_docx, args.figures_dir, stem, title_output)

    print("\nPipeline execution finished.")