import os
import argparse
import json
from typing import List

# Import specific functional agents
try:
    from title_agent import extract_title_page_info
    from table_processor_agent import run_table_processing_on_file
    from docx_writer import run_docx_creation
    from section_processor import run_section_processing_on_file
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    exit(1)

def get_document_stems(input_dir: str) -> List[str]:
    stems = set()
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist.")
        return []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            stems.add(os.path.splitext(filename)[0])
    return sorted(list(stems))

def extract_title_from_first_page(raw_input_path: str, output_path: str, llm_config: dict):
    if not os.path.exists(raw_input_path): return
    with open(raw_input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Helper to get text from page 1 regardless of format
    page_text = ""
    if isinstance(data, list) and len(data) > 0:
        data.sort(key=lambda x: int(x.get('page_num', 0) or x.get('page_Id', 0) or 0))
        page_dict = data[0].get('page_dict', data[0])
        page_text = " ".join([str(t) for t in page_dict.get('text', [])])
    elif isinstance(data, dict):
        # Try to find key '1' or the first key available
        key = "1" if "1" in data else next(iter(data))
        page_obj = data[key]
        page_dict = page_obj.get('page_dict', page_obj)
        page_text = " ".join([str(t) for t in page_dict.get('text', [])])

    if not page_text:
        with open(output_path, 'w') as f: json.dump([], f)
        return

    print("  - Extracting title info...")
    info = extract_title_page_info(
        page_text, llm_config['model_name'], llm_config['base_url'], 
        llm_config['api_key'], llm_config['provider']
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([info], f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all", choices=["title", "structure", "tables", "write", "all"])
    
    # --- UPDATED PATH HERE ---
    default_raw_dir = os.path.join("iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced")
    parser.add_argument("--raw_ocr_dir", type=str, default=default_raw_dir)
    
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

    print(f"Checking for documents in: {args.raw_ocr_dir}")
    doc_stems = get_document_stems(args.raw_ocr_dir)
    print(f"Found {len(doc_stems)} documents.")

    for stem in doc_stems:
        print(f"\n=== Processing: {stem} ===")
        raw_input = os.path.join(args.raw_ocr_dir, f"{stem}.json")
        title_out = os.path.join(args.results_dir, f"{stem}_title.json")
        organized_out = os.path.join(args.results_dir, f"{stem}_organized.json")
        tables_out = os.path.join(args.results_dir, f"{stem}_with_tables.json")
        final_docx = os.path.join(args.results_dir, f"{stem}.docx")

        if args.step in ["title", "all"]:
            extract_title_from_first_page(raw_input, title_out, llm_config)

        if args.step in ["structure", "all"]:
            print("  [Step: Structure]")
            run_section_processing_on_file(raw_input, organized_out)

        if args.step in ["tables", "all"]:
            print("  [Step: Tables]")
            # Prefer using the organized output as input for table processing
            inp = organized_out if os.path.exists(organized_out) else None
            if inp: 
                run_table_processing_on_file(inp, tables_out, args.figures_dir, stem, llm_config)
            else:
                print(f"    [Skipping] Missing input file: {organized_out}")

        if args.step in ["write", "all"]:
            print("  [Step: Write]")
            # Prefer table-processed file, fallback to organized file
            inp = tables_out if os.path.exists(tables_out) else organized_out
            if os.path.exists(inp):
                run_docx_creation(inp, final_docx, args.figures_dir, stem, title_out)
            else:
                print("    [Error] No input file for DOCX creation.")