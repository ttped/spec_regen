import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any

def save_results_to_json(data: List[Dict], output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {output_path}")

def load_single_document(filepath: str) -> List[Tuple[int, Dict]]:
    """
    Loads a single OCR JSON file. Robustly handles various key formats.
    """
    print(f"Loading document: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sorted_pages = []
    
    # Normalize List vs Dict structure
    if isinstance(data, list):
        for idx, page_data in enumerate(data):
            p_num = page_data.get('page_num', idx + 1)
            try:
                p_num = int(p_num)
            except (ValueError, TypeError):
                p_num = idx + 1
            sorted_pages.append((p_num, page_data))
            
    elif isinstance(data, dict):
        for idx, (page_key, page_val) in enumerate(data.items()):
            try:
                p_num = int(page_key)
            except ValueError:
                # Fallback for non-integer keys
                digits = re.findall(r'\d+', str(page_key))
                p_num = int(digits[0]) if digits else idx + 1
            
            actual_page_dict = page_val.get('page_dict', page_val)
            sorted_pages.append((p_num, actual_page_dict))

    sorted_pages.sort(key=lambda x: x[0])
    print(f"Loaded {len(sorted_pages)} pages.")
    return sorted_pages

def is_section_header_word(word: str, prev_word: str) -> bool:
    """
    Checks if a single word looks like a section number.
    Includes a 'Stop Word' check to avoid 'Figure 1.1', 'Table 2.0'.
    """
    # 1. Strict Regex for Section Numbers (e.g., "1.0", "2.1.3", "A.", "3.")
    # Must start with digit or single letter, contain dots, end with digit or dot.
    # Excludes strings like "100mm" or "3000".
    
    # Matches: "1.", "1.1", "1.1.", "A.1"
    # Rejects: "1,000", "1995" (years), "10-5"
    if not re.match(r'^[A-Z0-9]+(?:\.[A-Z0-9]+)+\.?$|^[A-Z0-9]+\.$', word):
        # Allow simple "1.0" or "2." cases specifically
        if not re.match(r'^\d+\.\d*$', word):
            return False

    # 2. Context Filter (Stop Words)
    # If the previous word indicates a reference, ignore this number.
    stop_words = {"figure", "fig", "table", "tab", "ref", "reference", "see", "section", "paragraph", "para"}
    clean_prev = prev_word.lower().strip(".,:;")
    if clean_prev in stop_words:
        return False

    return True

def process_document_word_stream(sorted_pages: List[Tuple[int, Dict]]) -> List[Dict]:
    """
    Iterates through the document word-by-word.
    Captures ANY word matching the section regex as a new section.
    """
    final_elements = []
    
    # State tracking
    current_section = None
    # If we find a header, we treat the next few words as the title
    capture_title_mode = False 
    title_word_count = 0
    MAX_TITLE_WORDS = 2
    
    last_line_id = None
    prev_word = ""

    for page_num, page_dict in sorted_pages:
        texts = page_dict.get('text', [])
        # We need geometry to track line breaks and bboxes
        lefts = page_dict.get('left', [])
        tops = page_dict.get('top', [])
        widths = page_dict.get('width', [])
        heights = page_dict.get('height', [])
        
        # Use block/par/line to detect visual breaks
        block_nums = page_dict.get('block_num', [])
        par_nums = page_dict.get('par_num', [])
        line_nums = page_dict.get('line_num', [])
        
        count = len(texts)
        
        for i in range(count):
            word = str(texts[i]).strip()
            if not word: continue

            # Construct a unique ID for the visual line to detect breaks
            # Default to 0 if keys missing
            b_n = block_nums[i] if i < len(block_nums) else 0
            p_n = par_nums[i] if i < len(par_nums) else 0
            l_n = line_nums[i] if i < len(line_nums) else 0
            current_line_id = f"{page_num}_{b_n}_{p_n}_{l_n}"
            
            # Insert newline if visual line changed (and it's not the very first word)
            separator = " "
            if last_line_id is not None and current_line_id != last_line_id:
                separator = "\n"
            last_line_id = current_line_id

            # --- HEADER DETECTION LOGIC ---
            if is_section_header_word(word, prev_word):
                # 1. Save previous section
                if current_section:
                    final_elements.append(current_section)
                
                # 2. Start New Section
                bbox = None
                if i < len(lefts) and i < len(tops) and i < len(widths) and i < len(heights):
                    bbox = [lefts[i], tops[i], widths[i], heights[i]]

                current_section = {
                    "type": "section",
                    "section_number": word,
                    "topic": "", # Will be filled by next words
                    "content": "",
                    "page": page_num,
                    "header_bbox": bbox
                }
                
                # Reset state
                capture_title_mode = True
                title_word_count = 0
                prev_word = word
                continue # Skip adding this word to content
            
            # --- CONTENT / TITLE LOGIC ---
            
            # If no section exists yet, treat as unassigned text
            if current_section is None:
                # Create a dummy section or unassigned block? 
                # User prefers ensuring nothing is skipped. Let's make an unassigned block.
                # However, for simplicity based on "group by section", we can just 
                # make a temporary "Preamble" section if needed, or wait.
                # Let's verify if we should append to a raw text block.
                if not final_elements or final_elements[-1]['type'] != 'unassigned_text_block':
                    final_elements.append({
                        "type": "unassigned_text_block", 
                        "content": word, 
                        "page": page_num
                    })
                    current_section = final_elements[-1] # Point to this block for appending
                else:
                    # Append to existing unassigned block
                    if separator == "\n":
                        final_elements[-1]['content'] += "\n" + word
                    else:
                        final_elements[-1]['content'] += " " + word
            
            else:
                # We have an active section (or unassigned block)
                if current_section['type'] == 'section' and capture_title_mode:
                    # Append to Title
                    current_section['topic'] += (" " + word).strip()
                    title_word_count += 1
                    if title_word_count >= MAX_TITLE_WORDS:
                        capture_title_mode = False
                
                else:
                    # Append to Content
                    if not current_section['content']:
                        current_section['content'] = word
                    else:
                        current_section['content'] += separator + word

            prev_word = word

    # Append the last section being built
    if current_section and current_section not in final_elements:
        final_elements.append(current_section)
        
    return final_elements

def run_algorithmic_organization(input_file_path: str, output_path: str):
    # 1. Load (Robust)
    sorted_pages = load_single_document(input_file_path)
    if not sorted_pages: return

    # 2. Process Word Stream (No spatial filtering)
    final_elements = process_document_word_stream(sorted_pages)

    print(f"Extracted {len(final_elements)} elements.")
    save_results_to_json(final_elements, output_path)

if __name__ == '__main__':
    project_root = os.getcwd() 
    doc_stem = "S-133-05737AF-SSS"
    input_file = os.path.join(project_root, "iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced", f"{doc_stem}.json")
    output_file = os.path.join(project_root, "results", f"{doc_stem}_algo_organized.json")

    print(f"--- Processing {doc_stem} (Word-Stream Mode) ---")
    run_algorithmic_organization(input_file, output_file)
    print("--- Finished ---")