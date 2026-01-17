import os
import json
import re
import statistics
from typing import List, Dict, Tuple, Optional, Any

def save_results_to_json(data: List[Dict], output_path: str):
    """Saves the data list to a JSON file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {output_path}")

def load_single_document(filepath: str) -> List[Tuple[int, Dict]]:
    """
    Loads a single OCR JSON file. 
    Returns a list of tuples: (page_number_int, page_dict).
    Sorted strictly by integer page number.
    """
    print(f"Loading document: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sorted_pages = []
    
    if isinstance(data, list):
        for idx, page_data in enumerate(data):
            p_num = page_data.get('page_num', idx + 1) 
            sorted_pages.append((int(p_num), page_data))
            
    elif isinstance(data, dict):
        for page_key, page_val in data.items():
            try:
                p_num = int(page_key)
            except ValueError:
                continue 
            
            actual_page_dict = page_val.get('page_dict', page_val)
            sorted_pages.append((p_num, actual_page_dict))

    sorted_pages.sort(key=lambda x: x[0])
    return sorted_pages

def reconstruct_rich_paragraphs(page_dict: Dict[str, List], page_num: int) -> List[Dict[str, Any]]:
    """
    Converts parallel arrays into 'Rich Paragraph' objects with spatial metadata.
    """
    if not page_dict or 'text' not in page_dict:
        return []

    rich_paragraphs = []
    current_words = []
    current_heights = []
    current_lefts = []
    
    texts = page_dict.get('text', [])
    block_nums = page_dict.get('block_num', [])
    par_nums = page_dict.get('par_num', [])
    heights = page_dict.get('height', [])
    lefts = page_dict.get('left', [])
    
    count = len(texts)
    if count == 0: return []

    last_block = block_nums[0] if block_nums else 0
    last_par = par_nums[0] if par_nums else 0

    for i in range(count):
        word_text = str(texts[i]).strip()
        
        if not word_text:
            continue
            
        block_num = block_nums[i]
        par_num = par_nums[i]
        
        if block_num != last_block or par_num != last_par:
            if current_words:
                rich_paragraphs.append({
                    "text": " ".join(current_words),
                    "avg_height": statistics.mean(current_heights) if current_heights else 0,
                    "indentation": min(current_lefts) if current_lefts else 0,
                    "page": page_num,
                    "block_id": f"{last_block}_{last_par}"
                })
            
            current_words = []
            current_heights = []
            current_lefts = []
            last_block, last_par = block_num, par_num

        current_words.append(word_text)
        if i < len(heights): current_heights.append(heights[i])
        if i < len(lefts): current_lefts.append(lefts[i])

    if current_words:
        rich_paragraphs.append({
            "text": " ".join(current_words),
            "avg_height": statistics.mean(current_heights) if current_heights else 0,
            "indentation": min(current_lefts) if current_lefts else 0,
            "page": page_num,
            "block_id": f"{last_block}_{last_par}"
        })
        
    return rich_paragraphs

def get_document_stats(all_paragraphs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates the 'Body Text' profile and the 'Left Margin'.
    """
    if not all_paragraphs:
        return {"body_size": 10, "margin_left": 0}

    heights = [round(p['avg_height']) for p in all_paragraphs if p['avg_height'] > 0]
    
    # We look for the global minimum indentation to find the "Left Margin"
    valid_indents = [p['indentation'] for p in all_paragraphs if p['indentation'] >= 0]
    
    try:
        margin_left = min(valid_indents) if valid_indents else 0
    except ValueError:
        margin_left = 0

    try:
        body_size = statistics.mode(heights)
    except statistics.StatisticsError:
        body_size = statistics.median(heights) if heights else 10

    print(f"Document Stats - Body Font: ~{body_size}px, Left Margin: ~{margin_left}px")
    return {"body_size": body_size, "margin_left": margin_left}

def check_if_header(para: Dict[str, Any], stats: Dict[str, float]) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Determines if a paragraph is a header using Regex + Spatial Logic.
    Returns: (is_header, section_number, title_text, run_in_content)
    """
    text = para['text']
    avg_height = para['avg_height']
    indent = para['indentation']
    
    # 1. Regex: Capture "1.0", "1.1", "A." at start of string
    match = re.match(r'^\s*([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*\.?)\s*(.*)', text)
    
    if not match:
        return False, None, None, None

    potential_num = match.group(1).strip()
    rest_of_line = match.group(2).strip()

    # 2. Heuristics to reject obvious non-headers
    if not any(c.isdigit() for c in potential_num): return False, None, None, None
    if len(potential_num) > 12: return False, None, None, None
    
    # 3. SPATIAL FILTER: List Item vs Section
    is_complex_number = potential_num.count('.') >= 2
    indent_tolerance = 25 # pixels
    
    if not is_complex_number:
        if indent > (stats['margin_left'] + indent_tolerance):
            # It's a single digit (1.) or simple (1.1) but indented -> Likely a List Item
            return False, None, None, None

    # 4. Title Extraction (User Rule: First 1-2 words are title)
    if not rest_of_line:
        # Just "1.0" or "2.0"
        return True, potential_num.rstrip('.'), "", ""

    words = rest_of_line.split()
    if len(words) <= 2:
        title_text = rest_of_line
        run_in_content = ""
    else:
        # Heuristic: First 2 words are title, rest is content
        title_text = " ".join(words[:2])
        run_in_content = " ".join(words[2:])

    return True, potential_num.rstrip('.'), title_text, run_in_content

def group_elements(classified_items: List[Dict]) -> List[Dict]:
    """
    Merges unassigned text. 
    """
    if not classified_items:
        return []

    merged_elements = []
    current_section = None

    for item in classified_items:
        if item['type'] == 'section':
            if current_section:
                merged_elements.append(current_section)
            
            # Start new section
            current_section = item
            # If the header detection extracted run-in content, start with that.
            initial_content = item.pop('run_in_content', '')
            current_section['content'] = initial_content
            
            # Ensure page metadata is top-level
            if 'page' not in current_section:
                current_section['page'] = item.get('page')

        elif item['type'] == 'unassigned_text_block':
            text = item.get('content', '')
            if current_section:
                if current_section['content']:
                    current_section['content'] += "\n\n" + text
                else:
                    current_section['content'] = text
            else:
                merged_elements.append(item)

    if current_section:
        merged_elements.append(current_section)
            
    return merged_elements

def run_algorithmic_organization(input_file_path: str, output_path: str):
    # 1. Load
    sorted_pages = load_single_document(input_file_path)
    if not sorted_pages: return

    # 2. Reconstruct Rich Paragraphs
    all_rich_paragraphs = []
    for page_num, page_dict in sorted_pages:
        paras = reconstruct_rich_paragraphs(page_dict, page_num)
        all_rich_paragraphs.extend(paras)

    # 3. Get Spatial Baseline
    doc_stats = get_document_stats(all_rich_paragraphs)

    # 4. Classify
    classified_items = []
    for para in all_rich_paragraphs:
        is_header_bool, sec_num, topic, run_in_content = check_if_header(para, doc_stats)

        if is_header_bool:
            classified_items.append({
                "type": "section",
                "section_number": sec_num,
                "topic": topic,
                "run_in_content": run_in_content, 
                "page": para['page']
            })
        else:
            classified_items.append({
                "type": "unassigned_text_block",
                "content": para['text'],
                "page": para['page']
            })

    # 5. Group
    final_elements = group_elements(classified_items)
    save_results_to_json(final_elements, output_path)

if __name__ == '__main__':
    project_root = os.getcwd() 
    
    # Updated Configuration
    doc_stem = "S-133-05737AF-SSS"
    
    # Updated input directory to 'raw_data_advanced'
    input_file = os.path.join(project_root, "iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced", f"{doc_stem}.json")
    output_file = os.path.join(project_root, "results", f"{doc_stem}_algo_organized.json")

    print(f"--- Processing {doc_stem} ---")
    run_algorithmic_organization(input_file, output_file)
    print("--- Algorithmic Organization Finished ---")