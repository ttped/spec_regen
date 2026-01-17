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

    # The structure is { "1": {...}, "2": {...} }
    # We must convert keys to integers for sorting: "10" comes after "2", not "1".
    sorted_pages = []
    
    # Handle the case where the JSON might be a list (rare but possible in some exports)
    if isinstance(data, list):
        for idx, page_data in enumerate(data):
            # Try to find a page number, otherwise use index + 1
            p_num = page_data.get('page_num', idx + 1) 
            sorted_pages.append((int(p_num), page_data))
            
    elif isinstance(data, dict):
        for page_key, page_val in data.items():
            try:
                p_num = int(page_key)
            except ValueError:
                continue # Skip non-integer keys if any metadata keys exist
            
            # The actual word data is often inside a 'page_dict' key, 
            # or sometimes the value *is* the page_dict.
            actual_page_dict = page_val.get('page_dict', page_val)
            sorted_pages.append((p_num, actual_page_dict))

    # STRICT SORT by page number
    sorted_pages.sort(key=lambda x: x[0])
    
    print(f"Loaded {len(sorted_pages)} pages in correct order.")
    return sorted_pages

def reconstruct_rich_paragraphs(page_dict: Dict[str, List], page_num: int) -> List[Dict[str, Any]]:
    """
    Converts parallel arrays into 'Rich Paragraph' objects.
    Now attaches 'page' metadata to every paragraph.
    """
    # Defensive check: ensure 'text' exists
    if not page_dict or 'text' not in page_dict:
        return []

    rich_paragraphs = []
    current_words = []
    current_heights = []
    current_lefts = []
    
    # Get the lists, defaulting to empty if missing
    texts = page_dict.get('text', [])
    block_nums = page_dict.get('block_num', [])
    par_nums = page_dict.get('par_num', [])
    heights = page_dict.get('height', [])
    lefts = page_dict.get('left', [])
    
    count = len(texts)
    if count == 0: return []

    # Initialize tracking with the first item's ID
    last_block = block_nums[0] if block_nums else 0
    last_par = par_nums[0] if par_nums else 0

    for i in range(count):
        word_text = str(texts[i]).strip()
        
        # Skip empty layout noise
        if not word_text:
            continue
            
        block_num = block_nums[i]
        par_num = par_nums[i]
        
        # Check for new paragraph (Block change or Paragraph ID change)
        if block_num != last_block or par_num != last_par:
            if current_words:
                rich_paragraphs.append({
                    "text": " ".join(current_words),
                    "avg_height": statistics.mean(current_heights) if current_heights else 0,
                    "indentation": min(current_lefts) if current_lefts else 0,
                    "page": page_num,  # <--- METADATA ADDED
                    "block_id": f"{last_block}_{last_par}"
                })
            
            # Reset
            current_words = []
            current_heights = []
            current_lefts = []
            last_block, last_par = block_num, par_num

        current_words.append(word_text)
        if i < len(heights): current_heights.append(heights[i])
        if i < len(lefts): current_lefts.append(lefts[i])

    # Flush the final paragraph
    if current_words:
        rich_paragraphs.append({
            "text": " ".join(current_words),
            "avg_height": statistics.mean(current_heights) if current_heights else 0,
            "indentation": min(current_lefts) if current_lefts else 0,
            "page": page_num, # <--- METADATA ADDED
            "block_id": f"{last_block}_{last_par}"
        })
        
    return rich_paragraphs

def get_document_stats(all_paragraphs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates Document-wide visual baselines."""
    if not all_paragraphs:
        return {"body_size": 10, "body_indent": 0}

    heights = [round(p['avg_height']) for p in all_paragraphs if p['avg_height'] > 0]
    indents = [round(p['indentation'] / 5) * 5 for p in all_paragraphs]

    try:
        body_size = statistics.mode(heights)
    except statistics.StatisticsError:
        body_size = statistics.median(heights) if heights else 10

    try:
        body_indent = statistics.mode(indents)
    except statistics.StatisticsError:
        body_indent = min(indents) if indents else 0

    return {"body_size": body_size, "body_indent": body_indent}

def check_if_header(para: Dict[str, Any], stats: Dict[str, float]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Determines if a paragraph is a header using Regex + Visual Stats."""
    text = para['text']
    avg_height = para['avg_height']
    indent = para['indentation']
    
    # 1. Regex: Must look like a header (e.g., "1.2 Title")
    match = re.match(r'^\s*([a-zA-Z0-9\.]+)\s+(.+)', text)
    match_num_only = re.match(r'^\s*([a-zA-Z0-9\.]+)\s*$', text)

    if match:
        potential_num, topic = match.groups()
    elif match_num_only:
        potential_num = match_num_only.group(1)
        topic = ""
    else:
        return False, None, None

    # 2. Cleanup
    if not any(c.isdigit() for c in potential_num): return False, None, None 
    if len(potential_num) > 10: return False, None, None 

    # 3. Visual Confirmation
    is_visually_prominent = False
    
    # Larger than body text?
    if avg_height > (stats['body_size'] * 1.1):
        is_visually_prominent = True
    # Or, same size but strictly aligned to margin (and Regex matched)?
    elif avg_height >= (stats['body_size'] * 0.9) and indent <= (stats['body_indent'] + 10):
        is_visually_prominent = True

    if not is_visually_prominent:
        return False, None, None

    return True, potential_num.strip().rstrip('.'), topic.strip()

def group_elements(classified_items: List[Dict]) -> List[Dict]:
    """
    Merges unassigned text. 
    NOW PRESERVES 'page' info from the start of the block.
    """
    if not classified_items:
        return []

    merged_elements = []
    current_section = None

    for item in classified_items:
        if item['type'] == 'section':
            if current_section:
                merged_elements.append(current_section)
            
            current_section = item
            current_section['content'] = "" 
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
                # Independent text block (preamble)
                merged_elements.append(item)

    if current_section:
        merged_elements.append(current_section)
            
    return merged_elements

def run_algorithmic_organization(input_file_path: str, output_path: str):
    """
    Main orchestration function for a SINGLE document file.
    """
    # 1. Load pages sorted strictly by integer page number
    sorted_pages = load_single_document(input_file_path)
    
    if not sorted_pages:
        print("No pages loaded. Exiting.")
        return

    # 2. Flatten into Rich Paragraphs with Page IDs
    all_rich_paragraphs = []
    for page_num, page_dict in sorted_pages:
        paras = reconstruct_rich_paragraphs(page_dict, page_num)
        all_rich_paragraphs.extend(paras)

    # 3. Calc Stats
    doc_stats = get_document_stats(all_rich_paragraphs)

    # 4. Classify
    classified_items = []
    for para in all_rich_paragraphs:
        is_header_bool, sec_num, topic = check_if_header(para, doc_stats)

        if is_header_bool:
            classified_items.append({
                "type": "section",
                "section_number": sec_num,
                "topic": topic,
                "content": "",
                "page": para['page']  # Track where the section started
            })
        else:
            classified_items.append({
                "type": "unassigned_text_block",
                "content": para['text'],
                "page": para['page']  # Track where this text block is
            })

    # 5. Group
    final_elements = group_elements(classified_items)
    save_results_to_json(final_elements, output_path)

if __name__ == '__main__':
    # Usage Example
    # In your pipeline, you would loop through files and call this for each one.
    
    # Just for testing locally:
    project_root = os.getcwd() 
    doc_stem = "S-133-06923" # Replace with your actual test file stem
    
    # Assuming file is at .../raw_data/S-133-06923.json
    input_file = os.path.join(project_root, "iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data", f"{doc_stem}.json")
    output_file = os.path.join(project_root, "results", f"{doc_stem}_algo_organized.json")

    print(f"--- Processing {doc_stem} ---")
    run_algorithmic_organization(input_file, output_file)