import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any

def flatten_advanced_ocr_to_lines(page_data: Dict[str, Any]) -> List[str]:
    """
    Pre-processes 'Advanced' OCR data to create a list of text lines.
    It groups words based on block_num, par_num, and line_num.
    
    Args:
        page_data: The dictionary containing 'text', 'block_num', etc.
        
    Returns:
        A list of strings, where each string is a reconstructed line of text.
    """
    # 1. robustly fetch the lists
    texts = page_data.get('text', [])
    
    # If there is no text, return empty
    if not texts:
        return []

    # Get structural markers. Default to 0s if missing (handling non-advanced data)
    count = len(texts)
    blocks = page_data.get('block_num', [0] * count)
    pars = page_data.get('par_num', [0] * count)
    lines_ids = page_data.get('line_num', [0] * count)

    # 2. Iterate and group
    reconstructed_lines = []
    current_line_words = []
    
    # safe_get helper to avoid index errors if lists are different lengths
    def safe_get(lst, idx, default=0):
        return lst[idx] if idx < len(lst) else default

    last_b, last_p, last_l = -1, -1, -1

    for i, word in enumerate(texts):
        word_str = str(word).strip()
        if not word_str: 
            continue

        curr_b = safe_get(blocks, i)
        curr_p = safe_get(pars, i)
        curr_l = safe_get(lines_ids, i)

        # Check if we have moved to a new visual line
        is_new_line = (curr_b != last_b) or (curr_p != last_p) or (curr_l != last_l)

        if is_new_line and current_line_words:
            # Flush current line
            reconstructed_lines.append(" ".join(current_line_words))
            current_line_words = []

        current_line_words.append(word_str)
        last_b, last_p, last_l = curr_b, curr_p, curr_l

    # Flush the final line
    if current_line_words:
        reconstructed_lines.append(" ".join(current_line_words))
        
    return reconstructed_lines

def check_if_line_is_header(line_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Regex and heuristic checks to determine if a line is a section header.
    Matches formats like: "1.0 Scope", "3.2.1 General", "A. Appendix".
    """
    # Regex for "1.2.3 Title" or just "1.2.3"
    match = re.match(r'^\s*([a-zA-Z0-9\.]+)\s+(.+)', line_text)
    match_no_title = re.match(r'^\s*([a-zA-Z0-9\.]+)\s*$', line_text)
    
    if match:
        potential_num, topic = match.groups()
    elif match_no_title:
        potential_num = match_no_title.group(1)
        topic = ""
    else:
        return False, None, None

    # --- Strict Heuristics to reduce false positives ---
    
    # 1. Date filter (Month names)
    MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    if topic and any(topic.lower().startswith(m) for m in MONTHS):
        return False, None, None

    # 2. Must contain at least one digit
    if not any(c.isdigit() for c in potential_num):
        return False, None, None

    # 3. Limit alpha characters (avoids regular words like "The" being seen as headers)
    if sum(c.isalpha() for c in potential_num) > 2:
        return False, None, None
        
    # 4. Length and Format checks
    if len(potential_num) > 20: return False, None, None
    if potential_num.isalpha(): return False, None, None

    # 5. Pure digits limit (e.g. "1995" is likely a year/quantity, not a section)
    if potential_num.isdigit() and len(potential_num) > 3:
        return False, None, None

    # 6. Mixed Alpha-Digit must have dot (reject "616A", accept "3.2.A")
    has_alpha = any(c.isalpha() for c in potential_num)
    has_digit = any(c.isdigit() for c in potential_num)
    has_dot = '.' in potential_num
    if has_alpha and has_digit and not has_dot:
        return False, None, None

    return True, potential_num.strip().rstrip('.'), topic.strip()

def group_elements(elements: List[Dict]) -> List[Dict]:
    """
    Groups 'unassigned_text_block' items into the preceding 'section' item.
    """
    if not elements:
        return []

    merged = []
    i = 0
    while i < len(elements):
        curr = elements[i]
        
        if curr['type'] == 'section':
            content_parts = []
            j = i + 1
            # Absorb following unassigned blocks
            while j < len(elements) and elements[j]['type'] == 'unassigned_text_block':
                content_parts.append(elements[j]['content'])
                j += 1
            
            curr['content'] = "\n\n".join(content_parts)
            merged.append(curr)
            i = j
        else:
            # Keep unassigned blocks that appear before the first section
            merged.append(curr)
            i += 1
    return merged

def load_pages_robustly(filepath: str) -> List[Dict]:
    """
    Loads JSON and normalizes it into a list of {'page_number': int, 'data': dict} objects.
    """
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sorted_pages = []

    # Handle List vs Dict structure
    if isinstance(data, list):
        for idx, page_data in enumerate(data):
            p_num = page_data.get('page_num') or page_data.get('page_Id') or (idx + 1)
            try: 
                p_num = int(p_num)
            except: 
                p_num = idx + 1
            
            # Key step: ensure we get the inner dictionary if it exists
            content = page_data.get('page_dict', page_data)
            sorted_pages.append({'page_number': p_num, 'data': content})

    elif isinstance(data, dict):
        for key, val in data.items():
            try:
                p_num = int(key)
            except ValueError:
                digits = re.findall(r'\d+', str(key))
                p_num = int(digits[0]) if digits else 0
            
            content = val.get('page_dict', val)
            sorted_pages.append({'page_number': p_num, 'data': content})

    sorted_pages.sort(key=lambda x: x['page_number'])
    return sorted_pages

def run_section_processing_on_file(input_path: str, output_path: str):
    """
    Main entry point for the pipeline.
    """
    print(f"  - Loading OCR data from: {input_path}")
    pages = load_pages_robustly(input_path)
    
    if not pages:
        print("  - [Warning] No pages loaded. Check input file format.")
        # Create empty file to ensure pipeline continues safely
        with open(output_path, 'w') as f: json.dump([], f)
        return

    raw_elements = []
    total_lines = 0

    for p in pages:
        p_num = p['page_number']
        p_data = p['data']
        
        # --- THE PREPROCESSING STEP ---
        # Convert advanced OCR structure into simple lines of text
        lines = flatten_advanced_ocr_to_lines(p_data)
        total_lines += len(lines)
        
        for line in lines:
            is_header, sec_num, topic = check_if_line_is_header(line)
            
            if is_header:
                raw_elements.append({
                    "type": "section",
                    "section_number": sec_num,
                    "topic": topic,
                    "content": "", # Populated in grouping step
                    "page_number": p_num
                })
            else:
                raw_elements.append({
                    "type": "unassigned_text_block",
                    "content": line,
                    "page_number": p_num
                })

    print(f"  - Pre-processed {len(pages)} pages into {total_lines} lines of text.")
    
    # Group text under sections
    final_elements = group_elements(raw_elements)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_elements, f, indent=4)
        
    print(f"  - Extraction complete. Found {len(final_elements)} structured elements.")
    print(f"  - Saved to: {output_path}")