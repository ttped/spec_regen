import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any

def reconstruct_lines_from_page_dict(page_dict: Dict[str, List]) -> List[str]:
    """
    Reconstructs a list of line strings from the word-level OCR data.
    A new line is identified by a change in 'block_num', 'par_num', or 'line_num'.
    (Logic restored strictly from section_detection_algo_orig.py)
    """
    # Safety check: if text is missing, return empty
    if not page_dict.get('text'):
        return []
    
    # Safety check: if structure keys are missing, we cannot use this logic.
    # The 'advanced' files usually have these. 
    if 'line_num' not in page_dict or 'block_num' not in page_dict:
        # Fallback for pages that might be malformed: just join all text
        # But per instruction, we stick to orig logic which assumes structure.
        return []

    lines = []
    current_line_words = []
    
    # Use safe access or try/except to handle potential length mismatches in OCR data
    try:
        last_block = page_dict['block_num'][0]
        last_par = page_dict['par_num'][0]
        last_line = page_dict['line_num'][0]
    except IndexError:
        return []

    text_len = len(page_dict['text'])
    
    for i in range(text_len):
        # Guard against index errors if lists are uneven
        if i >= len(page_dict['block_num']) or i >= len(page_dict['line_num']):
            break

        block_num = page_dict['block_num'][i]
        par_num = page_dict['par_num'][i]
        line_num = page_dict['line_num'][i]
        word_text = str(page_dict['text'][i])

        # If any of the block, paragraph, or line numbers change, we start a new line.
        if block_num != last_block or par_num != last_par or line_num != last_line:
            if current_line_words:
                lines.append(" ".join(current_line_words))
            current_line_words = [word_text]
            last_block, last_par, last_line = block_num, par_num, line_num
        else:
            current_line_words.append(word_text)
    
    # Append the last line being built
    if current_line_words:
        lines.append(" ".join(current_line_words))
        
    return lines

def check_if_paragraph_is_header(line_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Checks if a line of text is a section header using regex and heuristics.
    (Logic restored strictly from section_detection_algo_orig.py)
    """
    # Regex to find a potential section number at the start, followed by a topic.
    match = re.match(r'^\s*([a-zA-Z0-9\.]+)\s+(.+)', line_text)
    match_no_title = re.match(r'^\s*([a-zA-Z0-9\.]+)\s*$', line_text)
    
    if match:
        potential_num, topic = match.groups()
    elif match_no_title:
        potential_num = match_no_title.group(1)
        topic = ""
    else:
        return False, None, None

    # Heuristic 1: Reject if the topic looks like a date.
    MONTHS = [
        'january', 'february', 'march', 'april', 'may', 'june', 
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    if topic and any(topic.lower().startswith(m) for m in MONTHS):
        return False, None, None

    # Heuristic 2: Must contain at least one digit.
    if not any(c.isdigit() for c in potential_num):
        return False, None, None

    # Heuristic 3: Limit alpha characters to avoid matching regular text.
    if sum(c.isalpha() for c in potential_num) > 2:
        return False, None, None
        
    # Heuristic 4: Avoid excessively long "numbers".
    if len(potential_num) > 20:
        return False, None, None

    # Heuristic 5: Ensure it's not purely alphabetic.
    if potential_num.isalpha():
        return False, None, None

    # Heuristic 6: If it's all digits, limit length to 3. Rejects "1506073".
    if potential_num.isdigit() and len(potential_num) > 3:
        return False, None, None

    # Heuristic 7: If it contains both letters and digits, it must contain a dot.
    # Rejects "616A" but allows "3.23.A".
    has_alpha = any(c.isalpha() for c in potential_num)
    has_digit = any(c.isdigit() for c in potential_num)
    has_dot = '.' in potential_num
    if has_alpha and has_digit and not has_dot:
        return False, None, None

    section_num = potential_num.strip().rstrip('.')
    return True, section_num, topic.strip()

def group_elements(elements: List[Dict]) -> List[Dict]:
    """
    Merges consecutive content blocks and attaches them to preceding section headers.
    (Logic restored strictly from section_detection_algo_orig.py)
    """
    if not elements:
        return []

    merged_elements = []
    i = 0
    while i < len(elements):
        current_element = elements[i]

        if current_element['type'] == 'section':
            content_pieces = []
            j = i + 1
            while j < len(elements) and elements[j]['type'] == 'unassigned_text_block':
                content_pieces.append(elements[j]['content'])
                j += 1
            
            current_element['content'] = "\n\n".join(content_pieces)
            merged_elements.append(current_element)
            i = j
        else:
            merged_elements.append(current_element)
            i += 1
            
    return merged_elements

def load_pages_with_correct_ids(filepath: str) -> List[Dict]:
    """
    Loads pages and explicitly preserves the original page IDs (e.g. "12")
    instead of re-indexing them from 1.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pages_out = []

    # 1. Handle Dictionary Structure (Common in 'Advanced' OCR)
    if isinstance(data, dict):
        # We iterate items. keys might be "1", "2", or "page_1"
        for key, val in data.items():
            # The value 'val' usually contains 'page_dict' and 'page_Id'.
            # We trust 'page_Id' inside the dict first.
            
            # Check inside the value object
            inner_page_id = val.get('page_Id') or val.get('page_num')
            
            if inner_page_id:
                final_id = str(inner_page_id)
            else:
                # Fallback to the dictionary key
                final_id = str(key)

            # Ensure we get the actual word data
            page_content = val.get('page_dict', val)
            
            pages_out.append({
                'id': final_id,
                'data': page_content
            })

    # 2. Handle List Structure
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            # Trust internal ID first
            inner_page_id = item.get('page_Id') or item.get('page_num')
            
            if inner_page_id:
                final_id = str(inner_page_id)
            else:
                # If no ID in the object, use index + 1
                final_id = str(idx + 1)
            
            page_content = item.get('page_dict', item)
            
            pages_out.append({
                'id': final_id,
                'data': page_content
            })

    # Sort primarily by integer value of ID if possible, else string
    def sort_key(p):
        try:
            return int(p['id'])
        except ValueError:
            return 999999

    pages_out.sort(key=sort_key)
    return pages_out

def run_section_processing_on_file(input_path: str, output_path: str):
    """
    Main execution function using the restored 'orig' logic.
    """
    print(f"  - Loading raw OCR from: {input_path}")
    pages = load_pages_with_correct_ids(input_path)
    
    if not pages:
        print("  - [Warning] No pages found.")
        # Save empty list to keep pipeline alive
        with open(output_path, 'w') as f: json.dump([], f)
        return

    raw_elements = []
    
    for p in pages:
        page_id = p['id']
        page_dict = p['data']
        
        # 1. Reconstruct Lines (ORIG LOGIC)
        lines = reconstruct_lines_from_page_dict(page_dict)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 2. Check Header (ORIG LOGIC)
            is_header, section_num, topic = check_if_paragraph_is_header(line)

            if is_header:
                raw_elements.append({
                    "type": "section",
                    "section_number": section_num,
                    "topic": topic,
                    "content": "",
                    "page_number": page_id  # Using the preserved ID
                })
            else:
                raw_elements.append({
                    "type": "unassigned_text_block",
                    "content": line,
                    "page_number": page_id # Using the preserved ID
                })
        
    # 3. Group Elements (ORIG LOGIC)
    final_elements = group_elements(raw_elements)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_elements, f, indent=4)
    
    print(f"  - Extracted {len(final_elements)} elements.")
    print(f"  - Results saved to {output_path}")