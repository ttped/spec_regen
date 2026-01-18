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
    if 'line_num' not in page_dict or 'block_num' not in page_dict:
        # Fallback: just return all text joined as a single "line"
        # This ensures we don't silently skip pages
        all_text = " ".join(str(t) for t in page_dict.get('text', []))
        return [all_text] if all_text.strip() else []

    lines = []
    current_line_words = []
    
    # Use safe access or try/except to handle potential length mismatches in OCR data
    try:
        last_block = page_dict['block_num'][0]
        last_par = page_dict.get('par_num', [0])[0]  # par_num might be missing
        last_line = page_dict['line_num'][0]
    except (IndexError, KeyError):
        return []

    text_len = len(page_dict['text'])
    
    for i in range(text_len):
        # Guard against index errors if lists are uneven
        if i >= len(page_dict['block_num']) or i >= len(page_dict['line_num']):
            break

        block_num = page_dict['block_num'][i]
        par_num = page_dict.get('par_num', [0] * text_len)[i] if i < len(page_dict.get('par_num', [])) else 0
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


def get_page_id_from_object(page_obj: Dict, fallback_key: str = "0") -> int:
    """
    Extract page ID from a page object, matching the original algorithm's approach.
    Tries 'page_Id', then 'page_num', then falls back to the provided key.
    Returns an integer for proper sorting.
    """
    # Try internal fields first (like the original does)
    page_id = page_obj.get('page_Id') or page_obj.get('page_num')
    
    if page_id is not None:
        try:
            return int(page_id)
        except (ValueError, TypeError):
            pass
    
    # Fallback to the dictionary key
    try:
        return int(fallback_key)
    except (ValueError, TypeError):
        # Try to extract digits from the key (e.g., "page_12" -> 12)
        digits = re.findall(r'\d+', str(fallback_key))
        if digits:
            return int(digits[0])
        return 0


def get_page_dict_from_object(page_obj: Any) -> Optional[Dict]:
    """
    Extract the actual page_dict (containing 'text', 'block_num', etc.) from various structures.
    Handles:
    - Direct page_dict (has 'text' key directly)
    - Wrapped structure (has 'page_dict' key containing the data)
    """
    if not isinstance(page_obj, dict):
        return None
    
    # Check if this object directly contains word-level data
    if 'text' in page_obj and isinstance(page_obj.get('text'), list):
        # Check if it also has 'page_dict' - if so, prefer that
        if 'page_dict' in page_obj and isinstance(page_obj['page_dict'], dict):
            return page_obj['page_dict']
        # Otherwise, this IS the page_dict
        return page_obj
    
    # Check for wrapped structure
    if 'page_dict' in page_obj:
        inner = page_obj['page_dict']
        if isinstance(inner, dict) and 'text' in inner:
            return inner
    
    return None


def run_section_processing_on_file(input_path: str, output_path: str):
    """
    Main execution function - processes raw OCR file into organized sections.
    
    This version closely matches the original algorithm's data handling while
    being more robust to different input structures.
    """
    print(f"  - Loading raw OCR from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"  - [Error] File not found: {input_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build a list of (page_id, page_dict) tuples
    pages_to_process = []

    if isinstance(data, dict):
        # Dictionary structure: keys are typically page numbers ("1", "2", etc.)
        # Values may be the page_dict directly, or a wrapper containing page_dict
        for key, val in data.items():
            page_dict = get_page_dict_from_object(val)
            if page_dict is None:
                print(f"    [Warning] Could not extract page_dict for key '{key}', skipping.")
                continue
            
            # Get page ID - prefer internal fields, fall back to dict key
            page_id = get_page_id_from_object(val, fallback_key=key)
            pages_to_process.append((page_id, page_dict))
            
    elif isinstance(data, list):
        # List structure: each item is a page object
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
                
            page_dict = get_page_dict_from_object(item)
            if page_dict is None:
                # Fallback: if the item itself has 'text', use it directly
                if 'text' in item:
                    page_dict = item
                else:
                    print(f"    [Warning] Could not extract page_dict for index {idx}, skipping.")
                    continue
            
            page_id = get_page_id_from_object(item, fallback_key=str(idx + 1))
            pages_to_process.append((page_id, page_dict))
    else:
        print(f"  - [Error] Unexpected data type: {type(data)}")
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

    # Sort by page ID
    pages_to_process.sort(key=lambda x: x[0])
    
    print(f"  - Found {len(pages_to_process)} pages to process.")
    
    if not pages_to_process:
        print("  - [Warning] No valid pages found.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

    raw_elements = []
    
    for page_id, page_dict in pages_to_process:
        # Reconstruct lines from word-level OCR data
        lines = reconstruct_lines_from_page_dict(page_dict)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line is a section header
            is_header, section_num, topic = check_if_paragraph_is_header(line)

            if is_header:
                raw_elements.append({
                    "type": "section",
                    "section_number": section_num,
                    "topic": topic,
                    "content": "",
                    "page_number": page_id
                })
            else:
                raw_elements.append({
                    "type": "unassigned_text_block",
                    "content": line,
                    "page_number": page_id
                })
    
    # Group elements: merge unassigned blocks into preceding sections
    final_elements = group_elements(raw_elements)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_elements, f, indent=4)
    
    print(f"  - Extracted {len(final_elements)} elements ({sum(1 for e in final_elements if e['type'] == 'section')} sections).")
    print(f"  - Results saved to {output_path}")


if __name__ == '__main__':
    # Test with a sample file if run directly
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output_organized.json"
        run_section_processing_on_file(input_file, output_file)
    else:
        print("Usage: python section_processor_fixed.py <input.json> [output.json]")