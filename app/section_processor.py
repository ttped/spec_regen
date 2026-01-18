import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any

def reconstruct_lines_from_page_dict(page_dict: Dict[str, List]) -> List[str]:
    """
    Reconstructs a list of line strings from the word-level OCR data.
    A new line is identified by a change in 'block_num', 'par_num', or 'line_num'.
    """
    # Safety check: if text is missing, return empty
    if not page_dict.get('text'):
        return []
    
    # Safety check: if structure keys are missing, fallback to joining all text
    if 'line_num' not in page_dict or 'block_num' not in page_dict:
        all_text = " ".join(str(t) for t in page_dict.get('text', []))
        return [all_text] if all_text.strip() else []

    lines = []
    current_line_words = []
    
    try:
        last_block = page_dict['block_num'][0]
        last_par = page_dict.get('par_num', [0])[0]
        last_line = page_dict['line_num'][0]
    except (IndexError, KeyError):
        return []

    text_len = len(page_dict['text'])
    
    for i in range(text_len):
        if i >= len(page_dict['block_num']) or i >= len(page_dict['line_num']):
            break

        block_num = page_dict['block_num'][i]
        par_num = page_dict.get('par_num', [0] * text_len)[i] if i < len(page_dict.get('par_num', [])) else 0
        line_num = page_dict['line_num'][i]
        word_text = str(page_dict['text'][i])

        if block_num != last_block or par_num != last_par or line_num != last_line:
            if current_line_words:
                lines.append(" ".join(current_line_words))
            current_line_words = [word_text]
            last_block, last_par, last_line = block_num, par_num, line_num
        else:
            current_line_words.append(word_text)
    
    if current_line_words:
        lines.append(" ".join(current_line_words))
        
    return lines


def check_if_paragraph_is_header(line_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Checks if a line of text is a section header using regex and heuristics.
    """
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
    Extract page ID from a page object.
    Tries 'page_Id', then 'page_num', then falls back to the provided key.
    """
    page_id = page_obj.get('page_Id') or page_obj.get('page_num')
    
    if page_id is not None:
        try:
            return int(page_id)
        except (ValueError, TypeError):
            pass
    
    try:
        return int(fallback_key)
    except (ValueError, TypeError):
        digits = re.findall(r'\d+', str(fallback_key))
        if digits:
            return int(digits[0])
        return 0


def get_page_dict_from_object(page_obj: Any) -> Optional[Dict]:
    """
    Extract the actual page_dict (containing 'text', 'block_num', etc.) from various structures.
    """
    if not isinstance(page_obj, dict):
        return None
    
    if 'text' in page_obj and isinstance(page_obj.get('text'), list):
        if 'page_dict' in page_obj and isinstance(page_obj['page_dict'], dict):
            return page_obj['page_dict']
        return page_obj
    
    if 'page_dict' in page_obj:
        inner = page_obj['page_dict']
        if isinstance(inner, dict) and 'text' in inner:
            return inner
    
    return None


def load_raw_ocr_pages(input_path: str) -> List[Tuple[int, Dict]]:
    """
    Loads raw OCR data and returns a sorted list of (page_id, page_dict) tuples.
    This preserves the original page numbers from the PDF.
    """
    if not os.path.exists(input_path):
        print(f"  - [Error] File not found: {input_path}")
        return []

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pages_to_process = []

    if isinstance(data, dict):
        for key, val in data.items():
            page_dict = get_page_dict_from_object(val)
            if page_dict is None:
                continue
            page_id = get_page_id_from_object(val, fallback_key=key)
            pages_to_process.append((page_id, page_dict))
            
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            page_dict = get_page_dict_from_object(item)
            if page_dict is None:
                if 'text' in item:
                    page_dict = item
                else:
                    continue
            page_id = get_page_id_from_object(item, fallback_key=str(idx + 1))
            pages_to_process.append((page_id, page_dict))

    pages_to_process.sort(key=lambda x: x[0])
    return pages_to_process


def run_section_processing_on_file(
    input_path: str, 
    output_path: str, 
    content_start_page: int = 1
):
    """
    Main execution function - processes raw OCR file into organized sections.
    
    Args:
        input_path: Path to the raw OCR JSON file.
        output_path: Path to save the organized sections JSON.
        content_start_page: The page number where actual content begins (skips ToC).
                           Pages before this are omitted. Original page numbers are preserved.
    """
    print(f"  - Loading raw OCR from: {input_path}")
    print(f"  - Content starts at page: {content_start_page}")
    
    pages_to_process = load_raw_ocr_pages(input_path)
    
    if not pages_to_process:
        print("  - [Warning] No valid pages found.")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

    # Filter to only pages >= content_start_page
    content_pages = [(pid, pdict) for pid, pdict in pages_to_process if pid >= content_start_page]
    
    skipped_count = len(pages_to_process) - len(content_pages)
    print(f"  - Found {len(pages_to_process)} total pages, skipping {skipped_count} (ToC/Title), processing {len(content_pages)}.")
    
    if not content_pages:
        print("  - [Warning] No content pages after filtering.")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

    raw_elements = []
    
    for page_id, page_dict in content_pages:
        lines = reconstruct_lines_from_page_dict(page_dict)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            is_header, section_num, topic = check_if_paragraph_is_header(line)

            if is_header:
                raw_elements.append({
                    "type": "section",
                    "section_number": section_num,
                    "topic": topic,
                    "content": "",
                    "page_number": page_id  # Original page number preserved
                })
            else:
                raw_elements.append({
                    "type": "unassigned_text_block",
                    "content": line,
                    "page_number": page_id
                })
    
    final_elements = group_elements(raw_elements)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_elements, f, indent=4)
    
    section_count = sum(1 for e in final_elements if e['type'] == 'section')
    print(f"  - Extracted {len(final_elements)} elements ({section_count} sections).")
    print(f"  - Results saved to {output_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output_organized.json"
        start_page = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        run_section_processing_on_file(input_file, output_file, content_start_page=start_page)
    else:
        print("Usage: python section_processor.py <input.json> [output.json] [content_start_page]")