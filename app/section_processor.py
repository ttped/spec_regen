import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any
from json_repair import repair_json


def get_line_bbox(page_dict: Dict, word_indices: List[int]) -> Optional[Dict]:
    """
    Calculate the bounding box for a line given the indices of its words.
    Returns the raw OCR bbox without any modifications.
    """
    if not word_indices:
        return None
    
    lefts = page_dict.get('left', [])
    tops = page_dict.get('top', [])
    widths = page_dict.get('width', [])
    heights = page_dict.get('height', [])
    
    # Check if we have the required data
    if not all([lefts, tops, widths, heights]):
        return None
    
    try:
        line_lefts = [lefts[i] for i in word_indices if i < len(lefts)]
        line_tops = [tops[i] for i in word_indices if i < len(tops)]
        line_rights = [lefts[i] + widths[i] for i in word_indices if i < len(lefts) and i < len(widths)]
        line_bottoms = [tops[i] + heights[i] for i in word_indices if i < len(tops) and i < len(heights)]
        
        if not all([line_lefts, line_tops, line_rights, line_bottoms]):
            return None
        
        left = min(line_lefts)
        top = min(line_tops)
        right = max(line_rights)
        bottom = max(line_bottoms)
        
        return {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top,
            "right": right,
            "bottom": bottom
        }
    except (IndexError, TypeError, ValueError):
        return None


def reconstruct_lines_with_bbox(page_dict: Dict[str, List]) -> List[Dict]:
    """
    Reconstructs lines from word-level OCR data, including bounding box info.
    """
    if not page_dict.get('text'):
        return []
    
    # If structure keys are missing, fallback to simple join
    if 'line_num' not in page_dict or 'block_num' not in page_dict:
        all_text = " ".join(str(t) for t in page_dict.get('text', []))
        if all_text.strip():
            # Try to get overall bbox from all words
            all_indices = list(range(len(page_dict.get('text', []))))
            bbox = get_line_bbox(page_dict, all_indices)
            return [{"text": all_text, "bbox": bbox, "word_indices": all_indices}]
        return []

    lines = []
    current_line_words = []
    current_line_indices = []
    
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
            # Save current line
            if current_line_words:
                line_text = " ".join(current_line_words)
                bbox = get_line_bbox(page_dict, current_line_indices)
                lines.append({
                    "text": line_text,
                    "bbox": bbox,
                    "word_indices": current_line_indices.copy()
                })
            # Start new line
            current_line_words = [word_text]
            current_line_indices = [i]
            last_block, last_par, last_line = block_num, par_num, line_num
        else:
            current_line_words.append(word_text)
            current_line_indices.append(i)
    
    # Don't forget the last line
    if current_line_words:
        line_text = " ".join(current_line_words)
        bbox = get_line_bbox(page_dict, current_line_indices)
        lines.append({
            "text": line_text,
            "bbox": bbox,
            "word_indices": current_line_indices.copy()
        })
        
    return lines


def split_topic_at_period(text: str) -> Tuple[str, str]:
    """
    Splits text at the first period that appears to end a title.
    """
    if not text:
        return "", ""
    
    period_match = re.search(r'\.(?=\s+[A-Z0-9]|$)', text)
    
    if period_match:
        split_pos = period_match.end()
        topic = text[:split_pos].strip()
        remainder = text[split_pos:].strip()
        return topic, remainder
    
    return text.strip(), ""



def normalize_section_number(raw: str) -> str:
    """
    Normalize a section number by fixing common OCR separator errors.
    
    OCR commonly confuses separators:
    - '1,1.3' should be '1.1.3'
    - '1-1-3' should be '1.1.3'
    - '3,2,1' should be '3.2.1'
    
    Returns the normalized section number string.
    """
    if not raw:
        return raw
    
    normalized = raw
    
    # Replace comma with period
    normalized = normalized.replace(',', '.')
    
    # Replace hyphen with period, but only when between digits
    # This avoids breaking things like "A-1" or "Phase-2"
    # Use a loop to handle consecutive replacements like "3-2-1"
    while True:
        new_normalized = re.sub(r'(\d)-(\d)', r'\1.\2', normalized)
        if new_normalized == normalized:
            break
        normalized = new_normalized
    
    # Clean up any double periods that might result
    while '..' in normalized:
        normalized = normalized.replace('..', '.')
    
    return normalized

def check_if_paragraph_is_header(line_text: str, debug: bool = False) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Checks if a line of text is a section header using regex and heuristics.
    
    Args:
        line_text: The text line to check
        debug: If True, print why a line was rejected
    
    Returns:
        Tuple of (is_header, section_number, topic, remainder)
    """
    # Primary regex: number followed by space and text
    match = re.match(r'^\s*([a-zA-Z0-9\.\,\-]+)\s+(.+)', line_text)
    # Secondary regex: just a number alone on the line
    match_no_title = re.match(r'^\s*([a-zA-Z0-9\.\,\-]+)\s*$', line_text)
    
    if match:
        potential_num, full_topic = match.groups()
    elif match_no_title:
        potential_num = match_no_title.group(1)
        full_topic = ""
    else:
        if debug:
            print(f"      [DEBUG] Rejected (no regex match): '{line_text[:50]}...'")
        return False, None, None, None

    # Normalize the section number (fix OCR separator errors like 1,1.3 -> 1.1.3)
    original_num = potential_num
    potential_num = normalize_section_number(potential_num)
    
    if debug and original_num != potential_num:
        print(f"      [DEBUG] Normalized section number: '{original_num}' -> '{potential_num}'")

    # Heuristics to filter out false positives
    
    # 1. Reject if topic starts with a month (likely a date)
    MONTHS = [
        'january', 'february', 'march', 'april', 'may', 'june', 
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    if full_topic and any(full_topic.lower().startswith(m) for m in MONTHS):
        if debug:
            print(f"      [DEBUG] Rejected (month name): '{potential_num}' '{full_topic[:30]}'")
        return False, None, None, None

    # 2. Must contain at least one digit
    if not any(c.isdigit() for c in potential_num):
        if debug:
            print(f"      [DEBUG] Rejected (no digits): '{potential_num}'")
        return False, None, None, None

    # 3. Too many alpha characters (likely a word, not a section number)
    if sum(c.isalpha() for c in potential_num) > 2:
        if debug:
            print(f"      [DEBUG] Rejected (too many alpha chars): '{potential_num}'")
        return False, None, None, None
        
    # 4. Too long to be a section number
    if len(potential_num) > 20:
        if debug:
            print(f"      [DEBUG] Rejected (too long): '{potential_num}'")
        return False, None, None, None

    # 5. Pure alpha (no digits) - already caught by rule 2, but explicit
    if potential_num.isalpha():
        if debug:
            print(f"      [DEBUG] Rejected (pure alpha): '{potential_num}'")
        return False, None, None, None

    # 6. Pure digits but too many (like a year or ID number)
    if potential_num.isdigit() and len(potential_num) > 3:
        if debug:
            print(f"      [DEBUG] Rejected (pure digits > 3 chars): '{potential_num}'")
        return False, None, None, None

    # 7. Mixed alpha+digit without dots (like "A1" or "B2" - not section numbers)
    has_alpha = any(c.isalpha() for c in potential_num)
    has_digit = any(c.isdigit() for c in potential_num)
    has_dot = '.' in potential_num
    if has_alpha and has_digit and not has_dot:
        if debug:
            print(f"      [DEBUG] Rejected (mixed alpha+digit, no dot): '{potential_num}'")
        return False, None, None, None

    section_num = potential_num.strip().rstrip('.')
    topic, remainder = split_topic_at_period(full_topic)
    
    if debug:
        print(f"      [DEBUG] ACCEPTED: section='{section_num}' topic='{topic[:30] if topic else ''}...'")
    
    return True, section_num, topic.strip(), remainder.strip()


def group_elements_with_bbox(elements: List[Dict]) -> List[Dict]:
    """
    Merges consecutive content blocks and attaches them to preceding section headers.
    
    IMPORTANT: Preserves the ORIGINAL bbox of the section header line.
    Does NOT merge bboxes from content blocks into the section bbox.
    This ensures accurate positioning based on where the section header actually appears.
    """
    if not elements:
        return []

    merged_elements = []
    i = 0
    while i < len(elements):
        current_element = elements[i]

        if current_element['type'] == 'section':
            content_pieces = []
            
            # PRESERVE the original section header bbox - do NOT merge with content
            # The bbox should represent where the section HEADER is, not the entire section content
            original_bbox = current_element.get('bbox')
            
            j = i + 1
            while j < len(elements) and elements[j]['type'] == 'unassigned_text_block':
                content_pieces.append(elements[j]['content'])
                j += 1
            
            current_element['content'] = "\n\n".join(content_pieces)
            
            # Keep the original bbox unchanged
            current_element['bbox'] = original_bbox
            
            merged_elements.append(current_element)
            i = j
        else:
            merged_elements.append(current_element)
            i += 1
            
    return merged_elements


def get_page_id_from_object(page_obj: Dict, fallback_key: str = "0") -> int:
    """
    Extract page ID from a page object.
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


def extract_page_metadata(page_dict: Dict) -> Dict:
    """
    Extract useful metadata from a page_dict for debugging.
    """
    metadata = {}
    
    if page_dict.get('left') and page_dict.get('width'):
        try:
            rights = [l + w for l, w in zip(page_dict['left'], page_dict['width'])]
            metadata['inferred_page_width'] = max(rights) if rights else None
        except:
            pass
    
    if page_dict.get('top') and page_dict.get('height'):
        try:
            bottoms = [t + h for t, h in zip(page_dict['top'], page_dict['height'])]
            metadata['inferred_page_height'] = max(bottoms) if bottoms else None
        except:
            pass
    
    for key in ['dpi', 'resolution', 'scale', 'ppi']:
        if key in page_dict:
            metadata[key] = page_dict[key]
    
    return metadata


def load_raw_ocr_pages(input_path: str) -> List[Tuple[int, Dict, Dict]]:
    """
    Loads raw OCR data and returns a sorted list of (page_id, page_dict, page_metadata) tuples.
    Handles malformed JSON files gracefully using json_repair.
    """
    if not os.path.exists(input_path):
        print(f"  - [Error] File not found: {input_path}")
        return []

    data = None
    
    # 1. Try standard load (Fastest)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # 2. Try Repair (Slower but robust)
        if repair_json:
            print(f"  - [Notice] JSON malformed in {os.path.basename(input_path)}. Attempting repair...")
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    # repair_json returns a string of corrected JSON, so we parse it again
                    repaired_content = repair_json(file_content)
                    data = json.loads(repaired_content)
                print(f"  - [Success] File repaired successfully.")
            except Exception as repair_error:
                print(f"  - [Error] Repair failed: {repair_error}")
        else:
            print(f"  - [Error] JSON malformed and 'json_repair' library not found.")
            print(f"  - [Action] Run `pip install json_repair` to enable auto-fixing.")

    if data is None:
        return []

    pages_to_process = []

    # (The rest of your existing logic remains exactly the same)
    if isinstance(data, dict):
        for key, val in data.items():
            page_dict = get_page_dict_from_object(val)
            if page_dict is None:
                continue
            page_id = get_page_id_from_object(val, fallback_key=key)
            page_meta = extract_page_metadata(page_dict)
            pages_to_process.append((page_id, page_dict, page_meta))
            
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
            page_meta = extract_page_metadata(page_dict)
            pages_to_process.append((page_id, page_dict, page_meta))

    pages_to_process.sort(key=lambda x: x[0])
    return pages_to_process


def run_section_processing_on_file(
    input_path: str, 
    output_path: str, 
    content_start_page: int = 1,
    header_top_threshold: int = 0,
    footer_top_threshold: int = 0
):
    """
    Main execution function - processes raw OCR file into organized sections with bbox metadata.
    
    Args:
        input_path: Path to the raw OCR JSON file.
        output_path: Path to save the organized sections JSON.
        content_start_page: The page number where actual content begins (skips ToC).
        header_top_threshold: Filter out lines where bbox['top'] < this value (0 to disable).
        footer_top_threshold: Filter out lines where bbox['top'] > this value (0 to disable).
    """
    print(f"  - Loading raw OCR from: {input_path}")
    print(f"  - Content starts at page: {content_start_page}")
    if header_top_threshold > 0:
        print(f"  - Filtering headers: Dropping text with Top position < {header_top_threshold}")
    if footer_top_threshold > 0:
        print(f"  - Filtering footers: Dropping text with Top position > {footer_top_threshold}")
    
    pages_to_process = load_raw_ocr_pages(input_path)
    
    if not pages_to_process:
        print("  - [Warning] No valid pages found (or file was corrupt).")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

    # How does the discern whether it is a content page?
    content_pages = [(pid, pdict, pmeta) for pid, pdict, pmeta in pages_to_process if pid >= content_start_page]
    
    skipped_count = len(pages_to_process) - len(content_pages)
    print(f"  - Found {len(pages_to_process)} total pages, skipping {skipped_count} (ToC/Title), processing {len(content_pages)}.")
    
    if not content_pages:
        print("  - [Warning] No content pages after filtering.")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

    all_page_metadata = {}
    raw_elements = []
    dropped_header_lines = 0
    dropped_footer_lines = 0
    
    # These pages signal whether it is a content page?
    for page_id, page_dict, page_meta in content_pages:
        if page_meta:
            all_page_metadata[page_id] = page_meta
        
        # I'm assuming this attempt to convert the document in to something that is read from top to bottom?
        lines = reconstruct_lines_with_bbox(page_dict)
        
        # Header/foot logic seems correct
        for line_data in lines:
            line_text = line_data['text'].strip()
            if not line_text:
                continue
            
            line_bbox = line_data.get('bbox')
            
            # --- HEADER FILTER ---
            if header_top_threshold > 0 and line_bbox:
                if line_bbox.get('top', 9999) < header_top_threshold:
                    dropped_header_lines += 1
                    continue
            # ---------------------
            
            # --- FOOTER FILTER ---
            if footer_top_threshold > 0 and line_bbox:
                if line_bbox.get('top', 0) > footer_top_threshold:
                    dropped_footer_lines += 1
                    continue
            # ---------------------

            # Logic for determing if section like 1.1.1?
            is_header, section_num, topic, remainder = check_if_paragraph_is_header(line_text)

            if is_header:
                raw_elements.append({
                    "type": "section",
                    "section_number": section_num,
                    "topic": topic,
                    "content": "",
                    "page_number": page_id,
                    "bbox": line_bbox
                })
                
                if remainder: # What is remainder? Extra text?
                    raw_elements.append({
                        "type": "unassigned_text_block",
                        "content": remainder,
                        "page_number": page_id,
                        "bbox": line_bbox
                    })
            else:
                # Seems correct to preserve all text
                raw_elements.append({
                    "type": "unassigned_text_block",
                    "content": line_text,
                    "page_number": page_id,
                    "bbox": line_bbox
                })
    
    # What is the purpose of this?
    final_elements = group_elements_with_bbox(raw_elements)

    output_data = {
        "page_metadata": all_page_metadata,
        "elements": final_elements
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    section_count = sum(1 for e in final_elements if e['type'] == 'section')
    print(f"  - Extracted {len(final_elements)} elements ({section_count} sections).")
    if header_top_threshold > 0:
        print(f"  - Filtered out {dropped_header_lines} header lines (Top < {header_top_threshold}).")
    if footer_top_threshold > 0:
        print(f"  - Filtered out {dropped_footer_lines} footer lines (Top > {footer_top_threshold}).")
    print(f"  - Results saved to {output_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output_organized.json"
        start_page = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        header_thresh = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        footer_thresh = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        run_section_processing_on_file(
            input_file, 
            output_file, 
            content_start_page=start_page, 
            header_top_threshold=header_thresh,
            footer_top_threshold=footer_thresh
        )
    else:
        print("Usage: python section_processor.py <input.json> [output.json] [content_start_page] [header_threshold] [footer_threshold]")