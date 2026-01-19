import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any


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


def check_if_paragraph_is_header(line_text: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Checks if a line of text is a section header using regex and heuristics.
    """
    match = re.match(r'^\s*([a-zA-Z0-9\.]+)\s+(.+)', line_text)
    match_no_title = re.match(r'^\s*([a-zA-Z0-9\.]+)\s*$', line_text)
    
    if match:
        potential_num, full_topic = match.groups()
    elif match_no_title:
        potential_num = match_no_title.group(1)
        full_topic = ""
    else:
        return False, None, None, None

    # Heuristics
    MONTHS = [
        'january', 'february', 'march', 'april', 'may', 'june', 
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    if full_topic and any(full_topic.lower().startswith(m) for m in MONTHS):
        return False, None, None, None

    if not any(c.isdigit() for c in potential_num):
        return False, None, None, None

    if sum(c.isalpha() for c in potential_num) > 2:
        return False, None, None, None
        
    if len(potential_num) > 20:
        return False, None, None, None

    if potential_num.isalpha():
        return False, None, None, None

    if potential_num.isdigit() and len(potential_num) > 3:
        return False, None, None, None

    has_alpha = any(c.isalpha() for c in potential_num)
    has_digit = any(c.isdigit() for c in potential_num)
    has_dot = '.' in potential_num
    if has_alpha and has_digit and not has_dot:
        return False, None, None, None

    section_num = potential_num.strip().rstrip('.')
    topic, remainder = split_topic_at_period(full_topic)
    
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
    Handles malformed JSON files gracefully.
    """
    if not os.path.exists(input_path):
        print(f"  - [Error] File not found: {input_path}")
        return []

    # Defensive loading for malformed JSON
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  - [Error] Failed to load JSON from {input_path}: {e}")
        return []

    pages_to_process = []

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
    header_top_threshold: int = 0
):
    """
    Main execution function - processes raw OCR file into organized sections with bbox metadata.
    
    Args:
        input_path: Path to the raw OCR JSON file.
        output_path: Path to save the organized sections JSON.
        content_start_page: The page number where actual content begins (skips ToC).
        header_top_threshold: Filter out lines where bbox['top'] < this value (0 to disable).
    """
    print(f"  - Loading raw OCR from: {input_path}")
    print(f"  - Content starts at page: {content_start_page}")
    if header_top_threshold > 0:
        print(f"  - Filtering headers: Dropping text with Top position < {header_top_threshold}")
    
    pages_to_process = load_raw_ocr_pages(input_path)
    
    if not pages_to_process:
        print("  - [Warning] No valid pages found (or file was corrupt).")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

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
    
    for page_id, page_dict, page_meta in content_pages:
        if page_meta:
            all_page_metadata[page_id] = page_meta
        
        lines = reconstruct_lines_with_bbox(page_dict)
        
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
                
                if remainder:
                    raw_elements.append({
                        "type": "unassigned_text_block",
                        "content": remainder,
                        "page_number": page_id,
                        "bbox": line_bbox
                    })
            else:
                raw_elements.append({
                    "type": "unassigned_text_block",
                    "content": line_text,
                    "page_number": page_id,
                    "bbox": line_bbox
                })
    
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
    print(f"  - Results saved to {output_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output_organized.json"
        start_page = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        threshold = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        run_section_processing_on_file(input_file, output_file, content_start_page=start_page, header_top_threshold=threshold)
    else:
        print("Usage: python section_processor.py <input.json> [output.json] [content_start_page] [header_threshold]")