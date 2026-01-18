"""
section_processor.py - Processes raw OCR into organized sections with bbox metadata.

This module:
1. Loads raw OCR data preserving original page numbers
2. Reconstructs lines from word-level data
3. Captures bounding box metadata (top, left, width, height) for positioning
4. Detects section headers using regex heuristics
5. Groups content under sections

The bbox metadata enables proper positioning of elements relative to figures/tables.
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any


def get_line_bbox(page_dict: Dict, word_indices: List[int]) -> Optional[Dict]:
    """
    Calculate the bounding box for a line given the indices of its words.
    
    The bbox is computed as:
    - left: minimum left value of all words
    - top: minimum top value of all words  
    - right: maximum (left + width) of all words
    - bottom: maximum (top + height) of all words
    - width: right - left
    - height: bottom - top
    
    Args:
        page_dict: The page dictionary containing 'left', 'top', 'width', 'height' arrays
        word_indices: List of indices for words in this line
    
    Returns:
        Dict with bbox info, or None if data is missing
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
    
    Returns a list of dicts, each containing:
    - text: the line text
    - bbox: bounding box dict (left, top, width, height, right, bottom)
    - word_indices: indices of words in this line (for debugging)
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
    
    Returns (topic, remainder) where:
    - topic: everything up to and including the first sentence-ending period
    - remainder: everything after (to be prepended to content)
    
    Examples:
    - "SCOPE. This document..." -> ("SCOPE.", "This document...")
    - "SCOPE" -> ("SCOPE", "")
    - "GENERAL REQUIREMENTS. 1.1 Purpose. Text" -> ("GENERAL REQUIREMENTS.", "1.1 Purpose. Text")
    """
    if not text:
        return "", ""
    
    # Look for a period followed by a space (sentence end) or end of string
    # But avoid splitting on periods in abbreviations like "U.S." or numbers like "1.0"
    
    # Find the first period that looks like a sentence end
    # A sentence-ending period is typically followed by a space and uppercase letter, 
    # or followed by a space and a number (like "1.1"), or is at end of string
    
    period_match = re.search(r'\.(?=\s+[A-Z0-9]|$)', text)
    
    if period_match:
        split_pos = period_match.end()
        topic = text[:split_pos].strip()
        remainder = text[split_pos:].strip()
        return topic, remainder
    
    # No sentence-ending period found, return whole text as topic
    return text.strip(), ""


def check_if_paragraph_is_header(line_text: str) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Checks if a line of text is a section header using regex and heuristics.
    
    Returns:
        (is_header, section_number, topic, remainder)
        - is_header: True if this is a section header
        - section_number: The section number (e.g., "1.0", "2.1.3")
        - topic: The section title up to and including first period
        - remainder: Text after the topic that should be prepended to content
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

    # Heuristic 1: Reject if the topic looks like a date.
    MONTHS = [
        'january', 'february', 'march', 'april', 'may', 'june', 
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    if full_topic and any(full_topic.lower().startswith(m) for m in MONTHS):
        return False, None, None, None

    # Heuristic 2: Must contain at least one digit.
    if not any(c.isdigit() for c in potential_num):
        return False, None, None, None

    # Heuristic 3: Limit alpha characters to avoid matching regular text.
    if sum(c.isalpha() for c in potential_num) > 2:
        return False, None, None, None
        
    # Heuristic 4: Avoid excessively long "numbers".
    if len(potential_num) > 20:
        return False, None, None, None

    # Heuristic 5: Ensure it's not purely alphabetic.
    if potential_num.isalpha():
        return False, None, None, None

    # Heuristic 6: If it's all digits, limit length to 3. Rejects "1506073".
    if potential_num.isdigit() and len(potential_num) > 3:
        return False, None, None, None

    # Heuristic 7: If it contains both letters and digits, it must contain a dot.
    has_alpha = any(c.isalpha() for c in potential_num)
    has_digit = any(c.isdigit() for c in potential_num)
    has_dot = '.' in potential_num
    if has_alpha and has_digit and not has_dot:
        return False, None, None, None

    section_num = potential_num.strip().rstrip('.')
    
    # Split the topic at the first period
    topic, remainder = split_topic_at_period(full_topic)
    
    return True, section_num, topic.strip(), remainder.strip()


def merge_bboxes(bboxes: List[Dict]) -> Optional[Dict]:
    """
    Merge multiple bounding boxes into one encompassing bbox.
    """
    valid_bboxes = [b for b in bboxes if b is not None]
    if not valid_bboxes:
        return None
    
    left = min(b['left'] for b in valid_bboxes)
    top = min(b['top'] for b in valid_bboxes)
    right = max(b['right'] for b in valid_bboxes)
    bottom = max(b['bottom'] for b in valid_bboxes)
    
    return {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
        "right": right,
        "bottom": bottom
    }


def group_elements_with_bbox(elements: List[Dict]) -> List[Dict]:
    """
    Merges consecutive content blocks and attaches them to preceding section headers.
    Also merges bounding boxes to create an overall bbox for each section.
    """
    if not elements:
        return []

    merged_elements = []
    i = 0
    while i < len(elements):
        current_element = elements[i]

        if current_element['type'] == 'section':
            content_pieces = []
            content_bboxes = [current_element.get('bbox')]  # Start with section header bbox
            
            j = i + 1
            while j < len(elements) and elements[j]['type'] == 'unassigned_text_block':
                content_pieces.append(elements[j]['content'])
                if elements[j].get('bbox'):
                    content_bboxes.append(elements[j]['bbox'])
                j += 1
            
            current_element['content'] = "\n\n".join(content_pieces)
            
            # Merge all bboxes for this section
            merged_bbox = merge_bboxes(content_bboxes)
            if merged_bbox:
                current_element['bbox'] = merged_bbox
            
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
    
    Returns info about the page dimensions and OCR settings if available.
    """
    metadata = {}
    
    # Try to infer page dimensions from the max values
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
    
    # Check for any DPI or resolution info that might be in the data
    for key in ['dpi', 'resolution', 'scale', 'ppi']:
        if key in page_dict:
            metadata[key] = page_dict[key]
    
    return metadata


def load_raw_ocr_pages(input_path: str) -> List[Tuple[int, Dict, Dict]]:
    """
    Loads raw OCR data and returns a sorted list of (page_id, page_dict, page_metadata) tuples.
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
    content_start_page: int = 1
):
    """
    Main execution function - processes raw OCR file into organized sections with bbox metadata.
    
    Args:
        input_path: Path to the raw OCR JSON file.
        output_path: Path to save the organized sections JSON.
        content_start_page: The page number where actual content begins (skips ToC).
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
    content_pages = [(pid, pdict, pmeta) for pid, pdict, pmeta in pages_to_process if pid >= content_start_page]
    
    skipped_count = len(pages_to_process) - len(content_pages)
    print(f"  - Found {len(pages_to_process)} total pages, skipping {skipped_count} (ToC/Title), processing {len(content_pages)}.")
    
    if not content_pages:
        print("  - [Warning] No content pages after filtering.")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([], f)
        return

    # Collect page metadata for the output (useful for debugging)
    all_page_metadata = {}
    
    raw_elements = []
    
    for page_id, page_dict, page_meta in content_pages:
        # Store page metadata
        if page_meta:
            all_page_metadata[page_id] = page_meta
        
        # Reconstruct lines with bbox info
        lines = reconstruct_lines_with_bbox(page_dict)
        
        for line_data in lines:
            line_text = line_data['text'].strip()
            if not line_text:
                continue
            
            line_bbox = line_data.get('bbox')

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
                
                # If there's remainder text after the title, add it as an unassigned block
                # It will get merged into the section's content during grouping
                if remainder:
                    raw_elements.append({
                        "type": "unassigned_text_block",
                        "content": remainder,
                        "page_number": page_id,
                        "bbox": line_bbox  # Same bbox since it's from the same line
                    })
            else:
                raw_elements.append({
                    "type": "unassigned_text_block",
                    "content": line_text,
                    "page_number": page_id,
                    "bbox": line_bbox
                })
    
    # Group elements and merge bboxes
    final_elements = group_elements_with_bbox(raw_elements)

    # Add page metadata to output for debugging
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
    print(f"  - Page metadata captured for {len(all_page_metadata)} pages.")
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