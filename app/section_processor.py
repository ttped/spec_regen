import os
import json
import re
import statistics
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


def split_topic_at_separator(text: str) -> Tuple[str, str]:
    """
    Splits text at the first separator that appears to end a title.
    
    Handles common patterns where OCR doesn't have newlines after titles:
    - "Scope -- The something something"
    - "Scope - something something"
    - "Scope ~ Something"
    - "Scope = Something"
    - "Scope. Something" (period followed by space and capital)
    
    Returns:
        tuple: (title, remainder)
    """
    if not text:
        return "", ""
    
    # Pattern 1: Double dash separator (highest priority)
    # Matches: "Title -- rest" or "Title — rest" (em-dash)
    double_dash = re.search(r'\s+(?:--|—|––)\s+', text)
    if double_dash:
        title = text[:double_dash.start()].strip()
        remainder = text[double_dash.end():].strip()
        return title, remainder
    
    # Pattern 2: Single separator characters (with spaces around them)
    # Matches: "Title - rest", "Title ~ rest", "Title = rest"
    # Must have space before AND after to avoid matching hyphenated words
    single_sep = re.search(r'\s+[-~=]\s+', text)
    if single_sep:
        # Make sure we're not splitting too early (title should be at least a few chars)
        if single_sep.start() >= 3:
            title = text[:single_sep.start()].strip()
            remainder = text[single_sep.end():].strip()
            return title, remainder
    
    # Pattern 3: Period followed by space and capital letter (sentence start)
    # Matches: "Title. The rest of the sentence"
    # But NOT: "Dr. Smith" or "U.S. Army" (abbreviations)
    period_match = re.search(r'\.(?=\s+[A-Z][a-z])', text)
    if period_match:
        # Check it's not an abbreviation (single capital letter before period)
        before_period = text[:period_match.start()]
        # Skip if it looks like an abbreviation (ends with single letter or common abbrev)
        abbrev_pattern = re.search(r'(?:^|\s)(?:[A-Z]|Dr|Mr|Mrs|Ms|Jr|Sr|Inc|Ltd|etc|vs|Vol|No|Fig)$', before_period)
        if not abbrev_pattern:
            title = text[:period_match.end()].strip()
            remainder = text[period_match.end():].strip()
            return title, remainder
    
    # Pattern 4: Period at end of text or followed by end
    period_end = re.search(r'\.\s*$', text)
    if period_end:
        return text.strip(), ""
    
    # No separator found - return whole text as title
    return text.strip(), ""


def split_topic_at_period(text: str) -> Tuple[str, str]:
    """
    Splits text at the first period that appears to end a title.
    
    DEPRECATED: Use split_topic_at_separator instead.
    This function is kept for backward compatibility.
    """
    return split_topic_at_separator(text)



def normalize_section_number(raw: str) -> str:
    """
    Normalize a section number by fixing common OCR separator errors.
    """
    if not raw:
        return raw
    
    normalized = raw
    
    # Replace comma with period
    normalized = normalized.replace(',', '.')
    
    # Replace hyphen with period, but only when between digits
    while True:
        new_normalized = re.sub(r'(\d)-(\d)', r'\1.\2', normalized)
        if new_normalized == normalized:
            break
        normalized = new_normalized
    
    # Clean up any double periods that might result
    while '..' in normalized:
        normalized = normalized.replace('..', '.')
    
    return normalized

def clean_leading_junk(text: str) -> Tuple[str, str]:
    """
    Remove common OCR artifacts that appear before section numbers.
    
    Handles patterns like:
    - ". 1. Scope" -> "1. Scope"
    - "` 1 Scope" -> "1 Scope"
    - "' 1.2 Title" -> "1.2 Title"
    - "- 1 Scope" -> "1 Scope"
    - ": 1.1 Scope" -> "1.1 Scope"
    
    Returns:
        tuple: (cleaned_text, removed_junk)
    """
    if not text:
        return text, ""
    
    # Pattern: punctuation/symbol + optional space + digit
    # Common junk: . ` ' " - : ; | ! * # @ ^
    junk_match = re.match(r'^(\s*[.`\'":\-;|!*#@^,]+\s*)(?=\d)', text)
    if junk_match:
        junk = junk_match.group(1)
        cleaned = text[len(junk):]
        return cleaned, junk
    
    return text, ""


def check_if_paragraph_is_header(line_text: str, debug: bool = False) -> Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[Dict]]:
    """
    Checks if a line of text is a section header using regex.
    """
    context = {
        'raw_section_number': None,
        'had_leading_whitespace': False,
        'leading_whitespace_len': 0,
        'had_text_before_number': False,
        'text_before_number': '',
        'original_line': line_text,
        'rejection_reason': None,
        'line_length': len(line_text),
        'leading_junk_removed': '',
    }
    
    # Check for leading whitespace
    stripped = line_text.lstrip()
    leading_ws = len(line_text) - len(stripped)
    context['had_leading_whitespace'] = leading_ws > 0
    context['leading_whitespace_len'] = leading_ws
    
    # Clean leading junk (like ". 1.2" or "` 1")
    cleaned_text, junk_removed = clean_leading_junk(line_text)
    if junk_removed:
        context['leading_junk_removed'] = junk_removed.strip()
        if debug:
            print(f"      [DEBUG] Removed leading junk: '{junk_removed.strip()}' from '{line_text[:30]}...'")
    
    # ==========================================================================
    # GREEDY REGEX: Find section-number-like patterns ANYWHERE in the line
    # ==========================================================================
    
    # Use cleaned text for matching
    match_start = re.match(r'^(\s*)([0-9]+(?:[.\-,][0-9]+)*[.\-,]?[A-Za-z]?)\s+(.+)', cleaned_text)
    match_start_no_title = re.match(r'^(\s*)([0-9]+(?:[.\-,][0-9]+)*[.\-,]?[A-Za-z]?)\s*$', cleaned_text)
    match_mid = re.search(r'(?:^|[:\s])([0-9]+(?:\.[0-9]+)+)\s+([A-Z][A-Za-z].*?)(?:\.|$)', cleaned_text)
    
    potential_num = None
    full_topic = ""
    
    if match_start:
        ws, potential_num, full_topic = match_start.groups()
        context['had_text_before_number'] = False
    elif match_start_no_title:
        ws, potential_num = match_start_no_title.groups()
        full_topic = ""
        context['had_text_before_number'] = False
    elif match_mid:
        potential_num, full_topic = match_mid.groups()
        match_start_pos = match_mid.start(1)
        text_before = cleaned_text[:match_start_pos].strip()
        context['had_text_before_number'] = len(text_before) > 0
        context['text_before_number'] = text_before[:50]
    else:
        if debug:
            print(f"      [DEBUG] Rejected (no regex match): '{line_text[:50]}...'")
        context['rejection_reason'] = 'no_regex_match'
        return False, None, None, None, context
    
    context['raw_section_number'] = potential_num
    
    original_num = potential_num
    potential_num = normalize_section_number(potential_num)
    
    if debug and original_num != potential_num:
        print(f"      [DEBUG] Normalized section number: '{original_num}' -> '{potential_num}'")

    if not any(c.isdigit() for c in potential_num):
        if debug:
            print(f"      [DEBUG] Rejected (no digits): '{potential_num}'")
        context['rejection_reason'] = 'no_digits'
        return False, None, None, None, context

    if len(potential_num) > 30:
        if debug:
            print(f"      [DEBUG] Rejected (too long > 30): '{potential_num}'")
        context['rejection_reason'] = 'too_long'
        return False, None, None, None, context

    if potential_num.replace('.', '').replace('-', '').replace(',', '').isalpha():
        if debug:
            print(f"      [DEBUG] Rejected (pure alpha): '{potential_num}'")
        context['rejection_reason'] = 'pure_alpha'
        return False, None, None, None, context

    # ==========================================================================
    # LOGICAL VALIDATION (Added for strict filtering)
    # ==========================================================================
    
    section_num = potential_num.strip().rstrip('.')
    
    # 1. Reject "all zero" sections (0, 0.0, 0.0.0)
    #    Split by dot, check if all parts are 0
    parts = section_num.split('.')
    if all(p.isdigit() and int(p) == 0 for p in parts):
        if debug:
            print(f"      [DEBUG] Rejected (all zeros): '{section_num}'")
        context['rejection_reason'] = 'zero_section'
        return False, None, None, None, context

    # 2. Reject major sections > 10 (Safety cap)
    #    e.g., "12.1" or "52.2.1"
    if parts and parts[0].isdigit():
        major_section = int(parts[0])
        if major_section > 10:
            if debug:
                print(f"      [DEBUG] Rejected (major section > 10): '{section_num}'")
            context['rejection_reason'] = 'major_section_too_large'
            return False, None, None, None, context

    # Split title at separator (handles "--", "-", "~", "=", "." patterns)
    topic, remainder = split_topic_at_separator(full_topic)
    
    if debug:
        print(f"      [DEBUG] ACCEPTED: section='{section_num}' topic='{topic[:30] if topic else ''}...'")
        if remainder:
            print(f"      [DEBUG] Split off remainder: '{remainder[:50]}...'")
    
    return True, section_num, topic.strip(), remainder.strip(), context


def check_if_paragraph_is_header_legacy(line_text: str, debug: bool = False) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    LEGACY VERSION: Original function signature for backward compatibility.
    """
    is_header, section_num, topic, remainder, context = check_if_paragraph_is_header(line_text, debug)
    return is_header, section_num, topic, remainder


def group_elements_with_bbox(elements: List[Dict]) -> List[Dict]:
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
            
            original_bbox = current_element.get('bbox')
            
            j = i + 1
            while j < len(elements) and elements[j]['type'] == 'unassigned_text_block':
                content_pieces.append(elements[j]['content'])
                j += 1
            
            current_element['content'] = "\n\n".join(content_pieces)
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
    Succinctly handles nested or flat OCR structures.
    """
    if not isinstance(page_obj, dict):
        raise ValueError(f"Expected dict for page object, got {type(page_obj)}")
    
    # If the object itself contains the OCR keys, it's the page_dict
    if 'text' in page_obj and isinstance(page_obj.get('text'), list):
        return page_obj
    
    # If it's a wrapper, look inside for 'page_dict'
    inner = page_obj.get('page_dict')
    if isinstance(inner, dict) and 'text' in inner:
        return inner
    
    raise KeyError("Could not find 'page_dict' or 'text' keys. Data may be stale or incorrectly formatted.")


def extract_page_metadata(page_dict: Dict, page_obj: Dict = None) -> Dict:
    """
    Extract useful metadata from a page_dict and page_obj.
    DEBUG VERSION - prints what it's receiving.
    """
    metadata = {}
    
    print('extract_page_metadata')
    print('page_dict', page_dict.keys()) # contains level, page_num, block_num, left, top, width, height, etc
    print('page_obj', page_obj.keys()) # contains document_Id, page_Id, page_dict
    
    # === PRESERVE ORIGINAL IMAGE_META ===
    image_meta = page_obj.get('image_meta')
    # Store the FULL image_meta structure
    metadata['image_meta'] = image_meta
    
    # Also extract convenient top-level width/height from render_raw
    render_raw = image_meta.get('render_raw', {})
    if render_raw:
        if 'width_px' in render_raw:
            metadata['page_width'] = render_raw['width_px']
        if 'height_px' in render_raw:
            metadata['page_height'] = render_raw['height_px']
    
    # Fallback to canonical if render_raw not present
    if 'page_width' not in metadata:
        canonical = image_meta.get('canonical', {})
        if canonical:
            if 'width_px' in canonical:
                metadata['page_width'] = canonical['width_px']
            if 'height_px' in canonical:
                metadata['page_height'] = canonical['height_px']
    
    # Capture OCR confidence stats if available

    confs = [c for c in page_dict['conf'] if isinstance(c, (int, float)) and c >= 0]
    if confs:
        metadata['avg_confidence'] = sum(confs) / len(confs)
        metadata['min_confidence'] = min(confs)
        metadata['max_confidence'] = max(confs)

    
    # Count words on page
    if page_dict.get('text'):
        metadata['word_count'] = len(page_dict['text'])
    
    # Count unique blocks, paragraphs, lines
    if page_dict.get('block_num'):
        metadata['block_count'] = len(set(page_dict['block_num']))
    if page_dict.get('par_num'):
        metadata['paragraph_count'] = len(set(page_dict['par_num']))
    if page_dict.get('line_num'):
        metadata['line_count'] = len(set(page_dict['line_num']))
    
    # Preserve any DPI info if present
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

    print('input_path', input_path)
    
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

    # Handle dict structure (keyed by page number)

    for key, val in data.items():
        # val is the full page object containing 'page_dict' and 'image_meta'
        page_dict = get_page_dict_from_object(val)
        print('page val load_raw', val.keys())
        print('page_dict load_raw', page_dict.keys())
        
        if page_dict is not None:
            page_id = get_page_id_from_object(val, fallback_key=key)
            # Ensure we pass the original 'val' so extract_page_metadata sees 'image_meta'
            page_meta = extract_page_metadata(page_dict, page_obj=val)
            pages_to_process.append((page_id, page_dict, page_meta))
            


    pages_to_process.sort(key=lambda x: x[0])
    return pages_to_process


def run_section_processing_on_file(
    input_path: str, 
    output_path: str, 
    content_start_page: int = 1,
    header_rel_threshold: float = 0.06,
    footer_rel_threshold: float = 0.94
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
    dropped_footer_lines = 0
    
    for page_id, page_dict, page_meta in content_pages:
        if page_meta:
            all_page_metadata[page_id] = page_meta

        page_height = page_meta.get('page_height', 0)
        
        lines = reconstruct_lines_with_bbox(page_dict)
        
        # --- NEW: Calculate Page-Level Font Stats ---
        # We calculate the median line height for this specific page.
        # Headers are usually > 1.1x the median height.
        line_heights = []
        for line in lines:
            if line.get('bbox') and line['bbox'].get('height', 0) > 0:
                line_heights.append(line['bbox']['height'])
        
        median_height = statistics.median(line_heights) if line_heights else 10.0
        # --------------------------------------------
        
        for line_data in lines:
            line_text = line_data['text'].strip()
            if not line_text:
                continue
            
            line_bbox = line_data.get('bbox')
            
            # CALCULATE NORMALIZED TOP
            norm_top = 0.0
            if line_bbox and page_height > 0:
                norm_top = line_bbox.get('top', 0) / page_height

            # --- HEADER FILTER (UPDATED) ---
            # Old: if line_bbox.get('top', 9999) < header_top_threshold:
            if header_rel_threshold > 0 and norm_top < header_rel_threshold:
                dropped_header_lines += 1
                continue
            
            # --- FOOTER FILTER (UPDATED) ---
            # Old: if line_bbox.get('top', 0) > footer_top_threshold:
            # Note: For footers, you usually check if norm_top > 0.94 (or similar)
            if footer_rel_threshold > 0 and norm_top > footer_rel_threshold:
                dropped_footer_lines += 1
                continue

            is_header, section_num, topic, remainder, context = check_if_paragraph_is_header(line_text)
            
            # --- NEW: Calculate Visual Features ---
            current_height = line_bbox['height'] if line_bbox and 'height' in line_bbox else median_height
            relative_height = current_height / median_height if median_height > 0 else 1.0
            # --------------------------------------

            element_base = {
                "page_number": page_id,
                "bbox": line_bbox,
                "ml_features": {
                    "relative_height": relative_height,
                    "is_bold": context.get('is_bold', False) if context else False, # Placeholder
                    "line_length_chars": len(line_text),
                }
            }

            if is_header:
                element_base["type"] = "section"
                element_base["section_number"] = section_num
                element_base["topic"] = topic
                element_base["content"] = ""
                # Store context for ML features
                element_base["detection_context"] = {
                    "raw_section_number": context.get('raw_section_number'),
                    "had_leading_whitespace": context.get('had_leading_whitespace', False),
                    "leading_whitespace_len": context.get('leading_whitespace_len', 0),
                    "had_text_before_number": context.get('had_text_before_number', False),
                    "text_before_number": context.get('text_before_number', ''),
                    "original_line_length": context.get('line_length', 0),
                }
                raw_elements.append(element_base)
                
                if remainder:
                    # The remainder is body text, so likely median height (rel height = 1.0)
                    remainder_element = {
                        "type": "unassigned_text_block",
                        "content": remainder,
                        "page_number": page_id,
                        "bbox": line_bbox,
                        "ml_features": {
                            "relative_height": 1.0,
                            "line_length_chars": len(remainder)
                        }
                    }
                    raw_elements.append(remainder_element)
            else:
                element_base["type"] = "unassigned_text_block"
                element_base["content"] = line_text
                raw_elements.append(element_base)
    
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