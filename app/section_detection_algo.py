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

def reconstruct_lines(page_dict: Dict[str, List], page_num: int) -> List[Dict[str, Any]]:
    """
    Groups words into LINES instead of PARAGRAPHS.
    This prevents headers from being 'lumped' into previous text blocks.
    Calculates the Bounding Box (bbox) for each line.
    """
    if not page_dict or 'text' not in page_dict:
        return []

    rich_lines = []
    
    # Buffers for the current line
    current_words = []
    current_heights = []
    current_lefts = []
    current_tops = []
    current_rights = []  # To calc max width
    current_bottoms = [] # To calc max height
    
    texts = page_dict.get('text', [])
    # We use line_num to differentiate, fallback to par_num/block_num if needed
    block_nums = page_dict.get('block_num', [])
    par_nums = page_dict.get('par_num', [])
    line_nums = page_dict.get('line_num', [])
    
    # Geometry
    heights = page_dict.get('height', [])
    lefts = page_dict.get('left', [])
    tops = page_dict.get('top', [])
    widths = page_dict.get('width', [])
    
    count = len(texts)
    if count == 0: return []

    # Initialize with first item
    last_block = block_nums[0] if block_nums else 0
    last_par = par_nums[0] if par_nums else 0
    last_line = line_nums[0] if line_nums else 0

    for i in range(count):
        word_text = str(texts[i]).strip()
        
        # Skip empty strings, but DON'T skip logic checks 
        # (sometimes empty strings mark structure, but usually safe to skip in Tesseract)
        if not word_text:
            continue
            
        block_num = block_nums[i]
        par_num = par_nums[i]
        line_num = line_nums[i]
        
        # BREAK CONDITION: New Block OR New Paragraph OR New Line
        is_new_line = (block_num != last_block) or (par_num != last_par) or (line_num != last_line)
        
        if is_new_line:
            if current_words:
                # Calculate BBOX: [min_left, min_top, max_right, max_bottom]
                # width = right - left
                min_left = min(current_lefts)
                min_top = min(current_tops)
                max_right = max(current_rights)
                max_bottom = max(current_bottoms)
                
                rich_lines.append({
                    "text": " ".join(current_words),
                    "avg_height": statistics.mean(current_heights) if current_heights else 0,
                    "indentation": min_left,
                    "page": page_num,
                    "bbox": [min_left, min_top, max_right - min_left, max_bottom - min_top], # [x, y, w, h]
                    "line_id": f"{last_block}_{last_par}_{last_line}"
                })
            
            # Reset buffers
            current_words = []
            current_heights = []
            current_lefts = []
            current_tops = []
            current_rights = []
            current_bottoms = []
            
            last_block, last_par, last_line = block_num, par_num, line_num

        # Add word data
        current_words.append(word_text)
        
        # Safety checks for indices
        if i < len(heights): current_heights.append(heights[i])
        
        if i < len(lefts) and i < len(tops) and i < len(widths) and i < len(heights):
            l = lefts[i]
            t = tops[i]
            w = widths[i]
            h = heights[i]
            
            current_lefts.append(l)
            current_tops.append(t)
            current_rights.append(l + w)
            current_bottoms.append(t + h)

    # Flush final line
    if current_words:
        min_left = min(current_lefts)
        min_top = min(current_tops)
        max_right = max(current_rights)
        max_bottom = max(current_bottoms)
        
        rich_lines.append({
            "text": " ".join(current_words),
            "avg_height": statistics.mean(current_heights) if current_heights else 0,
            "indentation": min_left,
            "page": page_num,
            "bbox": [min_left, min_top, max_right - min_left, max_bottom - min_top],
            "line_id": f"{last_block}_{last_par}_{last_line}"
        })
        
    return rich_lines

def get_document_stats(all_lines: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates stats. We use LINES now, but the logic holds:
    Body text is the most common height.
    Left Margin is the most common minimum indentation.
    """
    if not all_lines:
        return {"body_size": 10, "margin_left": 0}

    heights = [round(p['avg_height']) for p in all_lines if p['avg_height'] > 0]
    
    # Filter valid indents (avoid negatives)
    valid_indents = [p['indentation'] for p in all_lines if p['indentation'] >= 0]
    
    try:
        # We use min() here to find the "hard" left edge of the document
        margin_left = min(valid_indents) if valid_indents else 0
    except ValueError:
        margin_left = 0

    try:
        body_size = statistics.mode(heights)
    except statistics.StatisticsError:
        body_size = statistics.median(heights) if heights else 10

    print(f"Document Stats - Body Font: ~{body_size}px, Left Margin: ~{margin_left}px")
    return {"body_size": body_size, "margin_left": margin_left}

def check_if_header_line(line: Dict[str, Any], stats: Dict[str, float]) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Checks if a LINE is a header.
    Returns: (is_header, section_number, title_text, run_in_content)
    """
    text = line['text']
    avg_height = line['avg_height']
    indent = line['indentation']
    
    # 1. Regex: Matches "1.0", "1.1", "A." at start of line
    # Expanded to better catch "1.0" or "2.1.3" followed by spaces
    match = re.match(r'^\s*([A-Z0-9]+(?:\.[A-Z0-9]+)*\.?)\s*(.*)', text)
    
    if not match:
        return False, None, None, None

    potential_num = match.group(1).strip()
    rest_of_line = match.group(2).strip()

    # 2. Basic Heuristics
    if not any(c.isdigit() for c in potential_num): return False, None, None, None
    if len(potential_num) > 12: return False, None, None, None
    
    # 3. SPATIAL FILTER
    # If the number is simple (1. or 1.1) AND indented -> likely a list item.
    is_complex_number = potential_num.count('.') >= 2
    indent_tolerance = 25 
    
    if not is_complex_number:
        if indent > (stats['margin_left'] + indent_tolerance):
            # Indented simple number -> List Item
            return False, None, None, None

    # 4. Title Extraction
    if not rest_of_line:
        # Case: "1.0" is on the line by itself.
        return True, potential_num.rstrip('.'), "", ""

    # Heuristic: If we matched, we treat the rest as the title.
    # Since we are processing by LINE now, it's safer to assume 
    # the rest of the line is the Title, and the Body starts on the next line.
    # However, for "1.2 Scope. The scope is..." we still split.
    
    words = rest_of_line.split()
    # If it looks like a sentence (long), split. If short, it's just the title.
    if len(words) > 10: 
        # Long line -> Run-in header
        title_text = " ".join(words[:2])
        run_in_content = " ".join(words[2:])
    else:
        # Short line -> Just Title
        title_text = rest_of_line
        run_in_content = ""

    return True, potential_num.rstrip('.'), title_text, run_in_content

def group_lines_into_sections(classified_lines: List[Dict]) -> List[Dict]:
    """
    Merges lines.
    - If Line is Section -> Start new Section object.
    - If Line is Text -> Append to current Section content.
    """
    if not classified_lines:
        return []

    merged_elements = []
    current_section = None

    for item in classified_lines:
        if item['type'] == 'section':
            if current_section:
                merged_elements.append(current_section)
            
            # Create new section
            current_section = item
            
            # Handle run-in content (content that was on the same line as the header)
            initial_content = item.pop('run_in_content', '')
            current_section['content'] = initial_content
            
            if 'page' not in current_section:
                current_section['page'] = item.get('page')

        elif item['type'] == 'unassigned_text_block':
            text = item.get('content', '')
            if current_section:
                if current_section['content']:
                    # We are merging lines, so we add a space if it seems continuous, 
                    # or newline if it was a distinct line. 
                    # For simplicity in this text-block phase, we use newline to preserve structure.
                    current_section['content'] += "\n" + text
                else:
                    current_section['content'] = text
            else:
                # Text before the first section (Preamble)
                merged_elements.append(item)

    if current_section:
        merged_elements.append(current_section)
            
    return merged_elements

def run_algorithmic_organization(input_file_path: str, output_path: str):
    # 1. Load
    sorted_pages = load_single_document(input_file_path)
    if not sorted_pages: return

    # 2. Reconstruct LINES (Not Paragraphs)
    all_rich_lines = []
    for page_num, page_dict in sorted_pages:
        lines = reconstruct_lines(page_dict, page_num)
        all_rich_lines.extend(lines)

    # 3. Stats
    doc_stats = get_document_stats(all_rich_lines)

    # 4. Classify Lines
    classified_items = []
    for line in all_rich_lines:
        is_header_bool, sec_num, topic, run_in_content = check_if_header_line(line, doc_stats)

        if is_header_bool:
            classified_items.append({
                "type": "section",
                "section_number": sec_num,
                "topic": topic,
                "run_in_content": run_in_content, 
                "page": line['page'],
                "header_bbox": line['bbox'] # <--- BBOX CAPTURED HERE
            })
        else:
            classified_items.append({
                "type": "unassigned_text_block",
                "content": line['text'],
                "page": line['page'],
                # "bbox": line['bbox'] # Optional: keep bbox for text blocks if needed later
            })

    # 5. Group
    final_elements = group_lines_into_sections(classified_items)
    save_results_to_json(final_elements, output_path)

if __name__ == '__main__':
    project_root = os.getcwd() 
    doc_stem = "S-133-05737AF-SSS"
    input_file = os.path.join(project_root, "iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced", f"{doc_stem}.json")
    output_file = os.path.join(project_root, "results", f"{doc_stem}_algo_organized.json")

    print(f"--- Processing {doc_stem} (Line-Based) ---")
    run_algorithmic_organization(input_file, output_file)
    print("--- Finished ---")