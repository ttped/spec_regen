import os
import json
import re
from typing import List, Dict, Tuple, Optional

def reconstruct_lines_from_page_dict(page_dict: Dict[str, List]) -> List[str]:
    """
    Reconstructs a list of line strings from the word-level OCR data.
    """
    if not page_dict.get('text') or 'line_num' not in page_dict:
        return []

    lines = []
    current_line_words = []
    
    # Handle potential empty arrays or mismatches safely
    try:
        last_block = page_dict['block_num'][0]
        last_par = page_dict['par_num'][0]
        last_line = page_dict['line_num'][0]
    except IndexError:
        return []

    count = len(page_dict['text'])
    for i in range(count):
        # Safety check for index out of bounds if lists are uneven
        if i >= len(page_dict['block_num']) or i >= len(page_dict['line_num']):
            break

        block_num = page_dict['block_num'][i]
        par_num = page_dict['par_num'][i]
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

def check_if_line_is_header(line_text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Checks if a line is a section header using your preferred heuristic logic.
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

    # --- Heuristics to filter out false positives ---
    
    # 1. Topic looks like a date (Month name check)
    MONTHS = [
        'january', 'february', 'march', 'april', 'may', 'june', 
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    if topic and any(topic.lower().startswith(m) for m in MONTHS):
        return False, None, None

    # 2. Must contain at least one digit
    if not any(c.isdigit() for c in potential_num):
        return False, None, None

    # 3. Limit alpha characters (avoids regular words being seen as headers)
    if sum(c.isalpha() for c in potential_num) > 2:
        return False, None, None
        
    # 4. Avoid excessively long "numbers"
    if len(potential_num) > 20:
        return False, None, None

    # 5. Ensure it's not purely alphabetic
    if potential_num.isalpha():
        return False, None, None

    # 6. If it's all digits, limit length to 3 (Rejects "1506073")
    if potential_num.isdigit() and len(potential_num) > 3:
        return False, None, None

    # 7. If it contains both letters and digits, it must contain a dot (Rejects "616A")
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
            # Absorb subsequent unassigned blocks into this section
            while j < len(elements) and elements[j]['type'] == 'unassigned_text_block':
                content_pieces.append(elements[j]['content'])
                j += 1
            
            # Join with double newlines for paragraph separation
            current_element['content'] = "\n\n".join(content_pieces)
            merged_elements.append(current_element)
            i = j
        else:
            merged_elements.append(current_element)
            i += 1
            
    return merged_elements

def run_section_processing_on_file(input_path: str, output_path: str):
    """
    Processes a single raw OCR file and organizes it into sections/content.
    """
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize input: handle list of pages or dict of pages
    pages = []
    if isinstance(data, list):
        pages = data
    elif isinstance(data, dict):
        # Assuming dict keys are page numbers or IDs
        pages = list(data.values())

    # Sort pages by ID to ensure correct order
    def get_page_id(p):
        try:
            # Try various keys that might hold the page number
            return int(p.get('page_Id', 0) or p.get('page_num', 0))
        except (ValueError, TypeError):
            return 0
    
    pages.sort(key=get_page_id)

    raw_elements = []

    for page in pages:
        page_dict = page.get('page_dict')
        # Fallback if the structure is flat (some OCR outputs differ)
        if not page_dict and 'text' in page:
            page_dict = page
            
        if not page_dict:
            continue
        
        lines = reconstruct_lines_from_page_dict(page_dict)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            is_header, section_num, topic = check_if_line_is_header(line)

            if is_header:
                # Capture the header
                raw_elements.append({
                    "type": "section",
                    "section_number": section_num,
                    "topic": topic,
                    "content": "", # Filled in grouping step
                    "page_number": get_page_id(page)
                })
            else:
                # Capture normal text
                raw_elements.append({
                    "type": "unassigned_text_block",
                    "content": line,
                    "page_number": get_page_id(page)
                })

    final_elements = group_elements(raw_elements)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_elements, f, indent=4)
    
    print(f"Processed {len(pages)} pages. Results saved to {output_path}")