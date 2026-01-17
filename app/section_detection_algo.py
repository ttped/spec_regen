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
    print(f"Algorithmic organization results saved to {output_path}")

def load_all_ocr_files(directory: str) -> List[Dict]:
    """Loads all OCR JSON files from a specified directory."""
    all_pages = []
    print(f"Loading OCR files from: {directory}")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at {directory}")
        return []
        
    filenames = sorted(os.listdir(directory))
    total_files = 0

    for filename in filenames:
        if filename.endswith(".json"):
            total_files += 1
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle structure where pages are values in a dict
                if isinstance(data, dict):
                    all_pages.extend(data.values())
                elif isinstance(data, list):
                    all_pages.extend(data)

    print(f"Loaded data for {len(all_pages)} pages from {total_files} files.")
    return all_pages

def reconstruct_rich_paragraphs(page_dict: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Converts Tesseract-style parallel arrays into 'Rich Paragraph' objects.
    Each object contains the text plus spatial metadata (font size, indent).
    """
    if not page_dict.get('text'):
        return []

    rich_paragraphs = []
    current_words = []
    current_heights = []
    current_lefts = []
    
    # Safely get lists or defaults
    texts = page_dict.get('text', [])
    block_nums = page_dict.get('block_num', [])
    par_nums = page_dict.get('par_num', [])
    heights = page_dict.get('height', [])
    lefts = page_dict.get('left', [])
    
    count = len(texts)
    if count == 0: return []

    last_block = block_nums[0]
    last_par = par_nums[0]

    for i in range(count):
        word_text = str(texts[i]).strip()
        block_num = block_nums[i]
        par_num = par_nums[i]
        
        if not word_text:
            continue
            
        # Check for new paragraph
        if block_num != last_block or par_num != last_par:
            if current_words:
                rich_paragraphs.append({
                    "text": " ".join(current_words),
                    "avg_height": statistics.mean(current_heights) if current_heights else 0,
                    "indentation": min(current_lefts) if current_lefts else 0,
                    # We can use these IDs for debugging order if needed
                    "block_id": f"{last_block}_{last_par}" 
                })
            
            current_words = []
            current_heights = []
            current_lefts = []
            last_block, last_par = block_num, par_num

        current_words.append(word_text)
        # Ensure we don't crash if metadata lists are shorter than text list (rare edge case)
        if i < len(heights): current_heights.append(heights[i])
        if i < len(lefts): current_lefts.append(lefts[i])

    # Add the final paragraph
    if current_words:
        rich_paragraphs.append({
            "text": " ".join(current_words),
            "avg_height": statistics.mean(current_heights) if current_heights else 0,
            "indentation": min(current_lefts) if current_lefts else 0,
            "block_id": f"{last_block}_{last_par}"
        })
        
    return rich_paragraphs

def get_document_stats(all_paragraphs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates the 'Body Text' profile for the entire document.
    Returns the mode (most common) font size and indentation.
    """
    if not all_paragraphs:
        return {"body_size": 10, "body_indent": 0}

    # Round heights to nearest integer to group similar fonts (e.g. 11.2 vs 11.5 -> 11)
    heights = [round(p['avg_height']) for p in all_paragraphs if p['avg_height'] > 0]
    # Round indentation to nearest 5 pixels to group slight misalignments
    indents = [round(p['indentation'] / 5) * 5 for p in all_paragraphs]

    # Calculate Mode (Most common value)
    try:
        body_size = statistics.mode(heights)
    except statistics.StatisticsError:
        # If multiple modes, take the median as a fallback
        body_size = statistics.median(heights) if heights else 10

    try:
        body_indent = statistics.mode(indents)
    except statistics.StatisticsError:
        body_indent = min(indents) if indents else 0

    print(f"Document Stats - Body Font Size: ~{body_size}px, Body Margin: ~{body_indent}px")
    return {"body_size": body_size, "body_indent": body_indent}

def check_if_header(para: Dict[str, Any], stats: Dict[str, float]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Determines if a paragraph is a header using Regex + Visual Stats.
    """
    text = para['text']
    avg_height = para['avg_height']
    indent = para['indentation']
    
    # 1. Regex Filter: Must start with a section number (e.g., "1.2", "A.")
    # We allow loose matching here because the visual check will filter false positives
    match = re.match(r'^\s*([a-zA-Z0-9\.]+)\s+(.+)', text)
    match_num_only = re.match(r'^\s*([a-zA-Z0-9\.]+)\s*$', text)

    if match:
        potential_num, topic = match.groups()
    elif match_num_only:
        potential_num = match_num_only.group(1)
        topic = ""
    else:
        return False, None, None

    # 2. Heuristics to clean up the Regex match
    if not any(c.isdigit() for c in potential_num): return False, None, None # Must have digit
    if len(potential_num) > 10: return False, None, None # Number too long
    if potential_num.count('.') > 4: return False, None, None # Too deep (1.2.3.4.5 is rare)

    # 3. Visual Confirmation (The "Algorithmic" Layer)
    is_visually_prominent = False
    
    # Rule A: It is significantly larger than body text (e.g. > 10% larger)
    if avg_height > (stats['body_size'] * 1.1):
        is_visually_prominent = True
        
    # Rule B: It matches body size, but is aligned to the left margin (common for sub-headers)
    # We allow a small tolerance (e.g. 10px) for scanner skew
    elif avg_height >= (stats['body_size'] * 0.9) and indent <= (stats['body_indent'] + 10):
        is_visually_prominent = True

    # If regex matched but visually it looks like indented body text or small footnote, reject it
    if not is_visually_prominent:
        return False, None, None

    return True, potential_num.strip().rstrip('.'), topic.strip()

def group_elements(classified_items: List[Dict]) -> List[Dict]:
    """
    Merges 'unassigned_text_block' items into the preceding 'section' item.
    """
    if not classified_items:
        return []

    merged_elements = []
    current_section = None

    for item in classified_items:
        if item['type'] == 'section':
            # If we were building a section, save it
            if current_section:
                merged_elements.append(current_section)
            
            # Start a new section
            current_section = item
            current_section['content'] = "" # Initialize content buffer
        
        elif item['type'] == 'unassigned_text_block':
            text = item.get('content', '')
            if current_section:
                # Append to current section
                if current_section['content']:
                    current_section['content'] += "\n\n" + text
                else:
                    current_section['content'] = text
            else:
                # No preceding section (preamble text), keep as unassigned
                merged_elements.append(item)

    # Append the last section being built
    if current_section:
        merged_elements.append(current_section)
            
    return merged_elements

def run_algorithmic_organization(input_dir: str, output_path: str):
    """
    Main orchestration function.
    1. Loads all pages.
    2. Converts raw OCR to Rich Paragraphs.
    3. Calculates document-wide font stats.
    4. Classifies and groups paragraphs.
    """
    all_page_data = load_all_ocr_files(input_dir)
    if not all_page_data:
        return
    
    # Sort pages (try by ID, fallback to file order)
    try:
        all_page_data.sort(key=lambda p: int(p.get('page_Id', 0)))
    except (ValueError, KeyError, TypeError):
        pass

    # --- PASS 1: Convert to Rich Paragraphs & Collect Stats ---
    all_rich_paragraphs = []
    for page in all_page_data:
        page_dict = page.get('page_dict')
        if page_dict:
            # Reconstruct paragraphs with metadata
            paras = reconstruct_rich_paragraphs(page_dict)
            all_rich_paragraphs.extend(paras)

    # Calculate baseline (what does "normal" text look like in this specific doc?)
    doc_stats = get_document_stats(all_rich_paragraphs)

    # --- PASS 2: Classification ---
    classified_items = []
    
    for para in all_rich_paragraphs:
        is_header_bool, sec_num, topic = check_if_header(para, doc_stats)

        if is_header_bool:
            classified_items.append({
                "type": "section",
                "section_number": sec_num,
                "topic": topic,
                "content": "" # To be filled in grouping
            })
        else:
            classified_items.append({
                "type": "unassigned_text_block",
                "content": para['text']
            })

    # --- PASS 3: Grouping ---
    final_elements = group_elements(classified_items)
    save_results_to_json(final_elements, output_path)

if __name__ == '__main__':
    project_root = os.getcwd() 
    
    input_directory = os.path.join(project_root, "iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced")
    results_dir = os.path.join(project_root, "results")
    output_filepath = os.path.join(results_dir, "algorithmic_organization_output.json")

    print("--- Starting Algorithmic Organization (Visual-Aware) ---")
    run_algorithmic_organization(input_directory, output_filepath)
    print("--- Algorithmic Organization Finished ---")