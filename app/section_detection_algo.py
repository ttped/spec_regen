import os
import json
import re
from typing import List, Dict, Tuple, Optional
import statistics
from typing import List, Dict, Any

def save_results_to_json(data: List[Dict], output_path: str):
    """
    Saves the given data list to a JSON file, creating the directory if needed.

    Args:
        data: The list of dictionary elements to save.
        output_path: The full path for the output JSON file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Algorithmic organization results saved to {output_path}")

def load_all_ocr_files(directory: str) -> List[Dict]:
    """
    Loads all OCR JSON files from a specified directory into a single list of pages.

    Args:
        directory: The path to the directory containing the OCR JSON files.

    Returns:
        A list of page data dictionaries.
    """
    all_pages = []
    print(f"Loading OCR files from: {directory}")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at {directory}")
        return []
        
    # Get a sorted list of filenames to ensure pages are loaded in a somewhat logical order
    # before the final sort.
    filenames = sorted(os.listdir(directory))
    total_files = 0

    for filename in filenames:
        if filename.endswith(".json"):
            total_files += 1
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # CORRECTED LINE: Use data.values() to get the page objects, not the keys.
                all_pages.extend(data.values())

    print(f"Loaded data for {len(all_pages)} pages from {total_files} files.")
    return all_pages

def reconstruct_rich_paragraphs(page_dict: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Converts Tesseract-style parallel arrays into 'Rich Paragraph' objects.
    Each object contains the text plus spatial metadata (font size, indent, y-pos).
    """
    if not page_dict.get('text'):
        return []

    rich_paragraphs = []
    
    # Initialize buffers for the current paragraph
    current_words = []
    current_heights = []
    current_lefts = []
    current_tops = []
    
    # We need to loop through the parallel arrays. 
    # The 'block_num' and 'par_num' keys define when a paragraph changes.
    count = len(page_dict['text'])
    last_block = page_dict['block_num'][0]
    last_par = page_dict['par_num'][0]

    for i in range(count):
        # Extract data for the current word
        word_text = str(page_dict['text'][i]).strip()
        block_num = page_dict['block_num'][i]
        par_num = page_dict['par_num'][i]
        
        # Skip empty strings (often just layout noise)
        if not word_text:
            continue
            
        # Check if we have started a new paragraph block
        if block_num != last_block or par_num != last_par:
            # SAVE PREVIOUS PARAGRAPH
            if current_words:
                rich_paragraphs.append({
                    "text": " ".join(current_words),
                    # Height is our best proxy for Font Size
                    "avg_height": statistics.mean(current_heights),
                    # Left is our proxy for Indentation
                    "indentation": min(current_lefts), 
                    # Top is used for sorting relative to images
                    "y_position": min(current_tops),
                    "block_id": f"{last_block}_{last_par}"
                })
            
            # RESET BUFFERS
            current_words = []
            current_heights = []
            current_lefts = []
            current_tops = []
            last_block, last_par = block_num, par_num

        # ADD WORD TO BUFFER
        # Tesseract visual data: left, top, width, height
        current_words.append(word_text)
        current_heights.append(page_dict['height'][i])
        current_lefts.append(page_dict['left'][i])
        current_tops.append(page_dict['top'][i])

    # Don't forget the very last paragraph after the loop finishes
    if current_words:
        rich_paragraphs.append({
            "text": " ".join(current_words),
            "avg_height": statistics.mean(current_heights),
            "indentation": min(current_lefts),
            "y_position": min(current_tops),
            "block_id": f"{last_block}_{last_par}"
        })
        
    return rich_paragraphs

def check_if_paragraph_is_header(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Checks if a paragraph is a section header using regex and heuristics.
    A header must start with a section number (e.g., "1.2.3", "2.A").

    Args:
        text: The paragraph text to check.

    Returns:
        A tuple containing:
        - A boolean indicating if it's a header.
        - The extracted section number string.
        - The extracted topic string.
    """
    # Regex to find a potential section number at the start, followed by a topic.
    match = re.match(r'^\s*([a-zA-Z0-9\.]+)\s+(.+)', text)
    match_no_title = re.match(r'^\s*([a-zA-Z0-9\.]+)\s*$', text)
    
    if match:
        potential_num, topic = match.groups()
    elif match_no_title:
        potential_num = match_no_title.group(1)
        topic = ""
    else:
        return False, None, None

    # Heuristic 1: Must contain at least one digit.
    if not any(c.isdigit() for c in potential_num):
        return False, None, None

    # Heuristic 2: Limit alpha characters to avoid matching regular text.
    if sum(c.isalpha() for c in potential_num) > 2:
        return False, None, None
        
    # Heuristic 3: Avoid excessively long "numbers".
    if len(potential_num) > 20:
        return False, None, None

    # Heuristic 4: Ensure it's not purely alphabetic.
    if potential_num.isalpha():
        return False, None, None

    section_num = potential_num.strip().rstrip('.')
    return True, section_num, topic.strip()

def group_elements(elements: List[Dict]) -> List[Dict]:
    """
    Merges consecutive content blocks and attaches them to preceding section headers.

    Args:
        elements: A flat list of 'section' and 'unassigned_text_block' elements.

    Returns:
        A structured list where content is merged into its parent section.
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

def run_algorithmic_organization(input_dir: str, output_path: str):
    """
    Main function to perform algorithmic document organization by loading OCR data,
    identifying section headers, and structuring the content.

    Args:
        input_dir: The directory containing advanced OCR JSON files.
        output_path: The path to save the final structured JSON file.
    """
    all_page_data = load_all_ocr_files(input_dir)
    if not all_page_data:
        return
    
    try:
        all_page_data.sort(key=lambda p: int(p['page_Id']))
    except (ValueError, KeyError, TypeError):
        print("Warning: Could not sort pages by 'page_Id'. Processing in file order.")

    raw_elements = []
    for page in all_page_data:
        page_dict = page.get('page_dict')
        if not page_dict:
            continue
            
        paragraphs = reconstruct_paragraphs_from_page_dict(page_dict)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            is_header, section_num, topic = check_if_paragraph_is_header(para)

            if is_header:
                raw_elements.append({
                    "type": "section",
                    "section_number": section_num,
                    "topic": topic,
                    "content": ""
                })
            else:
                raw_elements.append({
                    "type": "unassigned_text_block",
                    "content": para
                })

    final_elements = group_elements(raw_elements)
    save_results_to_json(final_elements, output_path)

if __name__ == '__main__':
    # This script is designed to be run from the project's root directory
    # or have its paths adjusted accordingly.
    project_root = os.getcwd() 
    
    input_directory = os.path.join(project_root, "iris_ocr", "CM_Spec_OCR_and_figtab_output", "raw_data_advanced")
    results_dir = os.path.join(project_root, "results")
    output_filepath = os.path.join(results_dir, "algorithmic_organization_output.json")

    print("--- Starting Algorithmic Organization ---")
    run_algorithmic_organization(input_directory, output_filepath)
    print("--- Algorithmic Organization Finished ---")