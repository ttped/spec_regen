"""
classify_agent.py - Determines where main content begins in a document.

This agent classifies the first N pages to find the transition from 
Table of Contents to actual Content. It returns the content start page number.

The flattened text is only used internally for LLM classification - the raw
OCR structure is preserved for downstream processing.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .utils import _extract_json_from_llm_string, call_llm, save_results_to_json
except ImportError:
    from utils import _extract_json_from_llm_string, call_llm, save_results_to_json


def flatten_page_text(page_dict: Dict) -> str:
    """
    Flattens a page_dict into a single text string for LLM classification.
    This is ONLY used for classification - not for downstream processing.
    """
    if not page_dict:
        return ""
    
    # Handle wrapped structure
    if 'page_dict' in page_dict and isinstance(page_dict['page_dict'], dict):
        page_dict = page_dict['page_dict']
    
    text_list = page_dict.get('text', [])
    if isinstance(text_list, list):
        return " ".join(str(t) for t in text_list)
    return str(text_list) if text_list else ""


def get_page_type_with_llm(page_text: str, llm_config: Dict) -> str:
    """
    Uses an LLM to classify a single page into one of three types:
    - TABLE_OF_CONTENTS
    - CONTENT_BODY  
    - AMBIGUOUS
    """
    prompt = f"""
You are a document page classifier. Your job is to classify the following page text into one of three specific types:

1.  `TABLE_OF_CONTENTS`: The page is a list of sections, figures, or tables, often with page numbers. It does NOT contain full paragraphs of body text.
2.  `CONTENT_BODY`: The page contains full, descriptive paragraphs and sentences. It is the main substance of the document. A heading like "1.0 SCOPE" might be present, but it must be followed by paragraph text.
3.  `AMBIGUOUS`: The page is unclear, mostly empty, or doesn't fit the other categories.

Analyze the text and respond with a single JSON object containing one key, "page_type", with one of the three values.

Page Text:
---
{page_text[:3000]}
---

Example Output:
{{"page_type": "TABLE_OF_CONTENTS"}}
"""
    try:
        llm_response = call_llm(
            prompt,
            llm_config['model_name'],
            llm_config['base_url'],
            llm_config['api_key'],
            llm_config['provider']
        )
    except Exception as e:
        print(f"    [Error] LLM Call failed: {e}")
        return "AMBIGUOUS"
    
    if not llm_response:
        return "AMBIGUOUS"

    # Try to extract JSON from the response
    json_string = _extract_json_from_llm_string(llm_response)
    
    # If extraction returned nothing, try the raw response (sometimes models output just JSON)
    if not json_string:
        json_string = llm_response

    try:
        response_data = json.loads(json_string)
        return response_data.get("page_type", "AMBIGUOUS")
    except Exception as e:
        # Log the error but don't crash the thread
        # print(f"    [Warning] JSON Parse Error: {e} | Raw: {json_string[:50]}...")
        return "AMBIGUOUS"


def classify_page_wrapper(args: Tuple[int, str, Dict]) -> Dict:
    """
    Wrapper for parallel execution of page classification.
    """
    page_num, page_text, llm_config = args
    page_type = get_page_type_with_llm(page_text, llm_config)
    return {'page': page_num, 'type': page_type}


def load_pages_for_classification(input_path: str) -> List[Tuple[int, str]]:
    """
    Loads raw OCR and returns a list of (page_id, flattened_text) for classification.
    Only the first ~20 pages are needed for finding the ToC->Content transition.
    """
    if not os.path.exists(input_path):
        return []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  - [Error] Failed to load input file {input_path}: {e}")
        print(f"  - [Action] Skipping classification for this file.")
        return []
    
    pages = []
    
    if isinstance(data, dict):
        for key, val in data.items():
            try:
                page_id = int(key)
            except ValueError:
                digits = re.findall(r'\d+', str(key))
                page_id = int(digits[0]) if digits else 0
            
            # Get page_dict if wrapped
            page_dict = val.get('page_dict', val) if isinstance(val, dict) else val
            flat_text = flatten_page_text(page_dict) if isinstance(page_dict, dict) else ""
            pages.append((page_id, flat_text))
            
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            page_id = item.get('page_Id') or item.get('page_num') or (idx + 1)
            try:
                page_id = int(page_id)
            except (ValueError, TypeError):
                page_id = idx + 1
            
            page_dict = item.get('page_dict', item) if isinstance(item, dict) else item
            flat_text = flatten_page_text(page_dict) if isinstance(page_dict, dict) else ""
            pages.append((page_id, flat_text))
    
    pages.sort(key=lambda x: x[0])
    return pages


def find_content_start_page(
    input_path: str, 
    llm_config: Dict, 
    max_workers: int = 4,
    max_pages_to_check: int = 20
) -> int:
    """
    Analyzes the first N pages to find where main content begins.
    """
    pages = load_pages_for_classification(input_path)
    
    if not pages:
        print(f"  - [Warning] No pages found in {input_path}")
        return 2
    
    # Skip page 1 (title page) and check next N pages
    pages_to_classify = pages[1:max_pages_to_check]
    
    if not pages_to_classify:
        print(f"  - [Warning] Only 1 page in document, defaulting to page 2")
        return 2
    
    print(f"  - Classifying pages 2-{min(len(pages), max_pages_to_check)} with {max_workers} workers...")
    
    # Build tasks for parallel execution
    tasks = [(page_id, page_text, llm_config) for page_id, page_text in pages_to_classify]
    
    page_classifications = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(classify_page_wrapper, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                page_classifications.append(result)
                print(f"    Page {result['page']}: {result['type']}")
            except Exception as exc:
                page_num = future_to_task[future][0]
                print(f"    Page {page_num}: ERROR - {exc}")
                page_classifications.append({'page': page_num, 'type': 'AMBIGUOUS'})
    
    # Sort by page number (results may arrive out of order)
    page_classifications.sort(key=lambda x: x['page'])
    
    # Find the ToC -> Content transition
    for i in range(1, len(page_classifications)):
        prev_type = page_classifications[i-1]['type']
        curr_type = page_classifications[i]['type']
        
        if prev_type == 'TABLE_OF_CONTENTS' and curr_type == 'CONTENT_BODY':
            start_page = page_classifications[i]['page']
            print(f"  - Transition found! Content starts on page {start_page}")
            return start_page
    
    # Fallback: if first classified page is CONTENT_BODY, start there
    if page_classifications and page_classifications[0]['type'] == 'CONTENT_BODY':
        start_page = page_classifications[0]['page']
        print(f"  - First page is content. Starting on page {start_page}")
        return start_page
    
    # Default fallback
    print(f"  - [Warning] No clear ToC->Content transition. Defaulting to page 2.")
    return 2


def run_classification_on_file(
    input_path: str, 
    output_path: str, 
    llm_config: Dict, 
    max_workers: int = 4
) -> int:
    """
    Runs classification and saves a simple result file with the content start page.
    """
    content_start_page = find_content_start_page(input_path, llm_config, max_workers)
    
    # Save a simple result file
    result = {
        "content_start_page": content_start_page,
        "source_file": os.path.basename(input_path)
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    
    print(f"  - Classification saved to {output_path}")
    return content_start_page


def load_content_start_page(classification_path: str, default: int = 2) -> int:
    """
    Loads the content start page from a classification result file.
    """
    if not os.path.exists(classification_path):
        return default
    
    try:
        with open(classification_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('content_start_page', default)
    except (json.JSONDecodeError, KeyError):
        return default


if __name__ == '__main__':
    llm_config = {
        "provider": "mission_assist",
        "model_name": "gemma3",
        "base_url": "http://devmissionassist.api.us.baesystems.com",
        "api_key": "aTOIT9hJM3DBYMQbEY"
    }
    
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "classification_result.json"
        run_classification_on_file(input_file, output_file, llm_config)
    else:
        print("Usage: python classify_agent.py <input.json> [output.json]")