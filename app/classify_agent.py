import os
import json
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .utils import _extract_json_from_llm_string, call_llm, load_pages_from_json, save_results_to_json
except ImportError:
    from utils import _extract_json_from_llm_string, call_llm, load_pages_from_json, save_results_to_json

def get_page_type_with_llm(page_text: str, llm_config: Dict) -> str:
    """
    Uses a simple LLM prompt to classify a single page into one of three types.
    (This function remains unchanged)
    """
    prompt = f"""
You are a document page classifier. Your job is to classify the following page text into one of three specific types:

1.  `TABLE_OF_CONTENTS`: The page is a list of sections, figures, or tables, often with page numbers. It does NOT contain full paragraphs of body text.
2.  `CONTENT_BODY`: The page contains full, descriptive paragraphs and sentences. It is the main substance of the document. A heading like "1.0 SCOPE" might be present, but it must be followed by paragraph text.
3.  `AMBIGUOUS`: The page is unclear, mostly empty, or doesn't fit the other categories.

Analyze the text and respond with a single JSON object containing one key, "page_type", with one of the three values.

Page Text:
---
{page_text}
---

Example Output:
{{"page_type": "TABLE_OF_CONTENTS"}}
"""
    llm_response = call_llm(
        prompt,
        llm_config['model_name'],
        llm_config['base_url'],
        llm_config['api_key'],
        llm_config['provider']
    )
    
    json_string = _extract_json_from_llm_string(llm_response)
    if not json_string:
        return "AMBIGUOUS"

    try:
        response_data = json.loads(json_string)
        return response_data.get("page_type", "AMBIGUOUS")
    except json.JSONDecodeError:
        return "AMBIGUOUS"

# --- NEW: Wrapper function for parallel execution ---
def classify_page_wrapper(args: Tuple[int, str, Dict]) -> Dict:
    """
    A wrapper to call get_page_type_with_llm and return a structured dictionary.
    
    Args:
        args: A tuple containing (page_num, page_text, llm_config).
        
    Returns:
        A dictionary {'page': page_num, 'type': page_type}.
    """
    page_num, page_text, llm_config = args
    page_type = get_page_type_with_llm(page_text, llm_config)
    return {'page': page_num, 'type': page_type}

def run_classification_on_file(input_path: str, output_path: str, llm_config: Dict, max_workers: int = 4):
    """
    Finds the content start page by classifying pages in parallel and then
    programmatically finding the transition from ToC to Content.
    This version ensures an output file is always created.
    """
    pages = load_pages_from_json(input_path)
    if not pages:
        print(f"No pages found in {input_path}. Saving empty classification file.")
        # Ensure the output file is created, even if it's empty.
        save_results_to_json([], output_path)
        return

    sorted_pages = sorted(pages.keys(), key=int)
    
    print(f"Classifying initial pages with {max_workers} parallel workers...")
    
    # Step 1: Classify the first 20 pages in parallel.
    tasks = []
    # Slicing up to the 20th page, or fewer if the document is short.
    for page_num_str in sorted_pages[1:20]: 
        page_num = int(page_num_str)
        tasks.append((page_num, pages[page_num_str], llm_config))

    page_classifications = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a map of futures to their original task to handle results
        future_to_task = {executor.submit(classify_page_wrapper, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                page_classifications.append(result)
                print(f"  - Page {result['page']} classified as: {result['type']}")
            except Exception as exc:
                page_num = future_to_task[future][0]
                print(f"  - Page {page_num} generated an exception: {exc}")
                page_classifications.append({'page': page_num, 'type': 'AMBIGUOUS'})

    # CRITICAL: Sort results by page number as they may complete out of order.
    page_classifications.sort(key=lambda x: x['page'])

    # Step 2: Programmatically find the start page (this logic remains the same).
    start_page = len(pages) + 1
    for i in range(1, len(page_classifications)):
        prev_page_type = page_classifications[i-1]['type']
        curr_page_type = page_classifications[i]['type']
        
        if prev_page_type == 'TABLE_OF_CONTENTS' and curr_page_type == 'CONTENT_BODY':
            start_page = page_classifications[i]['page']
            print(f"\nTransition found! Main content starts on page {start_page}.\n")
            break
    
    if start_page > len(pages):
        # If no transition is found, default to page 2 to avoid classifying everything as ToC.
        print("\nWarning: Could not find a clear ToC -> Content transition. Defaulting content start to page 2.\n")
        start_page = 2

    # Step 3: Apply final classifications (this logic remains the same).
    results = []
    for page_num_str in sorted_pages:
        page_num = int(page_num_str)
        page_content = pages.get(page_num_str, '')
        
        subject, reasoning = "", ""
        if page_num == 1:
            subject = "Title Page"
            reasoning = "This is the first page of the document."
        elif page_num < start_page:
            subject = "Table of Contents"
            reasoning = f"This page appears before the main content, which starts on page {start_page}."
        else:
            subject = "Contents"
            reasoning = f"This page is part of the main document body, starting from page {start_page}."

        classification_json = json.dumps({"subject": subject, "reasoning": reasoning})
        results.append({
            'page': page_num,
            'classification': classification_json,
            'text': page_content,
        })
    
    save_results_to_json(results, output_path)
    print(f"Final classification results saved to {output_path}")

if __name__ == '__main__':
    llm_config = {
        "provider": "mission_assist",
        "model_name": "gemma3",
        "base_url": "http://devmissionassist.api.us.baesystems.com",
        "api_key": "aTOIT9hJM3DBYMQbEY"
    }
    
    input_file = os.path.join("..", "iris_ocr", "CM_Spec_OCR_and_figtab_output", 'raw_data', "S-133-06923_A_CUI.json")
    output_file = os.path.join("..", "results", "S-133-06923_A_CUI_classified.json")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Example of calling with the new max_workers parameter
    run_classification_on_file(input_file, output_file, llm_config, max_workers=4)