# filename: title_agent.py
import os
import json
from typing import List, Dict, Any

try:
    from .utils import _extract_json_from_llm_string, call_llm, save_results_to_json
except ImportError:
    from utils import _extract_json_from_llm_string, call_llm, save_results_to_json

def extract_title_page_info(text: str, model_name: str, base_url: str, api_key: str, provider: str) -> Dict[str, Any]:
    """
    Extracts detailed, structured information from a document's title page using an LLM.
    This version is enhanced to capture more specific fields based on real-world examples.

    Args:
        text: The raw text content of the title page.
        model_name: The name of the language model to use.
        base_url: The base URL for the LLM API.
        api_key: The API key for the LLM service.
        provider: The provider of the LLM service.

    Returns:
        A dictionary containing the extracted information.
    """
    prompt = f"""
You are a document analysis expert specializing in government and defense specifications. Your task is to extract detailed, structured information from the text of a document's title page.

**Title Page Text to Analyze:**
---
{text}
---

**Instructions:**
Analyze the text and return a single JSON object with the following keys. If a field is not found, its value should be an empty string.

- "document_title": The main title block of the document, including any CI numbers.
- "approval_status": The line indicating the document's approval status (e.g., "Approved by Configuration Control Board...").
- "distribution_statement": The primary distribution statement (e.g., "DISTRIBUTION STATEMENT D...").
- "export_warning": The full paragraph that begins with "WARNING: THIS DOCUMENT CONTAINS TECHNICAL DATA...".
- "destruction_notice": The full paragraph that begins with "DESTRUCTION:".
- "controlled_by": The entity or entities listed after "CONTROLLED BY:".
- "cui_category": The value associated with "CUI CATEGORY:".
- "point_of_contact": The value associated with "POC:".

**Example Output Format:**

{{
  "document_title": "PRIME ITEM DEVELOPMENT SPECIFICATION FOR TEST STATION, GUIDED MISSLE COMPONENTS AN/DSM-175 CI 06903AA",
  "approval_status": "Approved by Configuration Control Board Directive dated DRAFT Date TBD.",
  "distribution_statement": "DISTRIBUTION STATEMENT D: DISTRIBUTION AUTHORIZED TO DEPARTMENT OF DEFENSE AND U.S. DOD CONTRACTORS ONLY FOR CONTROLLED TECHNICAL INFORMATION & EXPORT CONTROLLED: DETERMINED DRAFT Date TBD. OTHER REQUESTS FOR THIS DOCUMENT MUST BE REFERRED TO MINUTEMAN III PROGRAM OFFICE, HILL AFB, UT. 84056.",
  "export_warning": "WARNING: THIS DOCUMENT CONTAINS TECHNICAL DATA WHOSE EXPORT IS RESTRICTED BY THE ARMS EXPORT CONTROL ACT (SECTION 2751 OF TITLE 22, UNITED STATES CODE) OR THE EXPORT CONTROL REFORM ACT OF 2018 (CHAPTER 58 SECTIONS 4801-4852 OF TITLE 50, UNITED STATES CODE). VIOLATIONS OF THESE EXPORT-CONTROL LAWS ARE SUBJECT TO SEVERE CRIMINAL PENALTIES. DISSEMINATE IN ACCORDANCE WITH PROVISIONS OF DOD DIRECTIVE 5230.25 AND DOD INSTRUCTION 2040.02.",
  "destruction_notice": "DESTROY BY ANY METHOD THAT WILL PREVENT DISCLOSURE OF CONTENTS OR RECONSTRUCTION OF THE DOCUMENT.",
  "controlled_by": "USAF, AFNWC/NIM",
  "cui_category": "CONTROLLED TECHNICAL INFORMATION/ EXPORT CONTROLLED",
  "point_of_contact": "AFNWC Configuration Management (afnwc.nies.icbm.conm@us.af.mil)"
}}

"""
    llm_response = call_llm(prompt, model_name, base_url, api_key, provider)
    json_string = _extract_json_from_llm_string(llm_response)

    if json_string:
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON from title page extraction. Response: {json_string}")
            return {}
    return {}

def run_title_extraction_on_file(input_path: str, output_path: str, llm_config: Dict):
    """
    Loads classified page data, extracts detailed information from title pages,
    and saves the results to a new JSON file. This version is robust against
    empty or missing input files.
    """
    if not os.path.exists(input_path):
        print(f"Error: Classified file not found at {input_path}. Skipping title extraction.")
        # Create an empty output file to prevent downstream errors
        save_results_to_json([], output_path)
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            all_pages: List[Dict[str, Any]] = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {input_path}. It may be empty or malformed. Skipping.")
            all_pages = []

    if not all_pages:
        print(f"No pages found in {input_path}. Saving empty title data file.")
        save_results_to_json([], output_path)
        return

    title_page_data = []
    for page in all_pages:
        try:
            classification_data = json.loads(page.get('classification', '{}'))
            subject = classification_data.get('subject')
        except json.JSONDecodeError:
            subject = None

        if subject == "Title Page":
            print(f"Found Title Page on page {page.get('page')}. Extracting details...")
            page_text = page.get('text', '')
            extracted_info = extract_title_page_info(
                page_text,
                llm_config['model_name'],
                llm_config['base_url'],
                llm_config['api_key'],
                llm_config['provider']
            )
            
            if extracted_info:
                extracted_info['page_number'] = page.get('page')
                title_page_data.append(extracted_info)

    if title_page_data:
        save_results_to_json(title_page_data, output_path)
        print(f"Title page information successfully extracted and saved to {output_path}")
    else:
        print(f"No title pages found in {input_path}. Saving empty title data file.")
        save_results_to_json([], output_path)
