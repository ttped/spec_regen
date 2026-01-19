# filename: title_agent.py
"""
title_agent.py - Extracts structured information from document title pages.

Extracts:
- Document title (including CI numbers)
- Approval status with date
- Distribution statement with date
- Export warning
- Destruction notice
- Multiple CONTROLLED BY entries
- CUI category
- Point of contact (POC)
- LDP (Limited Distribution Program) info if present
- Document date/revision info
"""

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
Analyze the text and return a single JSON object with the following keys. If a field is not found, its value should be an empty string. Extract the COMPLETE text for each field, including any dates.

Fields to extract:

- "document_title": The main title block of the document. Include any specification type (e.g., "PRIME ITEM DEVELOPMENT SPECIFICATION"), the item name, and any CI/part numbers.

- "approval_status": The COMPLETE line about approval status, including the date. Look for text like "Approved by Configuration Control Board Directive dated..." Include any "DRAFT Date TBD" or actual dates.

- "approval_date": Just the date portion from the approval status (e.g., "DRAFT Date TBD" or "15 March 2024"). Extract this separately.

- "distribution_statement": The COMPLETE distribution statement, including any dates. Usually starts with "DISTRIBUTION STATEMENT" followed by a letter (A, B, C, D, etc.) and includes authorization details.

- "distribution_date": The date mentioned in the distribution statement (e.g., "DRAFT Date TBD" or an actual date).

- "export_warning": The full WARNING paragraph about export restrictions. Usually starts with "WARNING: THIS DOCUMENT CONTAINS TECHNICAL DATA..."

- "destruction_notice": Text about document destruction. Usually starts with "DESTRUCTION:" or "DESTROY BY..."

- "controlled_by": A list of ALL entities listed as controlling the document. Look for "CONTROLLED BY:" entries. Return as a JSON array of strings, e.g., ["USAF", "AFNWC/NIM"].

- "cui_category": The CUI (Controlled Unclassified Information) category. Look for "CUI CATEGORY:" followed by the category type.

- "point_of_contact": The POC information. Look for "POC:" followed by name/email/organization.

- "document_date": Any document date or revision date found on the title page.

- "supersedes": Any information about what this document supersedes.

**Example Output Format:**

{{
  "document_title": "PRIME ITEM DEVELOPMENT SPECIFICATION FOR TEST STATION, GUIDED MISSILE COMPONENTS AN/DSM-175 CI 06903AA",
  "approval_status": "Approved by Configuration Control Board Directive dated DRAFT Date TBD.",
  "approval_date": "DRAFT Date TBD",
  "distribution_statement": "DISTRIBUTION STATEMENT D: DISTRIBUTION AUTHORIZED TO DEPARTMENT OF DEFENSE AND U.S. DOD CONTRACTORS ONLY FOR CONTROLLED TECHNICAL INFORMATION & EXPORT CONTROLLED: DETERMINED DRAFT Date TBD. OTHER REQUESTS FOR THIS DOCUMENT MUST BE REFERRED TO MINUTEMAN III PROGRAM OFFICE, HILL AFB, UT. 84056.",
  "distribution_date": "DRAFT Date TBD",
  "export_warning": "WARNING: THIS DOCUMENT CONTAINS TECHNICAL DATA WHOSE EXPORT IS RESTRICTED BY THE ARMS EXPORT CONTROL ACT...",
  "destruction_notice": "DESTRUCTION: DESTROY BY ANY METHOD THAT WILL PREVENT DISCLOSURE OF CONTENTS OR RECONSTRUCTION OF THE DOCUMENT.",
  "controlled_by": ["USAF", "AFNWC/NIM"],
  "cui_category": "CONTROLLED TECHNICAL INFORMATION / EXPORT CONTROLLED",
  "point_of_contact": "AFNWC Configuration Management (afnwc.nies.icbm.conm@us.af.mil)",
  "document_date": "",
  "supersedes": ""
}}

Return ONLY the JSON object, no other text.
"""
    llm_response = call_llm(prompt, model_name, base_url, api_key, provider)
    json_string = _extract_json_from_llm_string(llm_response)

    if json_string:
        try:
            result = json.loads(json_string)
            
            # Normalize controlled_by to always be a list
            controlled_by = result.get('controlled_by', '')
            if isinstance(controlled_by, str):
                # Split on common separators if it's a string
                if controlled_by:
                    result['controlled_by'] = [c.strip() for c in controlled_by.replace(',', ';').split(';') if c.strip()]
                else:
                    result['controlled_by'] = []
            elif not isinstance(controlled_by, list):
                result['controlled_by'] = []
            
            return result
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON from title page extraction. Response: {json_string}")
            return {}
    return {}


def run_title_extraction_on_file(input_path: str, output_path: str, llm_config: Dict):
    """
    Loads classified page data, extracts detailed information from title pages,
    and saves the results to a new JSON file.
    """
    if not os.path.exists(input_path):
        print(f"Error: Classified file not found at {input_path}. Skipping title extraction.")
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


if __name__ == '__main__':
    import sys
    
    llm_config = {
        "provider": "mission_assist",
        "model_name": "gemma3",
        "base_url": "http://devmissionassist.api.us.baesystems.com",
        "api_key": "aTOIT9hJM3DBYMQbEY"
    }
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "title_extraction_result.json"
        run_title_extraction_on_file(input_file, output_file, llm_config)
    else:
        print("Usage: python title_agent.py <input.json> [output.json]")