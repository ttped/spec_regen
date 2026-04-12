# filename: title_agent.py
"""
title_agent.py - Extracts the document title from title pages.

Slimmed down: the LLM now only extracts `document_title`, the one field
that is genuinely unique per document. All other fields (distribution
statement, export warning, destruction notice, etc.) are emitted as
placeholder constants for downstream code or a human reviewer to fill.
"""

import os
import json
from typing import List, Dict, Any

try:
    from .utils import _extract_json_from_llm_string, call_llm, save_results_to_json
except ImportError:
    from utils import _extract_json_from_llm_string, call_llm, save_results_to_json


PLACEHOLDER = ""

TITLE_PAGE_PLACEHOLDERS: Dict[str, Any] = {
    "approval_status": PLACEHOLDER,
    "approval_date": PLACEHOLDER,
    "distribution_statement": PLACEHOLDER,
    "distribution_date": PLACEHOLDER,
    "export_warning": PLACEHOLDER,
    "destruction_notice": PLACEHOLDER,
    "controlled_by": [],
    "cui_category": PLACEHOLDER,
    "point_of_contact": PLACEHOLDER,
    "document_date": "",
    "supersedes": "",
}


def extract_title_page_info(text: str, model_name: str, base_url: str, api_key: str, provider: str) -> Dict[str, Any]:
    """Extract only the document title; fill the rest with placeholders."""
    prompt = f"""You are a document analysis expert. Extract the document title from the title page text below.

**Title Page Text:**
---
{text}
---

Return a single JSON object with one key: "document_title".

The title should include the specification type (e.g., "PRIME ITEM DEVELOPMENT SPECIFICATION"), the item name, and any CI/part numbers. Extract the complete title block as a single string.

Example:
{{"document_title": "PRIME ITEM DEVELOPMENT SPECIFICATION FOR TEST STATION, GUIDED MISSILE COMPONENTS AN/DSM-175 CI 06903AA"}}

Return ONLY the JSON object, no other text."""

    llm_response = call_llm(prompt, model_name, base_url, api_key, provider)
    json_string = _extract_json_from_llm_string(llm_response)

    result: Dict[str, Any] = {"document_title": ""}
    if json_string:
        try:
            parsed = json.loads(json_string)
            result["document_title"] = parsed.get("document_title", "")
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON from title extraction. Response: {json_string}")

    # Consistent schema for downstream consumers.
    for k, v in TITLE_PAGE_PLACEHOLDERS.items():
        result.setdefault(k, v)
    return result


def run_title_extraction_on_file(input_path: str, output_path: str, llm_config: Dict):
    """Load classified pages, extract titles from title pages, save results."""
    if not os.path.exists(input_path):
        print(f"Error: Classified file not found at {input_path}. Skipping title extraction.")
        save_results_to_json([], output_path)
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            all_pages: List[Dict[str, Any]] = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {input_path}. Skipping.")
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
            print(f"Found Title Page on page {page.get('page')}. Extracting title...")
            extracted_info = extract_title_page_info(
                page.get('text', ''),
                llm_config['model_name'],
                llm_config['base_url'],
                llm_config['api_key'],
                llm_config['provider']
            )
            extracted_info['page_number'] = page.get('page')
            title_page_data.append(extracted_info)

    if title_page_data:
        save_results_to_json(title_page_data, output_path)
        print(f"Title page information saved to {output_path}")
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