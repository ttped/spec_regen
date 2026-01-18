# table_processor_agent.py
"""
Processes table elements by performing OCR on their images to extract structured data.

Handles the new data format with page_metadata and elements structure.
"""

import os
import json
from typing import List, Dict, Optional

from PIL import Image

# Import the OCR function from the agent we already created
try:
    from .table_ocr_agent import ocr_table_image_with_llm
except ImportError:
    from table_ocr_agent import ocr_table_image_with_llm


def run_table_processing_on_file(input_path: str, output_path: str, figures_base_path: str, doc_stem: str, llm_config: Dict):
    """
    Loads document elements, finds tables, performs OCR on their images,
    and saves a new JSON file with the structured table data embedded.

    Handles both old format (list of elements) and new format (dict with page_metadata and elements).

    Args:
        input_path: Path to the JSON file containing document elements.
        output_path: Path to save the new JSON file with table data.
        figures_base_path: The root directory where figure/table images are stored.
        doc_stem: The base name of the document, used to find the correct image subfolder.
        llm_config: Configuration for the LLM provider.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}. Skipping table processing for {doc_stem}.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both old and new data formats
    if isinstance(data, dict) and 'elements' in data:
        elements = data.get('elements', [])
        page_metadata = data.get('page_metadata', {})
        is_new_format = True
        print(f"  - Found {len(elements)} elements (new format with metadata)")
    else:
        elements = data if isinstance(data, list) else []
        page_metadata = {}
        is_new_format = False
        print(f"  - Found {len(elements)} elements (legacy format)")

    figure_image_folder = os.path.join(figures_base_path, doc_stem)
    processed_elements = []
    tables_found = 0
    tables_processed = 0

    print(f"  - Searching for tables...")
    for element in elements:
        if element.get("type") == "table":
            tables_found += 1
            image_filename = element.get("export", {}).get("image_file")
            
            if not image_filename:
                processed_elements.append(element)
                continue

            image_path = os.path.join(figure_image_folder, image_filename)
            
            if os.path.exists(image_path):
                print(f"    - Found table '{element.get('asset_id')}'. Processing image: {image_filename}")
                try:
                    with Image.open(image_path) as img:
                        # Call the OCR agent to get structured data
                        table_data = ocr_table_image_with_llm(img, llm_config)
                    
                    if table_data:
                        # Embed the structured data directly into the element
                        element["table_data"] = table_data
                        tables_processed += 1
                except Exception as e:
                    print(f"    - [Error] Failed to process table image: {e}")
            else:
                print(f"    - [Warning] Image not found for table '{element.get('asset_id')}': {image_path}")

        processed_elements.append(element)

    # Preserve the data structure format
    if is_new_format:
        output_data = {
            "page_metadata": page_metadata,
            "elements": processed_elements
        }
    else:
        output_data = processed_elements

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"  - Table processing complete. {tables_processed}/{tables_found} tables processed.")
    print(f"  - Results saved to {output_path}")