# table_processor_agent.py
"""
Processes table elements by optionally performing OCR on their images to extract structured data.

This module:
- Finds table elements in the document
- Optionally uses a multimodal LLM to OCR table images into structured data
- Can skip OCR and render tables as images (for Table of Tables support in Word)

Handles the new data format with page_metadata and elements structure.
"""

import os
import json
import base64
import io
from typing import List, Dict, Optional

from PIL import Image

# Import utilities for LLM calls
try:
    from .llm_ocr_utils import call_multimodal_llm
    from .utils import _extract_json_from_llm_string
except ImportError:
    from llm_ocr_utils import call_multimodal_llm
    from utils import _extract_json_from_llm_string


def ocr_table_image_with_llm(image: Image.Image, llm_config: Dict) -> Optional[Dict]:
    """
    Performs OCR on a table image using a multimodal LLM and requests a structured JSON output.

    Args:
        image: The PIL Image object of the table.
        llm_config: Configuration for the LLM provider, model, base URL, and API key.

    Returns:
        A dictionary with 'columns' and 'rows' keys if successful, otherwise None.
    """
    print(f"      - OCR'ing table image with model '{llm_config['model_name']}'...")

    prompt = """
You are an expert AI assistant specializing in extracting structured data from tables in images.
Your task is to analyze the provided image of a table and convert its content into a structured JSON format.

**Instructions:**
1.  **Identify Column Headers:** Accurately identify the text for each column header.
2.  **Extract Rows:** Extract each row of data. The number of items in each row must match the number of column headers.
3.  **Handle Merged Cells:** If a cell spans multiple rows (a merged cell), repeat its value for each row it covers.
4.  **Handle Empty Cells:** Represent empty cells with an empty string `""`.
5.  **Output Format:** Return a single, valid JSON object with two keys:
    - `columns`: A list of strings, where each string is a column header.
    - `rows`: A list of lists, where each inner list represents a row and its values correspond to the columns.

**Example:**
{
    "columns": ["Part No.", "Description", "QTY"],
    "rows": [
        ["123-456", "Screw, Pan Head", "4"],
        ["789-012", "Washer, Flat", "4"]
    ]
}

Important Notes:
Be precise. The structure must be perfect for automated processing.
Provide ONLY the raw JSON object in your response. Do not include any other text, commentary, or markdown formatting.
Analyze the image and generate the JSON.
"""

    # Convert image to base64
    with io.BytesIO() as buffer:
        image.convert('RGB').save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Call the multimodal LLM
    llm_response = call_multimodal_llm(
        prompt=prompt,
        image_base64=encoded_image,
        provider=llm_config['provider'],
        model_name=llm_config['model_name'],
        base_url=llm_config['base_url'],
        api_key=llm_config.get('api_key')
    )

    # Extract JSON from response
    json_string = _extract_json_from_llm_string(llm_response)
    if not json_string:
        print("      - [Warning] LLM did not return a JSON object for the table.")
        return None

    try:
        table_data = json.loads(json_string)
    except json.JSONDecodeError:
        print(f"      - [Warning] LLM returned malformed JSON. Could not decode.")
        return None

    # Validate the structure
    if 'rows' in table_data and isinstance(table_data['rows'], list):
        num_rows = len(table_data['rows'])
        print(f"      - Successfully extracted table with {num_rows} rows.")
        return table_data

    print("      - [Warning] Extracted JSON is missing the 'rows' key or it's not a list.")
    return None


def run_table_processing_on_file(
    input_path: str, 
    output_path: str, 
    figures_base_path: str, 
    doc_stem: str, 
    llm_config: Dict,
    use_llm_ocr: bool = True
):
    """
    Loads document elements, finds tables, optionally performs OCR on their images,
    and saves a new JSON file with the table data embedded.

    When use_llm_ocr=False, tables are passed through without OCR - they will
    be rendered as images in the final document but still get Table captions
    (for Table of Tables support).

    Args:
        input_path: Path to the JSON file containing document elements.
        output_path: Path to save the new JSON file with table data.
        figures_base_path: The root directory where figure/table images are stored.
        doc_stem: The base name of the document, used to find the correct image subfolder.
        llm_config: Configuration for the LLM provider.
        use_llm_ocr: If True, use LLM to OCR tables into structured data. 
                     If False, skip OCR and keep tables as images.
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

    if not use_llm_ocr:
        print(f"  - LLM OCR disabled. Tables will be rendered as images.")
    
    print(f"  - Searching for tables...")
    for element in elements:
        if element.get("type") == "table":
            tables_found += 1
            image_filename = element.get("export", {}).get("image_file")
            
            if not image_filename:
                processed_elements.append(element)
                continue

            image_path = os.path.join(figure_image_folder, image_filename)
            
            if use_llm_ocr and os.path.exists(image_path):
                print(f"    - Found table '{element.get('asset_id')}'. Processing image: {image_filename}")
                try:
                    with Image.open(image_path) as img:
                        # Call the OCR function to get structured data
                        table_data = ocr_table_image_with_llm(img, llm_config)
                    
                    if table_data:
                        # Embed the structured data directly into the element
                        element["table_data"] = table_data
                        tables_processed += 1
                except Exception as e:
                    print(f"    - [Error] Failed to process table image: {e}")
            elif not use_llm_ocr:
                # Just note that we're skipping OCR
                print(f"    - Table '{element.get('asset_id')}' will use image: {image_filename}")
                # Mark it explicitly so docx_writer knows this is intentional
                element["_render_as_image"] = True
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
    
    if use_llm_ocr:
        print(f"  - Table processing complete. {tables_processed}/{tables_found} tables OCR'd.")
    else:
        print(f"  - Table processing complete. {tables_found} tables found (rendered as images).")
    print(f"  - Results saved to {output_path}")


if __name__ == '__main__':
    import sys
    
    llm_config = {
        "provider": "mission_assist",
        "model_name": "gemma3",
        "base_url": "http://devmissionassist.api.us.baesystems.com",
        "api_key": "aTOIT9hJM3DBYMQbEY"
    }
    
    if len(sys.argv) >= 4:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        figures_dir = sys.argv[3]
        doc_stem = sys.argv[4] if len(sys.argv) > 4 else "document"
        use_ocr = "--no-ocr" not in sys.argv
        
        run_table_processing_on_file(input_file, output_file, figures_dir, doc_stem, llm_config, use_llm_ocr=use_ocr)
    else:
        print("Usage: python table_processor_agent.py <input.json> <output.json> <figures_dir> [doc_stem] [--no-ocr]")