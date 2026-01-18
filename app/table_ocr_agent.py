# table_ocr_agent.py

import os
import json
import base64
import io
from typing import Dict, Optional

from PIL import Image

# Utility for calling the LLM and extracting JSON from its response.
# Assumes these are in an accessible path (e.g., a 'utils' package or the same directory).
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
    print(f"    - OCR'ing table image with model '{llm_config['model_name']}'...")

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
In the example, the second row has two cell objects. The third row only has one, because the second cell from the row above ("Data (spans 2 rows)") has a rowspan of 2, occupying its space.
Be precise. The structure must be perfect for automated processing.
Provide ONLY the raw JSON object in your response. Do not include any other text, commentary, or markdown formatting.
Analyze the image and generate the JSON.
"""

    with io.BytesIO() as buffer:
        image.convert('RGB').save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    llm_response = call_multimodal_llm(
        prompt=prompt,
        image_base64=encoded_image,
        provider=llm_config['provider'],
        model_name=llm_config['model_name'],
        base_url=llm_config['base_url'],
        api_key=llm_config.get('api_key')
    )

    json_string = _extract_json_from_llm_string(llm_response)
    if not json_string:
        print("      - [Warning] LLM did not return a JSON object for the table.")
        return None

    try:
        table_data = json.loads(json_string)
    except json.JSONDecodeError:
        print(f"      - [Warning] LLM returned malformed JSON. Could not decode.")
        return None

    # Validate the new structure
    if 'rows' in table_data and isinstance(table_data['rows'], list):
        num_rows = len(table_data['rows'])
        print(f"      - Successfully extracted complex table with {num_rows} rows.")
        return table_data

    print("      - [Warning] Extracted JSON is missing the 'rows' key or it's not a list.")
    return None
