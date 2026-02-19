"""
test_ocr_to_table.py

Loads OCR JSON, converts through ocr_to_table, and renders via
complex_table_maker to verify the full pipeline.

Lives in: root/tests/
Run from project root:
    python -m tests.test_ocr_to_table

Expects test data at: root/test_data/table_json_example.json
"""

import os
import json
from typing import Dict

import docx
from docx.shared import Pt

from app.ocr_to_table import convert_ocr_to_table_schema, convert_and_strip_empty
from app.complex_table_maker import add_complex_table


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def test_full_grid(ocr_data: Dict, doc):
    """Render the full grid as-is (no stripping)."""
    doc.add_paragraph("Full Grid (no trimming)", style="Heading 1")
    doc.add_paragraph(
        f"Grid: {ocr_data['n_rows']}x{ocr_data['n_cols']} | "
        f"Layout: {ocr_data.get('layout_type', '?')} | "
        f"Confidence: {ocr_data.get('confidence', 0):.1%}"
    )
    schema = convert_ocr_to_table_schema(ocr_data)
    add_complex_table(doc, schema)
    doc.add_paragraph()


def test_stripped_grid(ocr_data: Dict, doc):
    """Render with empty rows/columns stripped."""
    doc.add_paragraph("Stripped Grid (empty edges removed)", style="Heading 1")
    schema = convert_and_strip_empty(ocr_data)
    n_cols = len(schema["columns"])
    n_rows = len(schema["rows"])
    doc.add_paragraph(f"Trimmed to: {n_rows} rows x {n_cols} cols")
    add_complex_table(doc, schema)
    doc.add_paragraph()


def test_schema_dump(ocr_data: Dict):
    """Print converted schema summary for inspection."""
    schema = convert_and_strip_empty(ocr_data)

    total_cells = 0
    spanned_cells = 0
    low_conf = 0
    for row in schema["rows"]:
        for cell in row.get("cells", []):
            if cell is None:
                continue
            total_cells += 1
            if cell.get("colspan", 1) > 1 or cell.get("rowspan", 1) > 1:
                spanned_cells += 1
            if cell.get("italic"):
                low_conf += 1

    print(f"  Columns: {len(schema['columns'])}")
    print(f"  Rows: {len(schema['rows'])}")
    print(f"  Cells with content: {total_cells}")
    print(f"  Spanned cells: {spanned_cells}")
    print(f"  Low confidence (italic): {low_conf}")
    print(f"  Header rows: {schema['header_rows']}")

    return schema


def main():
    json_path = os.path.join(TEST_DATA_DIR, "table_json_example.json")

    with open(json_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    print(f"Loaded: {json_path}")
    print(f"Source grid: {ocr_data['n_rows']}x{ocr_data['n_cols']}")
    print()

    # Schema inspection
    print("Schema summary (stripped):")
    schema = test_schema_dump(ocr_data)
    print()

    # Build docx
    doc = docx.Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    doc.add_paragraph("OCR to Table Schema Test", style="Title")
    doc.add_paragraph(f"Source: {ocr_data.get('source_image', '?')}")
    doc.add_paragraph()

    test_full_grid(ocr_data, doc)
    test_stripped_grid(ocr_data, doc)

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    docx_path = os.path.join(OUTPUT_DIR, "ocr_table_test.docx")
    doc.save(docx_path)
    print(f"Docx: {docx_path}")

    schema_path = os.path.join(OUTPUT_DIR, "ocr_table_schema_output.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"Schema JSON: {schema_path}")


if __name__ == "__main__":
    main()