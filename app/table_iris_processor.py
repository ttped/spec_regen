"""
table_iris_processor.py - Process tables using IRIS table extraction JSONs.

Replaces the LLM-based table_processor_agent. For each table element in the
document stream, checks whether a pre-extracted IRIS JSON exists. If found,
converts it through reconstruct_to_df (JSON → Excel) then reads the Excel
back to populate table_data on the element. If not found, the element keeps
its image reference for fallback rendering in docx_writer.

Expected layout:
    table_jsons_dir/
        doc_name_page008_Table_0.json
        doc_name_page012_Table_1.json
        ...

Filenames match the YOLO crop image stems.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# IRIS JSON lookup (with fuzzy stem matching for space/underscore mismatches)
# ---------------------------------------------------------------------------

def _normalize_stem(name: str) -> str:
    """Collapse spaces, underscores, hyphens and case for fuzzy matching."""
    return name.replace(" ", "_").replace("-", "_").lower()


def _build_stem_index(table_jsons_dir: str) -> Dict[str, str]:
    """
    Scan the flat table_jsons directory once and build a normalized-stem → filepath map.
    This avoids repeated os.listdir calls per table element.
    """
    index = {}
    if not os.path.isdir(table_jsons_dir):
        return index

    for filename in os.listdir(table_jsons_dir):
        if not filename.endswith(".json"):
            continue
        stem = os.path.splitext(filename)[0]
        normalized = _normalize_stem(stem)
        index[normalized] = os.path.join(table_jsons_dir, filename)

    return index


def find_iris_json(image_stem: str, stem_index: Dict[str, str]) -> Optional[str]:
    """
    Look up an IRIS JSON path for a given YOLO crop image stem.

    Tries exact normalized match. Returns the file path or None.
    """
    normalized = _normalize_stem(image_stem)
    return stem_index.get(normalized)


# ---------------------------------------------------------------------------
# IRIS JSON → Excel conversion (calls reconstruct_to_df as subprocess)
# ---------------------------------------------------------------------------

def convert_iris_json_to_excel(
    iris_json_path: str,
    output_dir: str,
    reconstruct_script: str = "reconstruct_to_df.py",
) -> Optional[str]:
    """
    Run reconstruct_to_df.py in single-file mode to produce an Excel file.

    Returns the path to the generated .xlsx file, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    stem = Path(iris_json_path).stem
    # reconstruct_to_df outputs into: output_dir/<doc_name>/excel/<stem>.xlsx
    # We need to find it after the script runs.

    try:
        result = subprocess.run(
            [
                "python", reconstruct_script,
                iris_json_path,
                "--formats", "excel",
                "--out-dir", output_dir,
                "--no-meta-sheet",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"    [Warning] reconstruct_to_df failed for {stem}: {result.stderr[:300]}")
            return None
    except FileNotFoundError:
        print(f"    [Warning] reconstruct_to_df.py not found at: {reconstruct_script}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    [Warning] reconstruct_to_df timed out for {stem}")
        return None

    # Find the generated Excel file
    # The script creates: output_dir/<doc_name>/excel/<table_stem>.xlsx
    # Since we're in single-file mode, search for it
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".xlsx") and _normalize_stem(Path(f).stem) == _normalize_stem(stem):
                return os.path.join(root, f)

    print(f"    [Warning] Excel output not found after reconstruct_to_df for {stem}")
    return None


# ---------------------------------------------------------------------------
# Excel → table_data conversion
# ---------------------------------------------------------------------------

def read_excel_to_table_data(excel_path: str) -> Optional[Dict]:
    """
    Read an Excel file and convert it to the table_data format expected by
    complex_table_schema / docx_writer.

    Returns legacy-compatible format:
        {"columns": ["Header1", "Header2"], "rows": [["val1", "val2"], ...]}

    The first row of the Excel sheet is treated as the header row.
    Returns None if the file is empty or unreadable.
    """
    import openpyxl

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb.active

    all_rows = []
    for row in ws.iter_rows(values_only=True):
        all_rows.append([str(v) if v is not None else "" for v in row])

    if not all_rows:
        return None

    # First row = headers
    headers = all_rows[0]
    data_rows = all_rows[1:]

    if not headers or all(h == "" for h in headers):
        return None

    # Collect column width metadata for landscape decisions
    col_widths = []
    for col in range(1, ws.max_column + 1):
        col_letter = openpyxl.utils.get_column_letter(col)
        width = ws.column_dimensions[col_letter].width
        if width is None:
            width = 8.43
        col_widths.append(width)

    return {
        "columns": headers,
        "rows": data_rows,
        "_excel_col_widths": col_widths,
        "_num_columns": len(headers),
    }


# ---------------------------------------------------------------------------
# Direct IRIS JSON → table_data (skip Excel round-trip)
# ---------------------------------------------------------------------------

def read_iris_json_to_table_data(iris_json_path: str) -> Optional[Dict]:
    """
    Attempt to read an IRIS JSON directly and convert to table_data format,
    bypassing the Excel round-trip when reconstruct_to_df is unavailable.

    This is a fallback — the Excel path is preferred because reconstruct_to_df
    handles normalization, outlier cleanup, and header ontology mapping.
    """
    with open(iris_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # IRIS JSON structure varies, but commonly has a grid or cells structure.
    # This fallback handles the most common case where we can extract
    # columns and rows directly.

    # If the JSON already has columns/rows (pre-processed), use directly
    if isinstance(data, dict) and "columns" in data and "rows" in data:
        columns = data["columns"]
        rows = data["rows"]
        if isinstance(columns, list) and isinstance(rows, list):
            # Ensure columns are strings
            if columns and isinstance(columns[0], dict):
                columns = [c.get("name", "") for c in columns]
            return {
                "columns": [str(c) for c in columns],
                "rows": [[str(v) if v is not None else "" for v in row] for row in rows],
                "_num_columns": len(columns),
            }

    return None


# ---------------------------------------------------------------------------
# Per-element processing
# ---------------------------------------------------------------------------

def process_table_element(
    element: Dict,
    stem_index: Dict[str, str],
    excel_output_dir: str,
    reconstruct_script: str,
) -> Dict:
    """
    Process a single table element: look up IRIS JSON, convert to table_data.

    Modifies the element in place and returns it.
    """
    export_data = element.get("export") or {}
    image_file = export_data.get("image_file", "")

    if not image_file:
        return element

    image_stem = os.path.splitext(image_file)[0]
    iris_json_path = find_iris_json(image_stem, stem_index)

    if not iris_json_path:
        return element

    print(f"    [IRIS] Found table JSON for: {image_stem}")

    # Primary path: JSON → Excel → table_data
    excel_path = convert_iris_json_to_excel(
        iris_json_path, excel_output_dir, reconstruct_script
    )

    table_data = None
    if excel_path:
        table_data = read_excel_to_table_data(excel_path)
        if table_data:
            print(f"    [IRIS] Converted via Excel: {table_data['_num_columns']} columns, {len(table_data['rows'])} rows")

    # Fallback: read IRIS JSON directly
    if not table_data:
        print(f"    [IRIS] Excel path failed, attempting direct JSON read")
        table_data = read_iris_json_to_table_data(iris_json_path)
        if table_data:
            print(f"    [IRIS] Direct read: {table_data['_num_columns']} columns, {len(table_data['rows'])} rows")

    if table_data:
        element["table_data"] = table_data
        element["_table_source"] = "iris"

        # Flag wide tables for landscape rendering
        num_cols = table_data.get("_num_columns", 0)
        if num_cols > 7:
            element["_render_landscape"] = True
    else:
        print(f"    [IRIS] Could not extract table data for {image_stem}, will use image fallback")

    return element


# ---------------------------------------------------------------------------
# Main entry point (called from simple_pipeline.py step 6)
# ---------------------------------------------------------------------------

def run_iris_table_processing(
    input_path: str,
    output_path: str,
    table_jsons_dir: str,
    doc_stem: str,
    reconstruct_script: str = "reconstruct_to_df.py",
):
    """
    Process all table elements in a document's element stream using IRIS JSONs.

    Args:
        input_path: Path to the _with_assets.json file
        output_path: Path to write the _with_tables.json file
        table_jsons_dir: Flat directory containing IRIS table JSONs
        doc_stem: Document stem (used for Excel output subdirectory)
        reconstruct_script: Path to reconstruct_to_df.py
    """
    print(f"  - Reading elements: {input_path}")

    if not os.path.exists(input_path):
        print(f"  - [Error] Input file not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'elements' in data:
        elements = data.get('elements', [])
        page_metadata = data.get('page_metadata', {})
    else:
        elements = data if isinstance(data, list) else []
        page_metadata = {}

    # Build the IRIS JSON lookup index once
    stem_index = _build_stem_index(table_jsons_dir)
    print(f"  - IRIS table JSONs indexed: {len(stem_index)} files in {table_jsons_dir}")

    # Excel intermediate output goes alongside results
    excel_output_dir = os.path.join(
        os.path.dirname(output_path) or '.', "iris_excel", doc_stem
    )

    # Process each table element
    tables_found = 0
    tables_matched = 0

    for element in elements:
        if element.get("type") not in ("table", "table_layout"):
            continue

        tables_found += 1
        had_data_before = element.get("table_data") is not None

        process_table_element(
            element, stem_index, excel_output_dir, reconstruct_script
        )

        if not had_data_before and element.get("table_data") is not None:
            tables_matched += 1

    print(f"  - Tables in stream: {tables_found}")
    print(f"  - Tables matched to IRIS JSON: {tables_matched}")
    print(f"  - Tables falling back to image: {tables_found - tables_matched}")

    # Save output
    output_data = {
        "page_metadata": page_metadata,
        "elements": elements,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"  - Saved to: {output_path}")