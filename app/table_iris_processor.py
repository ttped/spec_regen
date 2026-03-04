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
import re
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# IRIS JSON lookup — bridges two different naming conventions:
#   YOLO crops:  {doc}_tab_p{page}_{id}.jpg     e.g. file_name_tab_p045_003.jpg
#   IRIS JSONs:  {doc}_page_{page}tab_{id}.json  e.g. file_name_page_045tab_003.json
#
# Both are decomposed into a canonical key (doc, page, table_id) for matching.
# Also handles exact stem matches and space/underscore normalization.
# ---------------------------------------------------------------------------

# Pattern for YOLO crop names: {doc}_tab_p{page}_{id}
_YOLO_PATTERN = re.compile(
    r'^(.+?)_tab_p(\d+)_(\d+)$',
    re.IGNORECASE,
)

# Pattern for IRIS JSON names: {doc}_page{page}_tab_{id}
_IRIS_PATTERN = re.compile(
    r'^(.+?)_page_?(\d+)_?tab_(\d+)$',
    re.IGNORECASE,
)


def _normalize_stem(name: str) -> str:
    """Collapse spaces, underscores, hyphens and case for fuzzy matching."""
    return name.replace(" ", "_").replace("-", "_").lower()


def _extract_canonical_key(stem: str) -> Optional[tuple]:
    """
    Extract (normalized_doc, page_int, table_id_int) from either naming convention.
    Returns None if the stem doesn't match any known pattern.
    """
    normalized = _normalize_stem(stem)

    # Try YOLO pattern: {doc}_tab_p{page}_{id}
    m = _YOLO_PATTERN.match(normalized)
    if m:
        doc = m.group(1)
        page = int(m.group(2))
        table_id = int(m.group(3))
        return (doc, page, table_id)

    # Try IRIS pattern: {doc}_page_{page}tab_{id}
    m = _IRIS_PATTERN.match(normalized)
    if m:
        doc = m.group(1)
        page = int(m.group(2))
        table_id = int(m.group(3))
        return (doc, page, table_id)

    return None


def _build_stem_index(table_jsons_dir: str) -> Dict[str, str]:
    """
    Scan the flat table_jsons directory and build lookup maps.

    Builds two indexes for maximum matching coverage:
      1. canonical key (doc, page, table_id) → filepath
      2. normalized stem → filepath  (fallback for exact-ish matches)
    """
    canonical_index = {}
    stem_index = {}

    if not os.path.isdir(table_jsons_dir):
        return {}

    for filename in os.listdir(table_jsons_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(table_jsons_dir, filename)
        stem = os.path.splitext(filename)[0]

        # Canonical key index
        key = _extract_canonical_key(stem)
        if key:
            canonical_index[key] = filepath

        # Normalized stem index (fallback)
        stem_index[_normalize_stem(stem)] = filepath

    # Store both indexes in a single dict with a sentinel key
    return {
        "_canonical": canonical_index,
        "_stem": stem_index,
    }


def find_iris_json(image_stem: str, index: Dict[str, str]) -> Optional[str]:
    """
    Look up an IRIS JSON path for a given YOLO crop image stem.

    Tries canonical key match first (handles different naming conventions),
    then falls back to normalized stem match.
    """
    canonical_index = index.get("_canonical", {})
    stem_index = index.get("_stem", {})

    # Try canonical key match (bridges YOLO ↔ IRIS naming)
    key = _extract_canonical_key(image_stem)
    if key and key in canonical_index:
        return canonical_index[key]

    # Fallback: direct normalized stem match
    normalized = _normalize_stem(image_stem)
    return stem_index.get(normalized)


# ---------------------------------------------------------------------------
# IRIS JSON → complex_table_schema rich format (direct conversion, no Excel)
# ---------------------------------------------------------------------------

def read_iris_json_to_table_data(iris_json_path: str) -> Optional[Dict]:
    """
    Convert an IRIS table extraction JSON directly into the rich format
    expected by complex_table_schema.

    IRIS JSON structure:
        {
            "n_rows": int, "n_cols": int,
            "header_rows": [0, 1, ...],
            "column_alignment": {"col_idx": "left"|"right"|"center", ...},
            "rows": [
                {
                    "row_index": int,
                    "cells": [
                        {
                            "row": int, "col": int,
                            "row_span": int, "col_span": int,
                            "text": str,
                            "confidence": float,
                            "alignment": str
                        }, ...
                    ]
                }, ...
            ],
            "table_title": str (optional),
            "context": {"header_rows": int, ...} (optional)
        }

    Returns complex_table_schema rich format:
        {
            "columns": [{"name": ""}, ...],
            "header_rows": int,
            "rows": [
                {"is_header": bool, "cells": [
                    {"text": str, "colspan": int, "rowspan": int,
                     "halign": str, "bold": bool} | null, ...
                ]}, ...
            ]
        }
    """
    with open(iris_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    n_rows = data.get("n_rows", 0)
    n_cols = data.get("n_cols", 0)
    iris_rows = data.get("rows", [])

    if n_rows == 0 or n_cols == 0 or not iris_rows:
        return None

    # Determine which rows are headers
    header_row_indices = set(data.get("header_rows", []))
    # Also check context block
    context = data.get("context", {})
    context_header_count = context.get("header_rows", 0)

    # Column alignment defaults
    col_alignment = data.get("column_alignment", {})

    # Build column definitions (IRIS doesn't provide column names directly,
    # but complex_table_schema doesn't require them for rendering)
    columns = [{"name": ""} for _ in range(n_cols)]

    # Build a grid to track which cells are covered by spans
    covered = [[False] * n_cols for _ in range(n_rows)]

    # Sort IRIS rows by row_index
    iris_rows_sorted = sorted(iris_rows, key=lambda r: r.get("row_index", 0))

    # Convert each IRIS row to complex_table_schema format
    output_rows = []

    for iris_row in iris_rows_sorted:
        row_idx = iris_row.get("row_index", 0)
        is_header = row_idx in header_row_indices

        # Build cell list for this row — initialize all positions as None (covered)
        row_cells = [None] * n_cols

        for cell_data in iris_row.get("cells", []):
            col = cell_data.get("col", 0)
            row_span = cell_data.get("row_span", 1)
            col_span = cell_data.get("col_span", 1)
            text = cell_data.get("text", "")
            alignment = cell_data.get("alignment", "left")
            confidence = cell_data.get("confidence", 0)

            # Map IRIS alignment to complex_table_schema halign
            halign = alignment if alignment in ("left", "center", "right") else "left"

            # Build the cell dict
            cell = {
                "text": str(text) if text is not None else "",
                "halign": halign,
            }

            if col_span > 1:
                cell["colspan"] = col_span
            if row_span > 1:
                cell["rowspan"] = row_span

            # Bold headers
            if is_header:
                cell["bold"] = True

            # Place in grid
            if 0 <= col < n_cols:
                row_cells[col] = cell

            # Mark spanned cells as covered
            for dr in range(row_span):
                for dc in range(col_span):
                    r, c = row_idx + dr, col + dc
                    if (r, c) != (row_idx, col) and 0 <= r < n_rows and 0 <= c < n_cols:
                        covered[r][c] = True

        output_row = {"cells": row_cells}
        if is_header:
            output_row["is_header"] = True

        output_rows.append(output_row)

    # Determine header_rows count for the schema
    num_header_rows = max(context_header_count, len(header_row_indices))

    result = {
        "columns": columns,
        "rows": output_rows,
        "header_rows": num_header_rows,
        "_num_columns": n_cols,
        "_iris_confidence": data.get("confidence"),
        "_iris_structure_confidence": data.get("structure_confidence"),
        "_iris_table_title": data.get("table_title"),
    }

    return result


# ---------------------------------------------------------------------------
# Per-element processing
# ---------------------------------------------------------------------------

def process_table_element(
    element: Dict,
    stem_index: Dict[str, str],
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

    table_data = read_iris_json_to_table_data(iris_json_path)

    if table_data:
        num_cols = table_data.get("_num_columns", 0)
        num_rows = len(table_data.get("rows", []))
        confidence = table_data.get("_iris_confidence", "?")
        title = table_data.get("_iris_table_title", "")
        print(f"    [IRIS] Converted: {num_cols} columns, {num_rows} rows, confidence={confidence}")
        if title:
            print(f"    [IRIS] Table title: {title}")

        element["table_data"] = table_data
        element["_table_source"] = "iris"

        # Flag wide tables for landscape rendering
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
):
    """
    Process all table elements in a document's element stream using IRIS JSONs.

    For each table element, looks up the corresponding IRIS JSON by matching
    the YOLO crop image stem. If found, converts directly to complex_table_schema
    rich format. If not found, the element retains its image for fallback rendering.

    Args:
        input_path: Path to the _with_assets.json file
        output_path: Path to write the _with_tables.json file
        table_jsons_dir: Flat directory containing IRIS table JSONs
        doc_stem: Document stem (for logging)
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
    canonical_count = len(stem_index.get("_canonical", {}))
    stem_count = len(stem_index.get("_stem", {}))
    print(f"  - IRIS table JSONs indexed: {stem_count} files ({canonical_count} with canonical keys) in {table_jsons_dir}")

    # Process each table element
    tables_found = 0
    tables_matched = 0

    for element in elements:
        if element.get("type") not in ("table", "table_layout"):
            continue

        tables_found += 1
        had_data_before = element.get("table_data") is not None

        process_table_element(element, stem_index)

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