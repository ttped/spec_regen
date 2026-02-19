"""
ocr_to_table.py

Converts the OCR pipeline JSON format (from the Tesseract-based table extractor)
into the complex_table_maker schema for rendering in Word documents.

Lives in: root/app/

The OCR JSON has:
  - n_rows, n_cols: grid dimensions
  - header_rows: list of row indices that are headers
  - column_alignment: {col_index_str: "left"|"center"|"right"}
  - rows: list of {row_index, cells: [{row, col, row_span, col_span, bbox, text, confidence, alignment}]}

The complex_table_maker schema expects:
  - columns: [{name, width_pct or width_dxa}]
  - rows: [{is_header, cells: [{text, colspan, rowspan, halign, bold, ...} | None]}]
  - header_rows: int count
"""

from typing import Dict, List, Optional, Tuple


def _build_cell_grid(
    ocr_data: Dict,
) -> Tuple[List[List[Optional[Dict]]], int, int]:
    """
    Flatten the OCR rows into a 2D grid of cell dicts.

    Returns:
        (grid, n_rows, n_cols) where grid[r][c] is the origin cell dict
        or None if that position is covered by a span or empty.
    """
    n_rows = ocr_data["n_rows"]
    n_cols = ocr_data["n_cols"]

    grid: List[List[Optional[Dict]]] = [
        [None for _ in range(n_cols)] for _ in range(n_rows)
    ]

    for row_entry in ocr_data.get("rows", []):
        for cell in row_entry.get("cells", []):
            r = cell["row"]
            c = cell["col"]
            if 0 <= r < n_rows and 0 <= c < n_cols:
                grid[r][c] = cell

    return grid, n_rows, n_cols


def _compute_column_widths(
    grid: List[List[Optional[Dict]]],
    n_rows: int,
    n_cols: int,
) -> List[Optional[Tuple[int, int]]]:
    """
    Infer column pixel extents [x_min, x_max] from bounding boxes.

    For each column, collects x1/x2 from cells that occupy exactly
    that column (col_span=1) and takes the median extent.

    Returns list of (x_min, x_max) per column, or None if no data.
    """
    col_x1s: Dict[int, List[int]] = {c: [] for c in range(n_cols)}
    col_x2s: Dict[int, List[int]] = {c: [] for c in range(n_cols)}

    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[r][c]
            if cell is None:
                continue
            cs = cell.get("col_span", 1)
            # Only use single-column cells for boundary estimation
            if cs == 1:
                x1, _, x2, _ = cell["bbox"]
                col_x1s[c].append(x1)
                col_x2s[c].append(x2)

    extents: List[Optional[Tuple[int, int]]] = []
    for c in range(n_cols):
        if col_x1s[c] and col_x2s[c]:
            x1 = sorted(col_x1s[c])[len(col_x1s[c]) // 2]
            x2 = sorted(col_x2s[c])[len(col_x2s[c]) // 2]
            extents.append((x1, x2))
        else:
            extents.append(None)

    return extents


def _extents_to_percentages(
    extents: List[Optional[Tuple[int, int]]],
    n_cols: int,
) -> List[float]:
    """
    Convert pixel extents to width percentages.

    Columns without data get equal share of the remaining space.
    """
    widths_px = []
    total_known = 0
    unknown_count = 0

    for ext in extents:
        if ext is not None:
            w = max(ext[1] - ext[0], 10)
            widths_px.append(w)
            total_known += w
        else:
            widths_px.append(None)
            unknown_count += 1

    # Estimate unknown columns as average of known ones
    avg_known = total_known / max(len(widths_px) - unknown_count, 1)
    for i, w in enumerate(widths_px):
        if w is None:
            widths_px[i] = avg_known

    total = sum(widths_px)
    return [round((w / total) * 100, 1) for w in widths_px]


def _is_occupied(
    grid: List[List[Optional[Dict]]],
    r: int,
    c: int,
    n_rows: int,
    n_cols: int,
) -> bool:
    """
    Check if grid position (r, c) is covered by a spanning cell
    originating from a different position.
    """
    for scan_r in range(r, -1, -1):
        for scan_c in range(c, -1, -1):
            cell = grid[scan_r][scan_c]
            if cell is None:
                continue
            rs = cell.get("row_span", 1)
            cs = cell.get("col_span", 1)
            if (scan_r + rs > r and scan_c + cs > c
                    and (scan_r, scan_c) != (r, c)):
                return True
            # Only need to check cells that could span to (r, c)
            if scan_c + cs <= c:
                continue
    return False


def convert_ocr_to_table_schema(
    ocr_data: Dict,
    confidence_threshold: float = 0.3,
    mark_low_confidence_italic: bool = True,
) -> Dict:
    """
    Convert OCR pipeline JSON into the complex_table_maker schema.

    Args:
        ocr_data: The full OCR JSON dict.
        confidence_threshold: Cells below this confidence get flagged.
        mark_low_confidence_italic: If True, low-confidence cells render italic.

    Returns:
        A dict compatible with complex_table_maker.add_complex_table().
    """
    grid, n_rows, n_cols = _build_cell_grid(ocr_data)
    header_row_indices = set(ocr_data.get("header_rows", []))
    col_alignment = ocr_data.get("column_alignment", {})

    # Column definitions
    extents = _compute_column_widths(grid, n_rows, n_cols)
    pcts = _extents_to_percentages(extents, n_cols)
    columns = [{"name": "", "width_pct": pcts[c]} for c in range(n_cols)]

    # Build rows
    schema_rows = []

    for r in range(n_rows):
        is_header = r in header_row_indices
        cells = []

        c = 0
        while c < n_cols:
            cell = grid[r][c]

            if cell is None:
                # Check if covered by a span from above/left
                if _is_occupied(grid, r, c, n_rows, n_cols):
                    cells.append(None)
                else:
                    cells.append({"text": ""})
                c += 1
                continue

            rs = cell.get("row_span", 1)
            cs = cell.get("col_span", 1)
            conf = cell.get("confidence", 1.0)
            text = cell.get("text", "")
            alignment = cell.get("alignment") or col_alignment.get(str(c), "left")

            schema_cell = {"text": text, "halign": alignment}

            if cs > 1:
                schema_cell["colspan"] = cs
            if rs > 1:
                schema_cell["rowspan"] = rs
            if is_header:
                schema_cell["bold"] = True
                schema_cell["shading"] = "D9E2F3"
            if conf < confidence_threshold and mark_low_confidence_italic:
                schema_cell["italic"] = True

            cells.append(schema_cell)

            # Skip columns covered by this cell's colspan
            c += cs
            continue

        schema_rows.append({
            "is_header": is_header,
            "cells": cells,
        })

    # Count header rows
    header_count = len(header_row_indices)

    return {
        "columns": columns,
        "rows": schema_rows,
        "header_rows": header_count,
    }


def convert_and_strip_empty(
    ocr_data: Dict,
    confidence_threshold: float = 0.3,
) -> Dict:
    """
    Convert OCR JSON and strip fully-empty leading/trailing rows and columns.

    This is useful since the OCR grid is often much larger than the actual
    table content (e.g. 23x19 grid but only a few columns have data).

    Returns a cleaned complex_table_maker schema dict.
    """
    schema = convert_ocr_to_table_schema(ocr_data, confidence_threshold)

    rows = schema["rows"]
    columns = schema["columns"]
    n_rows = len(rows)
    n_cols = len(columns)

    if n_rows == 0 or n_cols == 0:
        return schema

    def _cell_has_content(cell):
        if cell is None:
            return False
        text = cell.get("text", "")
        return bool(text.strip())

    # Find occupied column range
    col_has_data = [False] * n_cols
    for row in rows:
        for i, cell in enumerate(row.get("cells", [])):
            if i < n_cols and _cell_has_content(cell):
                col_has_data[i] = True

    first_col = next((i for i, v in enumerate(col_has_data) if v), 0)
    last_col = next((i for i in range(n_cols - 1, -1, -1) if col_has_data[i]), n_cols - 1)

    # Find occupied row range
    row_has_data = []
    for row in rows:
        cells = row.get("cells", [])
        has = any(_cell_has_content(cells[i]) for i in range(first_col, last_col + 1) if i < len(cells))
        row_has_data.append(has)

    first_row = next((i for i, v in enumerate(row_has_data) if v), 0)
    last_row = next((i for i in range(n_rows - 1, -1, -1) if row_has_data[i]), n_rows - 1)

    # Slice
    trimmed_columns = columns[first_col:last_col + 1]
    trimmed_rows = []
    for r in range(first_row, last_row + 1):
        old_row = rows[r]
        old_cells = old_row.get("cells", [])
        trimmed_cells = old_cells[first_col:last_col + 1]
        trimmed_rows.append({
            "is_header": old_row.get("is_header", False),
            "cells": trimmed_cells,
        })

    # Recount headers
    header_count = sum(1 for row in trimmed_rows if row.get("is_header"))

    # Renormalize column width percentages
    total_pct = sum(c.get("width_pct", 0) for c in trimmed_columns)
    if total_pct > 0:
        for col in trimmed_columns:
            col["width_pct"] = round((col["width_pct"] / total_pct) * 100, 1)

    return {
        "columns": trimmed_columns,
        "rows": trimmed_rows,
        "header_rows": header_count,
    }