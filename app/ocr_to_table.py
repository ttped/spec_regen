"""
ocr_to_table.py

Converts the OCR pipeline JSON format (from the Tesseract-based table extractor)
into the complex_table_schema format for rendering in Word documents.

Lives in: root/app/

Pipeline:
  OCR JSON → consolidate_columns (sparse grid → logical columns) →
  build position-indexed cell grid → compute widths from FINAL cells →
  emit complex_table_schema dict

IMPORTANT: The cell list for each row is **position-indexed**, meaning
cells[c] corresponds directly to column c. Positions covered by a span
(rowspan or colspan) are None.  The renderer MUST consume cells by index.
"""

from typing import Dict, List, Optional, Tuple

from consolidate_columns import consolidate_ocr_data


# ---------------------------------------------------------------------------
# Approximate DXA per character for width estimation.
#
# At 10-11pt Calibri, a character averages ~100-120 DXA (~0.07-0.08 in).
# We use 110 as a middle ground. The total table width is typically 9360 DXA
# (US Letter with 1" margins), so "Acceptance" (10 chars) needs ~1100 DXA
# which is ~11.7% of page width — about right for a readable column.
#
# Cell padding in Word eats ~216 DXA (108 each side) on top of this.
# ---------------------------------------------------------------------------
_DXA_PER_CHAR = 110
_CELL_PADDING_DXA = 216
_TABLE_WIDTH_DXA = 9360


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


def _build_occupied_map(
    grid: List[List[Optional[Dict]]],
    n_rows: int,
    n_cols: int,
) -> List[List[Optional[Tuple[int, int]]]]:
    """
    Build an occupation map: occupied[r][c] = (origin_r, origin_c) if
    position (r, c) is covered by a spanning cell from that origin.
    """
    occupied: List[List[Optional[Tuple[int, int]]]] = [
        [None for _ in range(n_cols)] for _ in range(n_rows)
    ]

    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[r][c]
            if cell is None:
                continue
            rs = cell.get("row_span", 1)
            cs = cell.get("col_span", 1)
            for dr in range(rs):
                for dc in range(cs):
                    rr, cc = r + dr, c + dc
                    if rr < n_rows and cc < n_cols:
                        occupied[rr][cc] = (r, c)

    return occupied


def _build_position_indexed_rows(
    grid: List[List[Optional[Dict]]],
    occupied: List[List[Optional[Tuple[int, int]]]],
    n_rows: int,
    n_cols: int,
    header_row_indices: set,
    col_alignment: Dict,
    confidence_threshold: float,
    mark_low_confidence_italic: bool,
) -> List[Dict]:
    """
    Build position-indexed schema rows from the cell grid.

    cells[c] maps directly to physical column c:
      None → covered by a row/col span
      {"text": ""} → empty cell
      {"text": "...", ...} → content cell
    """
    schema_rows = []

    for r in range(n_rows):
        is_header = r in header_row_indices

        cells: List[Optional[Dict]] = [{"text": ""} for _ in range(n_cols)]

        # Mark spanned positions
        for c in range(n_cols):
            origin = occupied[r][c]
            if origin is not None and origin != (r, c):
                cells[c] = None

        # Place content at exact positions
        for c in range(n_cols):
            cell = grid[r][c]
            if cell is None:
                continue

            rs = cell.get("row_span", 1)
            cs = cell.get("col_span", 1)
            conf = cell.get("confidence", 1.0)
            text = cell.get("text", "")
            alignment = (
                cell.get("alignment")
                or col_alignment.get(str(c), "left")
            )

            schema_cell: Dict = {"text": text, "halign": alignment}

            if cs > 1:
                schema_cell["colspan"] = cs
            if rs > 1:
                schema_cell["rowspan"] = rs
            if is_header:
                schema_cell["bold"] = True
                schema_cell["shading"] = "D9E2F3"
            if conf < confidence_threshold and mark_low_confidence_italic:
                schema_cell["italic"] = True

            cells[c] = schema_cell

        schema_rows.append({"is_header": is_header, "cells": cells})

    return schema_rows


def _compute_widths_from_final_cells(
    schema_rows: List[Dict],
    n_cols: int,
    bbox_extents: Optional[List] = None,
    min_col_pct: float = 5.0,
) -> List[float]:
    """
    Compute column width percentages from the FINAL position-indexed cells.

    This is the single source of truth for widths because it sees
    post-consolidation text, including cells that were multi-span in the
    original grid but became single-span after column merging.

    Width for each column = max(bbox_hint, text_content_need, floor).

    Args:
        schema_rows: The finalized position-indexed rows.
        n_cols: Number of columns.
        bbox_extents: Optional pixel extents from consolidation.
        min_col_pct: Minimum width as percentage of total.
    """
    # Pass 1: find longest single-span text per column
    max_text_len = [0] * n_cols
    for row in schema_rows:
        for c, cell in enumerate(row.get("cells", [])):
            if cell is None or c >= n_cols:
                continue
            cs = cell.get("colspan", 1)
            if cs != 1:
                continue
            text = cell.get("text", "").strip()
            max_text_len[c] = max(max_text_len[c], len(text))

    # Pass 2: compute width needs in DXA
    col_dxa_needs = []
    for c in range(n_cols):
        # Content-based: characters * dxa_per_char + cell padding
        content_dxa = max_text_len[c] * _DXA_PER_CHAR + _CELL_PADDING_DXA

        # Bbox-based: pixel extent scaled to DXA (rough: 1px ≈ 7.2 DXA at 200dpi)
        bbox_dxa = 0
        if bbox_extents and c < len(bbox_extents) and bbox_extents[c]:
            ext = bbox_extents[c]
            px_width = ext[1] - ext[0] if isinstance(ext, (list, tuple)) else 0
            bbox_dxa = int(px_width * 7.2)

        # Floor: at least enough for a few characters
        floor_dxa = int(_TABLE_WIDTH_DXA * min_col_pct / 100)

        col_dxa_needs.append(max(content_dxa, bbox_dxa, floor_dxa))

    # Normalize to percentages
    total_dxa = sum(col_dxa_needs)
    pcts = [(need / total_dxa) * 100 for need in col_dxa_needs]

    # Enforce minimum percentage floor
    needs_boost = [i for i, p in enumerate(pcts) if p < min_col_pct]
    if needs_boost and len(needs_boost) < n_cols:
        deficit = sum(min_col_pct - pcts[i] for i in needs_boost)
        donors = [i for i in range(n_cols) if i not in needs_boost]
        donor_total = sum(pcts[i] for i in donors)

        for i in needs_boost:
            pcts[i] = min_col_pct

        if donor_total > 0:
            shrink = (donor_total - deficit) / donor_total
            for i in donors:
                pcts[i] *= shrink

    return [round(p, 1) for p in pcts]


def convert_ocr_to_table_schema(
    ocr_data: Dict,
    confidence_threshold: float = 0.3,
    mark_low_confidence_italic: bool = True,
) -> Dict:
    """
    Convert OCR pipeline JSON into the complex_table_schema format.

    Args:
        ocr_data: The full OCR JSON dict (already consolidated if desired).
        confidence_threshold: Cells below this confidence get flagged.
        mark_low_confidence_italic: If True, low-confidence cells render italic.

    Returns:
        A dict compatible with complex_table_schema.add_complex_table().
    """
    grid, n_rows, n_cols = _build_cell_grid(ocr_data)
    header_row_indices = set(ocr_data.get("header_rows", []))
    col_alignment = ocr_data.get("column_alignment", {})
    occupied = _build_occupied_map(grid, n_rows, n_cols)

    # Build the final cell grid first
    schema_rows = _build_position_indexed_rows(
        grid, occupied, n_rows, n_cols,
        header_row_indices, col_alignment,
        confidence_threshold, mark_low_confidence_italic,
    )

    # Compute widths from the FINAL cells (not from raw OCR extents)
    bbox_extents = ocr_data.get("_column_extents")
    pcts = _compute_widths_from_final_cells(
        schema_rows, n_cols, bbox_extents,
    )

    columns = [{"name": "", "width_pct": pcts[c]} for c in range(n_cols)]
    header_count = len(header_row_indices)

    return {
        "columns": columns,
        "rows": schema_rows,
        "header_rows": header_count,
    }


def convert_and_strip_empty(
    ocr_data: Dict,
    confidence_threshold: float = 0.3,
    consolidate: bool = True,
    gap_threshold_px: int = 50,
) -> Dict:
    """
    Convert OCR JSON to schema, optionally consolidating phantom columns
    first, then stripping fully-empty leading/trailing rows and columns.
    """
    if consolidate:
        ocr_data = consolidate_ocr_data(ocr_data, gap_threshold_px)

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
        return bool(cell.get("text", "").strip())

    # Find occupied column range
    col_has_data = [False] * n_cols
    for row in rows:
        for i, cell in enumerate(row.get("cells", [])):
            if i < n_cols and _cell_has_content(cell):
                col_has_data[i] = True

    first_col = next((i for i, v in enumerate(col_has_data) if v), 0)
    last_col = next(
        (i for i in range(n_cols - 1, -1, -1) if col_has_data[i]),
        n_cols - 1,
    )

    # Find occupied row range
    row_has_data = []
    for row in rows:
        cells = row.get("cells", [])
        has = any(
            _cell_has_content(cells[i])
            for i in range(first_col, last_col + 1)
            if i < len(cells)
        )
        row_has_data.append(has)

    first_row = next((i for i, v in enumerate(row_has_data) if v), 0)
    last_row = next(
        (i for i in range(n_rows - 1, -1, -1) if row_has_data[i]),
        n_rows - 1,
    )

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

    header_count = sum(1 for row in trimmed_rows if row.get("is_header"))

    # Renormalize percentages to sum to 100
    total_pct = sum(c.get("width_pct", 0) for c in trimmed_columns)
    if total_pct > 0:
        for col in trimmed_columns:
            col["width_pct"] = round((col["width_pct"] / total_pct) * 100, 1)

    return {
        "columns": trimmed_columns,
        "rows": trimmed_rows,
        "header_rows": header_count,
    }