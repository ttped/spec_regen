"""
ocr_to_table.py

Converts the OCR pipeline JSON format (from the Tesseract-based table extractor)
into the complex_table_maker schema for rendering in Word documents.

Lives in: root/app/

Pipeline:
  OCR JSON → consolidate_columns (sparse grid → logical columns) →
  convert_ocr_to_table_schema → complex_table_maker schema

IMPORTANT: The cell list for each row is **position-indexed**, meaning
cells[c] corresponds directly to column c. Positions covered by a span
(rowspan or colspan) are None. The renderer in complex_table_maker must
consume cells by index, NOT by iterator.

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

from consolidate_columns import consolidate_ocr_data, compute_consolidated_widths_pct


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

    Every position touched by a cell (including the origin itself) is
    recorded.  Positions with no cell at all remain None.
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


def _compute_column_widths(
    grid: List[List[Optional[Dict]]],
    n_rows: int,
    n_cols: int,
) -> List[Optional[Tuple[int, int]]]:
    """
    Infer column pixel extents [x_min, x_max] from bounding boxes.

    Uses single-span cells only for clean measurement.
    """
    col_x1s: Dict[int, List[int]] = {c: [] for c in range(n_cols)}
    col_x2s: Dict[int, List[int]] = {c: [] for c in range(n_cols)}

    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[r][c]
            if cell is None:
                continue
            cs = cell.get("col_span", 1)
            if cs == 1:
                bbox = cell.get("bbox")
                if bbox and len(bbox) >= 4:
                    x1, _, x2, _ = bbox
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


def _distribute_spanning_widths(
    grid: List[List[Optional[Dict]]],
    extents: List[Optional[Tuple[int, int]]],
    n_rows: int,
    n_cols: int,
) -> List[Optional[Tuple[int, int]]]:
    """
    For columns with no single-span width data, estimate from
    multi-span cells that cross them.
    """
    result = list(extents)

    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[r][c]
            if cell is None:
                continue
            cs = cell.get("col_span", 1)
            if cs <= 1:
                continue

            bbox = cell.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            cell_x1, _, cell_x2, _ = bbox
            cell_width = cell_x2 - cell_x1

            spanned_cols = range(c, min(c + cs, n_cols))
            missing = [sc for sc in spanned_cols if result[sc] is None]

            if not missing:
                continue

            known_width = 0
            for sc in spanned_cols:
                if result[sc] is not None:
                    known_width += result[sc][1] - result[sc][0]

            remaining = max(cell_width - known_width, len(missing) * 10)
            per_missing = remaining // len(missing)

            cursor = cell_x1
            for sc in spanned_cols:
                if result[sc] is not None:
                    cursor = result[sc][1]
                elif sc in missing:
                    result[sc] = (cursor, cursor + per_missing)
                    cursor += per_missing

    return result


def _extents_to_percentages(
    extents: List[Optional[Tuple[int, int]]],
    n_cols: int,
    max_text_lengths: Optional[List[int]] = None,
    min_col_pct: float = 5.0,
) -> List[float]:
    """
    Convert pixel extents to width percentages with content-aware minimums.

    Args:
        extents: Pixel (x_min, x_max) per column, or None.
        n_cols: Number of columns.
        max_text_lengths: Longest text per column (character count).
        min_col_pct: Absolute minimum column width percentage.
    """
    approx_px_per_char = 12

    widths_px = []
    for i, ext in enumerate(extents):
        bbox_width = max(ext[1] - ext[0], 10) if ext is not None else 0

        # Content-aware minimum
        content_min = 0
        if max_text_lengths and i < len(max_text_lengths):
            content_min = max_text_lengths[i] * approx_px_per_char

        widths_px.append(max(bbox_width, content_min, 20))

    total = sum(widths_px)
    pcts = [(w / total) * 100 for w in widths_px]

    # Enforce minimum percentage
    needs_boost = [i for i, p in enumerate(pcts) if p < min_col_pct]
    if needs_boost and len(needs_boost) < n_cols:
        deficit = sum(min_col_pct - pcts[i] for i in needs_boost)
        donors = [i for i in range(n_cols) if i not in needs_boost]
        donor_total = sum(pcts[i] for i in donors)

        for i in needs_boost:
            pcts[i] = min_col_pct

        if donor_total > 0:
            shrink_factor = (donor_total - deficit) / donor_total
            for i in donors:
                pcts[i] *= shrink_factor

    return [round(p, 1) for p in pcts]


def _compute_max_text_lengths(
    grid: List[List[Optional[Dict]]],
    n_rows: int,
    n_cols: int,
) -> List[int]:
    """Find longest text string per column from single-span cells."""
    max_len = [0] * n_cols
    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[r][c]
            if cell is None:
                continue
            if cell.get("col_span", 1) != 1:
                continue
            text = cell.get("text", "").strip()
            max_len[c] = max(max_len[c], len(text))
    return max_len


def convert_ocr_to_table_schema(
    ocr_data: Dict,
    confidence_threshold: float = 0.3,
    mark_low_confidence_italic: bool = True,
) -> Dict:
    """
    Convert OCR pipeline JSON into the complex_table_maker schema.

    CRITICAL: The cells list for each row is POSITION-INDEXED.
    cells[c] maps directly to physical column c:
      - None = covered by a row/col span from another cell
      - {"text": "", ...} = empty cell (not spanned)
      - {"text": "...", ...} = content cell

    The renderer MUST consume cells by index, not by iterator.

    Args:
        ocr_data: The full OCR JSON dict (already consolidated if desired).
        confidence_threshold: Cells below this confidence get flagged.
        mark_low_confidence_italic: If True, low-confidence cells render italic.

    Returns:
        A dict compatible with complex_table_maker.add_complex_table().
    """
    grid, n_rows, n_cols = _build_cell_grid(ocr_data)
    header_row_indices = set(ocr_data.get("header_rows", []))
    col_alignment = ocr_data.get("column_alignment", {})

    # Precompute occupation map
    occupied = _build_occupied_map(grid, n_rows, n_cols)

    # Column widths
    if "_column_extents" in ocr_data:
        pcts = compute_consolidated_widths_pct(ocr_data)
    else:
        extents = _compute_column_widths(grid, n_rows, n_cols)
        extents = _distribute_spanning_widths(grid, extents, n_rows, n_cols)
        max_text_lens = _compute_max_text_lengths(grid, n_rows, n_cols)
        pcts = _extents_to_percentages(extents, n_cols, max_text_lens)

    columns = [{"name": "", "width_pct": pcts[c]} for c in range(n_cols)]

    # Build position-indexed rows
    schema_rows = []

    for r in range(n_rows):
        is_header = r in header_row_indices

        # Start with all cells as empty
        cells: List[Optional[Dict]] = [{"text": ""} for _ in range(n_cols)]

        # Mark spanned positions as None
        for c in range(n_cols):
            origin = occupied[r][c]
            if origin is not None and origin != (r, c):
                cells[c] = None

        # Place content cells at their exact position
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

        schema_rows.append({
            "is_header": is_header,
            "cells": cells,
        })

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

    Args:
        ocr_data: Raw OCR JSON dict.
        confidence_threshold: Cells below this get flagged.
        consolidate: If True, run column consolidation before conversion.
        gap_threshold_px: Pixel gap threshold for column merging.

    Returns a cleaned complex_table_maker schema dict.
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
        text = cell.get("text", "")
        return bool(text.strip())

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