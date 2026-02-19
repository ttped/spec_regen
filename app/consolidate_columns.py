"""
consolidate_columns.py

Reduces the sparse OCR pixel-grid to a tighter grid by removing
columns that are provably unused: no cell originates there AND no
cell's span covers it.

This is the SAFE consolidation — it never merges two content-bearing
columns, so it is guaranteed to never lose text.  The result may still
have more columns than the "true" table, but no data is destroyed.

Lives in: root/app/
"""

from typing import Dict, List, Optional, Set, Tuple


def _build_cell_lookup(ocr_data: Dict) -> List[Dict]:
    """Flatten all cells from the OCR rows into a single list."""
    cells = []
    for row_entry in ocr_data.get("rows", []):
        for cell in row_entry.get("cells", []):
            cells.append(cell)
    return cells


def _find_needed_columns(
    cells: List[Dict],
    n_cols: int,
) -> Set[int]:
    """
    A grid column is "needed" if any cell either originates there
    or spans across it.

    Returns the set of grid column indices that must be kept.
    """
    needed: Set[int] = set()

    for cell in cells:
        col = cell["col"]
        span = cell.get("col_span", 1)
        for dc in range(span):
            c = col + dc
            if 0 <= c < n_cols:
                needed.add(c)

    return needed


def _build_remap(
    needed: Set[int],
    n_cols: int,
) -> Tuple[Dict[int, int], int]:
    """
    Build a mapping from old grid column index → new dense column index.

    Only columns in `needed` get a new index; removed columns are absent
    from the map.

    Returns (remap_dict, new_n_cols).
    """
    remap: Dict[int, int] = {}
    new_idx = 0
    for c in range(n_cols):
        if c in needed:
            remap[c] = new_idx
            new_idx += 1
    return remap, new_idx


def consolidate_ocr_data(
    ocr_data: Dict,
    **_kwargs,
) -> Dict:
    """
    Remove empty, unspanned columns from the OCR grid.

    This is a safe transformation: it only removes columns that no cell
    touches (neither as origin nor as part of a span).  Two content-
    bearing columns are NEVER merged, so no text can be lost.

    Args:
        ocr_data: The raw OCR JSON dict (not mutated).

    Returns:
        A new OCR data dict with unused columns removed.
    """
    n_rows = ocr_data["n_rows"]
    n_cols = ocr_data["n_cols"]
    cells = _build_cell_lookup(ocr_data)

    needed = _find_needed_columns(cells, n_cols)
    remap, new_n_cols = _build_remap(needed, n_cols)

    if new_n_cols == n_cols:
        # Nothing to remove — return a shallow copy with debug info
        result = dict(ocr_data)
        result["_removed_cols"] = []
        result["_original_n_cols"] = n_cols
        return result

    removed = sorted(set(range(n_cols)) - needed)

    # Remap cells
    new_rows = []
    for row_entry in ocr_data.get("rows", []):
        new_cells = []
        for cell in row_entry.get("cells", []):
            old_col = cell["col"]
            old_span = cell.get("col_span", 1)

            # New start column
            new_col = remap.get(old_col)
            if new_col is None:
                # Origin column was removed — shouldn't happen since
                # we keep all columns that have cells, but guard anyway
                continue

            # New span: count how many of the spanned columns survived
            end_old = old_col + old_span - 1
            new_end = remap.get(end_old, new_col)
            new_span = max(new_end - new_col + 1, 1)

            new_cell = dict(cell)
            new_cell["col"] = new_col
            new_cell["col_span"] = new_span
            new_cells.append(new_cell)

        new_rows.append({
            "row_index": row_entry["row_index"],
            "cells": new_cells,
        })

    # Remap column metadata
    old_alignment = ocr_data.get("column_alignment", {})
    old_types = ocr_data.get("column_types", {})
    old_semantics = ocr_data.get("column_semantics", {})

    new_alignment = {}
    new_types = {}
    new_semantics = {}

    for old_c_str, val in old_alignment.items():
        new_c = remap.get(int(old_c_str))
        if new_c is not None:
            new_alignment[str(new_c)] = val

    for old_c_str, val in old_types.items():
        new_c = remap.get(int(old_c_str))
        if new_c is not None:
            new_types[str(new_c)] = val

    for old_c_str, val in old_semantics.items():
        new_c = remap.get(int(old_c_str))
        if new_c is not None:
            new_semantics[str(new_c)] = val

    result = dict(ocr_data)
    result["rows"] = new_rows
    result["n_cols"] = new_n_cols
    result["column_alignment"] = new_alignment
    result["column_types"] = new_types
    result["column_semantics"] = new_semantics
    result["_removed_cols"] = removed
    result["_original_n_cols"] = n_cols

    return result