"""
consolidate_columns.py

Transforms a sparse OCR pixel-grid into a true logical table structure by:
1. Profiling which grid columns actually carry content
2. Clustering adjacent grid columns into logical columns via bbox overlap analysis
3. Remapping cell positions and spans to the consolidated grid
4. Recomputing column widths from the merged extents with content-aware minimums

Sits between OCR JSON parsing and schema generation (ocr_to_table.py).

Lives in: root/app/
"""

from typing import Dict, List, Optional, Tuple, Set


# ---------------------------------------------------------------------------
# Grid analysis helpers
# ---------------------------------------------------------------------------


def _build_cell_lookup(ocr_data: Dict) -> List[Dict]:
    """Flatten all cells from the OCR rows into a single list."""
    cells = []
    for row_entry in ocr_data.get("rows", []):
        for cell in row_entry.get("cells", []):
            cells.append(cell)
    return cells


def _column_content_profile(
    cells: List[Dict],
    n_rows: int,
    n_cols: int,
) -> List[Set[int]]:
    """
    For each grid column, collect the set of rows that have content
    originating in that column (ignoring spanning).

    Returns: list of length n_cols, each element a set of row indices.
    """
    profile: List[Set[int]] = [set() for _ in range(n_cols)]

    for cell in cells:
        col = cell["col"]
        row = cell["row"]
        text = cell.get("text", "").strip()
        if text and 0 <= col < n_cols:
            profile[col].add(row)

    return profile


def _column_bbox_extents(
    cells: List[Dict],
    n_cols: int,
) -> List[Optional[Tuple[int, int]]]:
    """
    For each grid column, compute the (x_min, x_max) pixel extent
    from all cells originating there (single-span only).

    Returns: list of (x_min, x_max) or None per grid column.
    """
    col_x_min: Dict[int, List[int]] = {}
    col_x_max: Dict[int, List[int]] = {}

    for cell in cells:
        col = cell["col"]
        cs = cell.get("col_span", 1)
        if cs != 1:
            continue
        bbox = cell.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, _, x2, _ = bbox
        col_x_min.setdefault(col, []).append(x1)
        col_x_max.setdefault(col, []).append(x2)

    extents: List[Optional[Tuple[int, int]]] = []
    for c in range(n_cols):
        if c in col_x_min and c in col_x_max:
            x1 = sorted(col_x_min[c])[len(col_x_min[c]) // 2]
            x2 = sorted(col_x_max[c])[len(col_x_max[c]) // 2]
            extents.append((x1, x2))
        else:
            extents.append(None)

    return extents


def _column_max_text_length(
    cells: List[Dict],
    n_cols: int,
) -> List[int]:
    """
    For each grid column, find the longest text string (character count)
    from single-span cells. Used for minimum width estimation.
    """
    max_len: List[int] = [0] * n_cols
    for cell in cells:
        col = cell["col"]
        cs = cell.get("col_span", 1)
        if cs != 1 or col >= n_cols:
            continue
        text = cell.get("text", "").strip()
        max_len[col] = max(max_len[col], len(text))
    return max_len


# ---------------------------------------------------------------------------
# Core consolidation logic
# ---------------------------------------------------------------------------


def compute_column_groups(
    content_profile: List[Set[int]],
    extents: List[Optional[Tuple[int, int]]],
    n_cols: int,
    gap_threshold_px: int = 50,
) -> List[List[int]]:
    """
    Cluster adjacent grid columns into logical column groups.

    Two adjacent grid columns are merged if:
      - At least one of them has no content (phantom column), OR
      - They have no rows where BOTH have content (no conflict), OR
      - Their bbox extents overlap or are very close (< gap_threshold_px)

    Returns: list of groups, where each group is a sorted list of grid
             column indices. Groups are ordered left-to-right.
    """
    groups: List[List[int]] = [[c] for c in range(n_cols)]

    def _should_merge(group_a: List[int], group_b: List[int]) -> bool:
        rows_a: Set[int] = set()
        rows_b: Set[int] = set()
        for c in group_a:
            rows_a |= content_profile[c]
        for c in group_b:
            rows_b |= content_profile[c]

        # If either side is completely empty, merge (phantom gap)
        if not rows_a or not rows_b:
            return True

        # If they share content in the same rows, they're separate columns
        if rows_a & rows_b:
            return False

        # Check spatial proximity
        extent_a = _group_extent(group_a, extents)
        extent_b = _group_extent(group_b, extents)

        if extent_a is not None and extent_b is not None:
            gap = extent_b[0] - extent_a[1]
            if gap < gap_threshold_px:
                return True

        # No row conflicts and no strong spatial separation → merge
        return True

    # Greedy left-to-right merge
    merged = [groups[0]]
    for i in range(1, len(groups)):
        if _should_merge(merged[-1], groups[i]):
            merged[-1] = merged[-1] + groups[i]
        else:
            merged.append(groups[i])

    return merged


def _group_extent(
    group: List[int],
    extents: List[Optional[Tuple[int, int]]],
) -> Optional[Tuple[int, int]]:
    """Compute the combined (x_min, x_max) for a group of grid columns."""
    x_mins = []
    x_maxs = []
    for c in group:
        ext = extents[c] if c < len(extents) else None
        if ext is not None:
            x_mins.append(ext[0])
            x_maxs.append(ext[1])
    if x_mins and x_maxs:
        return (min(x_mins), max(x_maxs))
    return None


# ---------------------------------------------------------------------------
# Remapping
# ---------------------------------------------------------------------------


def build_column_remap(groups: List[List[int]]) -> Dict[int, int]:
    """
    Build a mapping from grid column index → logical column index.
    """
    remap: Dict[int, int] = {}
    for logical_col, group in enumerate(groups):
        for grid_col in group:
            remap[grid_col] = logical_col
    return remap


def consolidate_ocr_data(
    ocr_data: Dict,
    gap_threshold_px: int = 50,
) -> Dict:
    """
    Transform OCR data from the sparse pixel-grid into a consolidated
    logical-column structure.

    Modifies cell col indices, col_span values, and updates n_cols.
    Also updates column_alignment and column_types for the new indices.

    Args:
        ocr_data: The raw OCR JSON dict (not mutated).
        gap_threshold_px: Max pixel gap between columns to consider merging.

    Returns:
        A new OCR data dict with consolidated columns.
    """
    n_rows = ocr_data["n_rows"]
    n_cols = ocr_data["n_cols"]
    cells = _build_cell_lookup(ocr_data)

    content_profile = _column_content_profile(cells, n_rows, n_cols)
    extents = _column_bbox_extents(cells, n_cols)
    groups = compute_column_groups(content_profile, extents, n_cols, gap_threshold_px)
    remap = build_column_remap(groups)
    new_n_cols = len(groups)

    # Remap cells
    new_rows = []
    for row_entry in ocr_data.get("rows", []):
        new_cells = []
        for cell in row_entry.get("cells", []):
            new_cell = dict(cell)
            old_col = cell["col"]
            old_span = cell.get("col_span", 1)

            new_col = remap.get(old_col, old_col)
            end_grid_col = old_col + old_span - 1
            new_end_col = remap.get(end_grid_col, new_col)
            new_span = max(new_end_col - new_col + 1, 1)

            new_cell["col"] = new_col
            new_cell["col_span"] = new_span
            new_cells.append(new_cell)

        new_rows.append({
            "row_index": row_entry["row_index"],
            "cells": new_cells,
        })

    # Remap column metadata using first grid col in each group
    old_alignment = ocr_data.get("column_alignment", {})
    old_types = ocr_data.get("column_types", {})
    old_semantics = ocr_data.get("column_semantics", {})

    new_alignment = {}
    new_types = {}
    new_semantics = {}

    for logical_col, group in enumerate(groups):
        for grid_col in group:
            key = str(grid_col)
            if key in old_alignment:
                new_alignment[str(logical_col)] = old_alignment[key]
                break
        for grid_col in group:
            key = str(grid_col)
            if key in old_types:
                new_types[str(logical_col)] = old_types[key]
                break
        for grid_col in group:
            key = str(grid_col)
            if key in old_semantics:
                new_semantics[str(logical_col)] = old_semantics[key]
                break

    # Compute new column extents and max text lengths (for width calc)
    new_column_extents = []
    for group in groups:
        ext = _group_extent(group, extents)
        new_column_extents.append(ext)

    # Aggregate max text length across grouped grid columns
    raw_text_lengths = _column_max_text_length(cells, n_cols)
    new_max_text_lengths = []
    for group in groups:
        group_max = max((raw_text_lengths[gc] for gc in group), default=0)
        new_max_text_lengths.append(group_max)

    result = dict(ocr_data)
    result["rows"] = new_rows
    result["n_cols"] = new_n_cols
    result["column_alignment"] = new_alignment
    result["column_types"] = new_types
    result["column_semantics"] = new_semantics
    result["_column_groups"] = groups
    result["_column_extents"] = [
        list(e) if e else None for e in new_column_extents
    ]
    result["_max_text_lengths"] = new_max_text_lengths
    result["_original_n_cols"] = n_cols

    return result


# ---------------------------------------------------------------------------
# Column width estimation
# ---------------------------------------------------------------------------

# Approximate pixels per character at typical OCR DPI.
# Calibrated for ~10pt text at 200 DPI.
_APPROX_PX_PER_CHAR = 12


def compute_consolidated_widths_pct(
    ocr_data: Dict,
    min_col_pct: float = 5.0,
) -> List[float]:
    """
    Compute width percentages for consolidated columns.

    Uses bbox extents with a content-aware minimum: columns are guaranteed
    at least enough width for their longest text string (estimated from
    character count), and at least min_col_pct % of total width.

    Args:
        ocr_data: Consolidated OCR data dict.
        min_col_pct: Absolute minimum column width as percentage.

    Returns: list of width percentages summing to ~100.
    """
    extents = ocr_data.get("_column_extents", [])
    max_text_lens = ocr_data.get("_max_text_lengths", [])
    n_cols = ocr_data["n_cols"]

    widths_px = []
    for i in range(n_cols):
        # Start with bbox extent
        ext = extents[i] if i < len(extents) and extents[i] else None
        bbox_width = (ext[1] - ext[0]) if ext is not None else 0

        # Content-aware minimum from text length
        text_len = max_text_lens[i] if i < len(max_text_lens) else 0
        content_min = text_len * _APPROX_PX_PER_CHAR

        widths_px.append(max(bbox_width, content_min, 20))

    total = sum(widths_px)
    pcts = [(w / total) * 100 for w in widths_px]

    # Enforce minimum percentage per column
    # If any column is below min_col_pct, steal proportionally from others
    needs_boost = [i for i, p in enumerate(pcts) if p < min_col_pct]
    if needs_boost and len(needs_boost) < n_cols:
        deficit = sum(min_col_pct - pcts[i] for i in needs_boost)
        donors = [i for i in range(n_cols) if i not in needs_boost]
        donor_total = sum(pcts[i] for i in donors)

        for i in needs_boost:
            pcts[i] = min_col_pct

        # Proportionally shrink donors
        if donor_total > 0:
            shrink_factor = (donor_total - deficit) / donor_total
            for i in donors:
                pcts[i] *= shrink_factor

    return [round(p, 1) for p in pcts]