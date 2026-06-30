"""
table_ocr_llm.py - Vision-LLM table OCR.

Sends a table crop image to a vision-capable model, asks for an HTML <table>
(with rowspan/colspan), and deterministically expands that HTML into the
physical grid that complex_table_schema renders. Doing the grid arithmetic in
code — not in the LLM — is what makes irregular/merged tables reliable.

Public API:
    ocr_table_image(image_path, llm_config) -> (table_data|None, caption, raw_html)

`table_data` is the rich schema docx_writer understands:
    {"columns": [{"name": ...}], "rows": [{"cells": [...]}], "header_rows": N}

The HTML parser and grid expander are unit-tested in
tests/test_table_html_to_docx.py (which imports them from here).
"""

import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple


HTML_TABLE_PROMPT = (
    "Transcribe the single table in this image as HTML.\n\n"
    "- Output ONLY one <table>...</table> element. No <html>/<head>/<body>, no CSS, "
    "no prose, no markdown fences.\n"
    "- One <tr> per row. Use <th> for header cells, <td> for data cells.\n"
    "- CRITICAL: when a cell visually spans multiple columns or rows, represent it with a "
    "single <td>/<th> using colspan and/or rowspan — do NOT split it or repeat its text. "
    "Match the spans exactly as drawn.\n"
    "- Preserve the exact row and column order. Do not reorder.\n"
    "- Transcribe text verbatim (numbers, units, symbols, capitalization). Do not summarize, "
    "correct, translate, or invent.\n"
    "- Join a cell that wraps across printed lines into one line separated by spaces.\n"
    "- Empty cells should be empty (<td></td>). Do not add rows or columns not in the image.\n"
    "- Use <caption> ONLY for a title that sits OUTSIDE the table's ruled grid (above or "
    "below the box, with no cell border around it). If a title or heading is itself a cell "
    "INSIDE the grid — e.g. a bordered row that spans all columns across the top — keep it "
    "as a row in the table (a <th colspan=N> spanning all columns). Never both: do not copy "
    "a structural title cell into <caption>, and never drop a cell that is part of the table."
)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _to_int(v, default):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def extract_table_html(text: str) -> Optional[str]:
    """Pull the first <table>...</table> out of the model output (ignores fences/prose)."""
    m = re.search(r"<table\b.*?</table>", text, re.IGNORECASE | re.DOTALL)
    return m.group(0) if m else None


class _TableParser(HTMLParser):
    """Collect rows (list of cell dicts) and an optional caption from one HTML table."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: List[List[Dict]] = []
        self.caption = ""
        self._in_caption = False
        self._cur_row = None
        self._cur_cell = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        a = {k.lower(): v for k, v in attrs}
        if tag == "caption":
            self._in_caption = True
        elif tag == "tr":
            self._cur_row = []
        elif tag in ("td", "th"):
            self._cur_cell = {
                "parts": [],
                "colspan": _to_int(a.get("colspan"), 1),
                "rowspan": _to_int(a.get("rowspan"), 1),
                "is_header": tag == "th",
            }
        elif tag == "br" and self._cur_cell is not None:
            self._cur_cell["parts"].append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "caption":
            self._in_caption = False
        elif tag in ("td", "th") and self._cur_cell is not None:
            text = " ".join("".join(self._cur_cell["parts"]).split())
            if self._cur_row is not None:
                self._cur_row.append({
                    "text": text,
                    "colspan": max(1, self._cur_cell["colspan"]),
                    "rowspan": max(1, self._cur_cell["rowspan"]),
                    "is_header": self._cur_cell["is_header"],
                })
            self._cur_cell = None
        elif tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None

    def handle_data(self, data):
        if self._in_caption:
            self.caption += data
        elif self._cur_cell is not None:
            self._cur_cell["parts"].append(data)


def parse_html_table(html: str) -> Tuple[str, List[List[Dict]]]:
    """Return (caption, rows) where rows is a list of lists of cell dicts."""
    p = _TableParser()
    p.feed(html)
    return " ".join(p.caption.split()), p.rows


# ---------------------------------------------------------------------------
# HTML rows -> physical grid (the deterministic core)
# ---------------------------------------------------------------------------

def html_to_grid(rows: List[List[Dict]]) -> Tuple[int, List[List], Dict]:
    """
    Expand HTML rows (with colspan/rowspan) into a dense physical grid.

    Returns (ncols, grid_rows, report) where grid_rows[r][c] is:
        - the originating cell dict (at a span's top-left),
        - None (a position covered by a span), or
        - {"text": ""} (a gap the model under-specified — a short row).
    """
    occupied = {}
    max_col = 0
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in occupied:        # skip positions covered by earlier rowspans
                c += 1
            want_cs = max(1, _to_int(cell.get("colspan"), 1))
            want_rs = max(1, _to_int(cell.get("rowspan"), 1))

            # Clamp spans so the claimed rectangle is entirely free. Overlapping
            # spans (from noisy OCR) would otherwise produce a non-rectangular
            # merge that python-docx rejects with InvalidSpanError.
            cs = 1
            while cs < want_cs and (r, c + cs) not in occupied:
                cs += 1
            rs = 1
            while rs < want_rs and all((r + rs, c + dc) not in occupied for dc in range(cs)):
                rs += 1

            # Record the effective spans so downstream rendering stays consistent.
            cell["colspan"] = cs
            cell["rowspan"] = rs

            for dr in range(rs):
                for dc in range(cs):
                    occupied[(r + dr, c + dc)] = cell if (dr == 0 and dc == 0) else None
            c += cs
            max_col = max(max_col, c)

    ncols = max_col
    nrows = len(rows)
    grid_rows = []
    gaps = 0
    for r in range(nrows):
        gr = []
        for c in range(ncols):
            if (r, c) in occupied:
                gr.append(occupied[(r, c)])
            else:
                gr.append({"text": ""})      # never reached by any cell => short row
                gaps += 1
        grid_rows.append(gr)

    report = {"ncols": ncols, "nrows": nrows, "gaps": gaps}
    return ncols, grid_rows, report


def grid_to_table_data(ncols: int, grid_rows: List[List], caption: str = "") -> Dict:
    """Convert the physical grid into complex_table_schema's rich table_data dict."""
    nrows = len(grid_rows)
    columns = [{"name": ""} for _ in range(ncols)]
    out_rows = []
    header_rows = 0
    still_header = True

    for r, grid_row in enumerate(grid_rows):
        cells = []
        row_is_header = False
        for c, cell in enumerate(grid_row):
            if cell is None:
                cells.append(None)
                continue
            cd = {"text": cell.get("text", "")}
            cs = min(max(1, _to_int(cell.get("colspan"), 1)), ncols - c)
            rs = min(max(1, _to_int(cell.get("rowspan"), 1)), nrows - r)
            if cs > 1:
                cd["colspan"] = cs
            if rs > 1:
                cd["rowspan"] = rs
            if cell.get("is_header"):
                cd["bold"] = True
                row_is_header = True
            cells.append(cd)
        out_rows.append({"cells": cells, "is_header": row_is_header})
        if still_header and row_is_header:
            header_rows += 1
        else:
            still_header = False

    table_data = {"columns": columns, "rows": out_rows}
    if header_rows:
        table_data["header_rows"] = header_rows
    return table_data


# ---------------------------------------------------------------------------
# Vision call -> table_data
# ---------------------------------------------------------------------------

def ocr_table_image(image_path: str, cfg: Dict) -> Tuple[Optional[Dict], str, str]:
    """
    OCR one table crop into table_data via HTML, using the single configured LLM.

    `cfg` is the unified LLM config (see simple_pipeline.LLM_CONFIG). Returns
    (table_data, caption, raw_output); table_data is None when the model returned
    no usable table (caller should fall back to the crop image).
    """
    from utils import call_llm_vision  # lazy: keeps this module import-light

    raw = call_llm_vision(HTML_TABLE_PROMPT, image_path, cfg)

    html = extract_table_html(raw)
    if not html:
        return None, "", raw

    caption, rows = parse_html_table(html)
    if not rows:
        return None, caption, raw

    ncols, grid_rows, _report = html_to_grid(rows)
    if ncols == 0:
        return None, caption, raw

    return grid_to_table_data(ncols, grid_rows, caption), caption, raw
