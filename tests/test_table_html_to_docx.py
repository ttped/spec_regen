"""
test_table_html_to_docx.py

Higher-fidelity table OCR for irregular tables (varying column spans, merged
cells). Strategy:
  1. Ask the vision model for an HTML <table> with rowspan/colspan (LLMs handle
     HTML table structure far better than a bespoke JSON grid).
  2. Deterministically expand that HTML into the physical grid your renderer
     expects ({"columns":[{"name":..}], "rows":[{"cells":[...]}]} with null for
     covered positions). This grid arithmetic is the unreliable part when the
     LLM does it, so we do it in code instead.
  3. Render via app/complex_table_schema (through docx_writer), which already
     supports merged cells. Fall back to embedding the crop image when the
     model doesn't return a usable table.

Run from project root:
    python -m tests.test_table_html_to_docx path/to/table_crop.png
    python -m tests.test_table_html_to_docx crop.png --out out.docx --max-side 2048 --image-fallback-on-gaps

Outputs next to --out: the .docx, the raw .html, and the expanded grid .json.
"""

import os
import sys
import json
import re
import argparse
from html.parser import HTMLParser
from pathlib import Path

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, _TESTS_DIR)                        # sibling: test_table_ocr_mission_assist
sys.path.insert(0, os.path.join(_ROOT_DIR, "app"))   # docx_writer, complex_table_schema

from test_table_ocr_mission_assist import ocr_image, SEGMENT, MODEL, HOST, API_KEY, CA_CERT


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
    "- If the table has a printed title/caption, put it in a <caption> element."
)


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _to_int(v, default):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def extract_table_html(text: str):
    """Pull the first <table>...</table> out of the model output (ignores fences/prose)."""
    m = re.search(r"<table\b.*?</table>", text, re.IGNORECASE | re.DOTALL)
    return m.group(0) if m else None


class _TableParser(HTMLParser):
    """Collect rows (list of cell dicts) and an optional caption from one HTML table."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
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


def parse_html_table(html: str):
    """Return (caption, rows) where rows is a list of lists of cell dicts."""
    p = _TableParser()
    p.feed(html)
    return " ".join(p.caption.split()), p.rows


# ---------------------------------------------------------------------------
# HTML rows -> physical grid (the deterministic core)
# ---------------------------------------------------------------------------

def html_to_grid(rows):
    """
    Expand HTML rows (with colspan/rowspan) into a dense physical grid.

    Returns (ncols, grid_rows, report) where grid_rows[r][c] is:
        - the originating cell dict (at a span's top-left),
        - None (a position covered by a span), or
        - {"text": ""} (a gap the model under-specified — a short row).
    Uses the standard table-forming algorithm: place each cell at the next free
    column, marking every position it covers.
    """
    occupied = {}
    max_col = 0
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in occupied:        # skip positions covered by earlier rowspans
                c += 1
            cs = max(1, _to_int(cell.get("colspan"), 1))
            rs = max(1, _to_int(cell.get("rowspan"), 1))
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


def grid_to_table_data(ncols, grid_rows, caption=""):
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
                cd["shading"] = "D9E2F3"
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
# Rendering
# ---------------------------------------------------------------------------

def _write_docx(element, out_path, figures_dir):
    from docx_writer import create_docx_from_elements
    create_docx_from_elements([element], out_path,
                              figures_image_folder=figures_dir,
                              part_number="TABLE-HTML-TEST",
                              title_data=None)


def render_table(table_data, caption, out_path):
    element = {"type": "table", "caption_text": caption, "table_data": table_data}
    _write_docx(element, out_path, _TESTS_DIR)


def render_image_fallback(image_path, caption, out_path):
    element = {"type": "table", "caption_text": caption,
               "export": {"image_file": os.path.basename(image_path)}}
    _write_docx(element, out_path, os.path.dirname(os.path.abspath(image_path)))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(image_path, out_path, max_side=2048, prompt=HTML_TABLE_PROMPT,
        image_fallback_on_gaps=False, **conn):
    raw = ocr_image(image_path, prompt=prompt, max_side=max_side, **conn)
    print("\n--- RAW MODEL OUTPUT ---")
    print(raw)

    html = extract_table_html(raw)
    Path(out_path).with_suffix(".html").write_text(html or raw, encoding="utf-8")

    if not html:
        print("\n[fallback] No <table> found in output — embedding the crop image.")
        render_image_fallback(image_path, "", out_path)
        print(f"[OK] Wrote {out_path} (image fallback)")
        return

    caption, rows = parse_html_table(html)
    if not rows:
        print("\n[fallback] HTML had no rows — embedding the crop image.")
        render_image_fallback(image_path, caption, out_path)
        print(f"[OK] Wrote {out_path} (image fallback)")
        return

    ncols, grid_rows, report = html_to_grid(rows)
    table_data = grid_to_table_data(ncols, grid_rows, caption)

    print(f"\n--- GRID: {report['ncols']} cols x {report['nrows']} rows, "
          f"gaps={report['gaps']} ---")
    print("caption:", caption or "(none)")
    if report["gaps"]:
        print(f"  [warn] {report['gaps']} grid position(s) under-specified by the model.")

    Path(out_path).with_suffix(".json").write_text(
        json.dumps(table_data, indent=2, ensure_ascii=False), encoding="utf-8")

    if ncols == 0 or report["nrows"] == 0 or (image_fallback_on_gaps and report["gaps"]):
        reason = "empty grid" if ncols == 0 else "gaps present"
        print(f"\n[fallback] Rendering crop image instead ({reason}).")
        render_image_fallback(image_path, caption, out_path)
        print(f"[OK] Wrote {out_path} (image fallback)")
        return

    render_table(table_data, caption, out_path)
    print(f"\n[OK] Wrote {out_path}")
    print(f"[OK] Wrote {Path(out_path).with_suffix('.html')}")
    print(f"[OK] Wrote {Path(out_path).with_suffix('.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OCR a table image as HTML and render it to .docx with merged cells.")
    ap.add_argument("image", help="Path to the table image (e.g. a YOLO crop).")
    ap.add_argument("--out", default=None, help="Output .docx path (default: <image>_html.docx).")
    ap.add_argument("--max-side", type=int, default=2048, help="Downscale long edge to N px (tables need detail).")
    ap.add_argument("--prompt", default=HTML_TABLE_PROMPT)
    ap.add_argument("--image-fallback-on-gaps", action="store_true",
                    help="Embed the crop image instead of the reconstructed table if any grid gaps are detected.")
    ap.add_argument("--segment", default=SEGMENT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--ca-cert", default=CA_CERT)
    args = ap.parse_args()

    out = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_html.docx"))
    run(args.image, out, max_side=args.max_side, prompt=args.prompt,
        image_fallback_on_gaps=args.image_fallback_on_gaps,
        host=args.host, segment=args.segment, model=args.model,
        api_key=args.api_key, ca_cert=args.ca_cert)
