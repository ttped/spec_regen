"""
test_table_html_to_docx.py

Manual round-trip check for the HTML table path now used by the pipeline:
  1. OCR a table crop as HTML (reuses test_table_ocr_mission_assist for the call).
  2. Expand the HTML into the physical grid (app/table_ocr_llm — same code the
     pipeline's table_llm_processor uses).
  3. Render to a real .docx via the pipeline writer (docx_writer).

The deterministic HTML parser + grid expander live in app/table_ocr_llm and are
unit-tested separately; this file is the end-to-end visual harness.

Run from project root:
    python -m tests.test_table_html_to_docx path/to/table_crop.png
    python -m tests.test_table_html_to_docx crop.png --out out.docx --max-side 2048 --image-fallback-on-gaps

Outputs next to --out: the .docx, the raw .html, and the expanded grid .json.
"""

import os
import sys
import json
import argparse
from pathlib import Path

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, _TESTS_DIR)                        # sibling: test_table_ocr_mission_assist
sys.path.insert(0, os.path.join(_ROOT_DIR, "app"))   # table_ocr_llm, docx_writer

from test_table_ocr_mission_assist import ocr_image, SEGMENT, MODEL, HOST, API_KEY, CA_CERT
from table_ocr_llm import (
    HTML_TABLE_PROMPT, extract_table_html, parse_html_table,
    html_to_grid, grid_to_table_data,
)


# ---------------------------------------------------------------------------
# Rendering (via the pipeline's own writer)
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
