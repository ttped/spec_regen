"""
test_table_ocr_to_docx.py

End-to-end check for the planned LLM table path:
  1. OCR a single table image via Mission Assist (reuses test_table_ocr_mission_assist).
  2. Parse the model's JSON back into the pipeline's table schema.
  3. Render it into a real .docx using the pipeline's own writer (docx_writer).

Open the resulting .docx to see whether the round-trip preserved the table.
The parsed JSON is also written next to the .docx for inspection.

Run from project root:
    python -m tests.test_table_ocr_to_docx path/to/table_crop.png
    python -m tests.test_table_ocr_to_docx crop.png --out my_table.docx --max-side 1536
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, _TESTS_DIR)                        # sibling: test_table_ocr_mission_assist
sys.path.insert(0, os.path.join(_ROOT_DIR, "app"))   # docx_writer, complex_table_schema

from test_table_ocr_mission_assist import ocr_image, SEGMENT, MODEL, HOST, API_KEY, CA_CERT
from docx_writer import create_docx_from_elements


# Strict schema prompt — matches the {"columns": [...], "rows": [[...]]} shape
# that docx_writer's complex-schema path renders.
TABLE_PROMPT = (
    "Transcribe the single table in this image into JSON with EXACTLY this shape:\n"
    "{\n"
    '  "caption": "<the table title/caption if printed, else empty string>",\n'
    '  "columns": ["<header 1>", "<header 2>", ...],\n'
    '  "rows": [["<r1c1>", "<r1c2>", ...], ["<r2c1>", ...]]\n'
    "}\n\n"
    "Rules:\n"
    "1. Preserve exact left-to-right column order and top-to-bottom row order. Never reorder.\n"
    "2. columns is the header row; if there is no header row, use \"\" for every column.\n"
    "3. Every row MUST have the same number of elements as columns; use \"\" for blank cells.\n"
    "4. Transcribe verbatim — keep numbers, units, symbols, and capitalization. Do not invent or summarize.\n"
    "5. Join a cell that wraps across printed lines into one string with single spaces.\n"
    "6. For a merged/spanned cell, put the text in the top-left cell and \"\" in the others.\n"
    "7. If a character is illegible, give your best guess; never drop a cell.\n"
    "8. Return ONLY the JSON object, parseable by json.loads — no prose, no markdown fences."
)


def parse_table_json(raw: str) -> dict:
    """Strip code fences, parse, and force every row rectangular to match columns."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", s).strip()
    data = json.loads(s)  # let it raise — a parse failure should be loud

    columns = [str(c) for c in data.get("columns", [])]
    ncols = len(columns)
    rows = []
    for r in data.get("rows", []):
        cells = ["" if v is None else str(v) for v in (r or [])]
        if len(cells) != ncols:
            print(f"  [warn] row has {len(cells)} cells, expected {ncols}; padding/truncating")
        cells = (cells + [""] * ncols)[:ncols]
        rows.append(cells)
    return {"caption": str(data.get("caption", "")), "columns": columns, "rows": rows}


def ocr_table_to_docx(image_path, out_path, max_side=1536, prompt=TABLE_PROMPT, **conn):
    """OCR a table image, parse it, and render it to a .docx via the pipeline writer."""
    raw = ocr_image(image_path, prompt=prompt, max_side=max_side, **conn)

    print("\n--- RAW MODEL OUTPUT ---")
    print(raw)

    parsed = parse_table_json(raw)
    print(f"\n--- PARSED: {len(parsed['columns'])} columns x {len(parsed['rows'])} rows ---")
    print("caption:", parsed["caption"] or "(none)")

    # Save the parsed JSON next to the docx for inspection
    json_path = str(Path(out_path).with_suffix(".json"))
    Path(json_path).write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render through the pipeline's own writer as a single table element.
    # String columns route to the complex-schema renderer (no width metadata needed).
    element = {
        "type": "table",
        "caption_text": parsed["caption"],
        "table_data": {"columns": parsed["columns"], "rows": parsed["rows"]},
    }
    create_docx_from_elements(
        [element],
        out_path,
        figures_image_folder=_TESTS_DIR,   # unused for a text table, but must be a real dir
        part_number="TABLE-OCR-TEST",
        title_data=None,
    )

    print(f"\n[OK] Wrote {out_path}")
    print(f"[OK] Wrote {json_path}")
    return parsed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OCR a table image and render it to a .docx via the pipeline writer.")
    ap.add_argument("image", help="Path to the table image (e.g. a YOLO crop).")
    ap.add_argument("--out", default=None, help="Output .docx path (default: <image>_table.docx).")
    ap.add_argument("--max-side", type=int, default=1536, help="Downscale long edge to N px (tables need more detail).")
    ap.add_argument("--prompt", default=TABLE_PROMPT)
    # connection passthrough (defaults come from the OCR test module / .env)
    ap.add_argument("--segment", default=SEGMENT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--ca-cert", default=CA_CERT)
    args = ap.parse_args()

    out = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_table.docx"))
    ocr_table_to_docx(
        args.image, out, max_side=args.max_side, prompt=args.prompt,
        host=args.host, segment=args.segment, model=args.model,
        api_key=args.api_key, ca_cert=args.ca_cert,
    )
