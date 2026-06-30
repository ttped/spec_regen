"""
table_llm_processor.py - Pipeline step: OCR table crops with a vision LLM.

Drop-in alternative to table_iris_processor (step 7). For each table element
that has a YOLO crop image, OCRs the crop into a Word-ready table via
table_ocr_llm and attaches `table_data` (+ caption). On any failure the element
is left untouched, so docx_writer falls back to rendering the crop image.

Reads  : {doc}_with_assets.json
Writes : {doc}_with_tables.json   (same shape the rest of the pipeline expects)
"""

import os
import json
from typing import Dict

from table_ocr_llm import ocr_table_image
from asset_processor import resolve_asset_directory


TABLE_TYPES = {"table", "table_layout", "tab_layout"}


def run_llm_table_processing(
    input_path: str,
    output_path: str,
    figures_dir: str,
    doc_stem: str,
    vision_cfg: Dict,
    max_side: int = 2048,
):
    """
    Args:
        input_path:  Path to {doc}_with_assets.json
        output_path: Path to write {doc}_with_tables.json
        figures_dir: Root of YOLO exports (crops live in figures_dir/<doc_stem>/)
        doc_stem:    Resolved YOLO subdirectory name for this document
        vision_cfg:  Vision-LLM config (see simple_pipeline.VISION_LLM_CONFIG)
        max_side:    Long-edge downscale for the crop before sending
    """
    print(f"  - Reading elements: {input_path}")

    if not os.path.exists(input_path):
        print(f"  - [Error] Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "elements" in data:
        elements = data.get("elements", [])
        page_metadata = data.get("page_metadata", {})
    else:
        elements = data if isinstance(data, list) else []
        page_metadata = {}

    asset_dir = resolve_asset_directory(figures_dir, doc_stem) or os.path.join(figures_dir, doc_stem)

    found = reconstructed = fell_back = 0

    for element in elements:
        if str(element.get("type", "")).strip().lower() not in TABLE_TYPES:
            continue
        found += 1

        # Skip tables that already have data (e.g. a prior run / another source).
        if element.get("table_data") is not None:
            continue

        image_file = (element.get("export") or {}).get("image_file")
        if not image_file:
            fell_back += 1
            continue

        image_path = os.path.join(asset_dir, image_file)
        if not os.path.exists(image_path):
            print(f"    [skip] crop not found: {image_path}")
            fell_back += 1
            continue

        # API call — guard so one bad table doesn't abort the whole document.
        try:
            table_data, caption, _raw = ocr_table_image(image_path, vision_cfg, max_side=max_side)
        except Exception as e:
            print(f"    [warn] vision OCR failed for {image_file}: {e} — keeping crop image")
            fell_back += 1
            continue

        if table_data and table_data.get("rows"):
            element["table_data"] = table_data
            if caption and not str(element.get("caption_text", "")).strip():
                element["caption_text"] = caption
            reconstructed += 1
            print(f"    [ocr] {image_file}: "
                  f"{len(table_data['columns'])} cols x {len(table_data['rows'])} rows")
        else:
            fell_back += 1
            print(f"    [fallback] {image_file}: no usable table — keeping crop image")

    output_data = {"page_metadata": page_metadata, "elements": elements}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"  - Tables in stream: {found}")
    print(f"  - Tables reconstructed: {reconstructed}")
    print(f"  - Tables left as image: {fell_back}")
    print(f"  - Saved to: {output_path}")
