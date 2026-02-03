"""
asset_processor.py - Handles loading and positioning of figures and tables.

This module:
1. Loads figure/table metadata from exported JSON files
2. Filters out OCR text that lies INSIDE these assets (Double OCR Fix)
3. Integrates assets into the document element stream based on VISUAL POSITION
4. Interleaves assets between text blocks based on vertical (Y-axis) coordinates

The goal is to recreate the visual flow: Section 1.1 -> Image -> Section 1.2
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Any


def load_asset_metadata(json_path: str) -> Optional[Dict]:
    """
    Load and parse a single asset metadata JSON file.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"    [Warning] Could not load asset metadata: {json_path} - {e}")
        return None


def extract_asset_bbox_raw(meta: Dict) -> Optional[Dict]:
    """
    Extract bbox from asset metadata WITHOUT any scaling or normalization.
    Returns the raw bbox values as they appear in the metadata.
    """
    bbox_data = meta.get('bbox', {})
    
    raw_bbox = None
    bbox_source = None
    
    # Case 1: Nested dictionary (YOLO or Manual Export)
    if isinstance(bbox_data, dict):
        if 'pixels' in bbox_data:
            raw_bbox = bbox_data['pixels']
            bbox_source = 'pixels'
        elif 'pdf_units' in bbox_data:
            raw_bbox = bbox_data['pdf_units']
            bbox_source = 'pdf_units'
        
    # Case 2: Direct list (Simple format)
    elif isinstance(bbox_data, list):
        raw_bbox = bbox_data
        bbox_source = 'direct'
    
    if not raw_bbox or len(raw_bbox) < 4:
        return None
    
    # Interpret as [x, y, width, height]
    x, y, w, h = raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3]
    
    return {
        "left": x,
        "top": y,
        "width": w,
        "height": h,
        "right": x + w,
        "bottom": y + h,
        "_source": bbox_source,
        "_raw": raw_bbox
    }


def load_all_assets(exports_dir: str, doc_stem: str) -> List[Dict]:
    """
    Load all figure/table assets for a document.
    """
    assets = []
    
    # Construct full path to the document's asset folder
    asset_dir = os.path.join(exports_dir, doc_stem)
    
    if not os.path.isdir(asset_dir):
        print(f"    [Note] Asset directory not found: {asset_dir}")
        return assets
    
    print(f"    Scanning for assets in: {asset_dir}")
    
    loaded_count = 0
    for filename in os.listdir(asset_dir):
        if not filename.endswith(".json"):
            continue
        
        json_path = os.path.join(asset_dir, filename)
        meta = load_asset_metadata(json_path)
        
        if not meta:
            continue
        
        # Normalize the asset type (handle YOLO 'fig'/'tab' vs full names)
        raw_type = meta.get("asset_type", "").lower()
        if raw_type in ["fig", "figure"]:
            meta['type'] = 'figure'
        elif raw_type in ["tab", "table"]:
            meta['type'] = 'table'
        else:
            meta['type'] = 'figure'  # Default fallback
        
        # Extract raw bbox
        bbox = extract_asset_bbox_raw(meta)
        if bbox:
            meta['bbox'] = bbox
        
        # Get page number (normalize key)
        page = meta.get('page', meta.get('page_number', 9999))
        meta['page_number'] = page
        
        assets.append(meta)
        loaded_count += 1
    
    fig_count = sum(1 for a in assets if a['type'] == 'figure')
    tab_count = sum(1 for a in assets if a['type'] == 'table')
    print(f"    Loaded {loaded_count} assets ({fig_count} figures, {tab_count} tables)")
    
    return assets


def calculate_overlap_ratio(text_bbox: Dict, asset_bbox: Dict) -> float:
    """
    Calculates what fraction of the text block is covered by the asset block.
    Returns float 0.0 to 1.0.
    """
    # Determine intersection rectangle
    x_left = max(text_bbox['left'], asset_bbox['left'])
    y_top = max(text_bbox['top'], asset_bbox['top'])
    x_right = min(text_bbox['right'], asset_bbox['right'])
    y_bottom = min(text_bbox['bottom'], asset_bbox['bottom'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    text_area = text_bbox['width'] * text_bbox['height']
    
    if text_area <= 0:
        return 0.0
        
    return intersection_area / text_area


def filter_overlapped_text(elements: List[Dict], assets: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Removes text elements that are significantly overlapped by an asset (figure/table).
    This prevents "double OCR" where the text appears both in the image and as raw text.
    """
    if not assets:
        return elements, 0
        
    # Group assets by page for faster lookup
    assets_by_page = {}
    for asset in assets:
        page = asset.get('page_number', 9999)
        if page not in assets_by_page:
            assets_by_page[page] = []
        assets_by_page[page].append(asset)
        
    filtered_elements = []
    removed_count = 0
    
    for elem in elements:
        # Skip elements that don't have bboxes or aren't text/sections
        if 'bbox' not in elem or not elem['bbox']:
            filtered_elements.append(elem)
            continue
            
        page = elem.get('page_number', 9999)
        if page not in assets_by_page:
            filtered_elements.append(elem)
            continue
            
        # Check against all assets on this page
        is_overlapped = False
        text_bbox = elem['bbox']
        
        for asset in assets_by_page[page]:
            asset_bbox = asset.get('bbox')
            if not asset_bbox:
                continue
                
            # Check overlap
            overlap_ratio = calculate_overlap_ratio(text_bbox, asset_bbox)
            
            # THRESHOLD: If >60% of text area is inside the image, assume it's part of the image
            if overlap_ratio > 0.60:
                is_overlapped = True
                # print(f"      [Filter] Dropped text on p{page} (overlap {overlap_ratio:.2f} with {asset.get('asset_id')})")
                break
        
        if is_overlapped:
            removed_count += 1
        else:
            filtered_elements.append(elem)
            
    return filtered_elements, removed_count


def integrate_assets_with_elements(
    elements: List[Dict], 
    assets: List[Dict],
    page_metadata: Optional[Dict] = None
) -> List[Dict]:
    """
    Merge assets into the element stream by INTERLEAVING them based on vertical position.
    
    Algorithm:
    1. Filter out text elements that are visually covered by assets (Double OCR fix).
    2. Sort assets for a page by their 'top' (Y) coordinate.
    3. Iterate through text elements.
    4. If an asset's 'top' is LESS than the current text element's 'top', insert it BEFORE.
    5. Any remaining assets for the page are appended at the end of the page.
    """
    if not assets:
        return elements

    # --- STEP 1: Filter Double OCR Text ---
    clean_elements, removed_count = filter_overlapped_text(elements, assets)
    if removed_count > 0:
        print(f"    Filtered out {removed_count} text blocks overlapping with assets (Double OCR fix).")
    
    # --- STEP 2: Group assets by page and SORT by Top position ---
    assets_by_page: Dict[int, List[Dict]] = {}
    for asset in assets:
        # Robust page parsing
        page = asset.get('page_number', 9999)
        if isinstance(page, str) and page.isdigit():
            page = int(page)
        elif not isinstance(page, int):
            page = 9999
            
        if page not in assets_by_page:
            assets_by_page[page] = []
        assets_by_page[page].append(asset)
    
    # Sort assets within each page strictly by 'top' position
    for page in assets_by_page:
        assets_by_page[page].sort(key=lambda a: a.get('bbox', {}).get('top', 9999))
    
    result = []
    current_page = None
    
    # --- STEP 3: Iterate through existing text elements ---
    for element in clean_elements:
        elem_page = element.get('page_number', 9999)
        if isinstance(elem_page, str) and elem_page.isdigit():
            elem_page = int(elem_page)
        
        # If we moved to a NEW page, flush any remaining assets from the PREVIOUS page
        if current_page is not None and elem_page != current_page:
            if current_page in assets_by_page:
                remaining_assets = assets_by_page.pop(current_page)
                if remaining_assets:
                    result.extend(remaining_assets)
        
        current_page = elem_page
        
        # --- STEP 4: Check if any assets on CURRENT page belong BEFORE this element ---
        if current_page in assets_by_page:
            page_assets = assets_by_page[current_page]
            
            # Get current element's top position
            elem_top = element.get('bbox', {}).get('top', float('inf'))
            
            # Identify assets that appear visibly "above" this text block
            assets_to_insert = []
            while page_assets:
                next_asset = page_assets[0]
                asset_top = next_asset.get('bbox', {}).get('top', 9999)
                
                # If asset is definitely above the text, insert it now
                if asset_top < elem_top:
                    assets_to_insert.append(page_assets.pop(0))
                else:
                    # Next asset is below this text, stop checking
                    break
            
            if assets_to_insert:
                result.extend(assets_to_insert)

        # Add the text element itself
        result.append(element)
    
    # --- STEP 5: Handle assets for the very last page processed ---
    if current_page is not None and current_page in assets_by_page:
        remaining = assets_by_page.pop(current_page)
        result.extend(remaining)
    
    # --- STEP 6: Handle orphaned assets (pages with no text elements at all) ---
    for page in sorted(assets_by_page.keys()):
        page_assets = assets_by_page[page]
        result.extend(page_assets)
        print(f"      + Added {len(page_assets)} assets to text-less page {page}")
    
    return result


def run_asset_integration(
    input_path: str,
    output_path: str,
    exports_dir: str,
    doc_stem: str,
    positioning_mode: str = "bbox"
):
    """
    Main function to integrate assets.
    """
    print(f"  - Reading structure: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"  - [Error] Input file not found: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle structure formats
    if isinstance(data, dict) and 'elements' in data:
        elements = data.get('elements', [])
        page_metadata = data.get('page_metadata', {})
    else:
        elements = data if isinstance(data, list) else []
        page_metadata = {}
    
    # Load assets
    assets = load_all_assets(exports_dir, doc_stem)
    
    # Integrate
    integrated = integrate_assets_with_elements(elements, assets, page_metadata)
    
    # Save
    output_data = {
        "page_metadata": page_metadata,
        "elements": integrated
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    asset_count = sum(1 for e in integrated if e.get('type') in ('figure', 'table'))
    print(f"  - Saved to: {output_path}")
    print(f"  - Final Element Count: {len(integrated)} (Integrated {asset_count} assets)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 4:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        exports_dir = sys.argv[3]
        doc_stem = sys.argv[4] if len(sys.argv) > 4 else "unknown_doc"
        
        run_asset_integration(input_file, output_file, exports_dir, doc_stem)
    else:
        print("Usage: python asset_processor.py <input.json> <output.json> <exports_dir> [doc_stem]")