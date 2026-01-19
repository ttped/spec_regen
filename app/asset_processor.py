"""
asset_processor.py - Handles loading and positioning of figures and tables.

This module:
1. Loads figure/table metadata from exported JSON files
2. Uses raw bbox coordinates as-is (no scaling/normalization)
3. Integrates assets into the document element stream
4. Preserves the original order of OCR elements (sections/text)
5. Inserts assets at the end of their respective pages

The OCR element order is authoritative - assets are inserted without disrupting it.
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
    
    We accept whatever coordinate system the asset uses - the key purpose
    is just to have SOME position for page-level grouping.
    """
    bbox_data = meta.get('bbox', {})
    
    # Try different bbox formats - just extract raw values
    raw_bbox = None
    bbox_source = None
    
    if isinstance(bbox_data, dict):
        if 'pdf_units' in bbox_data:
            raw_bbox = bbox_data['pdf_units']
            bbox_source = 'pdf_units'
        elif 'pixels' in bbox_data:
            raw_bbox = bbox_data['pixels']
            bbox_source = 'pixels'
    elif isinstance(bbox_data, list):
        raw_bbox = bbox_data
        bbox_source = 'direct'
    
    if not raw_bbox or len(raw_bbox) < 4:
        return None
    
    # Just store the raw values - interpret as [x, y, width, height]
    # Don't try to detect or convert formats
    x, y, w, h = raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3]
    
    return {
        "left": x,
        "top": y,
        "width": w,
        "height": h,
        "right": x + w,
        "bottom": y + h,
        "_source": bbox_source,
        "_raw": raw_bbox  # Keep original for debugging
    }


def load_all_assets(exports_dir: str, doc_stem: str) -> List[Dict]:
    """
    Load all figure/table assets for a document.
    
    Args:
        exports_dir: Base directory containing exported assets
        doc_stem: Document stem name (subdirectory name)
    
    Returns:
        List of asset dictionaries with raw bbox (no normalization)
    """
    assets = []
    asset_dir = os.path.join(exports_dir, doc_stem)
    
    if not os.path.isdir(asset_dir):
        print(f"    [Note] No asset directory found at: {asset_dir}")
        return assets
    
    for filename in os.listdir(asset_dir):
        if not filename.endswith(".json"):
            continue
        
        json_path = os.path.join(asset_dir, filename)
        meta = load_asset_metadata(json_path)
        
        if not meta:
            continue
        
        # Normalize the asset type
        asset_type = meta.get("asset_type", "").lower()
        if asset_type == "fig":
            meta['type'] = 'figure'
        elif asset_type == "tab":
            meta['type'] = 'table'
        else:
            meta['type'] = 'figure'  # Default
        
        # Extract raw bbox (no scaling)
        bbox = extract_asset_bbox_raw(meta)
        if bbox:
            meta['bbox'] = bbox
        
        # Get page number
        page = meta.get('page', 9999)
        meta['page_number'] = page  # Normalize key name
        
        # Store export info for reference
        export_info = meta.get('export', {})
        meta['_export_dpi'] = export_info.get('dpi')
        
        assets.append(meta)
    
    fig_count = sum(1 for a in assets if a['type'] == 'figure')
    tab_count = sum(1 for a in assets if a['type'] == 'table')
    print(f"    Loaded {len(assets)} assets ({fig_count} figures, {tab_count} tables)")
    
    return assets


def integrate_assets_with_elements(
    elements: List[Dict], 
    assets: List[Dict],
    page_metadata: Optional[Dict] = None
) -> List[Dict]:
    """
    Merge assets into the element stream while PRESERVING the original element order.
    
    Strategy:
    - The OCR elements (sections, text blocks) maintain their exact order
    - Assets are inserted at the END of each page's elements
    - Within a page's assets, they are sorted by their raw 'top' value
    
    This ensures section ordering is never disrupted by asset integration.
    
    Args:
        elements: List of section/text block elements from section_processor (order is preserved)
        assets: List of figure/table assets
        page_metadata: Optional dict of page-level metadata (for debugging)
    
    Returns:
        Combined list with assets inserted after each page's elements
    """
    if not assets:
        return elements
    
    # Debug logging
    elements_with_bbox = sum(1 for e in elements if e.get('bbox'))
    assets_with_bbox = sum(1 for a in assets if a.get('bbox'))
    print(f"    Elements with bbox: {elements_with_bbox}/{len(elements)}")
    print(f"    Assets with bbox: {assets_with_bbox}/{len(assets)}")
    
    # Group assets by page
    assets_by_page: Dict[int, List[Dict]] = {}
    for asset in assets:
        page = asset.get('page_number', asset.get('page', 9999))
        if isinstance(page, str):
            try:
                page = int(page)
            except ValueError:
                page = 9999
        
        if page not in assets_by_page:
            assets_by_page[page] = []
        assets_by_page[page].append(asset)
    
    # Sort assets within each page by their top position (for consistent ordering)
    for page in assets_by_page:
        assets_by_page[page].sort(key=lambda a: a.get('bbox', {}).get('top', 9999))
    
    # Build result by going through elements and inserting assets at page boundaries
    result = []
    current_page = None
    
    for i, element in enumerate(elements):
        elem_page = element.get('page_number', element.get('page', 9999))
        if isinstance(elem_page, str):
            try:
                elem_page = int(elem_page)
            except ValueError:
                elem_page = 9999
        
        # Check if we've moved to a new page
        if current_page is not None and elem_page != current_page:
            # Insert any assets from the previous page
            if current_page in assets_by_page:
                page_assets = assets_by_page.pop(current_page)
                result.extend(page_assets)
                print(f"    Inserted {len(page_assets)} assets after page {current_page}")
        
        current_page = elem_page
        result.append(element)
    
    # Insert assets for the last page
    if current_page is not None and current_page in assets_by_page:
        page_assets = assets_by_page.pop(current_page)
        result.extend(page_assets)
        print(f"    Inserted {len(page_assets)} assets after page {current_page}")
    
    # Add any remaining assets (from pages that had no elements)
    for page in sorted(assets_by_page.keys()):
        page_assets = assets_by_page[page]
        result.extend(page_assets)
        print(f"    Inserted {len(page_assets)} assets from page {page} (no elements on this page)")
    
    return result


def run_asset_integration(
    input_path: str,
    output_path: str,
    exports_dir: str,
    doc_stem: str,
    positioning_mode: str = "bbox"
):
    """
    Main function to load sections and integrate assets.
    
    Args:
        input_path: Path to organized sections JSON (from section_processor)
        output_path: Path to save integrated output
        exports_dir: Base directory for exported figures/tables
        doc_stem: Document stem name
        positioning_mode: Ignored (kept for API compatibility) - always preserves element order
    """
    print(f"  - Loading sections from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"  - [Error] Input file not found: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle new output format with page_metadata and elements
    if isinstance(data, dict) and 'elements' in data:
        elements = data.get('elements', [])
        page_metadata = data.get('page_metadata', {})
        print(f"  - Found {len(elements)} section elements")
    else:
        # Old format - just a list of elements
        elements = data if isinstance(data, list) else []
        page_metadata = {}
        print(f"  - Found {len(elements)} section elements (legacy format)")
    
    # Load assets (with raw bboxes)
    print(f"  - Loading assets from: {exports_dir}/{doc_stem}")
    assets = load_all_assets(exports_dir, doc_stem)
    
    # Integrate - preserves element order, inserts assets at page boundaries
    print(f"  - Integrating assets (preserving element order)")
    integrated = integrate_assets_with_elements(elements, assets, page_metadata)
    
    # Prepare output - preserve page_metadata for downstream use
    output_data = {
        "page_metadata": page_metadata,
        "elements": integrated
    }
    
    # Save result
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    asset_count = sum(1 for e in integrated if e.get('type') in ('figure', 'table'))
    print(f"  - Saved {len(integrated)} elements ({asset_count} assets integrated)")
    print(f"  - Output: {output_path}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 4:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        exports_dir = sys.argv[3]
        doc_stem = sys.argv[4] if len(sys.argv) > 4 else os.path.splitext(os.path.basename(input_file))[0].replace('_organized', '')
        
        run_asset_integration(input_file, output_file, exports_dir, doc_stem)
    else:
        print("Usage: python asset_processor.py <input.json> <output.json> <exports_dir> [doc_stem]")