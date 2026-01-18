"""
asset_processor.py - Handles loading and positioning of figures and tables.

This module:
1. Loads figure/table metadata from exported JSON files
2. Uses bbox coordinates (top, left, width, height) for positioning
3. Integrates assets into the document element stream at correct positions
4. Sorts elements by page and vertical position (top/y-coordinate)

Both OCR elements and assets have bbox data:
- OCR elements: bbox with left, top, width, height (from OCR coordinate space)
- Assets: bbox.pdf_units with [x, y, width, height] or similar

The key for vertical ordering is the 'top' value - lower top = higher on page.
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


def extract_asset_bbox(meta: Dict) -> Optional[Dict]:
    """
    Extract and normalize bbox from asset metadata.
    
    Assets may have bbox in different formats:
    - bbox.pdf_units: [x, y, width, height] or [x1, y1, x2, y2]
    - bbox.pixels: [x, y, width, height]
    - bbox: [x, y, width, height] directly
    
    Returns a normalized dict with left, top, width, height, right, bottom.
    """
    bbox_data = meta.get('bbox', {})
    
    # Try different bbox formats
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
    
    # Interpret as [x, y, width, height] format
    # (Some systems use [x1, y1, x2, y2] - we may need to detect this)
    x, y, w, h = raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3]
    
    # Heuristic: if w and h are larger than x and y, it's probably [x1, y1, x2, y2]
    # In that case, x2 > x1 and y2 > y1, so w would be x2 and h would be y2
    if w > x * 2 and h > y * 2 and w > 100 and h > 100:
        # Likely [x1, y1, x2, y2] format
        x1, y1, x2, y2 = x, y, w, h
        return {
            "left": x1,
            "top": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "right": x2,
            "bottom": y2,
            "_source": bbox_source,
            "_format": "x1y1x2y2"
        }
    else:
        # Standard [x, y, width, height] format
        return {
            "left": x,
            "top": y,
            "width": w,
            "height": h,
            "right": x + w,
            "bottom": y + h,
            "_source": bbox_source,
            "_format": "xywh"
        }


def load_all_assets(exports_dir: str, doc_stem: str) -> List[Dict]:
    """
    Load all figure/table assets for a document.
    
    Args:
        exports_dir: Base directory containing exported assets
        doc_stem: Document stem name (subdirectory name)
    
    Returns:
        List of asset dictionaries with normalized bbox
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
        
        # Extract and normalize bbox
        bbox = extract_asset_bbox(meta)
        if bbox:
            meta['bbox'] = bbox
        
        # Get page number
        page = meta.get('page', 9999)
        meta['page_number'] = page  # Normalize key name
        
        # Store export DPI for reference
        export_info = meta.get('export', {})
        meta['_export_dpi'] = export_info.get('dpi')
        
        assets.append(meta)
    
    fig_count = sum(1 for a in assets if a['type'] == 'figure')
    tab_count = sum(1 for a in assets if a['type'] == 'table')
    print(f"    Loaded {len(assets)} assets ({fig_count} figures, {tab_count} tables)")
    
    return assets


def get_element_sort_key(element: Dict) -> Tuple[int, float]:
    """
    Get sort key for an element based on page and vertical position.
    
    Returns (page_number, top_position) tuple.
    Lower top = higher on page = should come first.
    """
    # Get page number (handle both key names)
    page = element.get('page_number', element.get('page', 9999))
    if isinstance(page, str):
        try:
            page = int(page)
        except:
            page = 9999
    
    # Get top position from bbox
    bbox = element.get('bbox')
    if bbox and isinstance(bbox, dict):
        top = bbox.get('top', 9999)
    else:
        top = 9999
    
    return (page, top)


def integrate_assets_with_elements(
    elements: List[Dict], 
    assets: List[Dict],
    page_metadata: Optional[Dict] = None
) -> List[Dict]:
    """
    Merge assets into the element stream, positioning them by page and bbox.top.
    
    Args:
        elements: List of section/text block elements from section_processor
        assets: List of figure/table assets
        page_metadata: Optional dict of page-level metadata (for debugging)
    
    Returns:
        Combined list of elements sorted by page and vertical position
    """
    if not assets:
        return elements
    
    # Log some debug info
    if page_metadata:
        print(f"    Page metadata available for {len(page_metadata)} pages")
        # Sample one page's metadata
        sample_page = next(iter(page_metadata.keys()), None)
        if sample_page:
            print(f"    Sample page {sample_page} metadata: {page_metadata[sample_page]}")
    
    # Log bbox info for debugging
    elements_with_bbox = sum(1 for e in elements if e.get('bbox'))
    assets_with_bbox = sum(1 for a in assets if a.get('bbox'))
    print(f"    Elements with bbox: {elements_with_bbox}/{len(elements)}")
    print(f"    Assets with bbox: {assets_with_bbox}/{len(assets)}")
    
    # Sample bbox values for debugging
    if elements and elements[0].get('bbox'):
        print(f"    Sample element bbox: {elements[0]['bbox']}")
    if assets and assets[0].get('bbox'):
        print(f"    Sample asset bbox: {assets[0]['bbox']}")
    
    # Combine and sort
    all_items = elements + assets
    all_items.sort(key=get_element_sort_key)
    
    return all_items


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
        positioning_mode: "bbox" for positioning by coordinates (only mode now)
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
    
    # Load assets
    print(f"  - Loading assets from: {exports_dir}/{doc_stem}")
    assets = load_all_assets(exports_dir, doc_stem)
    
    # Integrate
    print(f"  - Integrating assets by bbox position")
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