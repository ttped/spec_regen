"""
asset_processor.py - Handles loading and positioning of figures and tables.

NORMALIZED COORDINATES VERSION

This module uses normalized (0-1) coordinates to avoid dimension mismatch issues.
All bounding boxes are converted to relative coordinates before comparison.

Key insight:
- YOLO bbox: One large box covering entire figure/table
- OCR bbox: Many small boxes, one per text line
- Normalization makes coordinate systems irrelevant

Flow:
1. Load assets with their source dimensions (page_width_px, page_height_px)
2. Normalize asset bboxes to 0-1 range
3. Normalize text bboxes using render_raw dimensions from page_metadata
4. Compare normalized bboxes for overlap filtering
5. Interleave assets based on normalized Y position
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Any


def load_asset_metadata(json_path: str) -> Optional[Dict]:
    """Load and parse a single asset metadata JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"    [Warning] Could not load asset metadata: {json_path} - {e}")
        return None


def extract_and_normalize_asset_bbox(meta: Dict) -> Optional[Dict]:
    """
    Extract bbox from asset metadata and normalize to 0-1 range.
    
    Returns dict with both raw pixel values and normalized values.
    """
    bbox_data = meta.get('bbox', {})
    
    raw_bbox = None
    source_width = None
    source_height = None
    
    # Case 1: Nested dictionary (YOLO or Manual Export)
    if isinstance(bbox_data, dict):
        if 'pixels' in bbox_data:
            raw_bbox = bbox_data['pixels']
        elif 'pdf_units' in bbox_data:
            raw_bbox = bbox_data['pdf_units']
        elif 'xyxy' in bbox_data:
            # Handle xyxy format [x1, y1, x2, y2]
            xyxy = bbox_data['xyxy']
            raw_bbox = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
        
        source_width = bbox_data.get('page_width_px')
        source_height = bbox_data.get('page_height_px')
        
    # Case 2: Direct list
    elif isinstance(bbox_data, list):
        raw_bbox = bbox_data
    
    if not raw_bbox or len(raw_bbox) < 4:
        return None
    
    # Interpret as [x, y, width, height]
    x, y, w, h = raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3]
    
    # Calculate normalized coordinates if we have source dimensions
    norm_left = None
    norm_top = None
    norm_right = None
    norm_bottom = None
    
    if source_width and source_height and source_width > 0 and source_height > 0:
        norm_left = x / source_width
        norm_top = y / source_height
        norm_right = (x + w) / source_width
        norm_bottom = (y + h) / source_height
    
    return {
        # Raw pixel values
        "left": x,
        "top": y,
        "width": w,
        "height": h,
        "right": x + w,
        "bottom": y + h,
        # Source dimensions
        "_source_width": source_width,
        "_source_height": source_height,
        # Normalized 0-1 values
        "norm_left": norm_left,
        "norm_top": norm_top,
        "norm_right": norm_right,
        "norm_bottom": norm_bottom,
        "_is_normalized": norm_left is not None,
    }


def get_page_ocr_dimensions(page_metadata: Dict, page_number: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Get the OCR coordinate space dimensions (render_raw) for a page.
    """
    page_key = str(page_number)
    
    if page_key not in page_metadata:
        return None, None
    
    page_info = page_metadata[page_key]
    image_meta = page_info.get('image_meta', {})
    
    # Try render_raw first (what OCR uses)
    render_raw = image_meta.get('render_raw', {})
    if render_raw:
        width = render_raw.get('width_px')
        height = render_raw.get('height_px')
        if width and height:
            return width, height
    
    # Fall back to canonical
    canonical = image_meta.get('canonical', {})
    if canonical:
        width = canonical.get('width_px')
        height = canonical.get('height_px')
        if width and height:
            return width, height
    
    return None, None


def normalize_text_bbox(bbox: Dict, page_width: float, page_height: float) -> Dict:
    """
    Add normalized coordinates to a text element's bbox.
    """
    if not bbox or page_width <= 0 or page_height <= 0:
        return bbox
    
    result = dict(bbox)
    
    left = bbox.get('left', 0)
    top = bbox.get('top', 0)
    width = bbox.get('width', 0)
    height = bbox.get('height', 0)
    
    # Ensure right/bottom exist
    right = bbox.get('right', left + width)
    bottom = bbox.get('bottom', top + height)
    
    result['right'] = right
    result['bottom'] = bottom
    result['norm_left'] = left / page_width
    result['norm_top'] = top / page_height
    result['norm_right'] = right / page_width
    result['norm_bottom'] = bottom / page_height
    result['_is_normalized'] = True
    
    return result


def load_all_assets(exports_dir: str, doc_stem: str) -> List[Dict]:
    """Load all figure/table assets for a document."""
    assets = []
    asset_dir = os.path.join(exports_dir, doc_stem)
    
    if not os.path.isdir(asset_dir):
        print(f"    [Note] Asset directory not found: {asset_dir}")
        return assets
    
    print(f"    Scanning for assets in: {asset_dir}")
    
    for filename in os.listdir(asset_dir):
        if not filename.endswith(".json"):
            continue
        
        json_path = os.path.join(asset_dir, filename)
        meta = load_asset_metadata(json_path)
        
        if not meta:
            continue
        
        # Normalize asset type
        raw_type = meta.get("asset_type", "").lower()
        if raw_type in ["fig", "figure"]:
            meta['type'] = 'figure'
        elif raw_type in ["tab", "table"]:
            meta['type'] = 'table'
        else:
            meta['type'] = 'figure'
        
        # Extract and normalize bbox
        bbox = extract_and_normalize_asset_bbox(meta)
        if bbox:
            meta['bbox'] = bbox
        
        # Get page number
        page = meta.get('page', meta.get('page_number', 9999))
        meta['page_number'] = page
        
        assets.append(meta)
    
    fig_count = sum(1 for a in assets if a['type'] == 'figure')
    tab_count = sum(1 for a in assets if a['type'] == 'table')
    print(f"    Loaded {len(assets)} assets ({fig_count} figures, {tab_count} tables)")
    
    # Check normalization status
    normalized_count = sum(1 for a in assets if a.get('bbox', {}).get('_is_normalized'))
    if normalized_count < len(assets):
        print(f"    [Warning] Only {normalized_count}/{len(assets)} assets have normalized bboxes")
    
    return assets


def calculate_normalized_overlap(text_bbox: Dict, asset_bbox: Dict) -> float:
    """
    Calculate overlap ratio using normalized coordinates.
    Returns what fraction of the text box is covered by the asset.
    """
    # Check if both have normalized coordinates
    if not text_bbox.get('_is_normalized') or not asset_bbox.get('_is_normalized'):
        return 0.0
    
    # Get normalized bounds
    t_left = text_bbox['norm_left']
    t_top = text_bbox['norm_top']
    t_right = text_bbox['norm_right']
    t_bottom = text_bbox['norm_bottom']
    
    a_left = asset_bbox['norm_left']
    a_top = asset_bbox['norm_top']
    a_right = asset_bbox['norm_right']
    a_bottom = asset_bbox['norm_bottom']
    
    # Calculate intersection
    x_left = max(t_left, a_left)
    y_top = max(t_top, a_top)
    x_right = min(t_right, a_right)
    y_bottom = min(t_bottom, a_bottom)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    text_area = (t_right - t_left) * (t_bottom - t_top)
    
    if text_area <= 0:
        return 0.0
    
    return intersection / text_area


def filter_overlapped_text(
    elements: List[Dict], 
    assets: List[Dict],
    page_metadata: Dict,
    overlap_threshold: float = 0.60,
    verbose: bool = False
) -> Tuple[List[Dict], int]:
    """
    Remove text elements that overlap significantly with assets.
    Uses normalized coordinates for comparison.
    """
    if not assets:
        return elements, 0
    
    # Group assets by page
    assets_by_page = {}
    for asset in assets:
        page = asset.get('page_number', 9999)
        if page not in assets_by_page:
            assets_by_page[page] = []
        assets_by_page[page].append(asset)
    
    filtered_elements = []
    removed_count = 0
    
    for elem in elements:
        bbox = elem.get('bbox')
        if not bbox:
            filtered_elements.append(elem)
            continue
        
        page = elem.get('page_number', 9999)
        if page not in assets_by_page:
            filtered_elements.append(elem)
            continue
        
        # Get page dimensions and normalize text bbox
        page_width, page_height = get_page_ocr_dimensions(page_metadata, page)
        if not page_width or not page_height:
            filtered_elements.append(elem)
            continue
        
        norm_text_bbox = normalize_text_bbox(bbox, page_width, page_height)
        
        # Check overlap with each asset on this page
        is_overlapped = False
        for asset in assets_by_page[page]:
            asset_bbox = asset.get('bbox', {})
            
            overlap = calculate_normalized_overlap(norm_text_bbox, asset_bbox)
            
            if overlap > overlap_threshold:
                is_overlapped = True
                if verbose:
                    print(f"      [Filter] Text on p{page}: overlap={overlap:.2f}")
                    print(f"               Text norm: ({norm_text_bbox['norm_left']:.3f}, {norm_text_bbox['norm_top']:.3f}) - "
                          f"({norm_text_bbox['norm_right']:.3f}, {norm_text_bbox['norm_bottom']:.3f})")
                    print(f"               Asset norm: ({asset_bbox['norm_left']:.3f}, {asset_bbox['norm_top']:.3f}) - "
                          f"({asset_bbox['norm_right']:.3f}, {asset_bbox['norm_bottom']:.3f})")
                break
        
        if is_overlapped:
            removed_count += 1
        else:
            filtered_elements.append(elem)
    
    return filtered_elements, removed_count


def integrate_assets_with_elements(
    elements: List[Dict], 
    assets: List[Dict],
    page_metadata: Optional[Dict] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Merge assets into element stream using normalized Y positions for ordering.
    
    Algorithm:
    1. Filter overlapping text (using normalized coordinates)
    2. Sort assets by normalized top position
    3. Interleave assets before text elements they precede (by norm_top)
    """
    if not assets:
        return elements
    
    if not page_metadata:
        print(f"    [Warning] No page_metadata - cannot normalize coordinates")
        print(f"              Falling back to raw pixel comparison (may be inaccurate)")
    
    # --- STEP 1: Filter overlapping text ---
    if page_metadata:
        clean_elements, removed_count = filter_overlapped_text(
            elements, assets, page_metadata,
            overlap_threshold=0.60,
            verbose=verbose
        )
        if removed_count > 0:
            print(f"    Filtered {removed_count} text blocks overlapping with assets")
    else:
        clean_elements = elements
    
    # --- STEP 2: Group and sort assets by page and normalized top ---
    assets_by_page: Dict[int, List[Dict]] = {}
    for asset in assets:
        page = asset.get('page_number', 9999)
        if isinstance(page, str) and page.isdigit():
            page = int(page)
        
        if page not in assets_by_page:
            assets_by_page[page] = []
        assets_by_page[page].append(asset)
    
    # Sort by normalized top (fall back to raw top if not normalized)
    for page in assets_by_page:
        assets_by_page[page].sort(
            key=lambda a: a.get('bbox', {}).get('norm_top') or 
                          (a.get('bbox', {}).get('top', 9999) / 10000)
        )
    
    # --- STEP 3: Interleave assets with text elements ---
    result = []
    current_page = None
    
    for element in clean_elements:
        elem_page = element.get('page_number', 9999)
        if isinstance(elem_page, str) and elem_page.isdigit():
            elem_page = int(elem_page)
        
        # Flush remaining assets from previous page
        if current_page is not None and elem_page != current_page:
            if current_page in assets_by_page:
                remaining = assets_by_page.pop(current_page)
                result.extend(remaining)
        
        current_page = elem_page
        
        # Get element's normalized top position
        elem_bbox = element.get('bbox', {})
        if page_metadata:
            page_width, page_height = get_page_ocr_dimensions(page_metadata, current_page)
            if page_height and page_height > 0:
                elem_norm_top = elem_bbox.get('top', 0) / page_height
            else:
                elem_norm_top = elem_bbox.get('top', 0) / 10000
        else:
            elem_norm_top = elem_bbox.get('top', 0) / 10000
        
        # Insert assets that come before this element
        if current_page in assets_by_page:
            page_assets = assets_by_page[current_page]
            
            while page_assets:
                asset = page_assets[0]
                asset_norm_top = asset.get('bbox', {}).get('norm_top')
                
                if asset_norm_top is None:
                    asset_top = asset.get('bbox', {}).get('top', 9999)
                    asset_norm_top = asset_top / 10000
                
                if asset_norm_top < elem_norm_top:
                    result.append(page_assets.pop(0))
                    if verbose:
                        print(f"      Inserted asset at norm_top={asset_norm_top:.3f} "
                              f"before element at {elem_norm_top:.3f}")
                else:
                    break
        
        result.append(element)
    
    # Flush final page's remaining assets
    if current_page is not None and current_page in assets_by_page:
        result.extend(assets_by_page.pop(current_page))
    
    # Add orphaned assets (pages with no text)
    for page in sorted(assets_by_page.keys()):
        page_assets = assets_by_page[page]
        result.extend(page_assets)
        print(f"      Added {len(page_assets)} assets to text-less page {page}")
    
    return result


def run_asset_integration(
    input_path: str,
    output_path: str,
    exports_dir: str,
    doc_stem: str,
    positioning_mode: str = "bbox",
    verbose: bool = False
):
    """
    Main entry point for asset integration.
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
    
    if verbose:
        print(f"  - Loaded {len(elements)} elements")
        print(f"  - page_metadata has {len(page_metadata)} pages")
    
    # Load assets
    assets = load_all_assets(exports_dir, doc_stem)
    
    # Integrate
    integrated = integrate_assets_with_elements(
        elements, assets, page_metadata, verbose=verbose
    )
    
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
    print(f"  - Final count: {len(integrated)} elements ({asset_count} assets)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) >= 4:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        exports_dir = sys.argv[3]
        doc_stem = sys.argv[4] if len(sys.argv) > 4 else "unknown_doc"
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        
        run_asset_integration(input_file, output_file, exports_dir, doc_stem, verbose=verbose)
    else:
        print("Usage: python asset_processor.py <input.json> <output.json> <exports_dir> [doc_stem] [-v]")