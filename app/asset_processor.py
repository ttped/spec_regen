"""
asset_processor.py - Handles loading and positioning of figures and tables.

FIXED VERSION - Addresses coordinate system mismatch between YOLO and OCR

This module:
1. Loads figure/table metadata from exported JSON files
2. SCALES asset bboxes to match OCR coordinate system (render_full -> render_raw)
3. Filters out OCR text that lies INSIDE these assets (Double OCR Fix)
4. Integrates assets into the document element stream based on VISUAL POSITION
5. Interleaves assets between text blocks based on vertical (Y-axis) coordinates

COORDINATE SPACES:
- render_full: Original page images (~2530×3300) - YOLO runs on these (docs_images/)
- render_raw:  Upscaled images (~5052×6600) - OCR/Tesseract runs on these
- canonical:   Preprocessed (same dims as render_raw)

Key Fix:
- YOLO detects on render_full images (smaller)
- OCR text bboxes are in render_raw coordinates (larger, ~2x)
- This module scales YOLO bboxes UP to match OCR space before comparison
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
    
    Also extracts the source image dimensions for later scaling.
    """
    bbox_data = meta.get('bbox', {})
    
    raw_bbox = None
    bbox_source = None
    source_width = None
    source_height = None
    
    # Case 1: Nested dictionary (YOLO or Manual Export)
    if isinstance(bbox_data, dict):
        if 'pixels' in bbox_data:
            raw_bbox = bbox_data['pixels']
            bbox_source = 'pixels'
        elif 'pdf_units' in bbox_data:
            raw_bbox = bbox_data['pdf_units']
            bbox_source = 'pdf_units'
        
        # Get source image dimensions (from YOLO metadata)
        source_width = bbox_data.get('page_width_px')
        source_height = bbox_data.get('page_height_px')
        
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
        "_raw": raw_bbox,
        "_source_width": source_width,
        "_source_height": source_height,
    }


def scale_bbox_to_target(
    bbox: Dict, 
    source_width: float, 
    source_height: float,
    target_width: float, 
    target_height: float
) -> Dict:
    """
    Scale a bounding box from source coordinate system to target coordinate system.
    
    This is critical for aligning YOLO bboxes (render_raw) with OCR bboxes (canonical).
    
    Args:
        bbox: The bounding box dict with left, top, width, height, right, bottom
        source_width: Width of the image the bbox was detected on
        source_height: Height of the image the bbox was detected on
        target_width: Width of the target coordinate system
        target_height: Height of the target coordinate system
    
    Returns:
        A new bbox dict with scaled coordinates
    """
    if source_width <= 0 or source_height <= 0:
        return bbox  # Can't scale, return as-is
    
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    
    scaled = {
        "left": bbox["left"] * scale_x,
        "top": bbox["top"] * scale_y,
        "width": bbox["width"] * scale_x,
        "height": bbox["height"] * scale_y,
        "right": bbox["right"] * scale_x,
        "bottom": bbox["bottom"] * scale_y,
        "_source": bbox.get("_source"),
        "_raw": bbox.get("_raw"),
        "_scaled_from": (source_width, source_height),
        "_scaled_to": (target_width, target_height),
        "_scale_factors": (scale_x, scale_y),
    }
    
    return scaled


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
        
        # Extract raw bbox (includes source dimensions)
        bbox = extract_asset_bbox_raw(meta)
        if bbox:
            meta['bbox'] = bbox
            meta['_bbox_source_dims'] = (
                bbox.get('_source_width'),
                bbox.get('_source_height')
            )
        
        # Get page number (normalize key)
        page = meta.get('page', meta.get('page_number', 9999))
        meta['page_number'] = page
        
        assets.append(meta)
        loaded_count += 1
    
    fig_count = sum(1 for a in assets if a['type'] == 'figure')
    tab_count = sum(1 for a in assets if a['type'] == 'table')
    print(f"    Loaded {loaded_count} assets ({fig_count} figures, {tab_count} tables)")
    
    return assets


def get_page_ocr_dimensions(page_metadata: Dict, page_number: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract the OCR coordinate space dimensions for a given page from page_metadata.
    
    The page_metadata comes from the OCR JSON and contains image_meta with
    render_full, render_raw, and canonical dimensions.
    
    Priority order for OCR space (what text bboxes use):
    1. render_raw - the upscaled image that Tesseract actually processes
    2. canonical - preprocessed version (usually same dims as render_raw)
    
    Returns:
        Tuple of (width, height) or (None, None) if not found
    """
    page_key = str(page_number)
    
    if page_key not in page_metadata:
        return None, None
    
    page_info = page_metadata[page_key]
    image_meta = page_info.get('image_meta', {})
    
    # Try render_raw first (this is what OCR/Tesseract actually uses)
    render_raw = image_meta.get('render_raw', {})
    if render_raw:
        width = render_raw.get('width_px')
        height = render_raw.get('height_px')
        if width and height:
            return width, height
    
    # Fall back to canonical (usually same dimensions)
    canonical = image_meta.get('canonical', {})
    if canonical:
        width = canonical.get('width_px')
        height = canonical.get('height_px')
        if width and height:
            return width, height
    
    return None, None


def get_page_render_full_dimensions(page_metadata: Dict, page_number: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract the render_full (original) image dimensions for a given page.
    
    render_full is the original rendered image BEFORE upscaling for OCR.
    This is what YOLO processes (the images in docs_images/).
    
    Returns:
        Tuple of (width, height) or (None, None) if not found
    """
    page_key = str(page_number)
    
    if page_key not in page_metadata:
        return None, None
    
    page_info = page_metadata[page_key]
    image_meta = page_info.get('image_meta', {})
    
    render_full = image_meta.get('render_full', {})
    if render_full:
        width = render_full.get('width_px')
        height = render_full.get('height_px')
        if width and height:
            return width, height
    
    return None, None


def scale_assets_to_ocr_space(
    assets: List[Dict], 
    page_metadata: Dict
) -> Tuple[List[Dict], int]:
    """
    Scale asset bounding boxes from YOLO coordinate space to OCR coordinate space.
    
    COORDINATE SPACES:
    - YOLO runs on render_full images (e.g., 2530 × 3300) - stored in docs_images/
    - OCR text bboxes are in render_raw space (e.g., 5052 × 6600) - upscaled for Tesseract
    
    This function scales YOLO bboxes UP to match OCR coordinates.
    
    Source dimensions priority:
    1. render_full from page_metadata (most accurate)
    2. YOLO-recorded dimensions from bbox metadata (fallback)
    
    Target dimensions:
    - render_raw from page_metadata (what OCR uses)
    
    Args:
        assets: List of asset metadata dicts
        page_metadata: Dict mapping page numbers to page info (from OCR JSON)
    
    Returns:
        Tuple of (scaled_assets, num_scaled)
    """
    scaled_assets = []
    num_scaled = 0
    
    for asset in assets:
        asset_copy = dict(asset)
        bbox = asset_copy.get('bbox')
        page = asset_copy.get('page_number', 9999)
        
        if not bbox:
            scaled_assets.append(asset_copy)
            continue
        
        # Get SOURCE dimensions (what YOLO ran on)
        # Priority 1: render_full from page_metadata
        source_width, source_height = get_page_render_full_dimensions(page_metadata, page)
        
        # Priority 2: Fall back to YOLO-recorded dimensions
        if not source_width or not source_height:
            source_width = bbox.get('_source_width')
            source_height = bbox.get('_source_height')
        
        # Get TARGET dimensions (OCR coordinate space = render_raw)
        target_width, target_height = get_page_ocr_dimensions(page_metadata, page)
        
        # Only scale if we have all dimensions and they differ
        if (source_width and source_height and 
            target_width and target_height and
            (source_width != target_width or source_height != target_height)):
            
            scaled_bbox = scale_bbox_to_target(
                bbox, 
                source_width, source_height,
                target_width, target_height
            )
            asset_copy['bbox'] = scaled_bbox
            asset_copy['_bbox_scaled'] = True
            num_scaled += 1
            
            # Log scaling info
            scale_x = target_width / source_width
            scale_y = target_height / source_height
            print(f"      Scaled asset on page {page}: "
                  f"{source_width:.0f}×{source_height:.0f} -> {target_width:.0f}×{target_height:.0f} "
                  f"(scale: {scale_x:.2f}x, {scale_y:.2f}x)")
        
        scaled_assets.append(asset_copy)
    
    return scaled_assets, num_scaled


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


def calculate_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    This is a more robust overlap metric that considers both boxes' sizes.
    """
    x_left = max(bbox1['left'], bbox2['left'])
    y_top = max(bbox1['top'], bbox2['top'])
    x_right = min(bbox1['right'], bbox2['right'])
    y_bottom = min(bbox1['bottom'], bbox2['bottom'])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    area1 = bbox1['width'] * bbox1['height']
    area2 = bbox2['width'] * bbox2['height']
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def filter_overlapped_text(
    elements: List[Dict], 
    assets: List[Dict],
    overlap_threshold: float = 0.60,
    verbose: bool = False
) -> Tuple[List[Dict], int]:
    """
    Removes text elements that are significantly overlapped by an asset (figure/table).
    This prevents "double OCR" where the text appears both in the image and as raw text.
    
    Args:
        elements: List of document elements (sections, text blocks, etc.)
        assets: List of assets with scaled bboxes
        overlap_threshold: Minimum overlap ratio to filter (0.0-1.0)
        verbose: If True, print details about each filtered element
    
    Returns:
        Tuple of (filtered_elements, removed_count)
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
        
        # Ensure text_bbox has right/bottom calculated
        if 'right' not in text_bbox and 'width' in text_bbox:
            text_bbox = dict(text_bbox)
            text_bbox['right'] = text_bbox['left'] + text_bbox['width']
            text_bbox['bottom'] = text_bbox['top'] + text_bbox['height']
        
        for asset in assets_by_page[page]:
            asset_bbox = asset.get('bbox')
            if not asset_bbox:
                continue
                
            # Check overlap
            overlap_ratio = calculate_overlap_ratio(text_bbox, asset_bbox)
            
            # THRESHOLD: If >60% of text area is inside the asset, filter it
            if overlap_ratio > overlap_threshold:
                is_overlapped = True
                if verbose:
                    print(f"      [Filter] Dropped text on p{page} "
                          f"(overlap {overlap_ratio:.2f} with {asset.get('asset_id', 'unknown')})")
                    print(f"               Text bbox: ({text_bbox['left']:.0f}, {text_bbox['top']:.0f}) - "
                          f"({text_bbox['right']:.0f}, {text_bbox['bottom']:.0f})")
                    print(f"               Asset bbox: ({asset_bbox['left']:.0f}, {asset_bbox['top']:.0f}) - "
                          f"({asset_bbox['right']:.0f}, {asset_bbox['bottom']:.0f})")
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
    Merge assets into the element stream by INTERLEAVING them based on vertical position.
    
    Algorithm:
    1. Scale asset bboxes to OCR coordinate space (render_raw -> canonical)
    2. Filter out text elements that are visually covered by assets (Double OCR fix).
    3. Sort assets for a page by their 'top' (Y) coordinate.
    4. Iterate through text elements.
    5. If an asset's 'top' is LESS than the current text element's 'top', insert it BEFORE.
    6. Any remaining assets for the page are appended at the end of the page.
    
    Args:
        elements: List of document elements from OCR/structure processing
        assets: List of asset metadata dicts
        page_metadata: Dict mapping page numbers to page info (for coordinate scaling)
        verbose: If True, print detailed debug information
    """
    if not assets:
        return elements

    # --- STEP 0: Scale asset bboxes to OCR coordinate space ---
    if page_metadata:
        scaled_assets, num_scaled = scale_assets_to_ocr_space(assets, page_metadata)
        if num_scaled > 0:
            print(f"    Scaled {num_scaled} asset bboxes to OCR coordinate space.")
        assets = scaled_assets
    else:
        print(f"    [Warning] No page_metadata available - skipping bbox scaling")
        print(f"              Asset positions may not align correctly with text!")

    # --- STEP 1: Filter Double OCR Text ---
    clean_elements, removed_count = filter_overlapped_text(
        elements, assets, 
        overlap_threshold=0.60,
        verbose=verbose
    )
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
    positioning_mode: str = "bbox",
    verbose: bool = False
):
    """
    Main function to integrate assets.
    
    Args:
        input_path: Path to the structure JSON (from section_processor)
        output_path: Path to save the integrated JSON
        exports_dir: Base directory containing asset exports (e.g., yolo_exports/)
        doc_stem: Document stem name (for finding the right subfolder)
        positioning_mode: "bbox" (default) - future support for other modes
        verbose: If True, print detailed debug information
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
    
    # Integrate (now with coordinate scaling)
    integrated = integrate_assets_with_elements(
        elements, 
        assets, 
        page_metadata,
        verbose=verbose
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
    print(f"  - Final Element Count: {len(integrated)} (Integrated {asset_count} assets)")


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
        print("Usage: python asset_processor.py <input.json> <output.json> <exports_dir> [doc_stem] [-v|--verbose]")