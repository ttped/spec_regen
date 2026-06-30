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


def get_page_ocr_dimensions(page_metadata: Dict, page_number: int, verbose: bool = False) -> Tuple[Optional[float], Optional[float]]:
    """
    Get the OCR coordinate space dimensions for a page.
    
    Handles multiple metadata structures:
    - page_info['image_meta']['render_raw']['width_px'] (preferred)
    - page_info['image_meta']['canonical']['width_px']
    - page_info['page_width'] / page_info['page_height'] (convenience keys)
    """
    # Try both string and int versions of the key
    page_key_str = str(page_number)
    page_key_int = int(page_number) if isinstance(page_number, (int, str)) else None
    
    page_info = None
    
    # Try string key first (most common in JSON)
    if page_key_str in page_metadata:
        page_info = page_metadata[page_key_str]
    # Try integer key
    elif page_key_int is not None and page_key_int in page_metadata:
        page_info = page_metadata[page_key_int]
    
    if page_info is None:
        if verbose:
            print(f"    [Debug] Page {page_number} NOT FOUND in page_metadata")
        return None, None
    
    # === Try nested image_meta structure (from preserved OCR data) ===
    image_meta = page_info.get('image_meta', {})
    if image_meta:
        # Try render_raw first (what OCR uses)
        render_raw = image_meta.get('render_raw', {})
        if render_raw:
            width = render_raw.get('width_px')
            height = render_raw.get('height_px')
            if width and height:
                if verbose:
                    print(f"    [Debug] Found render_raw: {width} x {height}")
                return width, height
        
        # Try canonical
        canonical = image_meta.get('canonical', {})
        if canonical:
            width = canonical.get('width_px')
            height = canonical.get('height_px')
            if width and height:
                if verbose:
                    print(f"    [Debug] Found canonical: {width} x {height}")
                return width, height
    
    # === Try convenience keys at page_info level ===
    width = page_info.get('page_width')
    height = page_info.get('page_height')
    if width and height:
        if verbose:
            print(f"    [Debug] Found page_width/page_height: {width} x {height}")
        return width, height
    
    if verbose:
        print(f"    [Debug] Could not find dimensions for page {page_number}")
        print(f"    [Debug] page_info keys: {list(page_info.keys())}")
    
    return None, None


def normalize_text_bbox(bbox: Dict, page_width: float, page_height: float) -> Dict:
    """
    Add normalized coordinates to a text element's bbox.
    
    Returns a NEW dict with normalized values added.
    """
    if not bbox:
        raise ValueError("normalize_text_bbox called with empty bbox")
    
    if page_width <= 0 or page_height <= 0:
        raise ValueError(f"Invalid page dimensions: {page_width} x {page_height}")
    
    # Create a copy to avoid modifying the original
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
    result['_norm_page_dims'] = (page_width, page_height)
    
    return result


def resolve_asset_directory(exports_dir: str, doc_stem: str) -> Optional[str]:
    """
    Find the actual asset directory for a doc_stem, handling naming mismatches
    between OCR stems (may contain spaces) and YOLO export dirs (spaces → underscores).
    
    Returns the resolved directory path, or None if no match is found.
    """
    # Exact match first
    exact_path = os.path.join(exports_dir, doc_stem)
    if os.path.isdir(exact_path):
        return exact_path
    
    if not os.path.isdir(exports_dir):
        return None
    
    # Normalize for comparison: collapse both spaces and underscores
    def normalize(name: str) -> str:
        return name.replace(" ", "_").replace("-", "_").lower()
    
    target = normalize(doc_stem)
    
    for entry in os.listdir(exports_dir):
        entry_path = os.path.join(exports_dir, entry)
        if os.path.isdir(entry_path) and normalize(entry) == target:
            print(f"    [Resolved] Stem '{doc_stem}' matched directory '{entry}'")
            return entry_path
    
    return None


def load_all_assets(exports_dir: str, doc_stem: str) -> List[Dict]:
    """Load all figure/table assets for a document."""
    assets = []
    asset_dir = resolve_asset_directory(exports_dir, doc_stem)
    
    if not asset_dir:
        print(f"    [Note] Asset directory not found for stem '{doc_stem}' in {exports_dir}")
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
        elif raw_type in ["tab_layout"]:
            meta['type'] = 'table_layout'
        elif raw_type in ["eq", "equation", "isolate_formula", "formula"]:
            meta['type'] = 'equation'
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
    eq_count = sum(1 for a in assets if a['type'] == 'equation')
    tab_layout_count = sum(1 for a in assets if a['type'] == 'table_layout')
    print(f"    Loaded {len(assets)} assets ({fig_count} figures, {tab_count} tables, {eq_count} equations, {tab_layout_count} layout tables)")
    
    # Check normalization status
    normalized_count = sum(1 for a in assets if a.get('bbox', {}).get('_is_normalized'))
    if normalized_count < len(assets):
        print(f"    [Warning] Only {normalized_count}/{len(assets)} assets have normalized bboxes")
    
    return assets


def normalize_all_element_bboxes(
    elements: List[Dict],
    page_metadata: Dict,
    verbose: bool = False,
    strict: bool = False
) -> List[Dict]:
    """
    Add normalized bbox coordinates to ALL elements that have bboxes.
    This modifies elements in place and returns the list.
    
    Args:
        elements: List of document elements
        page_metadata: Dict mapping page numbers to page info
        verbose: Print debug info
        strict: If True, raise exceptions on failures instead of silently skipping
    """
    normalized_count = 0
    skipped_no_bbox = 0
    skipped_no_dims = 0
    failed_pages = set()
    
    for i, elem in enumerate(elements):
        bbox = elem.get('bbox')
        if not bbox:
            skipped_no_bbox += 1
            continue
        
        page = elem.get('page_number', 9999)

        page_width, page_height = get_page_ocr_dimensions(page_metadata, page, verbose=verbose)
        
        if not page_width or not page_height:
            skipped_no_dims += 1
            failed_pages.add(page)
            
            if strict:
                # Do one more verbose lookup to show exactly what's happening
                print(f"\n    [ERROR] Cannot normalize element {i} (type: {elem.get('type')}) on page {page}")
                get_page_ocr_dimensions(page_metadata, page, verbose=True)
                raise ValueError(
                    f"Cannot normalize element {i} on page {page}: "
                    f"No OCR dimensions found. See debug output above."
                )
            continue
        
        # Add normalized values to the bbox
        elem['bbox'] = normalize_text_bbox(bbox, page_width, page_height)
        normalized_count += 1
    
    print(f"    Normalized {normalized_count} element bboxes")
    if skipped_no_bbox > 0:
        print(f"    [WARNING] {skipped_no_bbox} elements had no bbox at all")
    if skipped_no_dims > 0:
        print(f"    [WARNING] Failed to normalize {skipped_no_dims} elements on pages: {sorted(failed_pages)}")
        print(f"    [WARNING] These elements will use fallback positioning (may be inaccurate)")
        print(f"    [WARNING] Check that page_metadata contains dimensions for these pages")
    
    # Verify normalization worked
    if normalized_count > 0:
        sample = next((e for e in elements if e.get('bbox', {}).get('_is_normalized')), None)
        if sample:
            bbox = sample['bbox']
            print(f"    [OK] Sample: norm_top={bbox.get('norm_top'):.4f}")
        else:
            print(f"    [WARNING] No elements have _is_normalized=True!")
    
    return elements



def _resolve_norm_top(bbox: Optional[Dict], page: int, page_metadata: Dict, default: float) -> float:
    """
    Best-effort normalized (0-1) top position for a bbox.

    Prefers an existing norm_top; otherwise normalizes the raw top using the
    page height; falls back to `default` when nothing is available so the item
    sorts to a sensible place (top for text, bottom for unplaceable assets).
    """
    bbox = bbox or {}
    norm_top = bbox.get('norm_top')
    if norm_top is not None:
        return norm_top

    raw_top = bbox.get('top')
    if raw_top is not None:
        _, page_height = get_page_ocr_dimensions(page_metadata, page)
        if page_height:
            return raw_top / page_height
        return raw_top / 10000

    return default


def integrate_assets_with_elements(
    elements: List[Dict],
    assets: List[Dict],
    page_metadata: Optional[Dict] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Merge assets into the element stream in true reading order.

    Note: Overlap filtering of text inside figures/tables is handled earlier
    in section_processor.py at the line level, before text aggregation.

    Algorithm:
    1. Normalize all element bboxes (so they have norm_* values).
    2. Expand each section into a heading unit plus one unit per body paragraph
       (using `_body_blocks`), each carrying its own vertical position. This
       lets a figure/table land *between* the paragraphs it sits between,
       instead of being pushed to the end of the whole section.
    3. Place every unit and asset with a single stable sort by
       (page_number, normalized_top). This keeps assets on their own page and
       in the right vertical slot — no more orphaned assets piling up at the
       end of the document.
    """
    if not assets:
        return elements

    if not page_metadata:
        print(f"    [Warning] No page_metadata - asset positioning will use raw fallback estimation.")
        page_metadata = {}

    # --- STEP 1: Normalize all element bboxes ---
    print(f"    Normalizing element bboxes...")
    elements = normalize_all_element_bboxes(elements, page_metadata, verbose=verbose)

    # --- STEP 2: Build positioned units (expanding sections by paragraph) ---
    # Each entry: (page, norm_top, sequence, item). `sequence` preserves input
    # order as a stable tiebreaker for items sharing a position.
    units: List[Tuple[int, float, int, Dict]] = []
    seq = 0

    for element in elements:
        page = int(element.get('page_number', 9999))
        body_blocks = element.get('_body_blocks')

        if element.get('type') == 'section' and body_blocks:
            # Heading unit: same section element minus its body (rendered as
            # separate paragraph units below) and minus the internal field.
            heading = {k: v for k, v in element.items() if k != '_body_blocks'}
            heading['content'] = ''
            head_top = _resolve_norm_top(element.get('bbox'), page, page_metadata, 0.0)
            units.append((page, head_top, seq, heading))
            seq += 1

            for block in body_blocks:
                text = str(block.get('text', '')).strip()
                if not text:
                    continue
                bbox = block.get('bbox')
                if bbox and not bbox.get('_is_normalized'):
                    pw, ph = get_page_ocr_dimensions(page_metadata, page)
                    if pw and ph:
                        bbox = normalize_text_bbox(bbox, pw, ph)
                body_top = _resolve_norm_top(bbox, page, page_metadata, head_top)
                body_unit = {
                    'type': 'unassigned_text_block',
                    'page_number': element.get('page_number'),
                    'content': text,
                    'bbox': bbox,
                }
                units.append((page, body_top, seq, body_unit))
                seq += 1
        else:
            # Non-section elements (and sections without body) pass through;
            # strip the internal field if it somehow lingers.
            item = {k: v for k, v in element.items() if k != '_body_blocks'} \
                if '_body_blocks' in element else element
            top = _resolve_norm_top(element.get('bbox'), page, page_metadata, 0.0)
            units.append((page, top, seq, item))
            seq += 1

    # --- STEP 3: Add assets (unplaceable ones sort to the page bottom) ---
    for asset in assets:
        page = int(asset.get('page_number', 9999))
        top = _resolve_norm_top(asset.get('bbox'), page, page_metadata, 1.0)
        units.append((page, top, seq, asset))
        seq += 1

    # --- STEP 4: Stable sort by (page, vertical position) ---
    units.sort(key=lambda u: (u[0], u[1], u[2]))

    if verbose:
        asset_count = sum(1 for _, _, _, item in units
                          if item.get('type') in ('figure', 'table', 'equation', 'table_layout'))
        print(f"      Ordered {len(units)} units ({asset_count} assets) by page and position")

    return [item for _, _, _, item in units]


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
    
    asset_count = sum(1 for e in integrated if e.get('type') in ('figure', 'table', 'equation', 'table_layout'))
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