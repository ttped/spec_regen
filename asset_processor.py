"""
asset_processor.py - Handles loading and positioning of figures and tables.

This module:
1. Loads figure/table metadata from exported JSON files
2. Normalizes bbox coordinates accounting for DPI differences
3. Integrates assets into the document element stream at correct positions
4. Sorts elements by page and vertical position (y-coordinate)

The bbox is typically [x, y, width, height] or [x1, y1, x2, y2] in PDF units.
We use the y-coordinate (top of the bbox) to determine vertical ordering within a page.
"""

import os
import json
from typing import List, Dict, Tuple, Optional, Any


# Common DPI values - adjust these based on your actual extraction settings
DEFAULT_OCR_DPI = 300      # DPI used for OCR text extraction
DEFAULT_EXPORT_DPI = 150   # DPI used for figure/table image export


def normalize_coordinate(value: float, from_dpi: float, to_dpi: float) -> float:
    """
    Normalize a coordinate from one DPI to another.
    
    Args:
        value: The coordinate value to normalize
        from_dpi: The DPI the value was captured at
        to_dpi: The target DPI to normalize to
    
    Returns:
        The normalized coordinate value
    """
    if from_dpi == to_dpi or from_dpi == 0:
        return value
    return value * (to_dpi / from_dpi)


def get_bbox_y_position(bbox: List[float], pdf_units: bool = True) -> float:
    """
    Extract the Y position (top edge) from a bbox for vertical ordering.
    
    Bbox formats supported:
    - [x, y, width, height] - y is the top position
    - [x1, y1, x2, y2] - y1 is the top position
    
    Args:
        bbox: The bounding box list
        pdf_units: Whether the bbox is in PDF coordinate space (origin at bottom-left)
    
    Returns:
        The Y coordinate representing vertical position (lower = higher on page)
    """
    if not bbox or len(bbox) < 2:
        return 9999  # Default to end of page if no bbox
    
    # bbox[1] is typically the Y coordinate (top of the box)
    return bbox[1]


def load_asset_metadata(json_path: str) -> Optional[Dict]:
    """
    Load and parse a single asset metadata JSON file.
    
    Expected structure:
    {
        "asset_id": "Figure 1",
        "asset_type": "fig" or "tab",
        "page": 12,
        "bbox": {
            "pdf_units": [x, y, width, height]
        },
        "export": {
            "image_file": "figure_1.png",
            "dpi": 150
        }
    }
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"    [Warning] Could not load asset metadata: {json_path} - {e}")
        return None


def load_all_assets(exports_dir: str, doc_stem: str) -> List[Dict]:
    """
    Load all figure/table assets for a document.
    
    Args:
        exports_dir: Base directory containing exported assets
        doc_stem: Document stem name (subdirectory name)
    
    Returns:
        List of asset dictionaries with normalized structure
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
            # Default to figure if unclear
            meta['type'] = 'figure'
        
        # Extract positioning info
        page = meta.get('page', 9999)
        bbox_data = meta.get('bbox', {})
        
        # Handle different bbox formats
        if isinstance(bbox_data, dict):
            bbox = bbox_data.get('pdf_units', bbox_data.get('pixels', []))
        elif isinstance(bbox_data, list):
            bbox = bbox_data
        else:
            bbox = []
        
        # Get export DPI for potential normalization
        export_info = meta.get('export', {})
        export_dpi = export_info.get('dpi', DEFAULT_EXPORT_DPI)
        
        # Calculate sort position (y-coordinate within page)
        y_position = get_bbox_y_position(bbox)
        
        # Store normalized positioning data
        meta['page'] = page
        meta['_bbox'] = bbox
        meta['_y_position'] = y_position
        meta['_export_dpi'] = export_dpi
        
        assets.append(meta)
    
    print(f"    Loaded {len(assets)} assets ({sum(1 for a in assets if a['type'] == 'figure')} figures, {sum(1 for a in assets if a['type'] == 'table')} tables)")
    return assets


def calculate_element_position(element: Dict, ocr_page_height: float = 1000) -> Tuple[int, float]:
    """
    Calculate the sort position for any element (section, text block, or asset).
    
    Returns:
        Tuple of (page_number, y_position) for sorting
    """
    page = element.get('page', element.get('page_number', 9999))
    
    # For assets with bbox
    if '_y_position' in element:
        return (page, element['_y_position'])
    
    # For sections/text blocks, we don't have precise y-position
    # Use a default that places them in document order
    # chunk_index can help preserve original order
    chunk_idx = element.get('chunk_index', 0)
    sub_chunk = element.get('sub_chunk_index', 0)
    
    # Create a synthetic y-position based on chunk ordering
    # This keeps text elements in their original order
    y_pos = chunk_idx * 100 + sub_chunk
    
    return (page, y_pos)


def integrate_assets_with_sections(
    sections: List[Dict], 
    assets: List[Dict],
    ocr_dpi: float = DEFAULT_OCR_DPI
) -> List[Dict]:
    """
    Merge assets into the section stream, positioning them correctly by page and bbox.
    
    This function interleaves figures/tables with sections based on their
    vertical position on each page.
    
    Args:
        sections: List of section/text block elements from section_processor
        assets: List of figure/table assets from load_all_assets
        ocr_dpi: The DPI used for OCR (for coordinate normalization)
    
    Returns:
        Combined list of elements sorted by page and vertical position
    """
    if not assets:
        return sections
    
    # Normalize asset coordinates if DPI differs
    for asset in assets:
        export_dpi = asset.get('_export_dpi', DEFAULT_EXPORT_DPI)
        if export_dpi != ocr_dpi and asset.get('_y_position', 0) < 9999:
            asset['_y_position'] = normalize_coordinate(
                asset['_y_position'], 
                export_dpi, 
                ocr_dpi
            )
    
    # Combine all elements
    all_elements = sections + assets
    
    # Sort by page first, then by y-position within page
    def sort_key(elem):
        page, y_pos = calculate_element_position(elem)
        return (page, y_pos)
    
    all_elements.sort(key=sort_key)
    
    return all_elements


def integrate_assets_by_page_end(
    sections: List[Dict], 
    assets: List[Dict]
) -> List[Dict]:
    """
    Alternative integration strategy: place all assets at the end of their respective pages.
    
    This is simpler but less precise than bbox-based positioning.
    Use this if bbox coordinates are unreliable.
    
    Args:
        sections: List of section/text block elements
        assets: List of figure/table assets
    
    Returns:
        Combined list with assets placed at page ends
    """
    if not assets:
        return sections
    
    # Group assets by page
    assets_by_page = {}
    for asset in assets:
        page = asset.get('page', 9999)
        if page not in assets_by_page:
            assets_by_page[page] = []
        assets_by_page[page].append(asset)
    
    # Sort assets within each page by y-position
    for page in assets_by_page:
        assets_by_page[page].sort(key=lambda a: a.get('_y_position', 9999))
    
    # Build result by inserting assets after each page's sections
    result = []
    current_page = None
    
    for section in sections:
        section_page = section.get('page', section.get('page_number', 9999))
        
        # If we've moved to a new page, insert assets from the previous page
        if current_page is not None and section_page != current_page:
            if current_page in assets_by_page:
                result.extend(assets_by_page[current_page])
                del assets_by_page[current_page]
        
        result.append(section)
        current_page = section_page
    
    # Don't forget assets from the last page
    if current_page in assets_by_page:
        result.extend(assets_by_page[current_page])
        del assets_by_page[current_page]
    
    # Add any remaining assets (from pages with no sections)
    for page in sorted(assets_by_page.keys()):
        result.extend(assets_by_page[page])
    
    return result


def run_asset_integration(
    input_path: str,
    output_path: str,
    exports_dir: str,
    doc_stem: str,
    positioning_mode: str = "bbox"  # "bbox" or "page_end"
):
    """
    Main function to load sections and integrate assets.
    
    Args:
        input_path: Path to organized sections JSON (from section_processor)
        output_path: Path to save integrated output
        exports_dir: Base directory for exported figures/tables
        doc_stem: Document stem name
        positioning_mode: "bbox" for precise positioning, "page_end" for simpler approach
    """
    print(f"  - Loading sections from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"  - [Error] Input file not found: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        sections = json.load(f)
    
    print(f"  - Found {len(sections)} section elements")
    
    # Load assets
    print(f"  - Loading assets from: {exports_dir}/{doc_stem}")
    assets = load_all_assets(exports_dir, doc_stem)
    
    # Integrate based on positioning mode
    if positioning_mode == "bbox":
        print(f"  - Integrating assets using bbox positioning")
        integrated = integrate_assets_with_sections(sections, assets)
    else:
        print(f"  - Integrating assets at page ends")
        integrated = integrate_assets_by_page_end(sections, assets)
    
    # Save result
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(integrated, f, indent=4)
    
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
        mode = sys.argv[5] if len(sys.argv) > 5 else "bbox"
        
        run_asset_integration(input_file, output_file, exports_dir, doc_stem, mode)
    else:
        print("Usage: python asset_processor.py <input.json> <output.json> <exports_dir> [doc_stem] [bbox|page_end]")