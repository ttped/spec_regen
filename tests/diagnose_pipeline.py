"""
Diagnostic script to check where page_metadata dimensions are being lost.
Run this after each step to verify the data.

Usage:
    python diagnose_pipeline.py <json_file>
"""

import json
import sys

def diagnose_file(filepath):
    print(f"\n{'='*60}")
    print(f"Diagnosing: {filepath}")
    print(f"{'='*60}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load file: {e}")
        return
    
    # Check structure
    if isinstance(data, dict):
        print(f"Root type: dict with keys {list(data.keys())}")
        
        # Check page_metadata
        page_metadata = data.get('page_metadata', {})
        if not page_metadata:
            print("⚠️  WARNING: No page_metadata found!")
        else:
            print(f"\npage_metadata has {len(page_metadata)} pages:")
            for page_key, page_info in list(page_metadata.items())[:3]:  # First 3
                print(f"\n  Page '{page_key}':")
                print(f"    Keys: {list(page_info.keys())}")
                
                # Check for dimensions
                width = None
                height = None
                
                # Check nested image_meta
                image_meta = page_info.get('image_meta', {})
                if image_meta:
                    print(f"    ✓ image_meta found with keys: {list(image_meta.keys())}")
                    render_raw = image_meta.get('render_raw', {})
                    if render_raw:
                        width = render_raw.get('width_px')
                        height = render_raw.get('height_px')
                        if width and height:
                            print(f"    ✓ render_raw dimensions: {width} x {height}")
                        else:
                            print(f"    ⚠️  render_raw missing width_px/height_px")
                else:
                    print(f"    ⚠️  No image_meta found")
                
                # Check convenience keys
                pw = page_info.get('page_width')
                ph = page_info.get('page_height')
                if pw and ph:
                    print(f"    ✓ page_width/page_height: {pw} x {ph}")
                else:
                    print(f"    ⚠️  No page_width/page_height")
                
                if not (width and height) and not (pw and ph):
                    print(f"    ❌ NO DIMENSIONS FOUND FOR PAGE {page_key}")
        
        # Check elements
        elements = data.get('elements', [])
        print(f"\nElements: {len(elements)}")
        if elements:
            pages_in_elements = set()
            for e in elements:
                p = e.get('page_number')
                if p:
                    pages_in_elements.add(p)
            print(f"  Pages referenced in elements: {sorted(pages_in_elements)}")
            
            # Check if those pages exist in metadata
            if page_metadata:
                missing = []
                for p in pages_in_elements:
                    str_key = str(p)
                    int_key = int(p) if isinstance(p, (int, str)) else None
                    if str_key not in page_metadata and int_key not in page_metadata:
                        missing.append(p)
                if missing:
                    print(f"  ❌ MISSING pages in metadata: {missing}")
                else:
                    print(f"  ✓ All element pages found in metadata")
    
    elif isinstance(data, list):
        print(f"Root type: list with {len(data)} items")
        print("This is the old format without page_metadata wrapper!")
        print("⚠️  page_metadata will not be available downstream")
    
    print()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for filepath in sys.argv[1:]:
            diagnose_file(filepath)
    else:
        print("Usage: python diagnose_pipeline.py <json_file> [more files...]")
        print()
        print("Run on your pipeline outputs to find where metadata is lost:")
        print("  python diagnose_pipeline.py results_simple/*_organized.json")
        print("  python diagnose_pipeline.py results_simple/*_ml_filtered.json")
        print("  python diagnose_pipeline.py results_simple/*_with_assets.json")