"""
yolo_asset_extractor.py - Extract figures and tables from document pages using DocLayout-YOLO.

UPDATES:
- Added `is_hole_punch` filter to remove hole punch artifacts.
- Improved `extract_caption_text` robustness for OCR mapping.
- Added `clean_caption_text` to remove "Figure X" prefix so it doesn't duplicate in Word.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from doclayout_yolo import YOLOv10
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

HUGGINGFACE_MODEL = "juliozhao/DocLayout-YOLO-DocStructBench"

YOLO_CLASSES = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

ASSET_CLASSES = {"figure", "table", "isolate_formula"}
CAPTION_CLASSES = {"figure_caption", "table_caption", "formula_caption"}

ASSET_TO_CAPTION = {
    "figure": "figure_caption",
    "table": "table_caption",
    "isolate_formula": "formula_caption",
}


@dataclass
class Detection:
    class_name: str
    bbox_pixels: Tuple[int, int, int, int]
    confidence: float
    class_id: int


@dataclass
class DetectedAsset:
    asset_type: str
    page_number: int
    bbox_pixels: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    doc_stem: str
    image_width: int
    image_height: int
    has_caption: bool = False
    caption_bbox: Optional[Tuple[int, int, int, int]] = None
    caption_text: str = ""
    validated_by: str = "none"


def get_paths():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name != "docs" else script_dir
    return {
        'images_dir': project_root / "docs_images",
        'exports_dir': project_root / "yolo_exports",
        'raw_ocr_dir': project_root / "iris_ocr" / "CM_Spec_OCR_and_figtab_output" / "raw_data_advanced",
    }


def load_model(local_path: str = None) -> 'YOLOv10':
    if local_path is None:
        local_path = str(Path(__file__).resolve().parent.parent / "doclayout_yolo_docstructbench_imgsz1024.pt")
    
    if local_path and Path(local_path).exists():
        print(f"  Loading local model: {local_path}")
        return YOLOv10(local_path)
    else:
        print(f"  Loading pre-trained model from HuggingFace: {HUGGINGFACE_MODEL}")
        return YOLOv10.from_pretrained(HUGGINGFACE_MODEL)


def parse_image_filename(filename: str) -> Optional[Tuple[str, int]]:
    match = re.match(r'^(.+)_page(\d+)\.[a-zA-Z]+$', filename)
    if match:
        return (match.group(1), int(match.group(2)))
    return None


def group_images_by_document(images_dir: Path) -> Dict[str, List[Tuple[Path, int]]]:
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    docs = defaultdict(list)
    
    if not images_dir.exists():
        print(f"[Error] Images directory not found: {images_dir}")
        return {}
    
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
        
        parsed = parse_image_filename(img_path.name)
        if parsed:
            doc_stem, page_num = parsed
            docs[doc_stem].append((img_path, page_num))
    
    for doc_stem in docs:
        docs[doc_stem].sort(key=lambda x: x[1])
    
    return dict(docs)


def is_hole_punch(bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> bool:
    """
    Heuristic to detect hole punches.
    Hole punches are typically:
    1. Small (relative to page)
    2. Roughly square aspect ratio
    3. Located near the left (or sometimes right/top) margins
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # 1. Size check: Hole punches are small
    # < 80px on a standard 200-300 DPI scan is a reasonable cutoff for a hole punch
    if w > 100 or h > 100:
        return False
        
    # 2. Aspect ratio check: Should be roughly square (0.5 to 2.0)
    aspect = w / h if h > 0 else 0
    if not (0.5 <= aspect <= 2.0):
        return False
        
    # 3. Margin check: Usually on the far left or very top
    # Left margin (common for binders)
    if x1 < img_width * 0.15: 
        return True
        
    # Top margin
    if y1 < img_height * 0.10:
        return True
        
    return False


def calculate_bbox_proximity(
    asset_bbox: Tuple[int, int, int, int],
    caption_bbox: Tuple[int, int, int, int],
    image_height: int
) -> Tuple[bool, float]:
    ax1, ay1, ax2, ay2 = asset_bbox
    cx1, cy1, cx2, cy2 = caption_bbox
    
    horizontal_overlap = min(ax2, cx2) - max(ax1, cx1)
    asset_width = ax2 - ax1
    
    # Loosened: Require only 20% horizontal overlap (sometimes captions are short/offset)
    if horizontal_overlap < asset_width * 0.2:
        return False, float('inf')
    
    if cy2 <= ay1:
        vertical_distance = ay1 - cy2
    elif cy1 >= ay2:
        vertical_distance = cy1 - ay2
    else:
        vertical_distance = 0
    
    distance_ratio = vertical_distance / image_height if image_height > 0 else float('inf')
    is_nearby = distance_ratio < 0.10
    
    return is_nearby, distance_ratio


def find_matching_caption(
    asset: Detection,
    captions: List[Detection],
    asset_type: str,
    image_height: int
) -> Optional[Detection]:
    expected_caption_type = ASSET_TO_CAPTION.get(asset_type)
    if not expected_caption_type:
        return None
    
    best_caption = None
    best_distance = float('inf')
    
    for caption in captions:
        if caption.class_name != expected_caption_type:
            continue
        
        is_nearby, distance = calculate_bbox_proximity(
            asset.bbox_pixels, caption.bbox_pixels, image_height
        )
        
        if is_nearby and distance < best_distance:
            best_distance = distance
            best_caption = caption
    
    return best_caption


def extract_caption_text(
    raw_ocr_dir: str,
    doc_stem: str,
    page_number: int,
    caption_bbox: Tuple[int, int, int, int],
    overlap_threshold: float = 0.3
) -> str:
    """
    Extract caption text by finding OCR words inside the caption bbox.
    Uses a lower overlap threshold (0.3) to capture words that might slightly spill over.
    """
    if not raw_ocr_dir:
        return ""
    
    ocr_path = os.path.join(raw_ocr_dir, f"{doc_stem}.json")
    if not os.path.exists(ocr_path):
        return ""
    
    try:
        with open(ocr_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return ""
    
    # Robustly find page data (handle str vs int keys)
    page_data = None
    if isinstance(data, dict):
        page_data = data.get(str(page_number)) or data.get(page_number)
    elif isinstance(data, list):
        # Handle list format (rare but possible)
        for p in data:
            if str(p.get('page_num', '')) == str(page_number):
                page_data = p
                break

    if not page_data:
        return ""
    
    # Handle nested page_dict
    page_dict = page_data.get('page_dict', page_data)
    
    texts = page_dict.get('text', [])
    lefts = page_dict.get('left', [])
    tops = page_dict.get('top', [])
    widths = page_dict.get('width', [])
    heights = page_dict.get('height', [])
    
    if not all([texts, lefts, tops, widths, heights]):
        return ""
    
    cx1, cy1, cx2, cy2 = caption_bbox
    
    caption_words = []
    num_words = min(len(texts), len(lefts), len(tops), len(widths), len(heights))
    
    for i in range(num_words):
        wl = lefts[i]
        wt = tops[i]
        wr = wl + widths[i]
        wb = wt + heights[i]
        
        # Calculate intersection
        inter_l = max(wl, cx1)
        inter_t = max(wt, cy1)
        inter_r = min(wr, cx2)
        inter_b = min(wb, cy2)
        
        if inter_r <= inter_l or inter_b <= inter_t:
            continue
        
        inter_area = (inter_r - inter_l) * (inter_b - inter_t)
        word_area = max((wr - wl) * (wb - wt), 1)
        
        # Lower threshold: Include word if 30% of it is in the box
        # Also include if the *center* of the word is in the box
        word_center_x = wl + (widths[i] / 2)
        word_center_y = wt + (heights[i] / 2)
        center_in_box = (cx1 <= word_center_x <= cx2) and (cy1 <= word_center_y <= cy2)

        if (inter_area / word_area >= overlap_threshold) or center_in_box:
            word_text = str(texts[i]).strip()
            if word_text and word_text != '-1':
                caption_words.append((wt, wl, word_text)) # Sort by top, then left
    
    if not caption_words:
        return ""
    
    # Sort words by line (Y) then position (X)
    # Simple line grouping threshold (10px)
    caption_words.sort(key=lambda w: (int(w[0] / 10), w[1]))
    
    return " ".join(w[2] for w in caption_words)


def clean_caption_text(text: str, asset_type: str) -> str:
    """
    Clean the extracted caption text.
    Removes the "Figure X" prefix so Word doesn't double-label it.
    Input: "Figure 1: System Diagram" -> Output: "System Diagram"
    """
    if not text:
        return ""
        
    text = text.strip()
    
    # Regex to match "Figure 1", "Figure 1.", "Figure 1:", "Fig 1", "Table 2-1"
    # Matches start of string
    if asset_type in ["figure", "fig"]:
        # Match "Figure 1" or "Fig 1" followed by optional punctuation
        pattern = r'^(?:Figure|Fig)\.?\s*[\d\.\-]+\s*[:\.\-]\s*'
    elif asset_type in ["table", "tab"]:
        pattern = r'^(?:Table|Tab)\.?\s*[\d\.\-]+\s*[:\.\-]\s*'
    elif asset_type in ["equation", "eq", "formula", "isolate_formula"]:
        pattern = r'^(?:Equation|Eq|Formula)\.?\s*[\d\.\-]+\s*[:\.\-]\s*'
    else:
        return text

    # Remove the matching prefix
    cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # If regex didn't match (e.g. just "Figure 1" with no text), return original
    # or empty if it was ONLY the identifier.
    if not cleaned:
        return text
        
    return cleaned.strip()


def check_ocr_for_keywords(
    raw_ocr_dir: str,
    doc_stem: str,
    page_number: int,
    asset_type: str
) -> bool:
    if not raw_ocr_dir:
        return False
    
    ocr_path = os.path.join(raw_ocr_dir, f"{doc_stem}.json")
    if not os.path.exists(ocr_path):
        return False
    
    try:
        with open(ocr_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False
    
    page_data = None
    if isinstance(data, dict):
        page_data = data.get(str(page_number)) or data.get(page_number)
    
    if not page_data:
        return False
    
    page_dict = page_data.get('page_dict', page_data)
    text_list = page_dict.get('text', [])
    
    if not text_list:
        return False
    
    page_text = ' '.join(str(t) for t in text_list).lower()
    
    if asset_type == "figure":
        pattern = r'\b(figure|fig\.?)\s*\d'
    elif asset_type == "table":
        pattern = r'\b(table|tab\.?)\s*\d'
    elif asset_type == "isolate_formula":
        pattern = r'\b(equation|eq\.?|formula)\s*\d'
    else:
        return False
    
    return bool(re.search(pattern, page_text))


def run_detection_on_image(
    model: 'YOLOv10',
    image_path: Path,
    doc_stem: str,
    page_number: int,
    confidence_threshold: float = 0.25,
    device: str = 'cpu',
    raw_ocr_dir: str = None
) -> List[DetectedAsset]:
    
    validated_assets = []
    
    results = model.predict(
        str(image_path),
        imgsz=1024,
        conf=confidence_threshold,
        save=False,
        device=device,
        verbose=False
    )
    
    if not results or len(results) == 0:
        return validated_assets
    
    result = results[0]
    img_height, img_width = result.orig_shape
    
    all_detections: List[Detection] = []
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = YOLOv10.names[class_id] if hasattr(YOLOv10, 'names') else YOLO_CLASSES.get(class_id, "unknown")
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        detection = Detection(
            class_name=class_name,
            bbox_pixels=(int(x1), int(y1), int(x2), int(y2)),
            confidence=confidence,
            class_id=class_id
        )
        all_detections.append(detection)
    
    asset_detections = [d for d in all_detections if d.class_name in ASSET_CLASSES]
    caption_detections = [d for d in all_detections if d.class_name in CAPTION_CLASSES]
    
    for detection in asset_detections:
        asset_class = detection.class_name
        
        # --- 1. FILTER HOLE PUNCHES ---
        # If it's a figure, check if it looks like a hole punch
        if asset_class == "figure":
            if is_hole_punch(detection.bbox_pixels, img_width, img_height):
                print(f"      [Filtered] Hole punch detected on page {page_number}")
                continue

        if asset_class == "figure":
            asset_type = "fig"
        elif asset_class == "table":
            asset_type = "tab"
        elif asset_class == "isolate_formula":
            asset_type = "eq"
        else:
            continue
        
        matching_caption = find_matching_caption(
            detection, caption_detections, asset_class, img_height
        )
        
        has_caption = matching_caption is not None
        caption_bbox = matching_caption.bbox_pixels if matching_caption else None
        validated_by = "caption" if has_caption else "none"
        
        if not has_caption and raw_ocr_dir:
            has_keyword = check_ocr_for_keywords(
                raw_ocr_dir, doc_stem, page_number, asset_class
            )
            if has_keyword:
                validated_by = "ocr_keyword"
        
        if validated_by == "none" and asset_class == "isolate_formula":
            validated_by = "standalone"
        
        if validated_by == "none" and asset_class == "table":
            asset_type = "tab_layout"
            validated_by = "layout_only"
        
        if validated_by == "none":
            print(f"      [Rejected] {asset_class} on page {page_number} - no caption or keyword found")
            continue
        
        # --- 2. EXTRACT CAPTION TEXT ---
        caption_text = ""
        if has_caption and caption_bbox and raw_ocr_dir:
            raw_text = extract_caption_text(
                raw_ocr_dir, doc_stem, page_number, caption_bbox
            )
            # Clean the text (remove "Figure X" prefix)
            caption_text = clean_caption_text(raw_text, asset_class)
            
            if not caption_text and raw_text:
                 # If cleaning removed everything, revert to raw (e.g. caption was just "Figure 1")
                 caption_text = raw_text
        
        asset = DetectedAsset(
            asset_type=asset_type,
            page_number=page_number,
            bbox_pixels=detection.bbox_pixels,
            confidence=detection.confidence,
            class_id=detection.class_id,
            doc_stem=doc_stem,
            image_width=img_width,
            image_height=img_height,
            has_caption=has_caption,
            caption_bbox=caption_bbox,
            caption_text=caption_text,
            validated_by=validated_by
        )
        validated_assets.append(asset)
    
    return validated_assets


def crop_and_save_asset(
    source_image_path: Path,
    asset: DetectedAsset,
    output_dir: Path,
    asset_index: int,
    padding: int = 5
) -> Tuple[str, Path]:
    filename = f"{asset.doc_stem}_{asset.asset_type}_p{asset.page_number:03d}_{asset_index:03d}.jpg"
    output_path = output_dir / filename
    
    try:
        with Image.open(source_image_path) as img:
            x1, y1, x2, y2 = asset.bbox_pixels
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.width, x2 + padding)
            y2 = min(img.height, y2 + padding)
            
            cropped = img.crop((x1, y1, x2, y2))
            cropped.save(output_path, "JPEG", quality=95)
    except Exception as e:
        print(f"      [Error] Failed to crop asset: {e}")
        return "", Path("")
    
    return filename, output_path


def crop_and_save_caption(
    source_image_path: Path,
    asset: DetectedAsset,
    output_dir: Path,
    asset_index: int,
    padding: int = 5
) -> Tuple[str, Path]:
    filename = f"{asset.doc_stem}_{asset.asset_type}_p{asset.page_number:03d}_{asset_index:03d}_caption.jpg"
    output_path = output_dir / filename
    
    try:
        with Image.open(source_image_path) as img:
            x1, y1, x2, y2 = asset.caption_bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.width, x2 + padding)
            y2 = min(img.height, y2 + padding)
            
            cropped = img.crop((x1, y1, x2, y2))
            cropped.save(output_path, "JPEG", quality=95)
    except Exception as e:
        print(f"      [Error] Failed to crop caption: {e}")
        return "", Path("")
    
    return filename, output_path


def create_asset_metadata(
    asset: DetectedAsset,
    image_filename: str,
    output_dpi: int = 200,
    caption_image_filename: str = None
) -> Dict:
    x1, y1, x2, y2 = asset.bbox_pixels
    width = x2 - x1
    height = y2 - y1
    
    # Default ID
    asset_id = f"{asset.doc_stem}_{asset.asset_type}_p{asset.page_number}_{x1}_{y1}"
    
    metadata = {
        "asset_id": asset_id,
        "asset_type": asset.asset_type,
        "page": asset.page_number,
        "bbox": {
            "pixels": [x1, y1, width, height],
            "page_width_px": asset.image_width,
            "page_height_px": asset.image_height,
        },
        "export": {
            "image_file": image_filename,
            "caption_image_file": caption_image_filename,
            "dpi": output_dpi,
            "format": "JPEG",
        },
        "detection": {
            "method": "DocLayout-YOLO",
            "model": HUGGINGFACE_MODEL,
            "confidence": round(asset.confidence, 4),
            "class_id": asset.class_id,
            "class_name": YOLO_CLASSES.get(asset.class_id, "unknown"),
            "validated_by": asset.validated_by,
            "has_caption": asset.has_caption,
        }
    }
    
    if asset.caption_bbox:
        cx1, cy1, cx2, cy2 = asset.caption_bbox
        metadata["detection"]["caption_bbox"] = {
            "pixels": [cx1, cy1, cx2 - cx1, cy2 - cy1],
        }
    
    # --- CRITICAL: Use extracted text as ID ---
    if asset.caption_text:
        metadata["caption_text"] = asset.caption_text
        metadata["asset_id"] = asset.caption_text
    
    return metadata


def process_document(
    model: 'YOLOv10',
    doc_stem: str,
    page_images: List[Tuple[Path, int]],
    output_dir: Path,
    confidence_threshold: float = 0.25,
    device: str = 'cpu',
    raw_ocr_dir: str = None
) -> List[Dict]:
    doc_output_dir = output_dir / doc_stem
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_assets_metadata = []
    asset_counters = {"fig": 0, "tab": 0, "eq": 0, "tab_layout": 0}
    validation_stats = {"caption": 0, "ocr_keyword": 0, "standalone": 0, "layout_only": 0}
    
    print(f"  Processing {len(page_images)} pages...")
    
    for image_path, page_number in page_images:
        detected_assets = run_detection_on_image(
            model=model,
            image_path=image_path,
            doc_stem=doc_stem,
            page_number=page_number,
            confidence_threshold=confidence_threshold,
            device=device,
            raw_ocr_dir=raw_ocr_dir
        )
        
        if detected_assets:
            print(f"    Page {page_number}: Found {len(detected_assets)} validated assets")
        
        for asset in detected_assets:
            asset_counters[asset.asset_type] += 1
            asset_index = asset_counters[asset.asset_type]
            validation_stats[asset.validated_by] += 1
            
            image_filename, _ = crop_and_save_asset(
                source_image_path=image_path,
                asset=asset,
                output_dir=doc_output_dir,
                asset_index=asset_index
            )
            
            caption_image_filename = None
            if asset.has_caption and asset.caption_bbox:
                caption_image_filename, _ = crop_and_save_caption(
                    source_image_path=image_path,
                    asset=asset,
                    output_dir=doc_output_dir,
                    asset_index=asset_index
                )
            
            metadata = create_asset_metadata(asset, image_filename, caption_image_filename=caption_image_filename)
            all_assets_metadata.append(metadata)
            
            json_filename = image_filename.replace('.jpg', '.json')
            json_path = doc_output_dir / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
    
    print(f"  Validation: {validation_stats['caption']} by caption, {validation_stats['ocr_keyword']} by OCR keyword, {validation_stats['standalone']} standalone, {validation_stats['layout_only']} layout-only")
    
    return all_assets_metadata


def run_yolo_extraction(
    images_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    doc_filter: Optional[str] = None,
    confidence_threshold: float = 0.25,
    device: str = 'cpu',
    model_path: Optional[str] = None,
    raw_ocr_dir: Optional[str] = None
) -> Dict[str, List[Dict]]:
    paths = get_paths()
    images_dir = Path(images_dir) if images_dir else paths['images_dir']
    output_dir = Path(output_dir) if output_dir else paths['exports_dir']
    raw_ocr_dir = str(paths['raw_ocr_dir']) if raw_ocr_dir is None else raw_ocr_dir
    
    if not os.path.isdir(raw_ocr_dir):
        print(f"  [WARNING] Raw OCR directory not found: {raw_ocr_dir}")
        print(f"  [WARNING] OCR keyword validation will not work - only caption detection will be used")
        raw_ocr_dir = None
    
    print("=" * 60)
    print("YOLO ASSET EXTRACTION")
    print("=" * 60)
    print(f"  Images directory: {images_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Device: {device}")
    
    docs = group_images_by_document(images_dir)
    if not docs:
        print("[Error] No page images found!")
        return {}
    
    if doc_filter:
        if doc_filter in docs:
            docs = {doc_filter: docs[doc_filter]}
        else:
            return {}
    
    print(f"  Found {len(docs)} document(s) to process")
    
    print("Loading YOLO model...")
    model = load_model(model_path)
    if model is None:
        return {}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for doc_stem, page_images in docs.items():
        print(f"[{doc_stem}]")
        assets = process_document(
            model=model,
            doc_stem=doc_stem,
            page_images=page_images,
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
            device=device,
            raw_ocr_dir=raw_ocr_dir
        )
        results[doc_stem] = assets
        
        fig_count = sum(1 for a in assets if a['asset_type'] == 'fig')
        tab_count = sum(1 for a in assets if a['asset_type'] == 'tab')
        print(f"  Extracted: {fig_count} figures, {tab_count} tables")

    print("=" * 60)
    print("EXTRACTION COMPLETE")
    return results


def get_yolo_exports_dir() -> Path:
    return get_paths()['exports_dir']


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', '-i', type=str, default=None)
    parser.add_argument('--output-dir', '-o', type=str, default=None)
    parser.add_argument('--doc', '-d', type=str, default=None)
    parser.add_argument('--conf', '-c', type=float, default=0.25)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--raw-ocr-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    run_yolo_extraction(
        images_dir=Path(args.images_dir) if args.images_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        doc_filter=args.doc,
        confidence_threshold=args.conf,
        device=args.device,
        raw_ocr_dir=args.raw_ocr_dir
    )