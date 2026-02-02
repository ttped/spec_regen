"""
DocLayout-YOLO Visual Testing Tool
===================================
Uses DocLayout-YOLO for document layout analysis - optimized for detecting
tables, figures, text blocks, and other document elements.

Installation:
    pip install doclayout-yolo

Usage:
    python test_yolo_visual_doclayout.py
    
    # Adjust confidence threshold:
    python test_yolo_visual_doclayout.py --conf 0.3
    
    # Use a local model file (for offline use):
    python test_yolo_visual_doclayout.py --model path/to/model.pt

Security Note:
    This runs entirely locally. After the initial model download from HuggingFace,
    no data is sent externally. For fully offline operation, download the model
    once and use --model to point to the local file.
"""

from doclayout_yolo import YOLOv10
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
from datetime import datetime

# ================= CONFIGURATION =================
def get_paths():
    """Calculates paths relative to this script file."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    return {
        'test_input': project_root / "docs_images",
        'test_output': project_root / "doclayout_test_results",
    }

PATHS = get_paths()

# DocLayout-YOLO DocStructBench model classes (10 classes)
# These are the classes detected by the pre-trained DocStructBench model
DOCLAYOUT_COLORS = {
    0: ("#FF6B6B", "title"),
    1: ("#4ECDC4", "plain text"),
    2: ("#FFE66D", "abandon"),
    3: ("#95E1D3", "figure"),
    4: ("#A8E6CF", "figure_caption"),
    5: ("#DDA0DD", "table"),
    6: ("#87CEEB", "table_caption"),
    7: ("#F0E68C", "table_footnote"),
    8: ("#FFA07A", "isolate_formula"),
    9: ("#98D8C8", "formula_caption"),
}

DEFAULT_COLOR = ("#888888", "Unknown")

# HuggingFace model identifier for automatic download
HUGGINGFACE_MODEL = "juliozhao/DocLayout-YOLO-DocStructBench"
# =================================================


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_font():
    """Try to load a nice font, fall back to default."""
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                return ImageFont.truetype(fp, 16)
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()


def draw_detections(image_path: Path, results, output_path: Path, conf_threshold: float = 0.25):
    """
    Draw bounding boxes and labels on an image.
    
    Returns: (num_detections, detection_summary)
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font()
    
    detections = []
    boxes = results[0].boxes
    
    # Get class names from model results
    class_names = results[0].names
    
    for box in boxes:
        confidence = float(box.conf)
        
        if confidence < conf_threshold:
            continue
        
        class_id = int(box.cls)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Get color and label - use model's class names
        if class_id in DOCLAYOUT_COLORS:
            color_hex, _ = DOCLAYOUT_COLORS[class_id]
        else:
            color_hex, _ = DEFAULT_COLOR
        
        # Use the class name from the model
        class_name = class_names.get(class_id, f"class_{class_id}")
        color_rgb = hex_to_rgb(color_hex)
        
        # Draw bounding box (thick outline)
        for i in range(3):
            draw.rectangle(
                [x1 - i, y1 - i, x2 + i, y2 + i],
                outline=color_rgb,
            )
        
        # Prepare label text
        label = f"{class_name} {confidence:.0%}"
        
        # Calculate label background size
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0] + 10
        label_height = bbox[3] - bbox[1] + 6
        
        # Draw label background
        label_y = y1 - label_height - 2 if y1 > label_height + 5 else y1 + 2
        draw.rectangle(
            [x1, label_y, x1 + label_width, label_y + label_height],
            fill=color_rgb,
        )
        
        # Draw label text
        draw.text(
            (x1 + 5, label_y + 2),
            label,
            fill=(255, 255, 255),
            font=font,
        )
        
        detections.append({
            'class': class_name,
            'confidence': confidence,
            'bbox': (x1, y1, x2, y2),
        })
    
    # Add summary text at top of image
    summary_text = f"Detections: {len(detections)}"
    if detections:
        counts = {}
        for d in detections:
            counts[d['class']] = counts.get(d['class'], 0) + 1
        summary_parts = [f"{v} {k}" for k, v in sorted(counts.items())]
        summary_text += f" ({', '.join(summary_parts)})"
    
    # Draw summary background
    bbox = draw.textbbox((0, 0), summary_text, font=font)
    draw.rectangle([5, 5, bbox[2] - bbox[0] + 15, bbox[3] - bbox[1] + 15], fill=(0, 0, 0, 180))
    draw.text((10, 8), summary_text, fill=(255, 255, 255), font=font)
    
    # Save annotated image
    img.save(output_path, quality=95)
    
    return len(detections), detections


def create_summary_report(results_dir: Path, all_results: list, model_info: str):
    """Create a text summary report of all detections."""
    report_path = results_dir / "detection_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("DocLayout-YOLO Detection Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_info}\n")
        f.write(f"Total images processed: {len(all_results)}\n")
        
        # Aggregate stats
        total_detections = sum(r['count'] for r in all_results)
        class_totals = {}
        for r in all_results:
            for d in r['detections']:
                cls = d['class']
                class_totals[cls] = class_totals.get(cls, 0) + 1
        
        f.write(f"Total detections: {total_detections}\n\n")
        
        f.write("Detection counts by class:\n")
        for cls, count in sorted(class_totals.items()):
            f.write(f"  {cls}: {count}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Per-image breakdown:\n")
        f.write("=" * 60 + "\n\n")
        
        for r in all_results:
            f.write(f"\n{r['filename']}:\n")
            f.write(f"  Detections: {r['count']}\n")
            
            if r['detections']:
                for i, d in enumerate(r['detections'], 1):
                    x1, y1, x2, y2 = d['bbox']
                    f.write(f"    {i}. {d['class']} ({d['confidence']:.1%}) ")
                    f.write(f"at ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})\n")
            else:
                f.write("    No detections\n")
    
    return report_path


def create_comparison_grid(results_dir: Path, image_results: list, max_images: int = 12):
    """Create a grid image showing multiple results side by side."""
    if not image_results:
        return None
    
    images_to_show = image_results[:max_images]
    
    annotated_images = []
    for r in images_to_show:
        img_path = results_dir / f"annotated_{r['filename']}"
        if img_path.exists():
            img = Image.open(img_path)
            annotated_images.append((img, r['filename']))
    
    if not annotated_images:
        return None
    
    n = len(annotated_images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    thumb_w, thumb_h = 400, 500
    
    grid_w = cols * thumb_w + (cols + 1) * 10
    grid_h = rows * thumb_h + (rows + 1) * 10 + 30
    grid = Image.new('RGB', (grid_w, grid_h), color=(40, 40, 40))
    draw = ImageDraw.Draw(grid)
    
    font = get_font()
    
    draw.text((10, 5), f"DocLayout-YOLO Results - {len(image_results)} images", fill=(255, 255, 255), font=font)
    
    for i, (img, filename) in enumerate(annotated_images):
        row = i // cols
        col = i % cols
        
        img.thumbnail((thumb_w - 10, thumb_h - 25), Image.Resampling.LANCZOS)
        
        x = col * thumb_w + (col + 1) * 10
        y = row * thumb_h + (row + 1) * 10 + 25
        
        grid.paste(img, (x, y))
        
        name_short = filename[:30] + "..." if len(filename) > 30 else filename
        draw.text((x, y + img.height + 2), name_short, fill=(200, 200, 200), font=font)
    
    grid_path = results_dir / "results_grid.jpg"
    grid.save(grid_path, quality=90)
    
    return grid_path


def main():
    parser = argparse.ArgumentParser(description="Test DocLayout-YOLO on document images")
    parser.add_argument('--model', type=str, default=None, 
                       help="Path to local model weights (optional - will download from HuggingFace if not provided)")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument('--imgsz', type=int, default=1024, help="Inference image size (default 1024)")
    parser.add_argument('--input', type=str, default=None, help="Input folder path")
    parser.add_argument('--output', type=str, default=None, help="Output folder path")
    parser.add_argument('--device', type=str, default='cpu', help="Device: 'cpu', 'cuda:0', or 'mps'")
    args = parser.parse_args()
    
    input_dir = Path(args.input) if args.input else PATHS['test_input']
    output_dir = Path(args.output) if args.output else PATHS['test_output']
    
    print("=" * 60)
    print("DOCLAYOUT-YOLO VISUAL TESTING")
    print("=" * 60)
    print(f"\nInput folder:  {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Device:        {args.device}")
    print(f"Image size:    {args.imgsz}")
    print(f"Confidence:    {args.conf:.0%}")
    
    # Check input directory
    if not input_dir.exists():
        print(f"\n✗ Input folder not found: {input_dir}")
        print("  Run prepare_images.py first to convert PDFs to images.")
        return
    
    # Find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    test_images = [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not test_images:
        print(f"\n✗ No images found in {input_dir}")
        return
    
    print(f"\nFound {len(test_images)} test images")
    
    # Load model
    print(f"\nLoading DocLayout-YOLO model...")
    if args.model and Path(args.model).exists():
        print(f"  Using local model: {args.model}")
        model = YOLOv10(args.model)
        model_info = args.model
    else:
        print(f"  Loading from HuggingFace: {HUGGINGFACE_MODEL}")
        print("  (This will be cached locally for future offline use)")
        model = YOLOv10.from_pretrained(HUGGINGFACE_MODEL)
        model_info = HUGGINGFACE_MODEL
    
    print("  Model loaded successfully!")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    print(f"\nProcessing images...")
    print("-" * 60)
    
    all_results = []
    
    for i, img_path in enumerate(sorted(test_images), 1):
        print(f"  [{i}/{len(test_images)}] {img_path.name}...", end=" ", flush=True)
        
        try:
            # Run inference
            results = model.predict(
                str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False
            )
            
            # Draw and save annotated image
            output_path = output_dir / f"annotated_{img_path.name}"
            num_detections, detections = draw_detections(
                img_path, results, output_path, args.conf
            )
            
            all_results.append({
                'filename': img_path.name,
                'count': num_detections,
                'detections': detections,
            })
            
            print(f"✓ {num_detections} detections")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            all_results.append({
                'filename': img_path.name,
                'count': 0,
                'detections': [],
                'error': str(e),
            })
    
    print("-" * 60)
    
    # Create summary report
    report_path = create_summary_report(output_dir, all_results, model_info)
    print(f"\n✓ Report saved: {report_path}")
    
    # Create comparison grid
    grid_path = create_comparison_grid(output_dir, all_results)
    if grid_path:
        print(f"✓ Grid saved:   {grid_path}")
    
    # Print summary
    total_detections = sum(r['count'] for r in all_results)
    images_with_detections = sum(1 for r in all_results if r['count'] > 0)
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Images processed:        {len(all_results)}")
    print(f"  Images with detections:  {images_with_detections}")
    print(f"  Total detections:        {total_detections}")
    
    # Class breakdown
    class_totals = {}
    for r in all_results:
        for d in r['detections']:
            cls = d['class']
            class_totals[cls] = class_totals.get(cls, 0) + 1
    
    if class_totals:
        print(f"\n  By class:")
        for cls, count in sorted(class_totals.items()):
            print(f"    {cls}: {count}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"\n  Open the folder to see annotated images!")
    
    # Security reminder
    print(f"\n" + "-" * 60)
    print("SECURITY NOTE: All processing was done locally.")
    print("No image data was transmitted externally.")
    print("-" * 60)


if __name__ == "__main__":
    main()