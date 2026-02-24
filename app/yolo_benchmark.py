"""
yolo_benchmark.py - Evaluates YOLO model performance against ground truth labels.

Filters evaluation to specific high-value classes (tables, figures, formulas).
Images without a corresponding .txt file in the labels directory are skipped.
"""

import os
import glob
from pathlib import Path
from doclayout_yolo import YOLOv10

# Configuration
IMAGES_DIR = "docs_images"
LABELS_DIR = "docs_labels"
MODEL_PATH = "doclayout_yolo_docstructbench_imgsz1024.pt"

# The classes we actually care about evaluating
TARGET_CLASSES = {
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes should be in (x1, y1, x2, y2) format. Works with normalized coords.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def yolo_to_coords(cx, cy, w, h):
    """Convert YOLO (center x, center y, width, height) to (x1, y1, x2, y2)."""
    x1 = cx - (w / 2)
    y1 = cy - (h / 2)
    x2 = cx + (w / 2)
    y2 = cy + (h / 2)
    return (x1, y1, x2, y2)

def load_ground_truth(txt_path):
    """Load ground truth boxes from YOLO .txt file. Ignores non-target classes."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
        
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id in TARGET_CLASSES:
                    cx, cy, w, h = map(float, parts[1:5])
                    bbox = yolo_to_coords(cx, cy, w, h)
                    boxes.append({'class_id': class_id, 'bbox': bbox})
    return boxes

def get_predictions(model, image_path, conf_threshold=0.25):
    """
    Run model inference and return bounding boxes.
    MODIFY THIS FUNCTION when testing DPI ensembles or multi-scale inference.
    """
    boxes = []
    results = model.predict(
        image_path,
        imgsz=1024,
        conf=conf_threshold,
        device='cpu',
        verbose=False
    )
    
    # Get original image dimensions to normalize predictions
    img_height, img_width = results[0].orig_shape
    
    for box in results[0].boxes:
        class_id = int(box.cls)
        if class_id in TARGET_CLASSES:
            conf = float(box.conf)
            # YOLO results are in pixel coords, convert to normalized 0-1 coords
            px1, py1, px2, py2 = box.xyxy[0].tolist()
            norm_bbox = (px1/img_width, py1/img_height, px2/img_width, py2/img_height)
            
            boxes.append({
                'class_id': class_id, 
                'bbox': norm_bbox, 
                'conf': conf
            })
            
    return boxes

def evaluate_image(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Match predictions to ground truth and calculate TP, FP, FN per class."""
    stats = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in TARGET_CLASSES}
    
    for class_id in TARGET_CLASSES:
        c_gts = [g for g in gt_boxes if g['class_id'] == class_id]
        c_preds = [p for p in pred_boxes if p['class_id'] == class_id]
        
        # Sort predictions by highest confidence first
        c_preds.sort(key=lambda x: x['conf'], reverse=True)
        matched_gt_indices = set()
        
        for pred in c_preds:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(c_gts):
                if i in matched_gt_indices:
                    continue
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
                    
            if best_iou >= iou_threshold:
                stats[class_id]['tp'] += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                stats[class_id]['fp'] += 1
                
        # Any GT box that didn't get matched is a False Negative
        stats[class_id]['fn'] += len(c_gts) - len(matched_gt_indices)
        
    return stats

def run_benchmark():
    print(f"Loading Model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found.")
        return
        
    model = YOLOv10(MODEL_PATH)
    
    image_paths = []
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
        
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        return
        
    overall_stats = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in TARGET_CLASSES}
    evaluated_count = 0
    
    print("\nRunning Benchmark...")
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
        
        # Skip images that have no corresponding manual label file
        if not os.path.exists(txt_path):
            continue
            
        gt_boxes = load_ground_truth(txt_path)
        pred_boxes = get_predictions(model, img_path)
        
        img_stats = evaluate_image(gt_boxes, pred_boxes, iou_threshold=0.5)
        
        for cid in TARGET_CLASSES:
            overall_stats[cid]['tp'] += img_stats[cid]['tp']
            overall_stats[cid]['fp'] += img_stats[cid]['fp']
            overall_stats[cid]['fn'] += img_stats[cid]['fn']
            
        evaluated_count += 1
        if evaluated_count % 50 == 0:
            print(f"Evaluated {evaluated_count} images...")
            
    print(f"\n========================================================")
    print(f"BENCHMARK RESULTS (Evaluated {evaluated_count} labeled images)")
    print(f"========================================================")
    print(f"{'Class Name':<20} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print(f"-" * 56)
    
    total_tp = total_fp = total_fn = 0
    
    for cid, name in TARGET_CLASSES.items():
        tp = overall_stats[cid]['tp']
        fp = overall_stats[cid]['fp']
        fn = overall_stats[cid]['fn']
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{name:<20} | {precision:.3f}     | {recall:.3f}     | {f1:.3f}")
        
    print(f"-" * 56)
    
    macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    macro_f1 = 2 * (macro_p * macro_r) / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    
    print(f"{'OVERALL':<20} | {macro_p:.3f}     | {macro_r:.3f}     | {macro_f1:.3f}")
    print(f"========================================================")

if __name__ == "__main__":
    run_benchmark()