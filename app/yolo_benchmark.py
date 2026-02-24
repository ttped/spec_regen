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


class EnsembleYOLO:
    """
    A drop-in replacement for YOLOv10 that performs multi-scale inference.
    Uses cross-scale voting to suppress false positives and averages bounding
    box coordinates for higher precision crops.
    """
    def __init__(self, model_path, scales=(800, 1024, 1280), min_votes=2, iou_thresh=0.5):
        self.model = YOLOv10(model_path)
        self.scales = scales
        self.min_votes = min_votes
        self.iou_thresh = iou_thresh

    def predict(self, image_path, conf_threshold=0.25):
        all_detections = []
        
        # Run inference at each specified scale
        for scale in self.scales:
            results = self.model.predict(
                image_path,
                imgsz=scale,
                conf=conf_threshold,
                device='cpu',
                verbose=False
            )
            
            img_height, img_width = results[0].orig_shape
            
            for box in results[0].boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                px1, py1, px2, py2 = box.xyxy[0].tolist()
                
                # Normalize coordinates so different scales can be compared
                norm_bbox = (px1/img_width, py1/img_height, px2/img_width, py2/img_height)
                
                all_detections.append({
                    'class_id': class_id,
                    'bbox': norm_bbox,
                    'conf': conf,
                    'scale': scale
                })
                
        return self._merge_detections(all_detections)
        
    def _merge_detections(self, detections):
        """Merges overlapping detections across different scales using NMS, Voting, and Geometric Heuristics."""
        merged_results = []
        unique_classes = set(d['class_id'] for d in detections)
        
        # 1. Merge and apply Box Expansion
        for cid in unique_classes:
            class_dets = [d for d in detections if d['class_id'] == cid]
            # Prioritize higher confidence boxes during merging
            class_dets.sort(key=lambda x: x['conf'], reverse=True)
            
            merged_for_class = []
            for det in class_dets:
                matched = False
                for m in merged_for_class:
                    if calculate_iou(det['bbox'], m['bbox']) >= self.iou_thresh:
                        # Only add weight if this is from a different scale
                        if det['scale'] not in m['scales']:
                            m['scales'].add(det['scale'])
                            m['votes'] += 1
                            
                            # BOX EXPANSION: Take the maximum outer bounds instead of averaging
                            m['bbox'] = (
                                min(m['bbox'][0], det['bbox'][0]),
                                min(m['bbox'][1], det['bbox'][1]),
                                max(m['bbox'][2], det['bbox'][2]),
                                max(m['bbox'][3], det['bbox'][3])
                            )
                            # Boost confidence to the highest seen
                            m['conf'] = max(m['conf'], det['conf'])
                        matched = True
                        break
                
                if not matched:
                    merged_for_class.append({
                        'class_id': cid,
                        'bbox': det['bbox'],
                        'conf': det['conf'],
                        'scales': {det['scale']},
                        'votes': 1
                    })
            
            # Filter strictly by minimum cross-scale agreement
            for m in merged_for_class:
                if m['votes'] >= self.min_votes:
                    merged_results.append(m)
                    
        # 2. Geometric Heuristics (Post-processing)
        final_results = []
        
        for det in merged_results:
            cid = det['class_id']
            bbox = det['bbox']
            
            # Heuristic A: Nesting Rule (Drop tables heavily contained inside figures)
            if cid == 5:  # table
                is_nested = False
                for other in merged_results:
                    if other['class_id'] == 3:  # figure
                        x_left = max(bbox[0], other['bbox'][0])
                        y_top = max(bbox[1], other['bbox'][1])
                        x_right = min(bbox[2], other['bbox'][2])
                        y_bottom = min(bbox[3], other['bbox'][3])
                        
                        if x_right > x_left and y_bottom > y_top:
                            inter_area = (x_right - x_left) * (y_bottom - y_top)
                            table_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            if inter_area / table_area > 0.80:  # 80% contained
                                is_nested = True
                                break
                if is_nested:
                    continue
                    
            # Heuristic B: Orphan Captions (Drop captions without nearby parent assets)
            if cid in [4, 6, 9]:  # figure_caption, table_caption, formula_caption
                parent_map = {4: 3, 6: 5, 9: 8}
                parent_cid = parent_map[cid]
                
                has_parent = False
                for other in merged_results:
                    if other['class_id'] == parent_cid:
                        # Expand parent box vertically by 15% to check for nearby captions
                        p_bbox = other['bbox']
                        expanded_parent = (p_bbox[0], p_bbox[1] - 0.15, p_bbox[2], p_bbox[3] + 0.15)
                        
                        x_left = max(bbox[0], expanded_parent[0])
                        y_top = max(bbox[1], expanded_parent[1])
                        x_right = min(bbox[2], expanded_parent[2])
                        y_bottom = min(bbox[3], expanded_parent[3])
                        
                        if x_right > x_left and y_bottom > y_top:
                            has_parent = True
                            break
                if not has_parent:
                    continue
                    
            final_results.append({
                'class_id': det['class_id'],
                'bbox': det['bbox'],
                'conf': det['conf']
            })
            
        return final_results

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
    Runs inference using the active model.
    Filters the output to only return classes specified in TARGET_CLASSES.
    """
    raw_detections = model.predict(image_path, conf_threshold=conf_threshold)
    return [d for d in raw_detections if d['class_id'] in TARGET_CLASSES]


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
    print(f"Loading Ensemble Model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found.")
        return
        
    # Using the new EnsembleYOLO class
    model = EnsembleYOLO(
        model_path=MODEL_PATH, 
        scales=(800, 1024, 1280), 
        min_votes=2
    )
    
    image_paths = []
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
        
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        return
        
    overall_stats = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in TARGET_CLASSES}
    evaluated_count = 0
    
    print("\nRunning Multi-Scale Benchmark...")
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