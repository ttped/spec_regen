"""
yolo_benchmark.py - Evaluates YOLO model performance against ground truth labels.

Filters evaluation to specific high-value classes (tables, figures, formulas).
Images without a corresponding .txt file in the labels directory are skipped.

Uses Weighted Box Fusion (WBF) for multi-scale ensemble merging instead of
naive NMS/voting. Supports per-class confidence thresholds.
"""

import os
import glob
from pathlib import Path
from dataclasses import dataclass, field
from doclayout_yolo import YOLOv10


# =============================================================================
# CONFIGURATION
# =============================================================================

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

# Per-class confidence thresholds (tune these on your validation set)
# Start with uniform, then adjust based on per-class precision/recall curves
CLASS_CONF_THRESHOLDS = {
    3: 0.25,   # figure
    4: 0.20,   # figure_caption — often low confidence, boost recall
    5: 0.25,   # table
    6: 0.20,   # table_caption
    7: 0.20,   # table_footnote
    8: 0.30,   # isolate_formula — higher to reduce false positives
    9: 0.20,   # formula_caption
}

DEFAULT_CONF_THRESHOLD = 0.25


# =============================================================================
# CORE GEOMETRY
# =============================================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes in (x1, y1, x2, y2) format. Works with normalized coords.
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
    union = box1_area + box2_area - intersection_area

    return intersection_area / union if union > 0 else 0.0


def yolo_to_coords(cx, cy, w, h):
    """Convert YOLO (center x, center y, width, height) to (x1, y1, x2, y2)."""
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# =============================================================================
# WEIGHTED BOX FUSION
# =============================================================================

def weighted_box_fusion(detection_lists, iou_thresh=0.55, skip_box_thresh=0.0):
    """
    Weighted Box Fusion (WBF) — merges overlapping boxes across N models/scales
    by computing a confidence-weighted average of coordinates.

    Unlike NMS (which discards boxes) or your old max-expansion approach (which
    inflates boxes), WBF produces tighter, more accurate fused boxes.

    Args:
        detection_lists: List of lists. Each inner list is the detections from
                         one scale/model, where each detection is a dict with
                         keys: 'class_id', 'bbox' (x1,y1,x2,y2 normalized),
                         'conf'.
        iou_thresh:      IoU threshold to consider two boxes as the same object.
        skip_box_thresh: Minimum fused confidence to keep a result.

    Returns:
        List of fused detections with 'class_id', 'bbox', 'conf', 'n_fused'.
    """
    # Flatten all detections, tagging which model/scale they came from
    all_dets = []
    for model_idx, dets in enumerate(detection_lists):
        for d in dets:
            all_dets.append({**d, 'model_idx': model_idx})

    n_models = len(detection_lists)
    fused_results = []
    unique_classes = set(d['class_id'] for d in all_dets)

    for cid in unique_classes:
        class_dets = [d for d in all_dets if d['class_id'] == cid]
        class_dets.sort(key=lambda x: x['conf'], reverse=True)

        # Greedy clustering: assign each detection to the best existing cluster
        # or start a new one
        clusters = []  # Each cluster: {'dets': [...], 'fused_bbox': ..., 'fused_conf': ...}

        for det in class_dets:
            best_cluster_idx = -1
            best_iou = iou_thresh

            for ci, cluster in enumerate(clusters):
                iou = calculate_iou(det['bbox'], cluster['fused_bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_cluster_idx = ci

            if best_cluster_idx >= 0:
                clusters[best_cluster_idx]['dets'].append(det)
                _recompute_fused_box(clusters[best_cluster_idx])
            else:
                clusters.append({
                    'dets': [det],
                    'fused_bbox': det['bbox'],
                    'fused_conf': det['conf'],
                })

        # Convert clusters to results
        for cluster in clusters:
            n_fused = len(cluster['dets'])
            # WBF paper: rescale confidence by (n_fused / n_models)
            # This naturally penalizes detections seen at only 1 scale
            adjusted_conf = cluster['fused_conf'] * min(n_fused, n_models) / n_models

            if adjusted_conf >= skip_box_thresh:
                fused_results.append({
                    'class_id': cid,
                    'bbox': cluster['fused_bbox'],
                    'conf': adjusted_conf,
                    'n_fused': n_fused,
                })

    return fused_results


def _recompute_fused_box(cluster):
    """Recompute the fused bbox as a confidence-weighted average of all
    detections in the cluster."""
    dets = cluster['dets']
    total_conf = sum(d['conf'] for d in dets)

    if total_conf == 0:
        return

    wx1 = sum(d['conf'] * d['bbox'][0] for d in dets) / total_conf
    wy1 = sum(d['conf'] * d['bbox'][1] for d in dets) / total_conf
    wx2 = sum(d['conf'] * d['bbox'][2] for d in dets) / total_conf
    wy2 = sum(d['conf'] * d['bbox'][3] for d in dets) / total_conf

    cluster['fused_bbox'] = (wx1, wy1, wx2, wy2)
    cluster['fused_conf'] = total_conf / len(dets)


# =============================================================================
# GEOMETRIC POST-PROCESSING
# =============================================================================

def apply_geometric_heuristics(detections, target_classes):
    """
    Post-processing rules for document layout:
    - Drop tables heavily nested inside figures (>80% containment)
    - Drop orphan captions with no nearby parent element
    """
    final = []

    for det in detections:
        cid = det['class_id']
        bbox = det['bbox']

        # Heuristic A: Nesting Rule — drop tables contained inside figures
        if cid == 5:  # table
            is_nested = _is_contained_in_class(bbox, detections, parent_cid=3, threshold=0.80)
            if is_nested:
                continue

        # Heuristic B: Orphan Captions — drop captions without nearby parent
        parent_map = {4: 3, 6: 5, 9: 8}  # caption_class -> parent_class
        if cid in parent_map:
            parent_cid = parent_map[cid]
            if not _has_nearby_parent(bbox, detections, parent_cid, vertical_expand=0.15):
                continue

        final.append(det)

    return final


def _is_contained_in_class(bbox, all_dets, parent_cid, threshold):
    """Check if bbox is >threshold% contained inside any detection of parent_cid."""
    box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if box_area <= 0:
        return False

    for other in all_dets:
        if other['class_id'] != parent_cid:
            continue
        x_left = max(bbox[0], other['bbox'][0])
        y_top = max(bbox[1], other['bbox'][1])
        x_right = min(bbox[2], other['bbox'][2])
        y_bottom = min(bbox[3], other['bbox'][3])

        if x_right > x_left and y_bottom > y_top:
            inter_area = (x_right - x_left) * (y_bottom - y_top)
            if inter_area / box_area > threshold:
                return True
    return False


def _has_nearby_parent(bbox, all_dets, parent_cid, vertical_expand):
    """Check if a caption bbox overlaps (with vertical expansion) with any parent."""
    for other in all_dets:
        if other['class_id'] != parent_cid:
            continue
        p = other['bbox']
        expanded = (p[0], p[1] - vertical_expand, p[2], p[3] + vertical_expand)

        x_left = max(bbox[0], expanded[0])
        y_top = max(bbox[1], expanded[1])
        x_right = min(bbox[2], expanded[2])
        y_bottom = min(bbox[3], expanded[3])

        if x_right > x_left and y_bottom > y_top:
            return True
    return False


# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

@dataclass
class EnsembleConfig:
    model_path: str
    scales: tuple = (600, 800, 1024, 1280)
    wbf_iou_thresh: float = 0.55
    apply_heuristics: bool = True
    # Use the lowest per-class threshold for inference, then filter per-class after
    inference_conf: float = 0.15
    class_conf_thresholds: dict = field(default_factory=lambda: dict(CLASS_CONF_THRESHOLDS))


class EnsembleYOLO:
    """
    Multi-scale YOLO ensemble using Weighted Box Fusion.

    Runs inference at each scale, normalizes coordinates, then fuses with WBF
    to get tighter bounding boxes and calibrated confidence scores.
    """

    def __init__(self, config: EnsembleConfig):
        self.model = YOLOv10(config.model_path)
        self.config = config

    def predict(self, image_path, conf_threshold=None):
        conf = conf_threshold or self.config.inference_conf

        # Collect per-scale detections as separate lists for WBF
        per_scale_dets = []

        for scale in self.config.scales:
            results = self.model.predict(
                image_path,
                imgsz=scale,
                conf=conf,
                device='cpu',
                verbose=False,
            )

            img_height, img_width = results[0].orig_shape
            scale_dets = []

            for box in results[0].boxes:
                px1, py1, px2, py2 = box.xyxy[0].tolist()
                scale_dets.append({
                    'class_id': int(box.cls),
                    'bbox': (px1 / img_width, py1 / img_height,
                             px2 / img_width, py2 / img_height),
                    'conf': float(box.conf),
                })

            per_scale_dets.append(scale_dets)

        # Fuse across scales
        fused = weighted_box_fusion(
            per_scale_dets,
            iou_thresh=self.config.wbf_iou_thresh,
        )

        # Apply per-class confidence thresholds
        thresholds = self.config.class_conf_thresholds
        fused = [
            d for d in fused
            if d['conf'] >= thresholds.get(d['class_id'], DEFAULT_CONF_THRESHOLD)
        ]

        # Geometric heuristics
        if self.config.apply_heuristics:
            fused = apply_geometric_heuristics(fused, TARGET_CLASSES)

        return fused


# =============================================================================
# SINGLE-SCALE MODEL (for baseline comparison)
# =============================================================================

class SingleScaleYOLO:
    """Single-scale inference — use as a baseline to compare against ensemble."""

    def __init__(self, model_path, imgsz=1024):
        self.model = YOLOv10(model_path)
        self.imgsz = imgsz

    def predict(self, image_path, conf_threshold=0.25):
        results = self.model.predict(
            image_path,
            imgsz=self.imgsz,
            conf=conf_threshold,
            device='cpu',
            verbose=False,
        )

        img_height, img_width = results[0].orig_shape
        detections = []

        for box in results[0].boxes:
            px1, py1, px2, py2 = box.xyxy[0].tolist()
            detections.append({
                'class_id': int(box.cls),
                'bbox': (px1 / img_width, py1 / img_height,
                         px2 / img_width, py2 / img_height),
                'conf': float(box.conf),
            })

        return detections


# =============================================================================
# GROUND TRUTH & EVALUATION
# =============================================================================

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
                    boxes.append({
                        'class_id': class_id,
                        'bbox': yolo_to_coords(cx, cy, w, h),
                    })
    return boxes


def get_predictions(model, image_path, conf_threshold=0.25):
    """Run inference and filter to TARGET_CLASSES."""
    raw = model.predict(image_path, conf_threshold=conf_threshold)
    return [d for d in raw if d['class_id'] in TARGET_CLASSES]


def evaluate_image(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Match predictions to ground truth. Returns per-class TP/FP/FN counts."""
    stats = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in TARGET_CLASSES}

    for class_id in TARGET_CLASSES:
        c_gts = [g for g in gt_boxes if g['class_id'] == class_id]
        c_preds = sorted(
            [p for p in pred_boxes if p['class_id'] == class_id],
            key=lambda x: x.get('conf', 0),
            reverse=True,
        )

        matched_gt = set()

        for pred in c_preds:
            best_iou = 0
            best_idx = -1

            for i, gt in enumerate(c_gts):
                if i in matched_gt:
                    continue
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_threshold:
                stats[class_id]['tp'] += 1
                matched_gt.add(best_idx)
            else:
                stats[class_id]['fp'] += 1

        stats[class_id]['fn'] += len(c_gts) - len(matched_gt)

    return stats


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def collect_labeled_images(images_dir, labels_dir):
    """Return list of (image_path, label_path) pairs where labels exist."""
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    pairs = []
    for img_path in sorted(image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(labels_dir, f"{base_name}.txt")
        if os.path.exists(txt_path):
            pairs.append((img_path, txt_path))

    return pairs


def compute_metrics(tp, fp, fn):
    """Compute precision, recall, F1 from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_benchmark(model, label="Model"):
    pairs = collect_labeled_images(IMAGES_DIR, LABELS_DIR)

    if not pairs:
        print(f"No labeled images found (images: {IMAGES_DIR}, labels: {LABELS_DIR})")
        return

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: {label}")
    print(f"Evaluating {len(pairs)} labeled images")
    print(f"{'=' * 60}")

    overall = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in TARGET_CLASSES}

    for idx, (img_path, txt_path) in enumerate(pairs, 1):
        gt_boxes = load_ground_truth(txt_path)
        pred_boxes = get_predictions(model, img_path)
        img_stats = evaluate_image(gt_boxes, pred_boxes, iou_threshold=0.5)

        for cid in TARGET_CLASSES:
            overall[cid]['tp'] += img_stats[cid]['tp']
            overall[cid]['fp'] += img_stats[cid]['fp']
            overall[cid]['fn'] += img_stats[cid]['fn']

        if idx % 50 == 0:
            print(f"  Evaluated {idx}/{len(pairs)} images...")

    # Print results
    print(f"\n{'Class Name':<20} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 68)

    total_tp = total_fp = total_fn = 0

    for cid, name in TARGET_CLASSES.items():
        tp = overall[cid]['tp']
        fp = overall[cid]['fp']
        fn = overall[cid]['fn']
        total_tp += tp
        total_fp += fp
        total_fn += fn

        p, r, f1 = compute_metrics(tp, fp, fn)
        print(f"{name:<20} | {p:.3f}   | {r:.3f}   | {f1:.3f}   | {tp:4d} {fp:4d} {fn:4d}")

    print("-" * 68)
    p, r, f1 = compute_metrics(total_tp, total_fp, total_fn)
    print(f"{'OVERALL':<20} | {p:.3f}   | {r:.3f}   | {f1:.3f}   | {total_tp:4d} {total_fp:4d} {total_fn:4d}")
    print("=" * 60)

    return overall


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found.")
        return

    # --- Baseline: single scale at 1024 ---
    print(f"Loading model: {MODEL_PATH}")
    baseline = SingleScaleYOLO(MODEL_PATH, imgsz=1024)
    run_benchmark(baseline, label="Single-Scale Baseline (1024)")

    # --- Ensemble: multi-scale with WBF ---
    config = EnsembleConfig(
        model_path=MODEL_PATH,
        scales=(600, 800, 1024, 1280),
        wbf_iou_thresh=0.55,
        apply_heuristics=True,
    )
    ensemble = EnsembleYOLO(config)
    run_benchmark(ensemble, label="Ensemble WBF (600+800+1024+1280)")


if __name__ == "__main__":
    main()