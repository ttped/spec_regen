"""
yolo_eval_compare.py - Compare base vs fine-tuned DocLayout-YOLO models.

Runs both models through the same evaluation framework (from yolo_benchmark.py)
and produces a side-by-side comparison report. Also generates per-image diffs
to see exactly where the fine-tuned model improved or regressed.

Usage:
    python yolo_eval_compare.py --finetuned runs/finetune/doclayout_domain_phase2/weights/best.pt
    python yolo_eval_compare.py --finetuned best.pt --base original.pt
    python yolo_eval_compare.py --finetuned best.pt --detailed  # Per-image breakdown
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from doclayout_yolo import YOLOv10

from yolo_benchmark import (
    SingleScaleYOLO,
    EnsembleYOLO,
    EnsembleConfig,
    TARGET_CLASSES,
    CLASS_CONF_THRESHOLDS,
    DEFAULT_CONF_THRESHOLD,
    collect_labeled_images,
    load_ground_truth,
    get_predictions,
    evaluate_image,
    compute_metrics,
    apply_geometric_heuristics,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

IMAGES_DIR = "docs_images"
LABELS_DIR = "docs_labels"
BASE_MODEL_PATH = "doclayout_yolo_docstructbench_imgsz1024.pt"


@dataclass
class CompareConfig:
    base_model_path: str = BASE_MODEL_PATH
    finetuned_model_path: str = ""
    images_dir: str = IMAGES_DIR
    labels_dir: str = LABELS_DIR
    iou_threshold: float = 0.5
    detailed: bool = False           # Per-image diff output
    output_report: str = ""          # Save JSON report (auto-named if empty)
    test_ensemble: bool = False      # Also test base+finetuned as ensemble


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, pairs, iou_threshold=0.5):
    """Run evaluation and return per-class stats + per-image details."""
    overall = {c: {"tp": 0, "fp": 0, "fn": 0} for c in TARGET_CLASSES}
    per_image = []

    for img_path, txt_path in pairs:
        gt_boxes = load_ground_truth(txt_path)
        pred_boxes = get_predictions(model, img_path)
        img_stats = evaluate_image(gt_boxes, pred_boxes, iou_threshold)

        for cid in TARGET_CLASSES:
            overall[cid]["tp"] += img_stats[cid]["tp"]
            overall[cid]["fp"] += img_stats[cid]["fp"]
            overall[cid]["fn"] += img_stats[cid]["fn"]

        per_image.append({
            "image": os.path.basename(img_path),
            "stats": img_stats,
            "n_gt": len(gt_boxes),
            "n_pred": len(pred_boxes),
        })

    return overall, per_image


def stats_to_metrics(overall):
    """Convert raw TP/FP/FN counts to precision/recall/F1 per class."""
    metrics = {}
    for cid, name in TARGET_CLASSES.items():
        tp = overall[cid]["tp"]
        fp = overall[cid]["fp"]
        fn = overall[cid]["fn"]
        p, r, f1 = compute_metrics(tp, fp, fn)
        metrics[cid] = {
            "name": name,
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    total_tp = sum(overall[c]["tp"] for c in TARGET_CLASSES)
    total_fp = sum(overall[c]["fp"] for c in TARGET_CLASSES)
    total_fn = sum(overall[c]["fn"] for c in TARGET_CLASSES)
    p, r, f1 = compute_metrics(total_tp, total_fp, total_fn)
    metrics["overall"] = {
        "name": "OVERALL",
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

    return metrics


# =============================================================================
# COMPARISON REPORTING
# =============================================================================

def print_comparison(base_metrics, ft_metrics, base_label="Base", ft_label="Fine-tuned"):
    """Print side-by-side comparison table."""
    print(f"\n{'=' * 80}")
    print(f"MODEL COMPARISON: {base_label} vs {ft_label}")
    print(f"{'=' * 80}")

    header = (
        f"{'Class':<18} | "
        f"{'P':>5} {'R':>5} {'F1':>5} | "
        f"{'P':>5} {'R':>5} {'F1':>5} | "
        f"{'ΔF1':>6}"
    )
    subheader = f"{'':18} | {'--- ' + base_label + ' ---':^17} | {'--- ' + ft_label + ' ---':^17} |"

    print(subheader)
    print(header)
    print("-" * 80)

    # Per-class rows
    for cid in TARGET_CLASSES:
        b = base_metrics[cid]
        f = ft_metrics[cid]
        delta_f1 = f["f1"] - b["f1"]
        arrow = "▲" if delta_f1 > 0.005 else ("▼" if delta_f1 < -0.005 else "─")

        print(
            f"{b['name']:<18} | "
            f"{b['precision']:>5.3f} {b['recall']:>5.3f} {b['f1']:>5.3f} | "
            f"{f['precision']:>5.3f} {f['recall']:>5.3f} {f['f1']:>5.3f} | "
            f"{arrow} {delta_f1:>+.3f}"
        )

    # Overall row
    print("-" * 80)
    b = base_metrics["overall"]
    f = ft_metrics["overall"]
    delta_f1 = f["f1"] - b["f1"]
    arrow = "▲" if delta_f1 > 0.005 else ("▼" if delta_f1 < -0.005 else "─")

    print(
        f"{b['name']:<18} | "
        f"{b['precision']:>5.3f} {b['recall']:>5.3f} {b['f1']:>5.3f} | "
        f"{f['precision']:>5.3f} {f['recall']:>5.3f} {f['f1']:>5.3f} | "
        f"{arrow} {delta_f1:>+.3f}"
    )
    print("=" * 80)


def print_per_image_diffs(base_per_image, ft_per_image):
    """Show images where fine-tuned model changed behavior."""
    print(f"\n{'=' * 60}")
    print("PER-IMAGE DIFFERENCES (fine-tuned vs base)")
    print(f"{'=' * 60}")

    improvements = []
    regressions = []

    for b_img, f_img in zip(base_per_image, ft_per_image):
        b_total_tp = sum(b_img["stats"][c]["tp"] for c in TARGET_CLASSES)
        b_total_fp = sum(b_img["stats"][c]["fp"] for c in TARGET_CLASSES)
        b_total_fn = sum(b_img["stats"][c]["fn"] for c in TARGET_CLASSES)

        f_total_tp = sum(f_img["stats"][c]["tp"] for c in TARGET_CLASSES)
        f_total_fp = sum(f_img["stats"][c]["fp"] for c in TARGET_CLASSES)
        f_total_fn = sum(f_img["stats"][c]["fn"] for c in TARGET_CLASSES)

        # Net score: TP gained - (FP gained + FN gained)
        delta = (f_total_tp - b_total_tp) - (f_total_fp - b_total_fp) - (f_total_fn - b_total_fn)

        if delta > 0:
            improvements.append((b_img["image"], delta, f_total_tp, f_total_fp, f_total_fn))
        elif delta < 0:
            regressions.append((b_img["image"], delta, f_total_tp, f_total_fp, f_total_fn))

    print(f"\nImprovements: {len(improvements)} images")
    for name, delta, tp, fp, fn in sorted(improvements, key=lambda x: x[1], reverse=True)[:20]:
        print(f"  ▲ {name:<40} +{delta}  (TP={tp} FP={fp} FN={fn})")

    print(f"\nRegressions:  {len(regressions)} images")
    for name, delta, tp, fp, fn in sorted(regressions, key=lambda x: x[1])[:20]:
        print(f"  ▼ {name:<40} {delta}  (TP={tp} FP={fp} FN={fn})")

    unchanged = len(base_per_image) - len(improvements) - len(regressions)
    print(f"\nUnchanged:    {unchanged} images")


# =============================================================================
# MAIN
# =============================================================================

def run_comparison(config: CompareConfig):
    """Run full comparison between base and fine-tuned models."""
    assert config.finetuned_model_path, "Must specify --finetuned model path"
    assert Path(config.finetuned_model_path).exists(), (
        f"Fine-tuned model not found: {config.finetuned_model_path}"
    )

    pairs = collect_labeled_images(config.images_dir, config.labels_dir)
    assert pairs, (
        f"No labeled images found.\n"
        f"  Images: {config.images_dir}\n"
        f"  Labels: {config.labels_dir}"
    )

    print(f"Evaluating on {len(pairs)} labeled images")
    print(f"IoU threshold: {config.iou_threshold}")

    # --- Base model ---
    print(f"\nLoading base model: {config.base_model_path}")
    base_model = SingleScaleYOLO(config.base_model_path, imgsz=1024)
    print("Running base model evaluation...")
    base_overall, base_per_image = evaluate_model(base_model, pairs, config.iou_threshold)
    base_metrics = stats_to_metrics(base_overall)

    # --- Fine-tuned model ---
    print(f"\nLoading fine-tuned model: {config.finetuned_model_path}")
    ft_model = SingleScaleYOLO(config.finetuned_model_path, imgsz=1024)
    print("Running fine-tuned model evaluation...")
    ft_overall, ft_per_image = evaluate_model(ft_model, pairs, config.iou_threshold)
    ft_metrics = stats_to_metrics(ft_overall)

    # --- Print comparison ---
    print_comparison(base_metrics, ft_metrics)

    if config.detailed:
        print_per_image_diffs(base_per_image, ft_per_image)

    # --- Optional: Ensemble (base + fine-tuned via WBF) ---
    if config.test_ensemble:
        print(f"\nLoading ensemble (base + fine-tuned)...")
        ensemble_config = EnsembleConfig(
            model_paths=[config.base_model_path, config.finetuned_model_path],
            imgsz=1024,
            wbf_iou_thresh=0.55,
            apply_heuristics=True,
        )
        ensemble_model = EnsembleYOLO(ensemble_config)
        print("Running ensemble evaluation...")
        ens_overall, ens_per_image = evaluate_model(ensemble_model, pairs, config.iou_threshold)
        ens_metrics = stats_to_metrics(ens_overall)

        print_comparison(base_metrics, ens_metrics, "Base", "Ensemble (base+ft)")

    # --- Save JSON report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_model": config.base_model_path,
            "finetuned_model": config.finetuned_model_path,
            "n_images": len(pairs),
            "iou_threshold": config.iou_threshold,
        },
        "base": {k: v for k, v in base_metrics.items()},
        "finetuned": {k: v for k, v in ft_metrics.items()},
    }

    if config.test_ensemble:
        report["ensemble"] = {k: v for k, v in ens_metrics.items()}

    report_path = config.output_report or f"eval_compare_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned YOLO models")
    parser.add_argument("--finetuned", required=True, help="Path to fine-tuned best.pt")
    parser.add_argument("--base", default=BASE_MODEL_PATH, help="Path to base model")
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--labels-dir", default=LABELS_DIR)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--detailed", action="store_true", help="Show per-image diffs")
    parser.add_argument("--ensemble", action="store_true", help="Also test base+ft ensemble")
    parser.add_argument("--output", default="", help="Report output path")

    args = parser.parse_args()

    config = CompareConfig(
        base_model_path=args.base,
        finetuned_model_path=args.finetuned,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        iou_threshold=args.iou,
        detailed=args.detailed,
        test_ensemble=args.ensemble,
        output_report=args.output,
    )

    run_comparison(config)


if __name__ == "__main__":
    main()