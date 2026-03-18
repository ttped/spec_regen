"""
yolo_finetune.py - Fine-tune DocLayout-YOLO on your labeled document pages.

Loads the pretrained DocStructBench checkpoint and fine-tunes it on your
domain-specific data (prepared by yolo_data_prep.py).

Key decisions:
    - Freezes backbone layers initially (transfer learning), then optionally
      unfreezes for a second phase of full fine-tuning.
    - Uses the same imgsz=1024 as your production pipeline.
    - Saves best.pt and last.pt to runs/finetune/weights/.

Prerequisites:
    1. Run yolo_data_prep.py to create yolo_dataset/dataset.yaml
    2. Have the pretrained .pt file locally (or it downloads from HuggingFace)

Usage:
    python yolo_finetune.py                                    # Default settings
    python yolo_finetune.py --epochs 50 --batch 4              # Adjust training
    python yolo_finetune.py --freeze-epochs 10 --epochs 30     # Two-phase training
    python yolo_finetune.py --model path/to/best.pt            # Resume from checkpoint
    python yolo_finetune.py --no-freeze                        # Skip frozen phase
"""

import os
from pathlib import Path
from dataclasses import dataclass

from doclayout_yolo import YOLOv10


# =============================================================================
# CONFIGURATION
# =============================================================================

HUGGINGFACE_MODEL = "juliozhao/DocLayout-YOLO-DocStructBench"

# Default local model path (same as yolo_asset_extractor.py)
DEFAULT_MODEL_PATH = "doclayout_yolo_docstructbench_imgsz1024.pt"


@dataclass
class FinetuneConfig:
    # Model
    model_path: str = DEFAULT_MODEL_PATH
    dataset_yaml: str = "yolo_dataset/dataset.yaml"

    # Training - Phase 1: Frozen backbone
    freeze_epochs: int = 15
    freeze_layers: int = 10  # Freeze first N layers of backbone

    # Training - Phase 2: Full fine-tune (unfrozen)
    epochs: int = 30  # Total epochs for phase 2 (set 0 to skip)

    # Shared training params
    batch_size: int = 4       # Adjust based on VRAM (4 for ~8GB, 8 for ~16GB)
    imgsz: int = 1024         # Match your production pipeline
    device: str = "0"         # CUDA device ("0", "0,1", or "cpu")
    workers: int = 4
    patience: int = 15        # Early stopping patience

    # Learning rates
    lr0_frozen: float = 0.001     # Higher LR when backbone frozen (only head trains)
    lr0_unfrozen: float = 0.0001  # Lower LR when full model trains
    lrf: float = 0.01             # Final LR as fraction of lr0

    # Augmentation (lighter than default — doc pages are structured, not natural images)
    hsv_h: float = 0.005   # Minimal hue shift (docs aren't color-sensitive)
    hsv_s: float = 0.1     # Light saturation variation
    hsv_v: float = 0.2     # Moderate brightness variation (scan quality varies)
    degrees: float = 0.0   # No rotation (docs are axis-aligned)
    translate: float = 0.05 # Slight translation
    scale: float = 0.15    # Slight scale variation
    flipud: float = 0.0    # Never flip vertically (text goes top-to-bottom)
    fliplr: float = 0.0    # Never flip horizontally (text goes left-to-right)
    mosaic: float = 0.0    # Disable mosaic (destroys document structure)
    mixup: float = 0.0     # Disable mixup (same reason)

    # Output
    project: str = "runs/finetune"
    name: str = "doclayout_domain"


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_pretrained_model(model_path: str) -> YOLOv10:
    """Load the pretrained DocLayout-YOLO checkpoint."""
    path = Path(model_path)

    if path.exists():
        print(f"Loading local model: {path}")
        return YOLOv10(str(path))

    print(f"Local model not found at {path}")
    print(f"Downloading from HuggingFace: {HUGGINGFACE_MODEL}")
    return YOLOv10.from_pretrained(HUGGINGFACE_MODEL)


# =============================================================================
# TRAINING
# =============================================================================

def run_frozen_phase(model: YOLOv10, config: FinetuneConfig) -> str:
    """
    Phase 1: Train with backbone frozen.

    Only the detection head learns your domain-specific patterns.
    This is fast, stable, and prevents catastrophic forgetting of
    the pretrained features.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Frozen backbone training")
    print(f"  Freezing first {config.freeze_layers} layers")
    print(f"  Epochs: {config.freeze_epochs}")
    print(f"  LR: {config.lr0_frozen}")
    print("=" * 60)

    results = model.train(
        data=config.dataset_yaml,
        epochs=config.freeze_epochs,
        batch=config.batch_size,
        imgsz=config.imgsz,
        device=config.device,
        workers=config.workers,
        patience=config.patience,

        # Frozen backbone
        freeze=config.freeze_layers,

        # Learning rate
        lr0=config.lr0_frozen,
        lrf=config.lrf,

        # Augmentation (document-appropriate)
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        degrees=config.degrees,
        translate=config.translate,
        scale=config.scale,
        flipud=config.flipud,
        fliplr=config.fliplr,
        mosaic=config.mosaic,
        mixup=config.mixup,

        # Output
        project=config.project,
        name=f"{config.name}_phase1",
        exist_ok=True,

        verbose=True,
    )

    # Return path to best weights from phase 1
    best_path = Path(config.project) / f"{config.name}_phase1" / "weights" / "best.pt"
    print(f"\nPhase 1 complete. Best weights: {best_path}")
    return str(best_path)


def run_unfrozen_phase(model_path: str, config: FinetuneConfig) -> str:
    """
    Phase 2: Full fine-tuning with all layers unfrozen.

    Starts from the Phase 1 best checkpoint and trains the entire
    model with a lower learning rate. This refines backbone features
    for your specific document style.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Full fine-tuning (all layers)")
    print(f"  Starting from: {model_path}")
    print(f"  Epochs: {config.epochs}")
    print(f"  LR: {config.lr0_unfrozen}")
    print("=" * 60)

    model = YOLOv10(model_path)

    results = model.train(
        data=config.dataset_yaml,
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.imgsz,
        device=config.device,
        workers=config.workers,
        patience=config.patience,

        # No freeze
        freeze=0,

        # Lower learning rate for full fine-tuning
        lr0=config.lr0_unfrozen,
        lrf=config.lrf,

        # Same augmentation
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        degrees=config.degrees,
        translate=config.translate,
        scale=config.scale,
        flipud=config.flipud,
        fliplr=config.fliplr,
        mosaic=config.mosaic,
        mixup=config.mixup,

        # Output
        project=config.project,
        name=f"{config.name}_phase2",
        exist_ok=True,

        verbose=True,
    )

    best_path = Path(config.project) / f"{config.name}_phase2" / "weights" / "best.pt"
    print(f"\nPhase 2 complete. Best weights: {best_path}")
    return str(best_path)


def run_finetuning(config: FinetuneConfig) -> str:
    """
    Execute the full fine-tuning pipeline.

    Returns path to the final best.pt checkpoint.
    """
    # Validate dataset exists
    assert Path(config.dataset_yaml).exists(), (
        f"Dataset YAML not found: {config.dataset_yaml}\n"
        f"Run yolo_data_prep.py first to create it."
    )

    model = load_pretrained_model(config.model_path)

    if config.freeze_epochs > 0:
        phase1_best = run_frozen_phase(model, config)
    else:
        phase1_best = config.model_path

    if config.epochs > 0:
        final_best = run_unfrozen_phase(phase1_best, config)
    else:
        final_best = phase1_best

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print(f"Final model: {final_best}")
    print("=" * 60)
    print("\nNext steps:")
    print(f"  1. Evaluate:  python yolo_eval_compare.py --finetuned {final_best}")
    print(f"  2. Use in pipeline:  update model path in simple_pipeline.py")
    print(f"  3. Ensemble:  add to model_paths in yolo_benchmark.py")

    return final_best


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune DocLayout-YOLO")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Pretrained .pt path")
    parser.add_argument("--dataset", default="yolo_dataset/dataset.yaml", help="Dataset YAML")
    parser.add_argument("--freeze-epochs", type=int, default=15, help="Frozen backbone epochs")
    parser.add_argument("--freeze-layers", type=int, default=10, help="Layers to freeze")
    parser.add_argument("--epochs", type=int, default=30, help="Unfrozen fine-tune epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--device", default="0", help="CUDA device")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--no-freeze", action="store_true", help="Skip frozen phase entirely")
    parser.add_argument("--project", default="runs/finetune")
    parser.add_argument("--name", default="doclayout_domain")

    args = parser.parse_args()

    config = FinetuneConfig(
        model_path=args.model,
        dataset_yaml=args.dataset,
        freeze_epochs=0 if args.no_freeze else args.freeze_epochs,
        freeze_layers=args.freeze_layers,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )

    run_finetuning(config)


if __name__ == "__main__":
    main()