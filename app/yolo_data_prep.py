"""
yolo_data_prep.py - Prepare labeled data for DocLayout-YOLO fine-tuning.

Reads images from docs_images/ and labels from docs_labels/ (YOLO .txt format),
splits into train/val sets, and creates the dataset.yaml required by ultralytics.

The split is stratified: it tries to balance the class distribution across splits
by scoring each image by its rarest-class contribution.

Usage:
    python yolo_data_prep.py                          # 80/20 split, default paths
    python yolo_data_prep.py --val-ratio 0.15         # 85/15 split
    python yolo_data_prep.py --images-dir my_images   # Custom paths
    python yolo_data_prep.py --seed 123               # Reproducible split
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

# DocLayout-YOLO DocStructBench class names (must match training order)
CLASS_NAMES = [
    "title",
    "plain_text",
    "abandon",
    "figure",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
]


@dataclass
class PrepConfig:
    images_dir: str = "docs_images"
    labels_dir: str = "docs_labels"
    output_dir: str = "yolo_dataset"
    val_ratio: float = 0.20
    seed: int = 42
    symlink: bool = False  # Use symlinks instead of copying (saves disk space)


# =============================================================================
# DATA INVENTORY
# =============================================================================

def collect_labeled_pairs(images_dir: str, labels_dir: str) -> list[tuple[Path, Path]]:
    """Find all (image, label) pairs where both files exist."""
    image_extensions = {".jpg", ".jpeg", ".png"}
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    pairs = []
    for img_file in sorted(images_path.iterdir()):
        if img_file.suffix.lower() not in image_extensions:
            continue

        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            pairs.append((img_file, label_file))

    return pairs


def parse_label_file(label_path: Path) -> list[int]:
    """Return list of class_ids present in a YOLO label file."""
    class_ids = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_ids.append(int(parts[0]))
    return class_ids


def compute_class_distribution(pairs: list[tuple[Path, Path]]) -> dict[int, int]:
    """Count total instances per class across all label files."""
    counts = Counter()
    for _, label_path in pairs:
        counts.update(parse_label_file(label_path))
    return dict(counts)


# =============================================================================
# STRATIFIED SPLIT
# =============================================================================

def stratified_split(
    pairs: list[tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """
    Split pairs into train/val with stratification.

    Strategy: score each image by the inverse frequency of its rarest class,
    then sort and interleave to ensure rare classes appear in both splits.
    """
    global_counts = compute_class_distribution(pairs)

    # Score each image: higher score = contains rarer classes
    scored = []
    for img_path, label_path in pairs:
        classes = parse_label_file(label_path)
        if not classes:
            # Empty label file — still include but low priority
            scored.append((0.0, img_path, label_path))
            continue

        rarity_score = max(1.0 / global_counts.get(c, 1) for c in classes)
        scored.append((rarity_score, img_path, label_path))

    # Shuffle within rarity tiers for randomness
    rng = random.Random(seed)
    rng.shuffle(scored)
    # Stable sort by rarity (rarest-class images spread across both splits)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Interleaved assignment: every Nth image goes to val
    val_every = max(1, round(1.0 / val_ratio))

    train_pairs, val_pairs = [], []
    for i, (_, img_path, label_path) in enumerate(scored):
        if i % val_every == 0:
            val_pairs.append((img_path, label_path))
        else:
            train_pairs.append((img_path, label_path))

    return train_pairs, val_pairs


# =============================================================================
# DATASET ASSEMBLY
# =============================================================================

def assemble_dataset(
    train_pairs: list[tuple[Path, Path]],
    val_pairs: list[tuple[Path, Path]],
    output_dir: str,
    use_symlink: bool = False,
) -> Path:
    """
    Create the YOLO dataset directory structure:
        output_dir/
            images/
                train/
                val/
            labels/
                train/
                val/
            dataset.yaml
    """
    out = Path(output_dir)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    link_fn = os.symlink if use_symlink else shutil.copy2

    def place_files(pairs, split):
        for img_path, label_path in pairs:
            img_dest = out / "images" / split / img_path.name
            lbl_dest = out / "labels" / split / label_path.name

            if not img_dest.exists():
                link_fn(str(img_path.resolve()), str(img_dest))
            if not lbl_dest.exists():
                link_fn(str(label_path.resolve()), str(lbl_dest))

    place_files(train_pairs, "train")
    place_files(val_pairs, "val")

    # Write dataset.yaml
    dataset_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = out / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    return yaml_path


# =============================================================================
# REPORTING
# =============================================================================

def print_split_report(
    train_pairs: list[tuple[Path, Path]],
    val_pairs: list[tuple[Path, Path]],
):
    """Print class distribution for each split."""
    train_counts = compute_class_distribution(train_pairs)
    val_counts = compute_class_distribution(val_pairs)

    print(f"\n{'Class':<20} {'Train':>7} {'Val':>7} {'Total':>7} {'Val %':>7}")
    print("-" * 52)

    for cid, name in enumerate(CLASS_NAMES):
        t = train_counts.get(cid, 0)
        v = val_counts.get(cid, 0)
        total = t + v
        pct = f"{v / total * 100:.1f}%" if total > 0 else "  -"
        print(f"{name:<20} {t:>7} {v:>7} {total:>7} {pct:>7}")

    total_t = sum(train_counts.values())
    total_v = sum(val_counts.values())
    total_all = total_t + total_v
    print("-" * 52)
    print(f"{'TOTAL':<20} {total_t:>7} {total_v:>7} {total_all:>7} {total_v / total_all * 100:.1f}%")
    print(f"\nImages: {len(train_pairs)} train, {len(val_pairs)} val")


# =============================================================================
# MAIN
# =============================================================================

def prepare_dataset(config: PrepConfig) -> Path:
    """End-to-end data preparation. Returns path to dataset.yaml."""
    print(f"Scanning for labeled pairs...")
    print(f"  Images: {config.images_dir}")
    print(f"  Labels: {config.labels_dir}")

    pairs = collect_labeled_pairs(config.images_dir, config.labels_dir)
    assert pairs, (
        f"No labeled image/label pairs found.\n"
        f"  Images dir: {config.images_dir}\n"
        f"  Labels dir: {config.labels_dir}\n"
        f"Make sure .txt label files exist in the labels directory."
    )

    print(f"Found {len(pairs)} labeled images.")

    # Global distribution
    global_dist = compute_class_distribution(pairs)
    print(f"\nGlobal class distribution:")
    for cid, name in enumerate(CLASS_NAMES):
        count = global_dist.get(cid, 0)
        if count > 0:
            print(f"  {name:<20} {count:>5}")

    # Split
    train_pairs, val_pairs = stratified_split(pairs, config.val_ratio, config.seed)
    print_split_report(train_pairs, val_pairs)

    # Assemble
    yaml_path = assemble_dataset(train_pairs, val_pairs, config.output_dir, config.symlink)
    print(f"\nDataset written to: {config.output_dir}/")
    print(f"YAML config: {yaml_path}")

    return yaml_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare YOLO dataset for fine-tuning")
    parser.add_argument("--images-dir", default="docs_images")
    parser.add_argument("--labels-dir", default="docs_labels")
    parser.add_argument("--output-dir", default="yolo_dataset")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying")

    args = parser.parse_args()

    config = PrepConfig(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        symlink=args.symlink,
    )

    prepare_dataset(config)


if __name__ == "__main__":
    main()