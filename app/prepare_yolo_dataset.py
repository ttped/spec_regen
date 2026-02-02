"""
YOLO Dataset Preparation Utility
=================================
After labeling, run this script to:
1. Split data into train/val sets
2. Generate the data.yaml configuration file
3. Verify label integrity

Usage:
    python prepare_yolo_dataset.py
"""

import os
import shutil
import random
from pathlib import Path
import yaml

# ================= CONFIGURATION =================
def get_paths():
    """Calculates paths relative to this script file."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    return {
        'images': project_root / "docs_images",
        'labels': project_root / "docs_labels",
        'dataset': project_root / "yolo_dataset",  # Output directory
    }

PATHS = get_paths()

# Class definitions - MUST match what you used in the tagger
CLASSES = {
    0: "Table",
    1: "Image", 
    2: "Chart",
    3: "Diagram",
}

# Train/validation split ratio
TRAIN_RATIO = 0.8
# =================================================


def verify_labels(images_dir: Path, labels_dir: Path):
    """Verify that labels are valid YOLO format."""
    print("Verifying labels...")
    
    issues = []
    valid_pairs = []
    
    # Find all images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            continue  # Skip unlabeled images
        
        # Check label file
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                issues.append(f"Empty label file: {label_path.name}")
                continue
            
            valid = True
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append(f"{label_path.name} line {i+1}: Expected 5 values, got {len(parts)}")
                    valid = False
                    continue
                
                try:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    
                    if class_id not in CLASSES:
                        issues.append(f"{label_path.name} line {i+1}: Unknown class {class_id}")
                        valid = False
                    
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        issues.append(f"{label_path.name} line {i+1}: Values out of range [0,1]")
                        valid = False
                        
                except ValueError as e:
                    issues.append(f"{label_path.name} line {i+1}: Parse error - {e}")
                    valid = False
            
            if valid:
                valid_pairs.append((img_path, label_path))
                
        except Exception as e:
            issues.append(f"Error reading {label_path.name}: {e}")
    
    return valid_pairs, issues


def split_dataset(pairs: list, train_ratio: float = 0.8):
    """Split dataset into train and validation sets."""
    random.seed(42)  # For reproducibility
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    return train_pairs, val_pairs


def copy_files(pairs: list, dest_images: Path, dest_labels: Path):
    """Copy image/label pairs to destination directories."""
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    
    for img_path, label_path in pairs:
        shutil.copy2(img_path, dest_images / img_path.name)
        shutil.copy2(label_path, dest_labels / label_path.name)


def generate_yaml(dataset_dir: Path, classes: dict):
    """Generate YOLO data.yaml configuration file."""
    yaml_content = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {k: v for k, v in classes.items()}
    }
    
    yaml_path = dataset_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def print_statistics(train_pairs, val_pairs, labels_dir):
    """Print dataset statistics."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nSplit:")
    print(f"  Training:   {len(train_pairs)} images")
    print(f"  Validation: {len(val_pairs)} images")
    print(f"  Total:      {len(train_pairs) + len(val_pairs)} images")
    
    # Count labels by class
    class_counts = {cid: 0 for cid in CLASSES}
    all_pairs = train_pairs + val_pairs
    
    for _, label_path in all_pairs:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    if cid in class_counts:
                        class_counts[cid] += 1
    
    print(f"\nLabel counts:")
    for cid, name in CLASSES.items():
        print(f"  {name}: {class_counts[cid]}")
    
    print(f"\nTotal annotations: {sum(class_counts.values())}")


def main():
    print("YOLO Dataset Preparation")
    print("="*50)
    
    images_dir = PATHS['images']
    labels_dir = PATHS['labels']
    dataset_dir = PATHS['dataset']
    
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Output directory: {dataset_dir}")
    
    # Step 1: Verify labels
    valid_pairs, issues = verify_labels(images_dir, labels_dir)
    
    if issues:
        print(f"\n⚠ Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    
    if not valid_pairs:
        print("\n❌ No valid image/label pairs found!")
        print("Make sure you have labeled some images with the tagger.")
        return
    
    print(f"\n✓ Found {len(valid_pairs)} valid image/label pairs")
    
    # Step 2: Split dataset
    train_pairs, val_pairs = split_dataset(valid_pairs, TRAIN_RATIO)
    
    # Step 3: Create directory structure
    print(f"\nCreating dataset at: {dataset_dir}")
    
    if dataset_dir.exists():
        response = input("Output directory exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(dataset_dir)
    
    # Create YOLO directory structure
    train_images = dataset_dir / "images" / "train"
    train_labels = dataset_dir / "labels" / "train"
    val_images = dataset_dir / "images" / "val"
    val_labels = dataset_dir / "labels" / "val"
    
    # Copy files
    print("Copying training files...")
    copy_files(train_pairs, train_images, train_labels)
    
    print("Copying validation files...")
    copy_files(val_pairs, val_images, val_labels)
    
    # Step 4: Generate YAML
    yaml_path = generate_yaml(dataset_dir, CLASSES)
    print(f"\n✓ Generated: {yaml_path}")
    
    # Step 5: Print statistics
    print_statistics(train_pairs, val_pairs, labels_dir)
    
    # Print next steps
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print(f"""
1. Your dataset is ready at: {dataset_dir}

2. To train YOLO, update your training script:

   from ultralytics import YOLO
   
   model = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy
   
   results = model.train(
       data='{yaml_path}',
       epochs=100,
       imgsz=640,
       batch=16,
       device='cpu',  # or '0' for GPU
       patience=20,
       plots=True
   )

3. Your trained model will be at: runs/detect/train/weights/best.pt

4. To use the trained model:
   
   model = YOLO('runs/detect/train/weights/best.pt')
   results = model.predict('path/to/new/image.jpg')
""")


if __name__ == "__main__":
    main()