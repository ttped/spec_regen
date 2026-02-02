"""
DocLayout-YOLO Training & Inference Script
===========================================
For document layout analysis - detects tables, figures, text, titles, etc.

Installation:
    pip install doclayout-yolo

Usage:
    # Quick inference using pre-trained model (recommended to start):
    python train_doclayout.py predict path/to/image.jpg
    
    # Evaluate pre-trained model on your dataset:
    python train_doclayout.py eval
    
    # Fine-tune on your own labeled data:
    python train_doclayout.py train

Security Note:
    All processing is local. The model downloads once from HuggingFace
    and caches locally. No data is sent externally during inference.
"""

from doclayout_yolo import YOLOv10
from pathlib import Path
import os

# ================= CONFIGURATION =================
def get_paths():
    """Calculates paths relative to this script file."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return {
        'dataset': project_root / "yolo_dataset",
        'data_yaml': project_root / "yolo_dataset" / "data.yaml",
        'images': project_root / "docs_images",
    }

PATHS = get_paths()

# HuggingFace model for pre-trained weights
HUGGINGFACE_MODEL = "juliozhao/DocLayout-YOLO-DocStructBench"

# Training parameters - adjust based on your hardware
CONFIG = {
    'epochs': 100,
    'imgsz': 1024,              # DocLayout-YOLO works best at 1024
    'batch': 8,                 # Reduce if memory errors
    'device': 'cpu',            # Use 'cuda:0' for GPU, 'mps' for Apple Silicon
    'patience': 20,
    'workers': 4,
    'project': 'runs/detect',
    'name': 'doclayout_finetuned',
    'lr0': 0.01,                # Initial learning rate
}
# =================================================


def load_model(local_path: str = None):
    """Load DocLayout-YOLO model."""
    if local_path and Path(local_path).exists():
        print(f"Loading local model: {local_path}")
        return YOLOv10(local_path)
    else:
        print(f"Loading pre-trained model from HuggingFace: {HUGGINGFACE_MODEL}")
        print("(Model will be cached locally for future use)")
        return YOLOv10.from_pretrained(HUGGINGFACE_MODEL)


def check_dataset():
    """Verify dataset exists before training."""
    data_yaml = PATHS['data_yaml']
    
    if not data_yaml.exists():
        print("✗ Dataset not found!")
        print(f"   Expected: {data_yaml}")
        print("\n   Run this first: python prepare_yolo_dataset.py")
        return False
    
    train_dir = PATHS['dataset'] / "images" / "train"
    val_dir = PATHS['dataset'] / "images" / "val"
    
    train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*"))) if val_dir.exists() else 0
    
    print(f"✓ Dataset found:")
    print(f"  Training images:   {train_count}")
    print(f"  Validation images: {val_count}")
    print(f"  Config: {data_yaml}")
    
    if train_count < 10:
        print("\n⚠ Warning: Very small dataset.")
        print("  Consider using the pre-trained model directly instead of fine-tuning.")
    
    return True


def train():
    """Fine-tune DocLayout-YOLO on your dataset."""
    print("\n" + "=" * 50)
    print("DOCLAYOUT-YOLO FINE-TUNING")
    print("=" * 50)
    
    if not check_dataset():
        return
    
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-" * 50)
    print("Loading pre-trained DocLayout-YOLO...")
    print("-" * 50 + "\n")
    
    # Load pre-trained model
    model = load_model()
    
    # Fine-tune on your data
    results = model.train(
        data=str(PATHS['data_yaml']),
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        device=CONFIG['device'],
        patience=CONFIG['patience'],
        workers=CONFIG['workers'],
        project=CONFIG['project'],
        name=CONFIG['name'],
        lr0=CONFIG['lr0'],
        plots=True,
        save=True,
        verbose=True,
    )
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    
    best_model = Path(CONFIG['project']) / CONFIG['name'] / "weights" / "best.pt"
    print(f"\n✓ Fine-tuned model saved to: {best_model}")
    
    return results


def evaluate(model_path: str = None):
    """Evaluate model on validation set."""
    print("\n" + "=" * 50)
    print("DOCLAYOUT-YOLO EVALUATION")
    print("=" * 50)
    
    if not check_dataset():
        return
    
    model = load_model(model_path)
    metrics = model.val(data=str(PATHS['data_yaml']))
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")
    
    return metrics


def predict_single(image_path: str, model_path: str = None, save: bool = True):
    """Run prediction on a single image."""
    print("\n" + "=" * 50)
    print("DOCLAYOUT-YOLO PREDICTION")
    print("=" * 50)
    
    model = load_model(model_path)
    
    print(f"\nProcessing: {image_path}")
    results = model.predict(
        image_path, 
        imgsz=1024,
        conf=0.25,
        save=save,
        device=CONFIG['device']
    )
    
    print(f"\nDetections:")
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = results[0].names[class_id]
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  {class_name}: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f}) [{confidence:.1%}]")
    
    if save:
        print(f"\n✓ Annotated image saved to: runs/detect/predict/")
    
    print("\n" + "-" * 50)
    print("SECURITY: All processing was done locally.")
    print("-" * 50)
    
    return results


def predict_folder(folder_path: str = None, model_path: str = None):
    """Run prediction on all images in a folder."""
    print("\n" + "=" * 50)
    print("DOCLAYOUT-YOLO BATCH PREDICTION")
    print("=" * 50)
    
    folder = Path(folder_path) if folder_path else PATHS['images']
    
    if not folder.exists():
        print(f"✗ Folder not found: {folder}")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"✗ No images found in: {folder}")
        return
    
    print(f"Found {len(images)} images in: {folder}")
    
    model = load_model(model_path)
    
    print(f"\nProcessing images...")
    
    # Process as batch for efficiency
    image_paths = [str(img) for img in images]
    results = model.predict(
        image_paths,
        imgsz=1024,
        conf=0.25,
        save=True,
        device=CONFIG['device'],
        verbose=False
    )
    
    # Summary
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n✓ Processed {len(images)} images")
    print(f"✓ Total detections: {total_detections}")
    print(f"✓ Results saved to: runs/detect/predict/")
    
    print("\n" + "-" * 50)
    print("SECURITY: All processing was done locally.")
    print("-" * 50)
    
    return results


def show_classes():
    """Display the classes detected by DocLayout-YOLO."""
    print("\n" + "=" * 50)
    print("DOCLAYOUT-YOLO CLASSES (DocStructBench)")
    print("=" * 50)
    print("""
The pre-trained DocStructBench model detects these document elements:

  0: title           - Document titles, section headers
  1: plain text      - Regular paragraph text
  2: abandon         - Artifacts, noise, page numbers, headers/footers
  3: figure          - Images, photos, illustrations
  4: figure_caption  - Captions below/above figures
  5: table           - Data tables
  6: table_caption   - Captions for tables
  7: table_footnote  - Footnotes within tables
  8: isolate_formula - Mathematical formulas
  9: formula_caption - Captions for formulas

This model is trained on diverse documents including:
- Academic papers, Technical reports
- Financial documents, Legal documents
- Books, Magazines, Newspapers
- And more...
""")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("""
DocLayout-YOLO - Document Layout Analysis
=========================================

Usage:
    python train_doclayout.py classes              # Show detected classes
    python train_doclayout.py predict <image>      # Predict single image
    python train_doclayout.py predict_folder       # Predict all images in docs_images/
    python train_doclayout.py eval                 # Evaluate on your dataset
    python train_doclayout.py train                # Fine-tune on your dataset

Quick Start:
    # Just want to try it? Run prediction on your images:
    python train_doclayout.py predict_folder

    # Or test on a single image:
    python train_doclayout.py predict path/to/document.jpg
""")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == 'classes':
        show_classes()
    elif command == 'train':
        train()
    elif command == 'eval':
        evaluate()
    elif command == 'predict' and len(sys.argv) > 2:
        predict_single(sys.argv[2])
    elif command == 'predict_folder':
        folder = sys.argv[2] if len(sys.argv) > 2 else None
        predict_folder(folder)
    else:
        print(f"Unknown command: {command}")
        print("Run without arguments to see usage.")