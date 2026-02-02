"""
YOLO Training Script for Document Element Detection
====================================================
Trains a YOLOv8 model to detect tables and images in PDF pages.

Prerequisites:
    pip install ultralytics

Usage:
    1. First run: python prepare_yolo_dataset.py
    2. Then run:  python train_yolo.py
"""

from ultralytics import YOLO
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
    }

PATHS = get_paths()

# Training parameters - adjust based on your hardware
CONFIG = {
    'model': 'yolov8n.pt',      # Options: yolov8n.pt (fast), yolov8s.pt (balanced), yolov8m.pt (accurate)
    'epochs': 100,               # More epochs = better, but diminishing returns after ~100
    'imgsz': 640,               # Image size - 640 is standard, can try 1280 for documents
    'batch': 16,                # Reduce to 8 or 4 if you get memory errors
    'device': 'cpu',            # Use '0' for NVIDIA GPU, 'mps' for Apple Silicon
    'patience': 20,             # Early stopping - stops if no improvement for N epochs
    'workers': 4,               # Data loading workers
    'project': 'runs/detect',   # Where to save results
    'name': 'doc_elements',     # Experiment name
}
# =================================================


def check_dataset():
    """Verify dataset exists before training."""
    data_yaml = PATHS['data_yaml']
    
    if not data_yaml.exists():
        print("❌ Dataset not found!")
        print(f"   Expected: {data_yaml}")
        print("\n   Run this first: python prepare_yolo_dataset.py")
        return False
    
    # Count images
    train_dir = PATHS['dataset'] / "images" / "train"
    val_dir = PATHS['dataset'] / "images" / "val"
    
    train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*"))) if val_dir.exists() else 0
    
    print(f"✓ Dataset found:")
    print(f"  Training images:   {train_count}")
    print(f"  Validation images: {val_count}")
    print(f"  Config: {data_yaml}")
    
    if train_count < 10:
        print("\n⚠ Warning: Very small dataset. Consider labeling more images.")
        print("  Recommended: 100+ images for decent results")
    
    return True


def train():
    """Run YOLO training."""
    print("\n" + "="*50)
    print("YOLO TRAINING")
    print("="*50)
    
    if not check_dataset():
        return
    
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-"*50)
    print("Starting training... (this may take a while)")
    print("-"*50 + "\n")
    
    # Load pretrained model
    model = YOLO(CONFIG['model'])
    
    # Train
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
        plots=True,
        save=True,
        verbose=True,
    )
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    
    best_model = Path(CONFIG['project']) / CONFIG['name'] / "weights" / "best.pt"
    print(f"\n✓ Best model saved to: {best_model}")
    
    print(f"""
Next steps:

1. Check training plots at:
   {Path(CONFIG['project']) / CONFIG['name']}

2. Test your model:
   
   from ultralytics import YOLO
   model = YOLO('{best_model}')
   results = model.predict('path/to/test/image.jpg', save=True)

3. Use in your pipeline:
   
   model = YOLO('{best_model}')
   results = model('page_image.jpg')
   
   for box in results[0].boxes:
       class_id = int(box.cls)
       confidence = float(box.conf)
       x1, y1, x2, y2 = box.xyxy[0].tolist()
       print(f"Found {{class_id}} at {{x1:.0f}},{{y1:.0f}} -> {{x2:.0f}},{{y2:.0f}} ({{confidence:.2%}})")
""")
    
    return results


def evaluate(model_path: str = None):
    """Evaluate a trained model on the validation set."""
    if model_path is None:
        model_path = Path(CONFIG['project']) / CONFIG['name'] / "weights" / "best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train a model first with: python train_yolo.py")
        return
    
    print(f"Evaluating: {model_path}")
    
    model = YOLO(model_path)
    metrics = model.val(data=str(PATHS['data_yaml']))
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")
    
    return metrics


def predict_single(image_path: str, model_path: str = None, save: bool = True):
    """Run prediction on a single image."""
    if model_path is None:
        model_path = Path(CONFIG['project']) / CONFIG['name'] / "weights" / "best.pt"
    
    model = YOLO(model_path)
    results = model.predict(image_path, save=save, conf=0.25)
    
    print(f"\nDetections in {image_path}:")
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = results[0].names[class_id]
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  {class_name}: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f}) [{confidence:.1%}]")
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'eval':
            evaluate()
        elif command == 'predict' and len(sys.argv) > 2:
            predict_single(sys.argv[2])
        else:
            print("Usage:")
            print("  python train_yolo.py          # Train model")
            print("  python train_yolo.py eval     # Evaluate model")
            print("  python train_yolo.py predict <image_path>  # Test on image")
    else:
        train()