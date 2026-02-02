from ultralytics import YOLO

def train_safe_yolo():
    # 1. Load the base model (downloads from GitHub Releases, NOT Hugging Face)
    # 'yolov8n.pt' is the "Nano" model - fastest and sufficient for documents.
    model = YOLO('yolov8n.pt') 

    # 2. Train on your local data
    # You must create a 'data.yaml' file pointing to your images (see below)
    results = model.train(
        data='path/to/your/data.yaml', 
        epochs=50,          # 50 epochs is usually plenty for this task
        imgsz=640,          # Document pages are large, but 640 often works
        plots=True,
        device='cpu'        # Use '0' if you have an NVIDIA GPU
    )
    
    # 3. Save the model
    # The trained model will be saved locally at runs/detect/train/weights/best.pt
    print("Training Complete. Model saved.")

if __name__ == '__main__':
    train_safe_yolo()