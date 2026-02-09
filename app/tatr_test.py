import os
import torch
from PIL import Image, ImageDraw
from transformers import TableTransformerForObjectDetection, DetrImageProcessor, AutoImageProcessor

def get_model_path():
    """
    Resolves the absolute path to the model directory.
    Assumes structure:
      repo_name/
      ├── app/tatr_test.py
      └── models/tatr-structure/
    """
    # Get the directory where this script lives (repo_name/app)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level (..) to repo root, then down into models
    model_path = os.path.join(script_dir, "..", "models", "tatr-structure")
    
    return os.path.normpath(model_path)

def load_local_model():
    """
    Loads the model and processor from the local directory.
    """
    model_dir = get_model_path()

    # Explicit check to avoid vague errors
    config_file = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"CRITICAL: Config not found at {config_file}. Check folder structure.")

    print(f"Loading model from: {model_dir}")

    # Load with local_files_only=True to prevent any internet connection attempts
    model = TableTransformerForObjectDetection.from_pretrained(
        model_dir, 
        local_files_only=True
    )
    processor = AutoImageProcessor.from_pretrained(
        model_dir, 
        local_files_only=True
    )
    
    return model, processor

def run_inference(image_path, model, processor):
    """
    Runs the model on the image and returns bounding box results.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = Image.open(image_path).convert("RGB")
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to bounding boxes
    target_sizes = [image.size[::-1]]
    results = processor.post_process_object_detection(
        outputs, 
        threshold=0.7, 
        target_sizes=target_sizes
    )[0]

    return image, results

def draw_boxes(image, results):
    """
    Draws red bounding boxes around detected table elements.
    """
    draw = ImageDraw.Draw(image)
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # Convert tensor to list and round
        box = [round(i, 2) for i in box.tolist()]
        xmin, ymin, xmax, ymax = box
        
        # Draw box
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=2)
        
        # Draw Label (Score)
        draw.text((xmin, ymin), f"{score:.2f}", fill="red")

    return image

if __name__ == "__main__":
    # 1. Setup
    # Ensure you have a 'test_table.jpg' in the same folder as this script (app/)
    # Or update this path to where your image is.
    input_image_path = os.path.join(os.path.dirname(__file__), "test_table.jpg")
    output_image_path = os.path.join(os.path.dirname(__file__), "output_result.jpg")

    # 2. Load
    model, processor = load_local_model()

    # 3. Run
    print(f"Processing image: {input_image_path}")
    original_image, results = run_inference(input_image_path, model, processor)

    # 4. Visualize
    final_image = draw_boxes(original_image, results)
    
    # 5. Save
    final_image.save(output_image_path)
    print(f"Done. Result saved to: {output_image_path}")