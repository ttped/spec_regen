import torch
from PIL import Image, ImageDraw
from transformers import TableTransformerForObjectDetection, DetrImageProcessor

def load_and_run_structure_model(image_path, model_directory):
    """
    Loads a local Table Transformer model and detects table structure (cells/rows).
    Returns the original image with detected bounding boxes drawn on it.
    """
    # Load model and processor from local files explicitly
    model = TableTransformerForObjectDetection.from_pretrained(
        model_directory, 
        local_files_only=True
    )
    processor = DetrImageProcessor.from_pretrained(
        model_directory, 
        local_files_only=True
    )

    image = Image.open(image_path).convert("RGB")
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")

    # Forward pass (inference)
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert outputs (logits/boxes) to actual coordinates
    target_sizes = [image.size[::-1]]
    results = processor.post_process_object_detection(
        outputs, 
        threshold=0.7, 
        target_sizes=target_sizes
    )[0]

    return _draw_boxes(image, results)

def _draw_boxes(image, results):
    """
    Helper to draw bounding boxes on the image for visual validation.
    """
    draw = ImageDraw.Draw(image)
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # Unpack box coordinates
        box = [round(i, 2) for i in box.tolist()]
        xmin, ymin, xmax, ymax = box
        
        # Draw rectangle (Red for high visibility)
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=2)
        
        # Optional: Label the box (0 usually maps to 'table')
        # In structure models, labels might be 'table row', 'table column', etc.
        draw.text((xmin, ymin), f"{score:.2f}", fill="red")

    return image

if __name__ == "__main__":
    # Example usage
    local_model_path = "./models/tatr-structure"
    test_image = "test_table.jpg" # Ensure this file exists
    
    result_img = load_and_run_structure_model(test_image, local_model_path)
    result_img.save("test_output.jpg")
    print("Test complete. Check 'test_output.jpg' for results.")