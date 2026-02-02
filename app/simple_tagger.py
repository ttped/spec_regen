import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import glob
from pathlib import Path

# ================= CONFIGURATION =================
def get_paths():
    """
    Calculates paths relative to this script file.
    Assumes directory structure:
    /project
      /src (where this script is)
      /docs_images
      /docs_labels
      /docs_crops
    """
    # Get the directory where THIS script is located
    script_dir = Path(__file__).resolve().parent
    
    # Go one level up to the project root
    project_root = script_dir.parent
    
    return (
        project_root / "docs_images",
        project_root / "docs_labels",
        project_root / "docs_crops"
    )

IMAGE_DIR, OUTPUT_LABELS_DIR, OUTPUT_CROPS_DIR = get_paths()

CLASSES = {
    "t": (0, "Table"),  # Press 't' for Table
    "i": (1, "Image")   # Press 'i' for Image
}
# =================================================

class SimpleTagger:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple PDF Tagger (Press 't' for Table, 'i' for Image)")
        
        # Setup Directories
        # Convert Path objects to strings for compatibility
        self.img_dir = str(IMAGE_DIR)
        self.lbl_dir = str(OUTPUT_LABELS_DIR)
        self.crop_dir = str(OUTPUT_CROPS_DIR)

        os.makedirs(self.lbl_dir, exist_ok=True)
        os.makedirs(self.crop_dir, exist_ok=True)
        
        # Load Images
        if not os.path.exists(self.img_dir):
             messagebox.showerror("Error", f"Image directory not found:\n{self.img_dir}")
             return

        # Use glob to find .jpg (case insensitive for safety)
        self.image_list = sorted(glob.glob(os.path.join(self.img_dir, "*.[jJ][pP][gG]")))
        
        if not self.image_list:
            messagebox.showerror("Error", f"No .jpg images found in:\n{self.img_dir}")
            return
            
        self.current_index = 0
        self.scale = 1.0
        self.rect = None
        self.start_x = None
        self.start_y = None
        
        # UI Setup
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.status_label = tk.Label(root, text="Welcome", bg="lightgray")
        self.status_label.pack(fill=tk.X)
        
        # Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        # Key Bindings
        root.bind("<Right>", self.next_image)
        root.bind("<Left>", self.prev_image)
        for key in CLASSES.keys():
            root.bind(key, lambda event, k=key: self.save_selection(k))
            
        self.load_image()

    def load_image(self):
        """Loads and scales the current image to fit window."""
        self.img_path = self.image_list[self.current_index]
        self.original_image = Image.open(self.img_path)
        
        # Calculate scale to fit screen (max 800px height for visibility)
        screen_h = 800
        w, h = self.original_image.size
        self.scale = min(1.0, screen_h / h)
        
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        
        self.tk_image = ImageTk.PhotoImage(self.original_image.resize((new_w, new_h)))
        
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        
        # Clear previous rectangles
        self.canvas.delete("rect")
        
        self.update_status(f"Image {self.current_index + 1}/{len(self.image_list)}: {os.path.basename(self.img_path)}")

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        # Create rectangle
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_move_press(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        self.end_x = event.x
        self.end_y = event.y

    def save_selection(self, key):
        if not self.rect:
            return

        class_id, class_name = CLASSES[key]
        
        # 1. Normalize Coordinates for YOLO
        # Convert screen coords back to original image coords
        x1 = min(self.start_x, self.end_x) / self.scale
        y1 = min(self.start_y, self.end_y) / self.scale
        x2 = max(self.start_x, self.end_x) / self.scale
        y2 = max(self.start_y, self.end_y) / self.scale
        
        # Get image dimensions
        img_w, img_h = self.original_image.size
        
        # Calculate YOLO center_x, center_y, w, h (normalized 0-1)
        center_x = ((x1 + x2) / 2) / img_w
        center_y = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        # 2. Append to Text File (YOLO Format)
        txt_filename = os.path.splitext(os.path.basename(self.img_path))[0] + ".txt"
        txt_path = os.path.join(self.lbl_dir, txt_filename)
        
        with open(txt_path, "a") as f:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
        # 3. Save the Actual Crop (For immediate use)
        crop_img = self.original_image.crop((x1, y1, x2, y2))
        
        # Count existing crops to make unique filename
        existing_crops = len(os.listdir(self.crop_dir))
        crop_name = f"{os.path.splitext(txt_filename)[0]}_{class_name}_{existing_crops}.jpg"
        
        crop_img.save(os.path.join(self.crop_dir, crop_name))
        
        # Visual Feedback
        self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline='green', width=2)
        self.update_status(f"Saved {class_name}! ({width:.2f}x{height:.2f})")
        self.rect = None # Reset active rect

    def next_image(self, event=None):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self, event=None):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def update_status(self, text):
        self.status_label.config(text=text)

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleTagger(root)
    root.mainloop()