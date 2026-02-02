"""
Improved YOLO Labeling Tool for PDF Document Elements
======================================================
A user-friendly tool for labeling tables and images in PDF page images.

Features:
- Clear visual interface with instructions
- Navigation buttons and keyboard shortcuts
- Progress tracking and statistics
- View/edit existing labels
- Zoom controls
- Dark mode option
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw
import os
import glob
from pathlib import Path
from datetime import datetime

# ================= CONFIGURATION =================
def get_paths():
    """Calculates paths relative to this script file."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    return (
        project_root / "docs_images",
        project_root / "docs_labels", 
        project_root / "docs_crops"
    )

DEFAULT_IMAGE_DIR, DEFAULT_LABELS_DIR, DEFAULT_CROPS_DIR = get_paths()

CLASSES = {
    0: ("Table", "#FF6B6B", "t"),      # Red for tables
    1: ("Image", "#4ECDC4", "i"),      # Teal for images
    2: ("Chart", "#FFE66D", "c"),      # Yellow for charts (optional)
    3: ("Diagram", "#95E1D3", "d"),    # Green for diagrams (optional)
}

# Reverse lookup: key -> class_id
KEY_TO_CLASS = {v[2]: k for k, v in CLASSES.items()}
# =================================================


class ImprovedTagger:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Document Labeler")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # State
        self.image_list = []
        self.current_index = 0
        self.scale = 1.0
        self.boxes = []  # List of (x1, y1, x2, y2, class_id) for current image
        self.current_rect = None
        self.start_x = None
        self.start_y = None
        self.selected_class = 0  # Default to Table
        
        # Directories
        self.img_dir = str(DEFAULT_IMAGE_DIR)
        self.lbl_dir = str(DEFAULT_LABELS_DIR)
        self.crop_dir = str(DEFAULT_CROPS_DIR)
        
        # Setup UI
        self._setup_styles()
        self._create_ui()
        self._bind_keys()
        
        # Try to load images
        self._load_directory()
    
    def _setup_styles(self):
        """Configure ttk styles for a modern look."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom button styles
        style.configure('Nav.TButton', font=('Segoe UI', 11), padding=10)
        style.configure('Action.TButton', font=('Segoe UI', 10, 'bold'), padding=8)
        style.configure('Class.TButton', font=('Segoe UI', 10), padding=6)
        
        # Labels
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 10))
        style.configure('Stats.TLabel', font=('Consolas', 10))
    
    def _create_ui(self):
        """Build the complete UI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ===== LEFT PANEL (Canvas + Navigation) =====
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # -- Top bar with file info --
        top_bar = ttk.Frame(left_frame)
        top_bar.pack(fill=tk.X, pady=(0, 10))
        
        self.file_label = ttk.Label(top_bar, text="No image loaded", style='Header.TLabel')
        self.file_label.pack(side=tk.LEFT)
        
        self.progress_label = ttk.Label(top_bar, text="0 / 0", style='Status.TLabel')
        self.progress_label.pack(side=tk.RIGHT)
        
        # -- Canvas with scrollbars --
        canvas_frame = ttk.Frame(left_frame, relief='sunken', borderwidth=2)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(
            canvas_frame, 
            cursor="cross",
            bg='#2d2d2d',
            xscrollcommand=self.h_scroll.set,
            yscrollcommand=self.v_scroll.set
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)
        
        # -- Navigation bar --
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(nav_frame, text="◀◀ First", command=self.first_image, style='Nav.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="◀ Prev (←)", command=self.prev_image, style='Nav.TButton').pack(side=tk.LEFT, padx=2)
        
        # Jump to specific image
        ttk.Label(nav_frame, text="Go to:").pack(side=tk.LEFT, padx=(20, 5))
        self.jump_var = tk.StringVar()
        self.jump_entry = ttk.Entry(nav_frame, textvariable=self.jump_var, width=6)
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind('<Return>', self.jump_to_image)
        ttk.Button(nav_frame, text="Go", command=self.jump_to_image).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(nav_frame, text="Next (→) ▶", command=self.next_image, style='Nav.TButton').pack(side=tk.RIGHT, padx=2)
        ttk.Button(nav_frame, text="Last ▶▶", command=self.last_image, style='Nav.TButton').pack(side=tk.RIGHT, padx=2)
        ttk.Button(nav_frame, text="Skip (Space)", command=self.skip_to_unlabeled, style='Nav.TButton').pack(side=tk.RIGHT, padx=20)
        
        # ===== RIGHT PANEL (Controls) =====
        right_frame = ttk.Frame(main_frame, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # -- Directory Selection --
        dir_frame = ttk.LabelFrame(right_frame, text="Directories", padding=10)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(dir_frame, text="📁 Select Image Folder", command=self._select_image_dir).pack(fill=tk.X, pady=2)
        self.dir_display = ttk.Label(dir_frame, text="...", wraplength=280)
        self.dir_display.pack(fill=tk.X, pady=2)
        
        # -- Class Selection --
        class_frame = ttk.LabelFrame(right_frame, text="Select Class (then draw box)", padding=10)
        class_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.class_var = tk.IntVar(value=0)
        self.class_buttons = {}
        
        for class_id, (name, color, key) in CLASSES.items():
            btn_frame = ttk.Frame(class_frame)
            btn_frame.pack(fill=tk.X, pady=2)
            
            # Color swatch
            swatch = tk.Canvas(btn_frame, width=20, height=20, bg=color, highlightthickness=1)
            swatch.pack(side=tk.LEFT, padx=(0, 5))
            
            # Radio button
            rb = ttk.Radiobutton(
                btn_frame, 
                text=f"{name} ({key.upper()})",
                variable=self.class_var,
                value=class_id,
                command=lambda cid=class_id: self._select_class(cid)
            )
            rb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.class_buttons[class_id] = rb
        
        # -- Current Labels --
        labels_frame = ttk.LabelFrame(right_frame, text="Labels on Current Image", padding=10)
        labels_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Listbox with scrollbar
        list_container = ttk.Frame(labels_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.labels_listbox = tk.Listbox(list_container, font=('Consolas', 10), selectmode=tk.SINGLE)
        self.labels_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        list_scroll = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.labels_listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.labels_listbox.config(yscrollcommand=list_scroll.set)
        
        # Label actions
        label_btn_frame = ttk.Frame(labels_frame)
        label_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(label_btn_frame, text="🗑 Delete Selected", command=self._delete_selected_label).pack(side=tk.LEFT, padx=2)
        ttk.Button(label_btn_frame, text="🗑 Clear All", command=self._clear_all_labels).pack(side=tk.LEFT, padx=2)
        
        # -- Statistics --
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=6, font=('Consolas', 9), state='disabled', bg='#f5f5f5')
        self.stats_text.pack(fill=tk.X)
        
        # -- Quick Actions --
        action_frame = ttk.LabelFrame(right_frame, text="Actions", padding=10)
        action_frame.pack(fill=tk.X)
        
        ttk.Button(action_frame, text="💾 Save Labels (S)", command=self.save_labels, style='Action.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="🔄 Reload Image (R)", command=self.reload_image, style='Action.TButton').pack(fill=tk.X, pady=2)
        
        # -- Instructions --
        instr_frame = ttk.LabelFrame(right_frame, text="Keyboard Shortcuts", padding=10)
        instr_frame.pack(fill=tk.X, pady=(10, 0))
        
        instructions = """
← / → : Navigate images
Space : Skip to next unlabeled
T : Select Table class
I : Select Image class
C : Select Chart class  
D : Select Diagram class
S : Save current labels
R : Reload current image
Delete : Remove last box
Escape : Cancel drawing
        """.strip()
        
        ttk.Label(instr_frame, text=instructions, font=('Consolas', 9), justify=tk.LEFT).pack(anchor='w')
    
    def _bind_keys(self):
        """Set up keyboard shortcuts."""
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<space>", lambda e: self.skip_to_unlabeled())
        self.root.bind("<Delete>", lambda e: self._delete_last_box())
        self.root.bind("<BackSpace>", lambda e: self._delete_last_box())
        self.root.bind("<Escape>", lambda e: self._cancel_drawing())
        self.root.bind("<s>", lambda e: self.save_labels())
        self.root.bind("<S>", lambda e: self.save_labels())
        self.root.bind("<r>", lambda e: self.reload_image())
        self.root.bind("<R>", lambda e: self.reload_image())
        
        # Class selection keys
        for key, class_id in KEY_TO_CLASS.items():
            self.root.bind(f"<{key}>", lambda e, cid=class_id: self._select_class(cid))
            self.root.bind(f"<{key.upper()}>", lambda e, cid=class_id: self._select_class(cid))
        
        # Canvas bindings
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        
        # Mouse wheel zoom
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)  # Linux
        self.canvas.bind("<Button-5>", self._on_mousewheel)  # Linux
    
    def _select_image_dir(self):
        """Open dialog to select image directory."""
        dir_path = filedialog.askdirectory(
            title="Select Image Directory",
            initialdir=self.img_dir if os.path.exists(self.img_dir) else os.getcwd()
        )
        if dir_path:
            self.img_dir = dir_path
            # Update label directories based on new image dir
            base = Path(dir_path).parent
            self.lbl_dir = str(base / "docs_labels")
            self.crop_dir = str(base / "docs_crops")
            self._load_directory()
    
    def _load_directory(self):
        """Load images from the current directory."""
        os.makedirs(self.lbl_dir, exist_ok=True)
        os.makedirs(self.crop_dir, exist_ok=True)
        
        if not os.path.exists(self.img_dir):
            self.dir_display.config(text=f"⚠ Not found: {self.img_dir}")
            return
        
        # Find images
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        self.image_list = []
        for pattern in patterns:
            self.image_list.extend(glob.glob(os.path.join(self.img_dir, pattern)))
        self.image_list = sorted(set(self.image_list))
        
        if not self.image_list:
            self.dir_display.config(text=f"⚠ No images in: {self.img_dir}")
            messagebox.showwarning("No Images", f"No images found in:\n{self.img_dir}")
            return
        
        self.dir_display.config(text=f"✓ {len(self.image_list)} images")
        self.current_index = 0
        self._load_image()
        self._update_stats()
    
    def _load_image(self):
        """Load and display the current image."""
        if not self.image_list:
            return
        
        self.img_path = self.image_list[self.current_index]
        self.original_image = Image.open(self.img_path)
        
        # Calculate scale to fit canvas (max 900px height)
        canvas_h = 700
        w, h = self.original_image.size
        self.scale = min(1.0, canvas_h / h)
        
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        
        self.display_image = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW, tags="image")
        
        # Load existing labels
        self._load_labels()
        self._draw_boxes()
        
        # Update UI
        filename = os.path.basename(self.img_path)
        labeled = "✓ LABELED" if self.boxes else ""
        self.file_label.config(text=f"{filename} {labeled}")
        self.progress_label.config(text=f"{self.current_index + 1} / {len(self.image_list)}")
        
        # Clear current drawing
        self.current_rect = None
        self.start_x = None
        self.start_y = None
    
    def _load_labels(self):
        """Load existing YOLO labels for current image."""
        self.boxes = []
        txt_filename = os.path.splitext(os.path.basename(self.img_path))[0] + ".txt"
        txt_path = os.path.join(self.lbl_dir, txt_filename)
        
        if os.path.exists(txt_path):
            img_w, img_h = self.original_image.size
            
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        # Convert YOLO format back to pixel coords
                        x1 = (cx - bw/2) * img_w
                        y1 = (cy - bh/2) * img_h
                        x2 = (cx + bw/2) * img_w
                        y2 = (cy + bh/2) * img_h
                        
                        self.boxes.append((x1, y1, x2, y2, class_id))
        
        self._update_labels_list()
    
    def _draw_boxes(self):
        """Draw all bounding boxes on the canvas."""
        self.canvas.delete("box")
        self.canvas.delete("box_label")
        
        for i, (x1, y1, x2, y2, class_id) in enumerate(self.boxes):
            # Scale to display coords
            dx1, dy1 = x1 * self.scale, y1 * self.scale
            dx2, dy2 = x2 * self.scale, y2 * self.scale
            
            color = CLASSES.get(class_id, ("Unknown", "#999999", "?"))[1]
            name = CLASSES.get(class_id, ("Unknown", "#999999", "?"))[0]
            
            # Draw rectangle
            self.canvas.create_rectangle(
                dx1, dy1, dx2, dy2,
                outline=color, width=3, tags="box"
            )
            
            # Draw label background
            self.canvas.create_rectangle(
                dx1, dy1 - 20, dx1 + 80, dy1,
                fill=color, outline=color, tags="box_label"
            )
            
            # Draw label text
            self.canvas.create_text(
                dx1 + 40, dy1 - 10,
                text=f"{i+1}. {name}",
                fill="white", font=('Segoe UI', 9, 'bold'),
                tags="box_label"
            )
    
    def _update_labels_list(self):
        """Update the listbox showing current labels."""
        self.labels_listbox.delete(0, tk.END)
        
        for i, (x1, y1, x2, y2, class_id) in enumerate(self.boxes):
            name = CLASSES.get(class_id, ("Unknown", "#999999", "?"))[0]
            w = int(x2 - x1)
            h = int(y2 - y1)
            self.labels_listbox.insert(tk.END, f"{i+1}. {name} ({w}x{h})")
    
    def _update_stats(self):
        """Update the statistics display."""
        total = len(self.image_list)
        labeled = 0
        label_counts = {cid: 0 for cid in CLASSES}
        
        for img_path in self.image_list:
            txt_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            txt_path = os.path.join(self.lbl_dir, txt_filename)
            if os.path.exists(txt_path):
                labeled += 1
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            if cid in label_counts:
                                label_counts[cid] += 1
        
        # Build stats text
        stats = f"Total Images:  {total}\n"
        stats += f"Labeled:       {labeled} ({100*labeled/total:.1f}%)\n"
        stats += f"Remaining:     {total - labeled}\n"
        stats += f"─────────────────────\n"
        for cid, (name, _, _) in CLASSES.items():
            stats += f"{name}s:  {label_counts[cid]}\n"
        
        self.stats_text.config(state='normal')
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', stats.strip())
        self.stats_text.config(state='disabled')
    
    def _select_class(self, class_id):
        """Select a class for labeling."""
        self.selected_class = class_id
        self.class_var.set(class_id)
        name = CLASSES[class_id][0]
        self.root.title(f"YOLO Document Labeler - Drawing: {name}")
    
    # ===== Drawing Methods =====
    
    def _on_mouse_down(self, event):
        """Start drawing a box."""
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        color = CLASSES[self.selected_class][1]
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline=color, width=2, dash=(4, 4), tags="drawing"
        )
    
    def _on_mouse_drag(self, event):
        """Update the drawing box."""
        if self.current_rect:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)
    
    def _on_mouse_up(self, event):
        """Finish drawing and add the box."""
        if not self.current_rect or self.start_x is None:
            return
        
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Check minimum size
        if abs(end_x - self.start_x) < 10 or abs(end_y - self.start_y) < 10:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            return
        
        # Convert to original image coordinates
        x1 = min(self.start_x, end_x) / self.scale
        y1 = min(self.start_y, end_y) / self.scale
        x2 = max(self.start_x, end_x) / self.scale
        y2 = max(self.start_y, end_y) / self.scale
        
        # Add to boxes
        self.boxes.append((x1, y1, x2, y2, self.selected_class))
        
        # Clean up and redraw
        self.canvas.delete(self.current_rect)
        self.current_rect = None
        self._draw_boxes()
        self._update_labels_list()
        
        # Auto-save feedback
        name = CLASSES[self.selected_class][0]
        self.file_label.config(text=f"{os.path.basename(self.img_path)} - Added {name}")
    
    def _cancel_drawing(self):
        """Cancel the current drawing."""
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
    
    def _delete_last_box(self):
        """Delete the last drawn box."""
        if self.boxes:
            self.boxes.pop()
            self._draw_boxes()
            self._update_labels_list()
    
    def _delete_selected_label(self):
        """Delete the selected label from the listbox."""
        selection = self.labels_listbox.curselection()
        if selection:
            idx = selection[0]
            self.boxes.pop(idx)
            self._draw_boxes()
            self._update_labels_list()
    
    def _clear_all_labels(self):
        """Clear all labels for current image."""
        if self.boxes:
            if messagebox.askyesno("Confirm", "Delete all labels for this image?"):
                self.boxes = []
                self._draw_boxes()
                self._update_labels_list()
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel for scrolling."""
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
    
    # ===== Save/Load Methods =====
    
    def save_labels(self):
        """Save all labels in YOLO format."""
        if not self.image_list:
            return
        
        txt_filename = os.path.splitext(os.path.basename(self.img_path))[0] + ".txt"
        txt_path = os.path.join(self.lbl_dir, txt_filename)
        
        img_w, img_h = self.original_image.size
        
        with open(txt_path, 'w') as f:
            for x1, y1, x2, y2, class_id in self.boxes:
                # Convert to YOLO format
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        
        # Save crops
        for i, (x1, y1, x2, y2, class_id) in enumerate(self.boxes):
            crop_img = self.original_image.crop((int(x1), int(y1), int(x2), int(y2)))
            class_name = CLASSES[class_id][0]
            crop_name = f"{os.path.splitext(txt_filename)[0]}_{class_name}_{i}.jpg"
            crop_img.save(os.path.join(self.crop_dir, crop_name), quality=95)
        
        self._update_stats()
        self.file_label.config(text=f"{os.path.basename(self.img_path)} ✓ SAVED")
    
    def reload_image(self):
        """Reload the current image and its labels."""
        self._load_image()
    
    # ===== Navigation Methods =====
    
    def first_image(self):
        self.save_labels()
        self.current_index = 0
        self._load_image()
    
    def last_image(self):
        self.save_labels()
        self.current_index = len(self.image_list) - 1
        self._load_image()
    
    def prev_image(self):
        self.save_labels()
        if self.current_index > 0:
            self.current_index -= 1
            self._load_image()
    
    def next_image(self):
        self.save_labels()
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self._load_image()
    
    def skip_to_unlabeled(self):
        """Skip to the next unlabeled image."""
        self.save_labels()
        
        start = self.current_index
        for i in range(self.current_index + 1, len(self.image_list)):
            txt_filename = os.path.splitext(os.path.basename(self.image_list[i]))[0] + ".txt"
            txt_path = os.path.join(self.lbl_dir, txt_filename)
            if not os.path.exists(txt_path):
                self.current_index = i
                self._load_image()
                return
        
        # Wrap around
        for i in range(0, start):
            txt_filename = os.path.splitext(os.path.basename(self.image_list[i]))[0] + ".txt"
            txt_path = os.path.join(self.lbl_dir, txt_filename)
            if not os.path.exists(txt_path):
                self.current_index = i
                self._load_image()
                return
        
        messagebox.showinfo("Done!", "All images have been labeled!")
    
    def jump_to_image(self, event=None):
        """Jump to a specific image number."""
        try:
            num = int(self.jump_var.get())
            if 1 <= num <= len(self.image_list):
                self.save_labels()
                self.current_index = num - 1
                self._load_image()
            else:
                messagebox.showwarning("Invalid", f"Enter a number between 1 and {len(self.image_list)}")
        except ValueError:
            messagebox.showwarning("Invalid", "Please enter a valid number")
        
        self.jump_var.set("")


def main():
    root = tk.Tk()
    app = ImprovedTagger(root)
    root.mainloop()


if __name__ == "__main__":
    main()