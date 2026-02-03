"""
DocLayout-YOLO Labeling Tool with Auto-Label
=============================================
A labeling tool configured for DocLayout-YOLO DocStructBench classes.
Includes auto-labeling using the pre-trained model for faster annotation.

Features:
- 10 classes matching DocStructBench model
- AUTO-LABEL: Use pre-trained model to generate initial labels
- Clear visual interface with instructions
- Navigation buttons and keyboard shortcuts
- Progress tracking and statistics
- View/edit existing labels
- Zoom controls

Classes (matching DocLayout-YOLO DocStructBench):
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

Setup for Auto-Label:
  1. pip install doclayout-yolo
  2. Download model from HuggingFace: juliozhao/DocLayout-YOLO-DocStructBench
  3. Place .pt file in same directory as this script, or set MODEL_PATH below
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw
import os
import glob
from pathlib import Path
from datetime import datetime

# Try to import DocLayout-YOLO (optional - only needed for auto-label)
DOCLAYOUT_AVAILABLE = False
try:
    from doclayout_yolo import YOLOv10
    DOCLAYOUT_AVAILABLE = True
except ImportError:
    pass

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

# DocLayout-YOLO DocStructBench classes
# Format: class_id: (name, color, hotkey)
CLASSES = {
    0: ("title", "#E74C3C", "1"),            # Red - titles/headers
    1: ("plain_text", "#3498DB", "2"),       # Blue - regular paragraphs
    2: ("abandon", "#95A5A6", "3"),          # Gray - noise/artifacts
    3: ("figure", "#2ECC71", "4"),           # Green - images/illustrations
    4: ("figure_caption", "#27AE60", "5"),   # Dark green - figure captions
    5: ("table", "#9B59B6", "6"),            # Purple - tables
    6: ("table_caption", "#8E44AD", "7"),    # Dark purple - table captions
    7: ("table_footnote", "#D35400", "8"),   # Orange - table footnotes
    8: ("isolate_formula", "#F39C12", "9"),  # Yellow - formulas
    9: ("formula_caption", "#E67E22", "0"),  # Dark yellow - formula captions
}

# Reverse lookup: key -> class_id
KEY_TO_CLASS = {v[2]: k for k, v in CLASSES.items()}

# ===== MODEL CONFIGURATION =====
# Path to DocLayout-YOLO model file (.pt)
# Options:
#   1. Place model in same directory as this script
#   2. Set full path below
#   3. Will search common locations automatically
MODEL_PATH = None  # Set this to your model path, e.g. "/path/to/doclayout_yolo_docstructbench_imgsz1024.pt"
MODEL_CONFIDENCE = 0.25  # Confidence threshold for auto-labeling
MODEL_IMAGE_SIZE = 1024  # Image size for inference
# =================================================


class DocLayoutTagger:
    def __init__(self, root):
        self.root = root
        self.root.title("DocLayout-YOLO Labeler")
        self.root.geometry("1500x950")
        self.root.minsize(1200, 800)
        
        # State
        self.image_list = []
        self.current_index = 0
        self.scale = 1.0
        self.boxes = []  # List of (x1, y1, x2, y2, class_id) for current image
        self.current_rect = None
        self.start_x = None
        self.start_y = None
        self.selected_class = 5  # Default to table (most common use case)
        
        # Model for auto-labeling
        self.model = None
        self.model_loaded = False
        
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
        
        # Try to load model in background
        self._try_load_model()
    
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
        
        # ===== RIGHT PANEL (Controls) - Make scrollable =====
        right_outer = ttk.Frame(main_frame, width=350)
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_outer.pack_propagate(False)
        
        # Create canvas for scrolling
        right_canvas = tk.Canvas(right_outer, highlightthickness=0, width=330)
        right_scrollbar = ttk.Scrollbar(right_outer, orient="vertical", command=right_canvas.yview)
        right_frame = ttk.Frame(right_canvas)
        
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_canvas_window = right_canvas.create_window((0, 0), window=right_frame, anchor="nw")
        
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        
        def configure_scroll(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        right_frame.bind("<Configure>", configure_scroll)
        
        def configure_width(event):
            right_canvas.itemconfig(right_canvas_window, width=event.width)
        right_canvas.bind("<Configure>", configure_width)
        
        # Mouse wheel scrolling for right panel
        def on_right_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        right_canvas.bind("<MouseWheel>", on_right_mousewheel)
        right_frame.bind("<MouseWheel>", on_right_mousewheel)
        
        # -- Model & Auto-Label (TOP PRIORITY) --
        model_frame = ttk.LabelFrame(right_frame, text="🤖 Auto-Label", padding=8)
        model_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.auto_label_btn = ttk.Button(
            model_frame, 
            text="Auto-Label This Image (A)", 
            command=self.auto_label,
            style='Action.TButton'
        )
        self.auto_label_btn.pack(fill=tk.X, pady=2)
        
        self.model_status = ttk.Label(model_frame, text="Model: Not loaded", font=('Segoe UI', 8))
        self.model_status.pack(fill=tk.X)
        
        ttk.Button(model_frame, text="📂 Load Model...", command=self._select_model_file).pack(fill=tk.X, pady=(4,0))
        
        # -- Directory Selection (compact) --
        dir_frame = ttk.LabelFrame(right_frame, text="Directory", padding=8)
        dir_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Button(dir_frame, text="📁 Select Image Folder", command=self._select_image_dir).pack(fill=tk.X)
        self.dir_display = ttk.Label(dir_frame, text="...", wraplength=300, font=('Segoe UI', 8))
        self.dir_display.pack(fill=tk.X)
        
        # -- Class Selection (compact) --
        class_frame = ttk.LabelFrame(right_frame, text="Class (1-9, 0)", padding=8)
        class_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.class_var = tk.IntVar(value=5)  # Default to table
        self.class_buttons = {}
        
        # Create 2-column grid for classes
        class_grid = ttk.Frame(class_frame)
        class_grid.pack(fill=tk.X)
        
        for i, (class_id, (name, color, key)) in enumerate(CLASSES.items()):
            row = i // 2
            col = i % 2
            
            btn_frame = ttk.Frame(class_grid)
            btn_frame.grid(row=row, column=col, sticky='w', pady=1, padx=2)
            
            # Color swatch (smaller)
            swatch = tk.Canvas(btn_frame, width=12, height=12, bg=color, highlightthickness=1)
            swatch.pack(side=tk.LEFT, padx=(0, 3))
            
            # Radio button with hotkey
            rb = ttk.Radiobutton(
                btn_frame, 
                text=f"{key}:{name[:8]}",  # Truncate long names
                variable=self.class_var,
                value=class_id,
                command=lambda cid=class_id: self._select_class(cid)
            )
            rb.pack(side=tk.LEFT)
            self.class_buttons[class_id] = rb
        
        # -- Current Labels (compact) --
        labels_frame = ttk.LabelFrame(right_frame, text="Current Labels", padding=8)
        labels_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Listbox with scrollbar (shorter)
        list_container = ttk.Frame(labels_frame)
        list_container.pack(fill=tk.X)
        
        self.labels_listbox = tk.Listbox(list_container, font=('Consolas', 8), selectmode=tk.SINGLE, height=6)
        self.labels_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        list_scroll = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.labels_listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.labels_listbox.config(yscrollcommand=list_scroll.set)
        
        # Label actions (horizontal)
        label_btn_frame = ttk.Frame(labels_frame)
        label_btn_frame.pack(fill=tk.X, pady=(4, 0))
        
        ttk.Button(label_btn_frame, text="Delete Sel", command=self._delete_selected_label, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(label_btn_frame, text="Clear All", command=self._clear_all_labels, width=10).pack(side=tk.LEFT, padx=1)
        
        # -- Quick Actions (compact) --
        action_frame = ttk.LabelFrame(right_frame, text="Actions", padding=8)
        action_frame.pack(fill=tk.X, pady=(0, 8))
        
        btn_row = ttk.Frame(action_frame)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="💾 Save (S)", command=self.save_labels, width=12).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_row, text="🔄 Reload (R)", command=self.reload_image, width=12).pack(side=tk.LEFT, padx=1)
        
        # -- Statistics (compact) --
        stats_frame = ttk.LabelFrame(right_frame, text="Stats", padding=8)
        stats_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.stats_text = tk.Text(stats_frame, height=5, font=('Consolas', 8), state='disabled', bg='#f5f5f5')
        self.stats_text.pack(fill=tk.X)
        
        # -- Instructions (compact) --
        instr_frame = ttk.LabelFrame(right_frame, text="Keys", padding=8)
        instr_frame.pack(fill=tk.X)
        
        instructions = "←/→:Nav | Space:Skip | A:AutoLabel\n1-0:Class | S:Save | Del:Remove"
        ttk.Label(instr_frame, text=instructions, font=('Consolas', 8), justify=tk.LEFT).pack(anchor='w')
    
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
        self.root.bind("<a>", lambda e: self.auto_label())
        self.root.bind("<A>", lambda e: self.auto_label())
        
        # Class selection keys (1-9 and 0) - bind directly to the key characters
        for key, class_id in KEY_TO_CLASS.items():
            self.root.bind(key, lambda e, cid=class_id: self._select_class(cid))
        
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
        canvas_h = 750
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
            
            color = CLASSES.get(class_id, ("unknown", "#999999", "?"))[1]
            name = CLASSES.get(class_id, ("unknown", "#999999", "?"))[0]
            
            # Draw rectangle
            self.canvas.create_rectangle(
                dx1, dy1, dx2, dy2,
                outline=color, width=3, tags="box"
            )
            
            # Draw label background
            label_text = f"{i+1}. {name}"
            label_width = max(80, len(label_text) * 7)
            self.canvas.create_rectangle(
                dx1, dy1 - 18, dx1 + label_width, dy1,
                fill=color, outline=color, tags="box_label"
            )
            
            # Draw label text
            self.canvas.create_text(
                dx1 + label_width/2, dy1 - 9,
                text=label_text,
                fill="white", font=('Segoe UI', 8, 'bold'),
                tags="box_label"
            )
    
    def _update_labels_list(self):
        """Update the listbox showing current labels."""
        self.labels_listbox.delete(0, tk.END)
        
        for i, (x1, y1, x2, y2, class_id) in enumerate(self.boxes):
            name = CLASSES.get(class_id, ("unknown", "#999999", "?"))[0]
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
        stats = f"Images: {labeled}/{total} labeled\n"
        stats += f"─────────────────────\n"
        for cid, (name, _, _) in CLASSES.items():
            if label_counts[cid] > 0:
                stats += f"{name}: {label_counts[cid]}\n"
        
        self.stats_text.config(state='normal')
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', stats.strip())
        self.stats_text.config(state='disabled')
    
    def _select_class(self, class_id):
        """Select a class for labeling."""
        self.selected_class = class_id
        self.class_var.set(class_id)
        name = CLASSES[class_id][0]
        self.root.title(f"DocLayout-YOLO Labeler - Drawing: {name}")
    
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
        
        # Save crops (optional)
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
    
    # ===== Model Methods =====
    
    def _try_load_model(self):
        """Try to find and load the DocLayout-YOLO model."""
        if not DOCLAYOUT_AVAILABLE:
            self.model_status.config(text="Model: doclayout-yolo not installed")
            self.auto_label_btn.config(state='disabled')
            return
        
        # Search for model file
        search_paths = []
        
        # User-specified path
        if MODEL_PATH:
            search_paths.append(Path(MODEL_PATH))
        
        # Try to get script directory safely
        try:
            script_dir = Path(__file__).resolve().parent
        except NameError:
            script_dir = Path.cwd()
        
        # Common model filenames to search for
        model_names = [
            "doclayout_yolo_docstructbench_imgsz1024.pt",
            "doclayout_yolo_docstructbench.pt",
            "best.pt",
        ]
        
        # Search in script directory
        for name in model_names:
            search_paths.append(script_dir / name)
        search_paths.append(script_dir / "models" / "doclayout_yolo_docstructbench_imgsz1024.pt")
        
        # Search in current working directory
        cwd = Path.cwd()
        for name in model_names:
            search_paths.append(cwd / name)
        search_paths.append(cwd / "models" / "doclayout_yolo_docstructbench_imgsz1024.pt")
        
        # Search in image directory's parent (project root)
        if self.img_dir and os.path.exists(self.img_dir):
            project_root = Path(self.img_dir).parent
            for name in model_names:
                search_paths.append(project_root / name)
            search_paths.append(project_root / "models" / "doclayout_yolo_docstructbench_imgsz1024.pt")
        
        # Try each path
        for model_path in search_paths:
            if model_path.exists():
                self._load_model(str(model_path))
                return
        
        self.model_status.config(text="Model: Not found (click Load Model)")
    
    def _load_model(self, model_path: str):
        """Load the DocLayout-YOLO model from a file."""
        try:
            self.model_status.config(text="Model: Loading...")
            self.root.update()
            
            self.model = YOLOv10(model_path)
            self.model_loaded = True
            
            # Shorten path for display
            display_path = Path(model_path).name
            self.model_status.config(text=f"Model: ✓ {display_path}")
            self.auto_label_btn.config(state='normal')
            
        except Exception as e:
            self.model_status.config(text=f"Model: Error - {str(e)[:30]}")
            self.auto_label_btn.config(state='disabled')
            messagebox.showerror("Model Error", f"Failed to load model:\n{e}")
    
    def _select_model_file(self):
        """Open dialog to select model file."""
        # Use current working directory or image directory as starting point
        start_dir = self.img_dir if os.path.exists(self.img_dir) else os.getcwd()
        
        file_path = filedialog.askopenfilename(
            title="Select DocLayout-YOLO Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
            initialdir=start_dir
        )
        if file_path:
            self._load_model(file_path)
    
    def auto_label(self):
        """Run the model on current image and populate boxes."""
        if not self.model_loaded:
            messagebox.showwarning(
                "Model Not Loaded",
                "Please load a DocLayout-YOLO model first.\n\n"
                "1. Click 'Load Model...' button\n"
                "2. Select your .pt model file\n\n"
                "Download from: huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench"
            )
            return
        
        if not self.image_list:
            return
        
        # Confirm if there are existing boxes
        if self.boxes:
            result = messagebox.askyesnocancel(
                "Existing Labels",
                "This image already has labels.\n\n"
                "Yes = Replace all labels\n"
                "No = Add to existing labels\n"
                "Cancel = Abort"
            )
            if result is None:  # Cancel
                return
            if result:  # Yes - replace
                self.boxes = []
        
        # Update UI
        self.file_label.config(text=f"{os.path.basename(self.img_path)} - Running model...")
        self.root.update()
        
        try:
            # Run inference
            results = self.model.predict(
                self.img_path,
                imgsz=MODEL_IMAGE_SIZE,
                conf=MODEL_CONFIDENCE,
                device='cpu',  # Use CPU for compatibility
                verbose=False
            )
            
            # Extract detections
            detections = results[0].boxes
            num_added = 0
            
            for box in detections:
                class_id = int(box.cls)
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Add to boxes
                self.boxes.append((x1, y1, x2, y2, class_id))
                num_added += 1
            
            # Redraw
            self._draw_boxes()
            self._update_labels_list()
            
            # Feedback
            self.file_label.config(
                text=f"{os.path.basename(self.img_path)} - Auto-labeled: {num_added} boxes"
            )
            
        except Exception as e:
            messagebox.showerror("Auto-Label Error", f"Failed to run model:\n{e}")
            self.file_label.config(text=f"{os.path.basename(self.img_path)} - Error")
    
    def auto_label_all_unlabeled(self):
        """Run auto-label on all unlabeled images."""
        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Please load a model first.")
            return
        
        # Count unlabeled
        unlabeled = []
        for img_path in self.image_list:
            txt_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            txt_path = os.path.join(self.lbl_dir, txt_filename)
            if not os.path.exists(txt_path):
                unlabeled.append(img_path)
        
        if not unlabeled:
            messagebox.showinfo("Done", "All images are already labeled!")
            return
        
        result = messagebox.askyesno(
            "Auto-Label All",
            f"Run auto-label on {len(unlabeled)} unlabeled images?\n\n"
            "This may take a while."
        )
        if not result:
            return
        
        # Process each
        for i, img_path in enumerate(unlabeled):
            self.file_label.config(text=f"Auto-labeling {i+1}/{len(unlabeled)}...")
            self.root.update()
            
            # Find index and load
            idx = self.image_list.index(img_path)
            self.current_index = idx
            self._load_image()
            
            # Auto-label
            self.auto_label()
            
            # Save
            self.save_labels()
        
        messagebox.showinfo("Done", f"Auto-labeled {len(unlabeled)} images!")
        self._update_stats()


def main():
    root = tk.Tk()
    app = DocLayoutTagger(root)
    root.mainloop()


if __name__ == "__main__":
    main()