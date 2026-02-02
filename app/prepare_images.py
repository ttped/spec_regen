import os
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm

def get_paths():
    """
    Calculates paths relative to this script file.
    Script location: .../project/src/prepare_images.py
    Docs location:   .../project/docs
    Images location: .../project/docs_images
    """
    # 1. Get the folder where THIS script lives
    script_dir = Path(__file__).resolve().parent
    
    # 2. Go one level up (..), then into 'docs'
    source_dir = script_dir.parent / "docs"
    
    # 3. Go one level up (..), then into 'docs_images'
    output_dir = script_dir.parent / "docs_images"
    
    return source_dir, output_dir

# Configuration
SOURCE_DIR, OUTPUT_DIR = get_paths()
DPI = 200     # 200 is good for LabelMe speed; 300 for higher precision
FORMAT = "jpg"

def sanitize_filename(name):
    """Replaces unsafe characters with underscores."""
    return "".join([c if c.isalnum() or c in "._-" else "_" for c in name])

def process_pdfs():
    print(f"Script location: {Path(__file__).resolve()}")
    print(f"Looking for PDFs in: {SOURCE_DIR}")
    
    # 1. Setup Output Directory
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    else:
        print(f"Output directory exists: {OUTPUT_DIR}")

    if not SOURCE_DIR.exists():
        print(f"[Error] Source directory not found: {SOURCE_DIR}")
        return

    # 2. Find all PDF files recursively
    pdf_files = list(SOURCE_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {SOURCE_DIR}")
        return
        
    print(f"Found {len(pdf_files)} PDF files to process.")

    # 3. Process each PDF
    for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
        try:
            # Create a unique prefix based on the folder structure
            # e.g., docs/finance/report.pdf -> finance_report
            relative_path = pdf_path.relative_to(SOURCE_DIR)
            
            # Flatten path: 'folder/subfolder/file.pdf' -> 'folder_subfolder_file'
            flat_name = str(relative_path.with_suffix('')).replace(os.sep, "_")
            flat_name = sanitize_filename(flat_name)

            # Convert to images
            images = convert_from_path(str(pdf_path), dpi=DPI, fmt=FORMAT)

            for i, image in enumerate(images):
                # Format: Folder_Filename_Page01.jpg
                page_num = f"{i + 1:03d}" 
                out_filename = f"{flat_name}_page{page_num}.{FORMAT}"
                out_path = OUTPUT_DIR / out_filename

                # Save the image
                image.save(out_path, FORMAT.upper())

        except Exception as e:
            print(f"\n[Error] Failed to process {pdf_path.name}: {e}")

    print(f"\nDone! Images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_pdfs()