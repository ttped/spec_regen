import os
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm

def get_paths():
    """
    Calculates paths relative to this script file.
    """
    script_dir = Path(__file__).resolve().parent
    source_dir = script_dir.parent / "docs"
    output_dir = script_dir.parent / "docs_images"
    return source_dir, output_dir

# Configuration
SOURCE_DIR, OUTPUT_DIR = get_paths()
DPI = 200     
FORMAT = "jpeg" # "jpeg" is required for the library to work

def sanitize_filename(name):
    """Replaces unsafe characters with underscores."""
    return "".join([c if c.isalnum() or c in "._-" else "_" for c in name])

def process_pdfs():
    print(f"Script location: {Path(__file__).resolve()}")
    print(f"Looking for PDFs in: {SOURCE_DIR}")
    
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)
    
    # Find all PDF files
    pdf_files = list(SOURCE_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {SOURCE_DIR}")
        return
        
    print(f"Found {len(pdf_files)} PDF files to process.")

    for pdf_path in tqdm(pdf_files, desc="Converting PDFs"):
        try:
            # Just get the filename without extension (e.g. 'report' from 'report.pdf')
            # and sanitize it just in case it has spaces or weird chars
            stem = sanitize_filename(pdf_path.stem)

            # Convert to images
            images = convert_from_path(str(pdf_path), dpi=DPI, fmt=FORMAT)

            for i, image in enumerate(images):
                # Format: filename_page001.jpg
                page_num = f"{i + 1:03d}" 
                out_filename = f"{stem}_page{page_num}.jpg" 
                out_path = OUTPUT_DIR / out_filename

                # Save explicitly as JPEG to fix the previous error
                image.save(out_path, "JPEG")

        except Exception as e:
            print(f"\n[Error] Failed to process {pdf_path.name}: {e}")

    print(f"\nDone! Images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_pdfs()