import os
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm

def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip())

_load_env(Path(__file__).resolve().parent.parent / ".env")


def get_paths():
    """
    Calculates paths relative to this script file.
    Source PDFs and output images both live in IMAGES_DIR from .env.
    """
    project_root = Path(__file__).resolve().parent.parent
    images_dir_env = os.environ.get("IMAGES_DIR")
    if images_dir_env is None:
        raise RuntimeError("Missing required environment variable: IMAGES_DIR  (check your .env file)")
    images_dir = project_root / images_dir_env
    return images_dir, images_dir

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

    pdf_files = list(SOURCE_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {SOURCE_DIR}")
        return

    # Filter out PDFs that already have at least one output image.
    # Conversion is the expensive step, so we decide before calling it.
    pending = []
    skipped = 0
    for pdf_path in pdf_files:
        stem = sanitize_filename(pdf_path.stem)
        if any(OUTPUT_DIR.glob(f"{stem}_page*.jpg")):
            skipped += 1
            continue
        pending.append((pdf_path, stem))

    print(f"Found {len(pdf_files)} PDFs — {skipped} already converted, {len(pending)} to process.")

    for pdf_path, stem in tqdm(pending, desc="Converting PDFs"):
        images = convert_from_path(str(pdf_path), dpi=DPI, fmt=FORMAT)
        for i, image in enumerate(images):
            page_num = f"{i + 1:03d}"
            out_path = OUTPUT_DIR / f"{stem}_page{page_num}.jpg"
            image.save(out_path, "JPEG")

    print(f"\nDone! Images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    process_pdfs()

