import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the directory above
env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

def process_ocr_jsons(source_folder: str | Path):
    """Finds nested adv_*.json files, removes the prefix, and copies them to RAW_OCR_DIR."""
    target_dir_str = os.getenv('RAW_OCR_DIR')
    
    if not target_dir_str:
        return
        
    target_dir = Path(target_dir_str)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_folder)
    
    # Recursively find all jsons starting with adv_
    for json_path in source_path.rglob('adv_*.json'):
        # Strip the prefix for the destination filename
        new_name = json_path.name.replace('adv_', '', 1)
        destination_path = target_dir / new_name
        
        # Copy the file to the flat destination directory
        shutil.copy2(json_path, destination_path)

if __name__ == "__main__":
    process_ocr_jsons('.')
