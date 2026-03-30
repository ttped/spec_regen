import os
import shutil
from pathlib import Path

# Try to import dotenv, fail gracefully if not installed
try:
    from dotenv import load_dotenv
except ImportError:
    print("The 'python-dotenv' package is required.")
    print("Please install it using: pip install python-dotenv")
    exit(1)

def flatten_adv_jsons(source_folder: Path, target_folder: Path):
    """
    Finds nested adv_*.json files in source_folder, removes the 'adv_' prefix, 
    and copies them to the flat target_folder.
    """
    # Create the target directory if it doesn't exist yet
    target_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Searching for 'adv_*.json' in {source_folder.resolve()}...")
    
    copied_count = 0
    
    # Recursively find all jsons starting with adv_
    for json_path in source_folder.rglob('adv_*.json'):
        # Strip the 'adv_' prefix for the destination filename (only the first occurrence)
        new_name = json_path.name.replace('adv_', '', 1)
        destination_path = target_folder / new_name
        
        # Safeguard: Prevent overwriting files with the same name from different folders
        counter = 1
        original_stem = destination_path.stem
        while destination_path.exists():
            destination_path = target_folder / f"{original_stem}_{counter}{destination_path.suffix}"
            counter += 1
            
        # Copy the file to the flat destination directory
        shutil.copy2(json_path, destination_path)
        print(f"Copied: {json_path.name} -> {destination_path.name}")
        copied_count += 1
        
    print(f"\nOperation complete. Copied {copied_count} files to {target_folder.resolve()}")

if __name__ == "__main__":
    # 1. Load .env from the directory above the current one
    env_path = Path('..') / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded environment from: {env_path.resolve()}")
    else:
        print(f"Warning: No .env file found at {env_path.resolve()}")

    # 2. Define the input directory (defaults to current directory)
    # You can change this to accept sys.argv[1] if you want to pass it via command line
    input_directory = Path('.')

    # 3. Read the output directory from the environment variable
    target_dir_str = os.getenv('RAW_OCR_DIR')
    
    if not target_dir_str:
        print("Error: 'RAW_OCR_DIR' environment variable is not set.")
        exit(1)
        
    output_directory = Path(target_dir_str)
    
    # 4. Pass both the input and output paths to the processing function
    flatten_adv_jsons(input_directory, output_directory)


