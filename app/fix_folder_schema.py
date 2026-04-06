import os
import shutil
from pathlib import Path

# Try to import dotenv, fail gracefully if not installed
try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    print("The 'python-dotenv' package is required.")
    print("Please install it using: pip install python-dotenv")
    exit(1)

def flatten_adv_jsons(source_folder: Path, target_folder: Path):
    """
    Finds nested adv_*.json files in source_folder, removes the 'adv_' prefix, 
    and copies them to the flat target_folder.
    """
    # Wipe and recreate the target directory to prevent duplicates on re-runs
    if target_folder.exists():
        shutil.rmtree(target_folder)
        print(f"Cleared existing target: {target_folder.resolve()}")
    target_folder.mkdir(parents=True, exist_ok=True)

    print(f"Searching for 'adv_*.json' in {source_folder.resolve()}...")
    
    copied_count = 0
    
    # Recursively find all jsons starting with adv_
    for json_path in source_folder.rglob('adv_*.json'):
        # Strip the 'adv_' prefix for the destination filename
        new_name = json_path.name.replace('adv_', '', 1)
        destination_path = target_folder / new_name

        # Copy the file to the flat destination directory
        shutil.copy2(json_path, destination_path)
        print(f"Copied: {json_path.name} -> {destination_path.name}")
        copied_count += 1
        
    print(f"\nOperation complete. Copied {copied_count} files to {target_folder.resolve()}")

if __name__ == "__main__":
    # 1. Reliably find the .env file in the folder above where this script lives
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir.parent / '.env'
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded environment from: {env_path}")
    else:
        print(f"Warning: No .env file found exactly at {env_path}")
        # Fallback: search up the directory tree for any .env file
        fallback_env = find_dotenv()
        if fallback_env:
            load_dotenv(fallback_env)
            print(f"Loaded environment using fallback search: {fallback_env}")
        else:
            print("No .env file could be found anywhere in the parent directories.")

    # 2. Define the input directory (using the script's directory as the starting point)
    # You can change this to input_directory = Path('.') if you prefer the working directory
    input_directory = script_dir

    # 3. Read the output directory from the environment variable
    target_dir_str = os.getenv('RAW_OCR_DIR')
    
    if not target_dir_str:
        print("\nError: 'RAW_OCR_DIR' environment variable is not set.")
        print("Please check your .env file or system environment variables.")
        exit(1)
        
    output_directory = Path(target_dir_str)
    
    # 4. Pass both the input and output paths to the processing function
    flatten_adv_jsons(input_directory, output_directory)


