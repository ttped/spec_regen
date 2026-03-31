import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def copy_file(src_file, dst_file):
    """Copies a single file, preserving metadata."""
    shutil.copy2(src_file, dst_file)

def fast_network_copy(source_dir, dest_dir, max_workers=8):
    """Copies a directory to another location using multithreading."""
    files_to_copy = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, source_dir)
            dst_path = os.path.join(dest_dir, rel_path)
            files_to_copy.append((src_path, dst_path))

    for _, dst_path in files_to_copy:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for src, dst in files_to_copy:
            executor.submit(copy_file, src, dst)

# Execution
source = r"C:\Your\Local\Folder1"
destination = r"Z:\Your\Network\Folder2"

fast_network_copy(source, destination, max_workers=8)
