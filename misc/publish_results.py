"""
validate_results.py - Validate 1:1:1 mapping of JSONs, PDFs, and generated DOCXs.

Checks for missing files across the three main pipeline directories and
generates a summary and a detailed text report.
"""

import os
import re
import sys
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_load_env(_PROJECT_ROOT / ".env")

RESULTS_DIR = _PROJECT_ROOT / os.environ.get("RESULTS_DIR", "results_simple")
PDF_DIR = _PROJECT_ROOT / os.environ.get("IMAGES_DIR", "docs/ci_repo")
RAW_OCR_DIR = _PROJECT_ROOT / os.environ.get("RAW_OCR_DIR", "iris_ocr/advanced/jsons")

OUTPUT_DRIVE = Path(os.environ.get("OUTPUT_DRIVE", r"Z:\Public\Mazer_Joel\Spec_Regen_Files\output"))
DRY_RUN = False   # Set True to preview without copying
WORKERS = 4      # Parallel copy threads

def _norm(name: str) -> str:
    """Collapse separators and lowercase for fuzzy matching."""
    base = re.sub(r'[\s\-_]+', '_', name.lower()).strip('_')
    return re.sub(r'_[12]$', '', base)

def validate_mappings():
    print("Scanning directories...")
    
    json_stems = { _norm(p.stem): p.name for p in RAW_OCR_DIR.glob("*.json") if p.is_file() }
    pdf_stems = { _norm(p.stem): p.name for p in PDF_DIR.glob("*.pdf") if p.is_file() }
    docx_stems = { _norm(p.stem): p.name for p in RESULTS_DIR.glob("*.docx") if p.is_file() }

    all_keys = set(json_stems.keys()) | set(pdf_stems.keys()) | set(docx_stems.keys())

    perfect_matches = []
    missing_docx = []
    missing_pdf = []
    missing_json = []

    for k in sorted(all_keys):
        has_json = k in json_stems
        has_pdf = k in pdf_stems
        has_docx = k in docx_stems

        if has_json and has_pdf and has_docx:
            perfect_matches.append(k)
        else:
            display_name = json_stems.get(k) or pdf_stems.get(k) or docx_stems.get(k)
            if not has_docx: missing_docx.append(display_name)
            if not has_pdf: missing_pdf.append(display_name)
            if not has_json: missing_json.append(display_name)

    # Console Summary
    print("\n--- Validation Summary ---")
    print(f"Total Unique Documents : {len(all_keys)}")
    print(f"Perfect 3-way Matches  : {len(perfect_matches)}")
    print(f"Missing DOCX Files     : {len(missing_docx)}")
    print(f"Missing PDF Files      : {len(missing_pdf)}")
    print(f"Missing JSON Files     : {len(missing_json)}")

    # Detailed Text Report
    report_path = _PROJECT_ROOT / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write("PIPELINE VALIDATION REPORT\n")
        f.write("==========================\n\n")
        
        f.write(f"Missing DOCX Files ({len(missing_docx)}):\n")
        for m in missing_docx: f.write(f"  - {m}\n")
        
        f.write(f"\nMissing PDF Files ({len(missing_pdf)}):\n")
        for m in missing_pdf: f.write(f"  - {m}\n")
            
        f.write(f"\nMissing JSON Files ({len(missing_json)}):\n")
        for m in missing_json: f.write(f"  - {m}\n")

    print(f"\nDetailed discrepancy list saved to: {report_path.relative_to(_PROJECT_ROOT)}")
    
    return perfect_matches, pdf_stems, docx_stems

def _copy_pair(stem: str, docx: Path, pdf: Path, dest_root: Path):
    folder = dest_root / stem
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(docx, folder / docx.name)
    shutil.copy2(pdf,  folder / pdf.name)

if __name__ == "__main__":
    perfect_matches, pdf_stems, docx_stems = validate_mappings()
    
    print("\n--- Publishing ---")
    if not OUTPUT_DRIVE.exists() and not DRY_RUN:
        OUTPUT_DRIVE.mkdir(parents=True, exist_ok=True)
        
    print(f"{'[DRY RUN] ' if DRY_RUN else ''}Destination : {OUTPUT_DRIVE}")
    print(f"Valid pairs ready to publish : {len(perfect_matches)}")
    
    if DRY_RUN:
        print(f"\nWould copy {len(perfect_matches)} pairs. Set DRY_RUN = False in script to apply.")
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {}
            for stem in perfect_matches:
                pdf_path = PDF_DIR / pdf_stems[stem]
                docx_path = RESULTS_DIR / docx_stems[stem]
                futures[pool.submit(_copy_pair, stem, docx_path, pdf_path, OUTPUT_DRIVE)] = stem
                
            for future in as_completed(futures):
                stem = futures[future]
                future.result()
                completed += 1
                print(f"  [{completed}/{len(perfect_matches)}] {stem}")
        
        print(f"\nCopied {completed}/{len(perfect_matches)} pairs to {OUTPUT_DRIVE}")