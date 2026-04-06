l"""
publish_results.py - Copy completed docx + source PDF pairs to an output drive.

For each generated .docx in results_dir, finds the matching source PDF in
pdf_dir and copies both into a paired folder on the destination drive:

    {OUTPUT_DRIVE}/{stem}/
        {stem}.docx
        {stem}.pdf

Configure via .env at project root. Edit the variables below to override.
"""

import os
import re
import sys
import shutil
import difflib
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
            os.environ.setdefault(key.strip(), value.strip())

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_load_env(_PROJECT_ROOT / ".env")

# =============================================================================
# CONFIG — edit here or set in .env
# =============================================================================

RESULTS_DIR  = str(_PROJECT_ROOT / "results_simple")
PDF_DIR      = str(_PROJECT_ROOT / "docs" / "ci_repo")
OUTPUT_DRIVE = r"Z:\\Public\\Mazer_Joel\\Spec_Regen_Files\\output"

DRY_RUN = True   # Set True to preview without copying
WORKERS = 8      # Parallel copy threads — increase for fast network drives

# =============================================================================


def _norm(name: str) -> str:
    """Collapse separators and lowercase for fuzzy matching."""
    return re.sub(r'[\s\-_]+', '_', name.lower()).strip('_')


def _build_pdf_index(pdf_dir: Path) -> tuple:
    exact = {}
    fuzzy = {}
    for entry in pdf_dir.iterdir():
        if entry.suffix.lower() == '.pdf' and entry.is_file():
            exact[entry.stem] = entry
            fuzzy[_norm(entry.stem)] = entry
    return exact, fuzzy


def _find_pairs(results_dir: Path, pdf_dir: Path) -> tuple:
    exact, fuzzy = _build_pdf_index(pdf_dir)
    paired = []
    missing = []
    
    fuzzy_keys = list(fuzzy.keys())

    for entry in sorted(results_dir.iterdir()):
        if entry.suffix != '.docx' or not entry.is_file():
            continue
        stem = entry.stem
        
        pdf = exact.get(stem)
        if not pdf:
            norm_stem = _norm(stem)
            pdf = fuzzy.get(norm_stem)
            
            if not pdf:
                # Use string similarity (80%+ match) to catch _1 or _2 files
                matches = difflib.get_close_matches(norm_stem, fuzzy_keys, n=1, cutoff=0.8)
                if matches:
                    pdf = fuzzy[matches[0]]

        if pdf:
            paired.append((stem, entry, pdf))
        else:
            missing.append(stem)

    return paired, missing


def _copy_pair(stem: str, docx: Path, pdf: Path, dest_root: Path):
    folder = dest_root / stem
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(docx, folder / docx.name)
    shutil.copy2(pdf,  folder / pdf.name)


if __name__ == "__main__":
    results_path = Path(RESULTS_DIR)
    pdf_path     = Path(PDF_DIR)
    dest_path    = Path(OUTPUT_DRIVE)

    for label, path in [("RESULTS_DIR", results_path), ("PDF_DIR", pdf_path)]:
        if not path.is_dir():
            print(f"[Error] {label} not found: {path}")
            sys.exit(1)

    paired, missing = _find_pairs(results_path, pdf_path)

    print(f"{'[DRY RUN] ' if DRY_RUN else ''}Destination : {dest_path}")
    print(f"Pairs found : {len(paired)}")
    if missing:
        print(f"Missing PDF : {len(missing)}")
        for stem in missing:
            print(f"  [NO PDF] {stem}")
    print()

    if DRY_RUN:
        for stem, docx, pdf in paired:
            print(f"  {stem}/")
            print(f"    {docx.name}")
            print(f"    {pdf.name}")
        print(f"\nWould copy {len(paired)} pairs. Set DRY_RUN = False to apply.")
        
    if not DRY_RUN:
        dest_path.mkdir(parents=True, exist_ok=True)
        completed = 0

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(_copy_pair, stem, docx, pdf, dest_path): stem
                for stem, docx, pdf in paired
            }
            for future in as_completed(futures):
                stem = futures[future]
                future.result()
                completed += 1
                print(f"  [{completed}/{len(paired)}] {stem}")

        print(f"\nCopied {completed}/{len(paired)} pairs to {dest_path}")


