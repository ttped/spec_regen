"""
publish_results.py - Copy completed docx + source PDF pairs to an output drive.

For each generated .docx in results_dir, finds the matching source PDF in
docs/ci_repo and copies both into a paired folder on the destination drive:

    {dest}/{stem}/
        {stem}.docx
        {stem}.pdf

Matching is done by stem name, with space/underscore/hyphen normalization as
fallback for minor naming differences between the OCR pipeline and the source PDFs.

Usage (from project root):
    python misc/publish_results.py P:\\Output\\Specs
    python misc/publish_results.py P:\\Output\\Specs --dry-run
    python misc/publish_results.py P:\\Output\\Specs --workers 32
"""

import os
import re
import sys
import argparse
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
            os.environ.setdefault(key.strip(), value.strip())

_load_env(Path(__file__).resolve().parent.parent / ".env")


def _norm(name: str) -> str:
    """Collapse separators and lowercase for fuzzy matching."""
    return re.sub(r'[\s\-_]+', '_', name.lower()).strip('_')


def _build_pdf_index(pdf_dir: Path) -> dict:
    """
    Scan pdf_dir for .pdf files.
    Returns two maps:
        exact:  stem → Path   (original casing)
        fuzzy:  normalized_stem → Path
    """
    exact = {}
    fuzzy = {}
    for entry in pdf_dir.iterdir():
        if entry.suffix.lower() == '.pdf' and entry.is_file():
            exact[entry.stem] = entry
            fuzzy[_norm(entry.stem)] = entry
    return exact, fuzzy


def _find_pairs(results_dir: Path, pdf_dir: Path):
    """
    Match every .docx in results_dir to a .pdf in pdf_dir.
    Returns:
        paired:   list of (stem, docx_path, pdf_path)
        missing:  list of stem (docx exists, no PDF found)
    """
    exact, fuzzy = _build_pdf_index(pdf_dir)

    paired = []
    missing = []

    for entry in sorted(results_dir.iterdir()):
        if entry.suffix != '.docx' or not entry.is_file():
            continue
        stem = entry.stem

        # 1. Exact stem match
        pdf = exact.get(stem)

        # 2. Normalized fallback
        if pdf is None:
            pdf = fuzzy.get(_norm(stem))

        if pdf:
            paired.append((stem, entry, pdf))
        else:
            missing.append(stem)

    return paired, missing


def _copy_pair(stem: str, docx: Path, pdf: Path, dest_root: Path, dry_run: bool):
    """Copy docx + pdf into dest_root/{stem}/."""
    folder = dest_root / stem
    if not dry_run:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(docx, folder / docx.name)
        shutil.copy2(pdf,  folder / pdf.name)
    return stem


def publish(results_dir: str, pdf_dir: str, dest: str, dry_run: bool, workers: int) -> None:
    results_path = Path(results_dir)
    pdf_path     = Path(pdf_dir)
    dest_path    = Path(dest)

    for label, path in [("results_dir", results_path), ("pdf_dir", pdf_path)]:
        if not path.is_dir():
            print(f"[Error] {label} not found: {path}")
            sys.exit(1)

    paired, missing = _find_pairs(results_path, pdf_path)

    if not paired:
        print("No matched docx/pdf pairs found.")
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Destination: {dest_path}")
    print(f"Pairs to copy : {len(paired)}")
    if missing:
        print(f"No PDF found  : {len(missing)}")
        for stem in missing:
            print(f"  [MISSING PDF] {stem}")
    print()

    if dry_run:
        for stem, docx, pdf in paired:
            print(f"  {stem}/")
            print(f"    {docx.name}")
            print(f"    {pdf.name}")
        print(f"\nWould copy {len(paired)} pairs. Run without --dry-run to apply.")
        return

    dest_path.mkdir(parents=True, exist_ok=True)

    completed = 0
    errors = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_copy_pair, stem, docx, pdf, dest_path, dry_run): stem
            for stem, docx, pdf in paired
        }
        for future in as_completed(futures):
            stem = futures[future]
            try:
                future.result()
                completed += 1
                print(f"  [{completed}/{len(paired)}] {stem}")
            except Exception as e:
                errors.append((stem, e))
                print(f"  [ERROR] {stem}: {e}")

    print(f"\nCopied {completed}/{len(paired)} pairs to {dest_path}")
    if errors:
        print(f"{len(errors)} errors — check output above.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Publish docx + PDF pairs to output drive")
    parser.add_argument("dest", help="Destination root directory (e.g. P:\\Output\\Specs)")
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("RESULTS_DIR", str(project_root / "results_simple")),
        help="Directory containing .docx files (default: $RESULTS_DIR)",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(project_root / "docs" / "ci_repo"),
        help="Directory containing source PDFs (default: docs/ci_repo)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be copied without copying",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel copy threads (default: 16, increase for fast network drives)",
    )
    args = parser.parse_args()

    publish(args.results_dir, args.pdf_dir, args.dest, args.dry_run, args.workers)
