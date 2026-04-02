"""
reset_pipeline.py - Delete intermediate pipeline files, keeping only classification and title.

Removes everything produced by steps 3-8 so the pipeline can be rerun from
structure onwards. Preserves:
    {stem}_classification.json   (step 1)
    {stem}_title.json            (step 2)

Deletes:
    {stem}_organized.json        (step 3)
    {stem}_ml_filtered.json      (step 4)
    {stem}_with_assets.json      (step 5)
    {stem}_with_tables.json      (step 6)
    {stem}.docx                  (step 7)

Usage (from project root):
    python app/reset_pipeline.py
    python app/reset_pipeline.py --results-dir path/to/results
    python app/reset_pipeline.py --dry-run
"""

import os
import sys
import argparse
from pathlib import Path


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


DELETABLE_SUFFIXES = [
    "_organized.json",
    "_ml_filtered.json",
    "_with_assets.json",
    "_with_tables.json",
    ".docx",
]


def reset_results_dir(results_dir: str, dry_run: bool = False) -> None:
    results_path = Path(results_dir)
    if not results_path.is_dir():
        print(f"[Error] Results directory not found: {results_dir}")
        sys.exit(1)

    targets = []
    for entry in results_path.iterdir():
        if not entry.is_file():
            continue
        for suffix in DELETABLE_SUFFIXES:
            if entry.name.endswith(suffix):
                targets.append(entry)
                break

    if not targets:
        print("Nothing to delete.")
        return

    targets.sort()
    print(f"{'[DRY RUN] ' if dry_run else ''}Deleting {len(targets)} files from {results_dir}:\n")

    for path in targets:
        print(f"  {'(would delete)' if dry_run else 'Deleting'} {path.name}")
        if not dry_run:
            path.unlink()

    print(f"\n{'Would have deleted' if dry_run else 'Deleted'} {len(targets)} files.")
    if dry_run:
        print("Run without --dry-run to apply.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset pipeline to after title extraction")
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("RESULTS_DIR", "results_simple"),
        help="Directory containing pipeline output files (default: $RESULTS_DIR)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without actually deleting",
    )
    args = parser.parse_args()

    reset_results_dir(args.results_dir, dry_run=args.dry_run)
