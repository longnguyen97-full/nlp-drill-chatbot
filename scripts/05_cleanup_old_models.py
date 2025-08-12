#!/usr/bin/env python3
"""
Cleanup Script for LawBot v8.2
==============================

Deletes all generated artifacts, including models, indexes, logs, reports,
and processed data, to ensure a clean state for a fresh pipeline run.
"""

import sys
import shutil
from pathlib import Path
import argparse

# Ensure we can import from the project root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# No config import needed to avoid dependency issues. Paths are defined directly.
# This makes the script more robust against config changes.

# Define logger-like print functions for consistent output
def log_info(message):
    print(f"[INFO] {message}")

def log_warning(message):
    print(f"[WARN] {message}")

def log_success(message):
    print(f"[SUCCESS] {message}")

def log_error(message):
    print(f"[ERROR] {message}")

def get_paths_to_clean():
    """Returns a list of Path objects for directories and files to be cleaned."""
    return [
        project_root / "models",
        project_root / "indexes",
        project_root / "reports",
        project_root / "logs",
        project_root / "data" / "processed",
        project_root / "data" / "validation",
    ]

def cleanup_paths(paths_to_clean: list[Path], dry_run: bool = False):
    """
    Iterates through a list of paths and deletes them.
    Handles both files and directories.
    """
    log_info("Starting cleanup process...")
    deleted_count = 0
    skipped_count = 0

    for path in paths_to_clean:
        if not path.exists():
            log_info(f"Skipping non-existent path: {path.relative_to(project_root)}")
            skipped_count += 1
            continue

        try:
            if path.is_dir():
                log_info(f"Preparing to delete directory: {path.relative_to(project_root)}")
                if not dry_run:
                    shutil.rmtree(path)
                    log_success(f"Successfully deleted directory: {path.relative_to(project_root)}")
                else:
                    log_info(f"DRY RUN: Would delete directory: {path.relative_to(project_root)}")
                deleted_count += 1
            elif path.is_file():
                log_info(f"Preparing to delete file: {path.relative_to(project_root)}")
                if not dry_run:
                    path.unlink()
                    log_success(f"Successfully deleted file: {path.relative_to(project_root)}")
                else:
                    log_info(f"DRY RUN: Would delete file: {path.relative_to(project_root)}")
                deleted_count += 1
        except Exception as e:
            log_error(f"Failed to delete {path.relative_to(project_root)}: {e}")

    log_info("="*50)
    if dry_run:
        log_success(f"DRY RUN COMPLETE. Would have attempted to delete {deleted_count} items.")
    else:
        log_success(f"Cleanup complete. Deleted {deleted_count} items. Skipped {skipped_count} non-existent items.")

def main():
    """Main function to parse arguments and run the cleanup."""
    parser = argparse.ArgumentParser(
        description="Clean up all generated artifacts for the LawBot project.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Bypass the confirmation prompt and proceed with deletion immediately.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting anything.",
    )
    args = parser.parse_args()

    paths_to_clean = get_paths_to_clean()

    print("="*60)
    print("⚠️  LawBot Project Cleanup Utility ⚠️")
    print("="*60)
    print("The following directories and all their contents will be PERMANENTLY DELETED:")
    for path in paths_to_clean:
        # Check existence for a more accurate message
        status = " (exists)" if path.exists() else " (does not exist)"
        print(f"  - {path.relative_to(project_root)}{status}")
    print("="*60)

    if args.dry_run:
        cleanup_paths(paths_to_clean, dry_run=True)
        return

    if not args.yes:
        try:
            confirm = input("Are you sure you want to continue? (yes/no): ").lower().strip()
            if confirm != "yes":
                log_warning("Cleanup cancelled by user.")
                return
        except KeyboardInterrupt:
            print("\nCleanup cancelled by user.")
            return

    cleanup_paths(paths_to_clean)

if __name__ == "__main__":
    main()
