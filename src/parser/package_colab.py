"""
VM.AI - Colab Package Generator
Creates a zip with src/parser, data, models/finetuned_parser, and config.yaml
for easy Colab training setup.
Run: python src/parser/package_colab.py
"""

import os
import zipfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INCLUDE_DIRS = [
    "src/parser",
    "data",
    "models/finetuned_parser",
]

INCLUDE_FILES = [
    "config.yaml",
]

EXCLUDE_EXTENSIONS = {".pyc", ".pth", ".pt", ".bin"}
EXCLUDE_PATTERNS = {"__pycache__", ".ipynb_checkpoints", ".git"}


def should_skip(path):
    name = os.path.basename(path)
    if name in EXCLUDE_PATTERNS:
        return True
    _, ext = os.path.splitext(name)
    if ext in EXCLUDE_EXTENSIONS:
        return True
    return False


def main():
    zip_name = "colab.zip"
    zip_path = os.path.join(ROOT, zip_name)

    total_files = 0
    total_size = 0

    print(f"Packing VM.AI for Colab...")
    print(f"Output: {zip_path}\n")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dir_path in INCLUDE_DIRS:
            full_path = os.path.join(ROOT, dir_path)
            if not os.path.exists(full_path):
                print(f"  SKIP: {dir_path} (not found)")
                continue

            for root_dir, dirs, files in os.walk(full_path):
                dirs[:] = [
                    d for d in dirs if not should_skip(os.path.join(root_dir, d))
                ]
                for f in files:
                    if should_skip(f):
                        continue
                    file_path = os.path.join(root_dir, f)
                    arc_name = os.path.relpath(file_path, ROOT)
                    zf.write(file_path, arc_name)
                    total_files += 1
                    total_size += os.path.getsize(file_path)
                    print(f"  + {arc_name}")

        for file_path in INCLUDE_FILES:
            full_path = os.path.join(ROOT, file_path)
            if not os.path.exists(full_path):
                print(f"  SKIP: {file_path} (not found)")
                continue
            zf.write(full_path, file_path)
            total_files += 1
            total_size += os.path.getsize(full_path)
            print(f"  + {file_path}")

    size_mb = total_size / (1024 * 1024)
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)

    print(f"\nDone: {total_files} files")
    print(f"  Uncompressed: {size_mb:.1f} MB")
    print(f"  Zip size: {zip_size_mb:.1f} MB")
    print(f"  Saved to: {zip_path}")


if __name__ == "__main__":
    main()
