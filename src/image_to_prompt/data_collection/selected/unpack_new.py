"""
VM.AI — Unpack new/ folders into their parent category directories.

Moves all files from selected/{category}/new/ to selected/{category}/new_{filename}
then removes the new/ directory.
"""

import sys
from pathlib import Path

SELECTED = Path("data/image_to_prompt/selected")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}


def _is_image(file: Path) -> bool:
    return file.suffix.lower() in IMAGE_EXTENSIONS


def unpack_category(category: str):
    cat_dir = SELECTED / category
    new_dir = cat_dir / "new"

    if not new_dir.is_dir():
        print(f"  {category}: no new/ folder")
        return 0

    files = sorted([f for f in new_dir.iterdir() if f.is_file() and _is_image(f)])
    if not files:
        new_dir.rmdir()
        print(f"  {category}: new/ was empty, removed")
        return 0

    moved = 0
    for f in files:
        dest = cat_dir / f"new_{f.name}"
        counter = 1
        while dest.exists():
            stem = f.stem
            dest = cat_dir / f"new_{stem}_{counter}{f.suffix}"
            counter += 1
        f.rename(dest)
        moved += 1

    new_dir.rmdir()
    print(f"  {category}: {moved} files unpacked")
    return moved


def main():
    categories = sys.argv[1:] if len(sys.argv) > 1 else sorted(
        d.name for d in SELECTED.iterdir() if d.is_dir()
    )

    total = 0
    for cat in categories:
        total += unpack_category(cat)

    print(f"\nTotal files unpacked: {total}")
    print("Done.")


if __name__ == "__main__":
    main()
