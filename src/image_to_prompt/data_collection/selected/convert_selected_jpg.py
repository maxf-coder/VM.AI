"""
VM.AI — Convert every image in selected/ to JPEG.

For each image in selected/{cat}/**/*:
  - Opens, converts to RGB, saves as .jpg (quality=95)
  - Deletes original if extension was not .jpg
  - Skips files that already end in .jpg
"""

from pathlib import Path

from PIL import Image

SELECTED = Path("data/image_to_prompt/selected")
QUALITY = 95
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _is_image(f: Path) -> bool:
    return f.suffix.lower() in IMAGE_EXTENSIONS


def main():
    if not SELECTED.is_dir():
        print("selected/ not found")
        return

    image_paths = sorted(
        [f for f in SELECTED.rglob("*") if f.is_file() and _is_image(f)]
    )
    print(f"Found {len(image_paths)} images in selected/")

    converted = 0
    skipped = 0

    for path in image_paths:
        if path.suffix.lower() in (".jpg", ".jpeg"):
            # Check if it's already a valid JPEG
            try:
                img = Image.open(path)
                img.verify()
                skipped += 1
                continue
            except Exception:
                pass  # Corrupted — re-encode

        try:
            img = Image.open(path).convert("RGB")
            new_path = path.with_suffix(".jpg")
            img.save(new_path, "JPEG", quality=QUALITY)

            if new_path != path:
                path.unlink()

            converted += 1
        except Exception as e:
            print(f"  FAILED {path.relative_to(SELECTED)}: {e}")

        if converted % 1000 == 0:
            print(f"  ... {converted} converted")

    print(f"\nDone — {converted} converted, {skipped} already JPEG")


if __name__ == "__main__":
    main()
