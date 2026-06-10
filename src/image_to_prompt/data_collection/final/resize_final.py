"""
VM.AI — Resize final/ images to 380x380.

Center-crops every image in final/ (train/val/test) to a square,
then resizes to 380x380 with Lanczos. Overwrites in place.
"""

from pathlib import Path

from PIL import Image

FINAL = Path("data/image_to_prompt/final")
TARGET_SIZE = 380
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _is_image(f: Path) -> bool:
    return f.suffix.lower() in IMAGE_EXTENSIONS


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def main():
    if not FINAL.is_dir():
        print("final/ not found")
        return

    image_paths = sorted(
        [f for f in FINAL.rglob("*") if f.is_file() and _is_image(f)]
    )
    if not image_paths:
        print("No images found in final/")
        return

    total = len(image_paths)
    print(f"Resizing {total} images to {TARGET_SIZE}x{TARGET_SIZE} ...")

    for i, path in enumerate(image_paths, 1):
        try:
            img = Image.open(path)
            img = img.convert("RGBA")

            square = center_crop_square(img)
            resized = square.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

            ext = path.suffix.lower()
            if ext in (".jpg", ".jpeg", ".bmp"):
                resized = resized.convert("RGB")

            resized.save(path)
        except Exception as e:
            print(f"  [{i}/{total}] FAILED {path.relative_to(FINAL)}: {e}")
            continue

        if i % 500 == 0:
            print(f"  [{i}/{total}] ...")

    print(f"Done — {total} images resized.")


if __name__ == "__main__":
    main()
