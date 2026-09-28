"""Build a compact 3-by-4 contact sheet from saved YOLO prediction images."""

from pathlib import Path
import sys

from PIL import Image


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_prediction_montage.py INPUT_DIR OUTPUT_FILE")

    source_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    images = sorted(source_dir.glob("*.jpg"))
    if len(images) != 12:
        raise ValueError(f"expected 12 prediction images, found {len(images)}")

    columns, tile_width, tile_height, gap = 3, 480, 270, 8
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_width + (columns + 1) * gap, rows * tile_height + (rows + 1) * gap),
        "white",
    )
    for index, image_path in enumerate(images):
        with Image.open(image_path) as image:
            tile = image.convert("RGB").resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = gap + (index % columns) * (tile_width + gap)
        y = gap + (index // columns) * (tile_height + gap)
        canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92, optimize=True)
    print(f"saved {output_path} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
