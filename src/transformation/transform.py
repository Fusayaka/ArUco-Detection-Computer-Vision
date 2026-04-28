"""
transform.py — Augmentation pipeline combining rotation and blur.
"""

import json
import random
from pathlib import Path

import cv2

from src.transformation.blur import apply_blur
from src.transformation.rotate import rotate_image, rotate_json_labels

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT_DIR = _REPO_ROOT / "data" / "raw" / "flyingarucov2"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "processed" / "transform"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def transform_sample(
    image_path: Path | str,
    json_path: Path | str,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    angle: float | None = None,
) -> tuple[Path, Path]:
    """
    Applies rotation then Gaussian blur to a single image + label pair.

    Processing order: rotate → blur. Blur is intentionally applied after
    rotation so that label corner coordinates remain accurate (blurring
    does not shift pixel positions).

    Args:
        image_path: Path to the source image (.jpg / .jpeg / .png).
        json_path:  Path to the corresponding JSON label file.
        output_dir: Destination directory (created if it does not exist).
                    Defaults to ``data/processed/transform/``.
        angle:      Rotation angle in degrees. A random integer in [1, 359]
                    is used when *angle* is None.

    Returns:
        (out_image_path, out_json_path) — paths to the saved files.
    """
    image_path = Path(image_path)
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if angle is None:
        angle = random.randint(1, 359)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    with open(json_path, "r") as f:
        json_data = json.load(f)

    rotated_image, rotation_matrix = rotate_image(image, angle)
    transformed_image = apply_blur(rotated_image)
    transformed_json = rotate_json_labels(json_data, rotation_matrix)

    stem = image_path.stem
    out_image_path = output_dir / f"{stem}.jpg"
    out_json_path = output_dir / f"{stem}.json"

    cv2.imwrite(str(out_image_path), transformed_image)
    with open(out_json_path, "w") as f:
        json.dump(transformed_json, f, indent=4)

    return out_image_path, out_json_path


def transform_dir(
    input_dir: Path | str = _DEFAULT_INPUT_DIR,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    angle: float | None = None,
    debug: bool = False
) -> list[tuple[Path, Path]]:
    """
    Applies :func:`transform_sample` to every image in *input_dir*.

    Args:
        input_dir:  Directory containing source images and JSON labels.
                    Defaults to ``data/raw/flyingarucov2/``.
        output_dir: Destination directory for transformed outputs.
                    Defaults to ``data/processed/transform/``.
        angle:      Fixed rotation angle in degrees for all images.
                    A random angle per image is used when None.

    Returns:
        List of (out_image_path, out_json_path) for every processed pair.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS]

    results = []
    for image_path in image_files:
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            print(f"[!] No JSON found for {image_path.name}, skipping.")
            continue

        out_img, out_json = transform_sample(image_path, json_path, output_dir, angle=angle)
        if debug:
            print(f"[*] Saved: {out_img.name}, {out_json.name}")
        results.append((out_img, out_json))

    return results


def _main() -> None:
    transform_dir()


if __name__ == "__main__":
    _main()