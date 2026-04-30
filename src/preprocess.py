import glob
import json
import os
import random
import shutil
from pathlib import Path
 
import cv2
import numpy as np
 
_REPO_ROOT = Path(__file__).resolve().parents[1]
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _remove_test_duplicates(raw_dir: Path, test_dir: Path) -> None:
    """
    Removes files from *raw_dir* whose stems also appear in *test_dir*.
 
    This prevents test images from leaking into the training set.
 
    Args:
        raw_dir:  Directory containing raw training data.
        test_dir: Directory containing test images.
    """
    if not os.path.exists(test_dir):
        print(f"[!] Test folder not found at '{test_dir}', skipping duplicate removal.")
        return
 
    test_stems = {
        f.stem
        for f in test_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
    }
 
    removed = 0
    for raw_file in raw_dir.iterdir():
        if raw_file.is_file() and raw_file.stem in test_stems:
            raw_file.unlink()
            removed += 1
 
    print(f"[*] Removed {removed // 2} duplicate file(s) from '{raw_dir}'.")


def convert_to_yolo_file(dirpath: str | Path) -> None:
    """
    Parses JSON annotations in *dirpath* and converts them to YOLO format .txt files.
 
    Skips conversion if a non-empty ``labels/`` subdirectory already exists.
 
    Args:
        dirpath: Directory containing .jpg images and matching .json annotations.
    """
    dirpath = Path(dirpath)
    label_dir = dirpath / "labels"
    
    if label_dir.exists() and any(label_dir.iterdir()):
        print(f"[*] Labels already exist at '{label_dir}', skipping conversion.")
        return
 
    label_dir.mkdir(parents=True, exist_ok=True)
    json_files = glob.glob(str(dirpath / "*.json"))

    for json_file in json_files:
        stem = Path(json_file).stem
        img_path = dirpath / f"{stem}.jpg"
 
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        height, width = img.shape[:2]
 
        with open(json_file, "r") as f:
            data = json.load(f)

        lines = []
        for marker in data['markers']:
            xs = [c[0] for c in marker["corners"]]
            ys = [c[1] for c in marker["corners"]]
 
            x_mid_norm = ((min(xs) + max(xs)) / 2) / width
            y_mid_norm = ((min(ys) + max(ys)) / 2) / height
            w_norm     = (max(xs) - min(xs)) / width
            h_norm     = (max(ys) - min(ys)) / height
 
            lines.append(f"0 {x_mid_norm:.6f} {y_mid_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(label_dir / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines))
 
    print("[*] Converted annotations to YOLO format successfully.")

def split_dataset(
    src_dir: str | Path,
    base_dir: str | Path = "data/processed/dataset",
    train_ratio: float = 0.8,
    apply_transform: bool = False,
    angle: float | None = None,
) -> None:
    """
    Splits the dataset into train/val sets and organises the directory structure.
 
    Steps:
        1. Remove any raw files whose stems match test images (deduplication).
        2. Optionally apply rotation + blur augmentation via :func:`transform_dir`.
        3. Copy images and labels into ``images/train``, ``images/val``,
           ``labels/train``, ``labels/val`` under *base_dir*.
 
    Args:
        src_dir:          Source directory with images and a ``labels/`` subfolder.
        base_dir:         Root output directory for the split dataset.
        train_ratio:      Fraction of data used for training (default 0.8).
        apply_transform:  When True, runs the rotate+blur augmentation pipeline
                          on *src_dir* before splitting.
        angle:            Fixed rotation angle passed to :func:`transform_dir`.
                          A random angle is used per image when None.
    """
    src_dir  = Path(src_dir)
    base_dir = Path(base_dir)
    
    # Step 1 — remove test duplicates from raw
    test_dir = _REPO_ROOT / "data" / "raw" / "test"
    _remove_test_duplicates(src_dir, Path(test_dir))

    # Step 2 — optional augmentation
    if apply_transform:
        from src.transformation.transform import transform_dir
        print("[*] Applying rotation + blur augmentation...")
        transform_dir(input_dir=src_dir, output_dir=src_dir, angle=angle)
        print("[*] Augmentation completed.")

    convert_to_yolo_file(src_dir)

    # Step 3 — create output folder structure
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (base_dir / sub).mkdir(parents=True, exist_ok=True)
        
    all_ids = [f.stem for f in src_dir.iterdir() if f.suffix.lower() == ".jpg"]
    random.seed(42)
    random.shuffle(all_ids)

    split_index = int(len(all_ids) * train_ratio)
    train_ids = all_ids[:split_index]
    val_ids = all_ids[split_index:]

    def _copy_split(ids: list[str], split_name: str) -> None:
        label_src_dir = src_dir / "labels"
        for stem in ids:
            img_src = src_dir / f"{stem}.jpg"
            img_dst = base_dir / "images" / split_name / f"{stem}.jpg"
            if img_src.exists():
                shutil.copy(img_src, img_dst)
 
            txt_src = label_src_dir / f"{stem}.txt"
            txt_dst = base_dir / "labels" / split_name / f"{stem}.txt"
            if txt_src.exists():
                shutil.copy(txt_src, txt_dst)
 
    _copy_split(train_ids, "train")
    _copy_split(val_ids, "val")
    print("[*] Dataset split successfully.")

def enhance_image(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
    correct_gradient: bool = False,
) -> np.ndarray:
    """Apply CLAHE-based contrast enhancement to a BGR image.

    Args:
        image: Input BGR image (uint8, H×W×3).
        clip_limit: CLAHE clip limit – higher values give stronger enhancement
            but risk amplifying noise.  2.0 is a safe default.
        tile_grid: Grid size for CLAHE tiles.  (8, 8) works well for typical
            marker sizes relative to full-image resolution.
        correct_gradient: If True, also apply coarse gradient correction before
            CLAHE.  Helps when a single bright light source creates a strong
            brightness ramp across the frame.

    Returns:
        Enhanced BGR image (uint8, same shape as input).
    """
    if image.ndim == 2:
        # Grayscale input – enhance directly
        return _clahe_gray(image, clip_limit, tile_grid)

    if correct_gradient:
        image = _correct_gradient(image)

    return _clahe_lab(image, clip_limit, tile_grid)


def enhance_gray(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a grayscale image (or convert BGR → gray first).

    Used by the decoding stage which operates on single-channel patches.

    Args:
        image: BGR (H×W×3) or grayscale (H×W) uint8 image.
        clip_limit: CLAHE clip limit.
        tile_grid: CLAHE tile grid size.

    Returns:
        Enhanced grayscale image (uint8, H×W).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return _clahe_gray(gray, clip_limit, tile_grid)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Normalise a rectified marker patch to [0, 1] float32.

    Used just before the ID-decoding step:  stretches the min–max range of the
    patch so that bit thresholding at 0.5 is independent of absolute brightness.

    Args:
        patch: Grayscale uint8 or float image of any size.

    Returns:
        Float32 image in [0.0, 1.0].
    """
    p = patch.astype(np.float32)
    lo, hi = p.min(), p.max()
    if hi - lo < 1e-6:
        # Flat patch (fully saturated or black) – return zeros to signal bad crop
        return np.zeros_like(p)
    return (p - lo) / (hi - lo)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clahe_lab(
    image: np.ndarray,
    clip_limit: float,
    tile_grid: tuple[int, int],
) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB, return BGR.

    Operating in LAB separates luminance from chrominance so colour hues are
    preserved while only the perceived brightness distribution is adjusted.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _clahe_gray(
    gray: np.ndarray,
    clip_limit: float,
    tile_grid: tuple[int, int],
) -> np.ndarray:
    """Apply CLAHE to a single-channel uint8 image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def _correct_gradient(image: np.ndarray) -> np.ndarray:
    """Divide out a coarse illumination gradient estimated from the image.

    Strategy: downsample to a tiny thumbnail, apply heavy blur, upsample back
    to original size.  This gives a smooth estimate of the background brightness
    ramp (i.e. the DC lighting component).  Dividing by it flattens the gradient
    without touching fine-grained detail.

    Why not just use a global gamma?  Gamma lifts all dark pixels uniformly;
    gradient correction specifically targets *spatially varying* illumination,
    which is the dominant challenge in FlyingArUco v2.
    """
    # Work in float [0, 1]
    img_f = image.astype(np.float32) / 255.0

    # Estimate illumination: heavily blurred downsampled version
    small = cv2.resize(img_f, (64, 64), interpolation=cv2.INTER_AREA)
    blurred_small = cv2.GaussianBlur(small, (31, 31), 0)
    illum = cv2.resize(blurred_small, (image.shape[1], image.shape[0]),
                       interpolation=cv2.INTER_LINEAR)

    # Avoid division by near-zero; clamp illumination floor to 0.05
    illum = np.clip(illum, 0.05, None)

    # Divide and re-normalise to [0, 1]
    corrected = img_f / illum
    corrected = np.clip(corrected, 0.0, 1.0)

    # Rescale so the mean brightness is preserved (avoids over-brightening)
    mean_orig = img_f.mean()
    mean_corr = corrected.mean()
    if mean_corr > 1e-6:
        corrected *= mean_orig / mean_corr
    corrected = np.clip(corrected, 0.0, 1.0)

    return (corrected * 255).astype(np.uint8)

if __name__ == "__main__":
    RAW_DIR = "data/raw/flyingarucov2"
    convert_to_yolo_file(RAW_DIR)
    split_dataset(RAW_DIR)