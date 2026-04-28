import os
import json
import glob
import shutil
import random
import cv2
import numpy as np

def convert_to_yolo_file(dirpath):
    """Parse JSON annotations and convert them into standard YOLO format (.txt)."""
    label_dir = os.path.join(dirpath, 'labels')
    if os.path.exists(label_dir) and len(os.listdir(label_dir)) > 0:
        print(f'{label_dir} already exists.')
        return
        
    os.makedirs(label_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(dirpath, '*.json'))

    for json_file in json_files:
        file_id = os.path.basename(json_file).replace('.json', '')
        img_path = os.path.join(dirpath, f'{file_id}.jpg')

        img = cv2.imread(img_path)
        if img is None: continue
        height, width, _ = img.shape

        with open(json_file, 'r') as f:
            data = json.load(f)

        lines = []
        for marker in data['markers']:
            corner = marker['corners']
            xs = [c[0] for c in corner]
            ys = [c[1] for c in corner]
                
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x_mid_norm = ((x_min + x_max) / 2) / width
            y_mid_norm = ((y_min + y_max) / 2) / height
            w_norm = (x_max - x_min) / width
            h_norm = (y_max - y_min) / height

            line = f'0 {x_mid_norm:.6f} {y_mid_norm:.6f} {w_norm:.6f} {h_norm:.6f}'
            lines.append(line)

        txt_file = os.path.join(label_dir, f'{file_id}.txt')
        with open(txt_file, 'w') as f:
            f.write('\n'.join(lines))
    print("Convert to YOLO format successfully.")

def convert_csv_to_yolo_file(dirpath):
    """Parse CSV annotations and convert them into standard YOLO format (.txt)."""

def split_dataset(src_dir, base_dir="data/processed/dataset", train_ratio=0.8):
    """Split data into train/val sets and organize directory structure."""
    sub_dirs = ["images/train", "images/val", "labels/train", "labels/val"]

    for sub in sub_dirs:
        try:
            os.makedirs(os.path.join(base_dir, sub))
        except FileExistsError:
            pass
        
    all_ids = [f.replace('.jpg', '') for f in os.listdir(src_dir) if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(all_ids)

    split_index = int(len(all_ids) * train_ratio)
    train_ids = all_ids[:split_index]
    val_ids = all_ids[split_index:]

    def move_files(ids, split_name):
        label_src_dir = os.path.join(src_dir, "labels")
        for file_id in ids:
            img_src = os.path.join(src_dir, f"{file_id}.jpg")
            img_dst = os.path.join(base_dir, f"images/{split_name}/{file_id}.jpg")
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dst)

            txt_src = os.path.join(label_src_dir, f"{file_id}.txt")
            txt_dst = os.path.join(base_dir, f"labels/{split_name}/{file_id}.txt")
            if os.path.exists(txt_src):
                shutil.copy(txt_src, txt_dst)

    move_files(train_ids, "train")
    move_files(val_ids, "val")
    print("Dataset split successfully.")

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