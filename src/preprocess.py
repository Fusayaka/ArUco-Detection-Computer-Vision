import glob
import json
import os
import random
import shutil
from pathlib import Path
 
import cv2
 
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

if __name__ == "__main__":
    RAW_DIR = "data/raw/flyingarucov2"
    convert_to_yolo_file(RAW_DIR)
    split_dataset(RAW_DIR)