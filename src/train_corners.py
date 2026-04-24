"""
Training script for the Corner Refinement CNN (src/corners.CornerRefinementCNN).

Run from the repository root:
    conda run -p C:/Users/Admin/anaconda3/envs/deeparuco ^
        python train/train_corners.py --data_dir <path/to/flyingarucov2>

---------------------------------------------------------------------------
DATASET FORMAT
---------------------------------------------------------------------------
Each image in the FlyingArUco v2 dataset has a paired JSON sidecar file:

    000000000089.jpg
    000000000089.json

The JSON looks like:

    {
        "markers": [
            {
                "id": 29,
                "corners": [
                    [481.78, 261.83],   ← corner 0 (pixel x, y)
                    [516.16, 203.50],   ← corner 1
                    [451.48, 169.94],   ← corner 2
                    [425.14, 229.83]    ← corner 3
                ],
                "rot": 2
            },
            ...
        ]
    }

The "corners" field gives all four corner pixel coordinates.
The "rot" field (0–3) records which corners[rot] is the canonical top-left
corner (i.e. the top-left of the marker in its readable orientation).

WHY THE ROT FIELD IS NOT USED FOR CORNER CNN TRAINING:
The CNN's job is purely geometric — given a 64×64 crop it predicts where
the four corners are.  It does not need to know *which* corner is canonical
TL; that is resolved at inference time by the decoding stage (decode.py uses
the Hamming-distance rotation to select corners[rotation] as canonical TL).
What the CNN needs is a *consistent spatial ordering* of the four output
values: we always sort into [TL, TR, BR, BL] by image position using
sort_corners_tl_tr_br_bl(), regardless of the marker's physical rotation.

---------------------------------------------------------------------------
TRAINING OVERVIEW
---------------------------------------------------------------------------

  Data loading  :  scan data_dir for all (*.jpg, *.json) pairs
                   → for every real marker in each JSON, extract a 64×64
                     crop (expanded bbox) and normalise its 4 corners to [0,1]

  Augmentation  :  per crop in __getitem__:
                   • geometric: 90° rotation × 4, H-flip, V-flip
                     (corner coordinates transformed in lock-step)
                   • photometric: random gradient multiply, Gaussian blur

  Model         :  CornerRefinementCNN (see src/corners.py)
                   ~144 K parameters, input (3,64,64), output (8,)

  Loss          :  MAE (L1) — linear in error, robust to hard/occluded crops

  Optimiser     :  Adam, initial lr=1e-3
  LR schedule   :  ReduceLROnPlateau — halve after 5 epochs no improvement
  Early stop    :  stop after 10 epochs no improvement on val loss
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corners import CornerRefinementCNN


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CROP_SIZE = 64  # CNN input spatial size (pixels)
BBOX_MARGIN = 0.20  # Expand tight bbox by this fraction on each side


# ──────────────────────────────────────────────────────────────────────────────
# Corner ordering utility
# ──────────────────────────────────────────────────────────────────────────────


def sort_corners_tl_tr_br_bl(corners: np.ndarray) -> np.ndarray:
    """Sort four (x, y) points into [TL, TR, BR, BL] spatial order.

    WHY THIS IS NEEDED:
    After a rotation or flip augmentation the four corner coordinates change
    position.  To keep a consistent output convention — the CNN always
    outputs [TL, TR, BR, BL] — we re-sort by spatial position after every
    geometric transform.

    HOW IT WORKS:
    Two projections onto the diagonal axes separate the four corners cleanly
    for any square marker at any multiple-of-90° rotation:

        x + y  →  minimum = TL (closest to image origin)
                   maximum = BR (farthest from origin)
        x − y  →  maximum = TR (most right, least down)
                   minimum = BL (most left, most down)

    Args:
        corners: (4, 2) float32 in any order, values in [0, 1].

    Returns:
        (4, 2) float32 in [TL, TR, BR, BL] order.
    """
    s = corners[:, 0] + corners[:, 1]  # x + y
    d = corners[:, 0] - corners[:, 1]  # x − y

    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]
    tr = corners[np.argmax(d)]
    bl = corners[np.argmin(d)]

    return np.stack([tl, tr, br, bl]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _random_lighting_gradient(h: int, w: int) -> np.ndarray:
    """Random linear brightness gradient, shape (H, W, 1), float32.

    WHY:
    FlyingArUco v2 images have markers under directional lighting (background
    luma is projected onto the marker).  Multiplying crops by a random gradient
    teaches the CNN to locate corners even when one side of the marker is much
    brighter than the other — exactly the failure mode for classical methods.

    HOW:
    Sample a random angle θ ∈ [0, 2π).  Project each pixel's (x, y) position
    onto the direction (cos θ, sin θ), linearly scale to [min_val, max_val].
    """
    angle = random.uniform(0, 2 * np.pi)
    min_val = random.uniform(0.1, 0.8)
    max_val = random.uniform(1.0, 2.0)

    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(xs, ys)
    raw = xx * np.cos(angle) + yy * np.sin(angle)  # (H, W)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    return (min_val + raw * (max_val - min_val)).astype(np.float32)[:, :, None]


def augment_crop_and_corners(
    crop: np.ndarray,
    corners_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random augmentation to a 64×64 crop and its normalised corners.

    THE CARDINAL RULE: every geometric operation applied to the image pixels
    must be applied identically to the corner coordinates, otherwise the
    model learns a wrong mapping between image content and corner positions.

    Augmentations:
      Geometric (always applied randomly):
        • 90° rotation  — uniform choice from {0°, 90°, 180°, 270°}
        • Horizontal flip — 50 % probability
        • Vertical flip   — 50 % probability

      Photometric (always applied):
        • Random linear gradient multiply
        • Gaussian blur — 20 % probability

    HOW COORDINATE ROTATION WORKS:
    One 90° CCW rotation maps (x, y) → (y, 1−x) in normalised [0,1] space.
    Applying this k times gives the coordinate transform for k × 90° CCW.
    After each geometric step we call sort_corners_tl_tr_br_bl() to restore
    the [TL, TR, BR, BL] convention — which corner label points to which
    physical corner can change after a flip or rotation.

    Args:
        crop:         (64, 64, 3) uint8 BGR.
        corners_norm: (4, 2) float32 in [0, 1] — [TL, TR, BR, BL].

    Returns:
        (augmented_crop uint8, augmented_corners float32) — same shapes.
    """
    img = crop.copy().astype(np.float32)
    pts = corners_norm.copy()

    # ── Geometric ────────────────────────────────────────────────────────────

    # 90° CCW rotation, 0–3 steps
    k = random.randint(0, 3)
    if k > 0:
        img = np.rot90(img, k=k).copy()
        for _ in range(k):
            # (x, y) → (y, 1−x)  for one 90° CCW step
            pts = np.column_stack([pts[:, 1], 1.0 - pts[:, 0]])
        pts = sort_corners_tl_tr_br_bl(pts)

    # Horizontal flip: new_x = 1 − old_x
    if random.random() < 0.5:
        img = img[:, ::-1].copy()
        pts[:, 0] = 1.0 - pts[:, 0]
        pts = sort_corners_tl_tr_br_bl(pts)

    # Vertical flip: new_y = 1 − old_y
    if random.random() < 0.5:
        img = img[::-1, :].copy()
        pts[:, 1] = 1.0 - pts[:, 1]
        pts = sort_corners_tl_tr_br_bl(pts)

    # ── Photometric ───────────────────────────────────────────────────────────

    # Random gradient — always applied (core augmentation for lighting robustness)
    grad = _random_lighting_gradient(img.shape[0], img.shape[1])
    img = np.clip(img * grad, 0, 255)

    # Gaussian blur — 20 % probability
    if random.random() < 0.2:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return np.clip(img, 0, 255).astype(np.uint8), pts.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading from FlyingArUco v2 JSON sidecar files
# ──────────────────────────────────────────────────────────────────────────────


def load_dataset_from_json(data_dir: str | Path) -> list[dict]:
    """Scan data_dir for (*.jpg, *.json) pairs and build the sample list.

    For every real marker in every JSON file we:
      1. Compute the tight axis-aligned bounding box from the 4 corners.
      2. Expand it by BBOX_MARGIN (20 %) on each side and clamp to image bounds.
      3. Crop that region from the image and resize to CROP_SIZE × CROP_SIZE.
      4. Normalise the 4 corner pixel coordinates to [0, 1] relative to the
         expanded crop dimensions.
      5. Sort into [TL, TR, BR, BL] spatial order.

    WHY EXPAND THE BBOX?
    The CNN must be able to *see* all four corners.  If we crop exactly to the
    tight bbox, any small detection error at inference time will clip a corner
    out of the crop.  A 20 % margin ensures corners remain visible even when
    the detection is a few pixels off — consistent with how crops are produced
    at inference time in src/detect.crop_detection().

    WHY SORT CORNERS BY SPATIAL POSITION?
    The JSON corners have a dataset-internal ordering that is NOT necessarily
    [TL, TR, BR, BL].  Sorting spatially gives the CNN a consistent target
    convention that is independent of the marker's rotation in the scene.

    WHY SKIP TINY MARKERS?
    Markers smaller than ~10 × 10 px in the original image produce very
    coarse crops after resize to 64 × 64.  The interpolation artefacts are
    worse than the annotation noise, so we skip them.  The threshold is
    min_side = 10 px on the tight bbox.

    Args:
        data_dir: Directory containing interleaved *.jpg and *.json files.

    Returns:
        List of dicts, each with:
            'crop'         — (CROP_SIZE, CROP_SIZE, 3) uint8 BGR
            'corners_norm' — (4, 2) float32 in [0, 1], [TL, TR, BR, BL] order
            'image_id'     — str, the filename stem (for debugging)
            'marker_id'    — int, the ArUco marker ID
    """
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON annotation files found in {data_dir}. "
            "Make sure data_dir points to the flyingarucov2 folder."
        )

    samples: list[dict] = []
    skipped_no_image = 0
    skipped_too_small = 0

    for json_path in json_files:
        image_id = json_path.stem
        img_path = json_path.with_suffix(".jpg")

        if not img_path.exists():
            skipped_no_image += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            skipped_no_image += 1
            continue

        img_h, img_w = image.shape[:2]

        with open(json_path, "r") as f:
            annotation = json.load(f)

        for marker in annotation.get("markers", []):
            corners_px = np.array(marker["corners"], dtype=np.float32)  # (4, 2)

            # ── Compute tight bbox ────────────────────────────────────────
            bx0 = int(np.floor(corners_px[:, 0].min()))
            by0 = int(np.floor(corners_px[:, 1].min()))
            bx1 = int(np.ceil(corners_px[:, 0].max()))
            by1 = int(np.ceil(corners_px[:, 1].max()))

            tight_w = bx1 - bx0
            tight_h = by1 - by0

            # Skip markers that are too small to produce a useful crop
            if tight_w < 10 or tight_h < 10:
                skipped_too_small += 1
                continue

            # ── Expand bbox by BBOX_MARGIN and clamp to image ────────────
            dx = int(tight_w * BBOX_MARGIN)
            dy = int(tight_h * BBOX_MARGIN)
            ex0 = max(0, bx0 - dx)
            ey0 = max(0, by0 - dy)
            ex1 = min(img_w, bx1 + dx)
            ey1 = min(img_h, by1 + dy)

            crop_w = ex1 - ex0
            crop_h = ey1 - ey0

            if crop_w < 4 or crop_h < 4:
                skipped_too_small += 1
                continue

            # ── Extract and resize crop ───────────────────────────────────
            crop = image[ey0:ey1, ex0:ex1]
            crop = cv2.resize(
                crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR
            )

            # ── Normalise corners to [0,1] relative to expanded crop ──────
            # corners_px are in full-image pixel coordinates.
            # We shift by the crop origin (ex0, ey0) then divide by the
            # crop dimensions so (0,0)=top-left of crop, (1,1)=bottom-right.
            corners_norm = corners_px.copy()
            corners_norm[:, 0] = (corners_px[:, 0] - ex0) / crop_w
            corners_norm[:, 1] = (corners_px[:, 1] - ey0) / crop_h

            # Clamp: if a corner is slightly outside the expanded crop due
            # to clamping against the image boundary, keep it at the edge.
            corners_norm = np.clip(corners_norm, 0.0, 1.0)

            # ── Sort into consistent [TL, TR, BR, BL] order ──────────────
            corners_norm = sort_corners_tl_tr_br_bl(corners_norm)

            samples.append(
                {
                    "image_id": image_id,
                    "marker_id": marker["id"],
                    "crop": crop,
                    "corners_norm": corners_norm,
                }
            )

    print(f"Loaded {len(samples)} marker crops from {len(json_files)} images.")
    if skipped_no_image:
        print(f"  Skipped {skipped_no_image} images (file not found or unreadable).")
    if skipped_too_small:
        print(f"  Skipped {skipped_too_small} markers (tight bbox < 10 px).")

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────


class CornerDataset(Dataset):
    """PyTorch Dataset wrapping pre-loaded (crop, corners) samples.

    WHY PRE-LOAD CROPS?
    Each sample requires a full image read, bbox expansion, crop, and resize.
    Doing this inside __getitem__ (called once per sample per epoch) repeats
    slow disk I/O thousands of times.  Pre-loading pays the cost once at
    construction and then only runs the cheap augmentation in __getitem__.

    Memory footprint: ~50 K samples × 64 × 64 × 3 bytes ≈ 600 MB — fine for
    a modern workstation.  If RAM is tight, store (image_path, bbox) tuples
    instead and reload lazily.

    Args:
        samples: List of dicts from load_dataset_from_json().
        augment: Whether to apply random augmentation in __getitem__.
                 Set False for the validation split.
    """

    def __init__(self, samples: list[dict], augment: bool = True):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (image_tensor, label_tensor) pair.

        image_tensor : (3, 64, 64) float32 in [0, 1]  — CNN input (RGB)
        label_tensor : (8,)        float32 in [0, 1]
                        = [TL_x, TL_y, TR_x, TR_y, BR_x, BR_y, BL_x, BL_y]

        WHY FLATTEN TO (8,)?
        nn.L1Loss operates on flat tensors by default.  We flatten here and
        reshape to (4, 2) only in post-processing when we need (x, y) pairs.

        WHY BGR → RGB?
        PyTorch's standard image convention is RGB.  OpenCV reads as BGR.
        Swapping channels here keeps the CNN consistent with the standard
        ImageNet colour order, which matters if we ever want to add a
        pre-trained backbone later.
        """
        s = self.samples[idx]
        crop = s["crop"]  # (64, 64, 3) uint8 BGR
        pts = s["corners_norm"].copy()  # (4, 2) float32

        if self.augment:
            crop, pts = augment_crop_and_corners(crop, pts)

        # BGR → RGB, HWC → CHW, scale to [0, 1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        # Flatten (4, 2) → (8,)
        label_tensor = torch.from_numpy(pts.flatten())

        return img_tensor, label_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────────────────────────────────────

# WHY MAE (L1) AND NOT MSE (L2)?
# MAE penalises errors linearly: a 10 px error hurts 10× more than 1 px.
# MSE penalises a 10 px error 100× more than 1 px, letting the small
# fraction of hard/occluded markers dominate the gradient and destabilise
# training.  FlyingArUco v2 includes partially occluded and extreme-lighting
# markers — MAE stops these outliers from overwhelming the clean samples.
criterion = nn.L1Loss()


# ──────────────────────────────────────────────────────────────────────────────
# Training and validation loops
# ──────────────────────────────────────────────────────────────────────────────


def run_epoch(
    model: CornerRefinementCNN,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer | None,
    device: str,
) -> float:
    """Run one full pass over a DataLoader, returning the mean MAE loss.

    When optimiser is not None (training mode):
      zero_grad → forward → loss → backward → step

    When optimiser is None (validation mode):
      forward only, no gradient computation (@torch.no_grad() applied).

    WHY COMBINED INTO ONE FUNCTION?
    The forward pass and loss calculation are identical for train and val.
    The only difference is whether we update weights.  One function with an
    optional optimiser argument avoids duplicating ~10 lines of code.
    """
    is_training = optimiser is not None
    model.train(is_training)
    total_loss = 0.0

    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            if is_training:
                optimiser.zero_grad()

            preds = model(imgs)
            loss = criterion(preds, labels)

            if is_training:
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    """End-to-end training pipeline for the Corner Refinement CNN."""

    device = args.device
    print(f"Device : {device}")
    print(f"Data   : {args.data_dir}")

    # ── 1. Load all samples from JSON sidecar files ───────────────────────────
    all_samples = load_dataset_from_json(args.data_dir)

    if len(all_samples) == 0:
        raise RuntimeError(
            "No samples were loaded. Check that --data_dir points to the "
            "flyingarucov2 folder containing *.jpg and *.json files."
        )

    # ── 2. Train / validation split ───────────────────────────────────────────
    # WHY RANDOM SHUFFLE BEFORE SPLIT?
    # JSON files are sorted by COCO image ID.  Images with similar IDs may
    # share background characteristics.  A shuffle breaks this correlation
    # so val and train images are drawn from the full distribution.
    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * args.val_fraction))
    train_samples = all_samples[n_val:]
    val_samples = all_samples[:n_val]
    print(f"Train samples : {len(train_samples)}")
    print(f"Val   samples : {len(val_samples)}")

    train_ds = CornerDataset(train_samples, augment=True)
    val_ds = CornerDataset(val_samples, augment=False)

    # WHY num_workers=0 ON WINDOWS?
    # Python's 'spawn' start method on Windows has high per-worker overhead.
    # num_workers=0 loads data in the main process — slightly slower per batch
    # but avoids the spawn delay and CUDA context issues on Windows.
    # On Linux/macOS you can safely raise this to 4.
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )

    # ── 3. Model, optimiser, LR scheduler ────────────────────────────────────
    model = CornerRefinementCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters    : {n_params:,}")

    # WHY ADAM?  It adapts the effective learning rate per parameter using
    # running gradient moment estimates.  For regression on a moderately-sized
    # dataset it converges faster and more reliably than vanilla SGD.
    optimiser = Adam(model.parameters(), lr=args.lr)

    # WHY ReduceLROnPlateau?
    # After the initial fast descent the loss landscape becomes flatter.
    # Halving the LR (factor=0.5) after 5 stagnant val epochs lets the
    # optimiser take smaller, more precise steps without manual intervention.
    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        # verbose=True,
    )

    # ── 4. Checkpoint directory ───────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_corners.pth"

    # ── 5. Training loop ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("\nEpoch  Train-MAE   Val-MAE    LR")
    print("-" * 46)

    for epoch in range(1, args.max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimiser, device)
        val_loss = run_epoch(model, val_loader, None, device)
        current_lr = optimiser.param_groups[0]["lr"]

        print(f"{epoch:5d}  {train_loss:.6f}  {val_loss:.6f}  {current_lr:.2e}")

        # Step the LR scheduler with current validation loss
        scheduler.step(val_loss)

        # ── Checkpoint & early stopping ───────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
            print(f"         ** New best saved to {best_ckpt}")
        else:
            epochs_no_improve += 1

        # WHY STOP EARLY?
        # Once validation loss stagnates the model is overfitting.  The best
        # checkpoint was already saved — additional epochs only make the last
        # checkpoint (which we do NOT use) worse.  Stopping saves compute.
        if epochs_no_improve >= args.patience:
            print(
                f"\nEarly stopping triggered after {args.patience} "
                f"epochs without improvement."
            )
            break

    print(f"\nDone.  Best val MAE : {best_val_loss:.6f}")
    print(f"Best model         : {best_ckpt}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train the CornerRefinementCNN on FlyingArUco v2 data."
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the flyingarucov2 folder containing *.jpg + *.json files.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="models/corners",
        help="Directory where the best checkpoint (best_corners.pth) is saved.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device: 'cpu', 'cuda', 'cuda:0', ...",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--lr", type=float, default=1e-3, help="Initial learning rate for Adam."
    )
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument(
        "--patience", type=int, default=10, help="Early-stopping patience in epochs."
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of samples held out for validation.",
    )

    train(p.parse_args())
