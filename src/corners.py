"""
Corner Refinement module — improves the (x, y) accuracy of the four marker
corners produced by the detection stage.

---------------------------------------------------------------------------
WHY REFINEMENT IS A SEPARATE STAGE
---------------------------------------------------------------------------
The detection stage (classical ArUco or YOLO) tells us roughly *where* a
marker is, but its corner coordinates are not precise enough for two reasons:

  1. The YOLO bounding box doesn't give corners at all — only a rectangle.
  2. Classical ArUco's polygon approximation snaps corners to the nearest
     pixel, and under blur / lighting variation the contour can be off by
     several pixels.

Why does this matter?  The assignment evaluation uses a Gaussian score:

    φ(d_norm) = exp(−d² / 2σ²),  σ = 0.02

where d_norm is the distance to the ground-truth top-left corner divided by
the image diagonal.  For a 640×480 image the diagonal is ≈ 800 px, so σ ≈ 16 px.
The score falls to ≈ 0.78 at d = σ (16 px error) and to ≈ 0.37 at d = 2σ.
Even a 10 px improvement in corner accuracy measurably lifts the final score.

---------------------------------------------------------------------------
TWO APPROACHES
---------------------------------------------------------------------------
We implement two refinement strategies and can switch between them:

  A. Classical — cv2.cornerSubPix
     OpenCV's built-in sub-pixel corner localiser.  It works by iteratively
     moving each corner estimate to the point where the gradient is
     *orthogonal* to the direction towards that point (a.k.a. the "optical
     flow constraint").  It needs a reasonable initial estimate (within ~5 px)
     and clean edges.  Fast, zero training required, but breaks under blur or
     low contrast because the gradient field becomes noisy.

  B. CNN (our trained model)
     A small convolutional network that takes a 64×64 crop and directly
     regresses the (x, y) of all four corners.  Because it is trained on the
     FlyingArUco v2 dataset (with gradient lighting, colour shifts, blur),
     it learns to find corners even when the gradient signal is weak.

     Architecture (our own design, trained from scratch — *not* a pre-trained
     model, so this does not consume our pre-trained allowance):

         Input  64×64×3 BGR crop
           │
           ▼
         Conv(32 filters, 3×3) → BatchNorm → ReLU → MaxPool(2×2)  → 32×32×32
         Conv(64 filters, 3×3) → BatchNorm → ReLU → MaxPool(2×2)  → 16×16×64
         Conv(128 filters, 3×3) → BatchNorm → ReLU → GlobalAvgPool →    128
           │
           ▼  (fully-connected head)
         Linear(128 → 256) → ReLU → Dropout(0.3)
         Linear(256 → 64)  → ReLU
         Linear(64  → 8)   → Sigmoid      ← 4 corners × (x, y) in [0, 1]

     WHY THIS ARCHITECTURE?
     - Three conv blocks give a receptive field that covers the full 64×64
       input after pooling, so the network "sees" the whole marker.
     - GlobalAvgPool instead of Flatten keeps the parameter count small
       (~150 K) and reduces overfitting on a limited dataset.
     - Sigmoid on the output constrains predictions to [0, 1], which maps
       cleanly to pixel fractions of the crop width/height.
     - Dropout(0.3) regularises the dense head against the lighting
       diversity in the training data.

  The inference path uses the CNN when a trained checkpoint is available,
  and falls back to the classical method otherwise.

---------------------------------------------------------------------------
COORDINATE CONVENTION
---------------------------------------------------------------------------
All functions in this module work in *image pixel coordinates*.  The CNN
internally uses normalised [0, 1] coordinates (relative to the expanded
crop), and `refine_corners` maps them back to full-image pixel coordinates
before returning.

Output corner order:  top-left, top-right, bottom-right, bottom-left
(ArUco convention — same as Detection.corners from detect.py).
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.detect import Detection, crop_detection


# ──────────────────────────────────────────────────────────────────────────────
# CNN Architecture
# ──────────────────────────────────────────────────────────────────────────────


class CornerRefinementCNN(nn.Module):
    """Lightweight CNN that regresses 4 marker corner coordinates.

    Input : (B, 3, 64, 64) float32 tensor, values in [0, 1]
    Output: (B, 8)          float32 tensor, values in [0, 1]
              → reshaped to (B, 4, 2) → 4 corners, each (x, y) fraction

    See module docstring for architecture rationale.
    """

    def __init__(self):
        super().__init__()

        # ── Convolutional backbone ──────────────────────────────────────────
        # Each block: Conv → BatchNorm → ReLU → MaxPool
        #
        # WHY BatchNorm?  It normalises each feature map's activations to
        # zero-mean unit-variance, which stabilises training and lets us use
        # a higher learning rate.  It also acts as a mild regulariser.
        #
        # WHY MaxPool?  Halves spatial resolution, doubling the effective
        # receptive field cheaply.  After three 2×2 MaxPools the 64×64 input
        # becomes 8×8 before GlobalAvgPool, so each output unit "sees" a
        # large portion of the input.
        self.backbone = nn.Sequential(
            # Block 1: 64×64×3 → 32×32×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32×32×32 → 16×16×64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 16×16×64 → 8×8×128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # GlobalAveragePool: 8×8×128 → 128
        # WHY GlobalAvgPool instead of Flatten?
        # Flatten would give 8×8×128 = 8192 values → a huge dense layer.
        # GlobalAvgPool averages each of the 128 feature maps to a single
        # number, keeping only 128 values.  This drastically reduces
        # parameters (8192→128 = 64× fewer) and prevents overfitting.
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Fully-connected head ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),  # randomly zeros 30% of neurons during training
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
            nn.Sigmoid(),  # constrains output to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 64, 64) float in [0,1] → (B, 8) float in [0,1]."""
        feat = self.backbone(x)  # (B, 128, 8, 8)
        feat = self.gap(feat)  # (B, 128, 1, 1)
        feat = feat.flatten(1)  # (B, 128)
        return self.head(feat)  # (B, 8)


def load_corner_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> CornerRefinementCNN:
    """Load a trained CornerRefinementCNN from a .pth checkpoint.

    Args:
        checkpoint_path: Path to a file saved with torch.save(model.state_dict(), ...).
        device: 'cpu', 'cuda', or 'cuda:0' etc.

    Returns:
        Model in eval mode on the specified device.
    """
    model = CornerRefinementCNN()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Approach A — Classical sub-pixel refinement (cv2.cornerSubPix)
# ──────────────────────────────────────────────────────────────────────────────


def refine_corners_classical(
    image: np.ndarray,
    corners: np.ndarray,
    win_size: int = 5,
    max_iter: int = 30,
    epsilon: float = 0.01,
) -> np.ndarray:
    """Refine corner positions to sub-pixel accuracy using the classical method.

    HOW cornerSubPix WORKS:
    For each corner estimate p, OpenCV searches a (2·win_size+1)² window.
    It solves for the point q where the image gradient G satisfies:

        Σ_i  G(p_i) · G(p_i)ᵀ · (q − p_i) = 0   for all pixels p_i in window

    Intuitively: the gradient vectors around a true corner point *away* from
    the corner.  The algorithm finds q such that the dot product of every
    local gradient with (q − pixel_position) sums to zero — i.e. q is the
    point that the gradients collectively point toward.

    WHY win_size=5?  A 11×11 search window (2·5+1) is large enough to pull
    a slightly mis-placed estimate onto the true corner, but small enough to
    avoid being confused by the marker's interior edges.

    Args:
        image:    BGR or grayscale full image.
        corners:  (4, 2) float32 array of initial corner estimates (pixels).
        win_size: Half-width of the search window.
        max_iter: Maximum iterations of the sub-pixel solver.
        epsilon:  Stop when corner moves less than this many pixels.

    Returns:
        (4, 2) float32 array of refined corners in image pixel coordinates.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # cornerSubPix expects shape (N, 1, 2) float32
    pts = corners.reshape(-1, 1, 2).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        epsilon,
    )

    refined = cv2.cornerSubPix(gray, pts, (win_size, win_size), (-1, -1), criteria)
    return refined.reshape(4, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Approach B — CNN-based refinement
# ──────────────────────────────────────────────────────────────────────────────


def _preprocess_crop_for_cnn(crop: np.ndarray) -> torch.Tensor:
    """Convert a 64×64 BGR uint8 crop to a (1, 3, 64, 64) float32 tensor.

    WHY divide by 255?  Neural networks train better when inputs are in a
    small numeric range (here [0, 1]).  Large raw pixel values (0–255) cause
    large activations early in the network, which destabilises gradient flow.
    """
    # BGR → RGB (PyTorch convention; model was trained with RGB)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # HWC uint8 → CHW float32 in [0, 1]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)  # add batch dimension


def refine_corners_cnn(
    crop: np.ndarray,
    expanded_bbox: tuple[int, int, int, int],
    model: CornerRefinementCNN,
    device: str = "cpu",
) -> np.ndarray:
    """Refine corners using the trained CNN.

    The CNN predicts corner positions as fractions of the crop dimensions.
    We then map those fractions back to full-image pixel coordinates using
    the expanded bounding box.

    WHY fraction space instead of pixels?
    Training in [0, 1] makes the model size-agnostic: a marker that is
    40 px wide and one that is 200 px wide both produce the same normalised
    representation, so the same model handles all scales.

    Args:
        crop:          64×64 BGR crop produced by detect.crop_detection().
        expanded_bbox: The (x0, y0, x1, y1) region that was cropped —
                       needed to unmap fractions → image coords.
        model:         Loaded CornerRefinementCNN in eval mode.
        device:        Torch device string.

    Returns:
        (4, 2) float32 array of refined corners in full-image pixel coords.
    """
    x0, y0, x1, y1 = expanded_bbox
    crop_w = x1 - x0
    crop_h = y1 - y0

    tensor = _preprocess_crop_for_cnn(crop).to(device)

    with torch.no_grad():
        pred = model(tensor)  # (1, 8)

    # pred is 8 values in [0, 1]: (x0,y0, x1,y1, x2,y2, x3,y3)
    coords = pred.squeeze(0).cpu().numpy().reshape(4, 2)  # (4, 2) in [0,1]

    # Un-normalise: fraction × crop_size + crop_offset → image pixels
    corners_px = coords * np.array([crop_w, crop_h], dtype=np.float32)
    corners_px += np.array([x0, y0], dtype=np.float32)

    return corners_px.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Unified refinement entry point
# ──────────────────────────────────────────────────────────────────────────────


def refine_corners(
    image: np.ndarray,
    detection: Detection,
    model: Optional[CornerRefinementCNN] = None,
    device: str = "cpu",
    crop_size: int = 64,
    margin: float = 0.20,
) -> np.ndarray:
    """Refine the corners of a single Detection.

    Automatically selects CNN refinement when a trained model is available,
    otherwise falls back to the classical cornerSubPix method.

    Args:
        image:     Full BGR image.
        detection: A Detection from the detect stage.
        model:     Trained CornerRefinementCNN, or None to use classical.
        device:    Torch device (only used when model is not None).
        crop_size: Expected CNN input size (64 px).
        margin:    Bounding box expansion margin (must match training, 0.20).

    Returns:
        (4, 2) float32 array — refined corners in image pixel coordinates.
    """
    if model is not None:
        # CNN path: crop the expanded bbox region, run the network
        crop, exp_bbox = crop_detection(
            image, detection.bbox, target_size=crop_size, margin=margin
        )
        return refine_corners_cnn(crop, exp_bbox, model, device)
    else:
        # Classical path: cornerSubPix on the raw corners
        return refine_corners_classical(image, detection.corners)


def refine_all(
    image: np.ndarray,
    detections: list[Detection],
    model: Optional[CornerRefinementCNN] = None,
    device: str = "cpu",
) -> list[np.ndarray]:
    """Refine corners for every detection in a list.

    Args:
        image:      Full BGR image.
        detections: List of Detection objects from the detect stage.
        model:      Trained model or None (falls back to classical).
        device:     Torch device string.

    Returns:
        List of (4, 2) float32 arrays, one per detection, same order.
    """
    return [refine_corners(image, det, model, device) for det in detections]
