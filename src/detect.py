"""
Detection module — finds ArUco marker candidates in an image and returns
their bounding boxes + (rough) corner locations.

---------------------------------------------------------------------------
WHY THIS MODULE EXISTS
---------------------------------------------------------------------------
Detection is the first real CV problem in our pipeline: "Where in this
640×360 image are the markers?"  We tackle it in two ways:

  1. Classical OpenCV ArUco (baseline)
     Fast, zero training required.  Internally it does:
       a) Adaptive thresholding → binary image
       b) Contour extraction (find all connected dark regions)
       c) Polygon approximation → keep only quadrilaterals
       d) Perspective warp of each quad → check if it looks like a marker
     The weakness: step (a) fails badly under uneven illumination.  If the
     left half of the image is dark and the right half is bright, a single
     threshold (even an adaptive one) cannot cleanly separate marker borders
     from background everywhere simultaneously.  This is exactly the problem
     FlyingArUco v2 was designed to stress-test.

  2. YOLOv8 (our trained detector) — added in a later step
     A convolutional network that learns *from data* what a marker looks like
     regardless of lighting.  Because it sees thousands of training examples
     with gradient lighting, colour shifts, and noise, it generalises far
     better.  We fine-tune the public YOLOv8 nano weights (pretrained on
     COCO) on our Kaggle dataset — that is our one allowed "pre-trained" step.

Both detectors return the same data structure so the rest of the pipeline
is detector-agnostic.

---------------------------------------------------------------------------
OUTPUT FORMAT
---------------------------------------------------------------------------
Each detector returns a list of Detection namedtuples:
  Detection(
      corners   – np.ndarray shape (4, 2) float32, in pixel coords
                  ordered: top-left, top-right, bottom-right, bottom-left
                  (ArUco convention, counterclockwise from top-left)
      bbox      – (x_min, y_min, x_max, y_max) int, tight bounding box
      marker_id – int or None  (OpenCV gives this directly; YOLO does not)
      confidence– float 0–1 (always 1.0 for classical; YOLO box score)
  )

Having corners from the detector is a rough first estimate.  The dedicated
corner-refinement stage (corners.py) will improve their sub-pixel accuracy.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os

# ──────────────────────────────────────────────────────────────────────────────
# Data structure returned by every detector
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Detection:
    """One detected marker candidate."""

    corners: np.ndarray  # shape (4, 2) float32, pixel coords
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    marker_id: Optional[int]  # None when the detector doesn't decode IDs
    confidence: float  # detection confidence score [0, 1]

    def top_left(self) -> tuple[float, float]:
        """Return the top-left corner (x, y).

        WHY: The assignment asks for the top-left corner specifically.
        ArUco's corner ordering guarantees corners[0] is always top-left
        *after* the marker has been oriented to its canonical rotation.
        """
        return float(self.corners[0, 0]), float(self.corners[0, 1])


# ──────────────────────────────────────────────────────────────────────────────
# Helper: convert OpenCV's (1, 4, 2) corner array to our (4, 2) shape
# ──────────────────────────────────────────────────────────────────────────────


def _opencv_corners_to_array(cv_corners) -> np.ndarray:
    """OpenCV returns corners as shape (1, 4, 2); we want (4, 2)."""
    return cv_corners[0].astype(np.float32)


def _corners_to_bbox(corners: np.ndarray) -> tuple[int, int, int, int]:
    """Compute an axis-aligned bounding box from 4 corner points.

    WHY axis-aligned: the corner-refinement CNN expects a square crop, so we
    need the bounding box of the *rotated* quad, not just the quad itself.
    """
    x_min = int(np.floor(corners[:, 0].min()))
    y_min = int(np.floor(corners[:, 1].min()))
    x_max = int(np.ceil(corners[:, 0].max()))
    y_max = int(np.ceil(corners[:, 1].max()))
    return x_min, y_min, x_max, y_max


# ──────────────────────────────────────────────────────────────────────────────
# Classical OpenCV ArUco detector
# ──────────────────────────────────────────────────────────────────────────────

# We use DICT_ARUCO_MIP_36H12 because that is the dictionary specified in the
# assignment.  It has 250 valid codewords, each encoded as a 6×6 bit matrix
# (36 bits total), with a minimum Hamming distance of 12 between any two
# codewords.  The large minimum distance makes false-positive IDs rare —
# a marker would need at least 6 bit errors before being mistaken for another.
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36H12)


def _make_classical_params(
    adaptive_thresh_win_size_min: int = 3,
    adaptive_thresh_win_size_max: int = 23,
    adaptive_thresh_win_size_step: int = 10,
    min_marker_perimeter_rate: float = 0.03,
    error_correction_rate: float = 0.6,
) -> cv2.aruco.DetectorParameters:
    """Build OpenCV ArUco detector parameters.

    WHY TUNE THESE?
    ---------------
    The default parameters work well for markers photographed in controlled
    lab conditions.  Our dataset has:
      - Small markers (sometimes only ~30px wide in a 640px image)
      - Strong lighting gradients

    adaptive_thresh_win_size_*: OpenCV tries multiple window sizes for its
        internal adaptive threshold.  More window sizes = more chances to
        pick up markers at different scales/contrasts, at the cost of speed.

    min_marker_perimeter_rate: Fraction of the image diagonal that a marker
        contour must span to be considered valid.  Setting it low (0.03)
        keeps small markers; setting it too low invites noise.

    error_correction_rate: How many bit errors to tolerate when matching
        against the dictionary.  0.6 means up to ~6/10 of the Hamming
        distance threshold (≈7 bits for MIP_36h12) may be wrong.  Higher
        = more recall but more false positives.
    """
    params = cv2.aruco.DetectorParameters()

    params.adaptiveThreshWinSizeMin = adaptive_thresh_win_size_min
    params.adaptiveThreshWinSizeMax = adaptive_thresh_win_size_max
    params.adaptiveThreshWinSizeStep = adaptive_thresh_win_size_step
    params.minMarkerPerimeterRate = min_marker_perimeter_rate
    params.errorCorrectionRate = error_correction_rate

    return params


def detect_classical(
    image: np.ndarray,
    params: cv2.aruco.DetectorParameters | None = None,
) -> list[Detection]:
    """Run the classical OpenCV ArUco detector on a BGR image.

    HOW IT WORKS (step by step):
    1. Convert to grayscale — ArUco's threshold operates on intensity only.
    2. Apply adaptive thresholding over a sliding window — each pixel is
       compared to the mean of its local neighbourhood, so areas with
       different overall brightness are handled independently.
    3. Find contours in the binary image, keep only quadrilaterals.
    4. Perspective-warp each quad into a small square, then read the bits.
    5. Compare bits against DICT_ARUCO_MIP_36H12 (Hamming distance).

    Args:
        image: BGR uint8 image.
        params: Optional custom detector parameters.  Uses tuned defaults.

    Returns:
        List of Detection objects, one per detected marker.
    """
    if params is None:
        params = _make_classical_params()

    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)

    # detectMarkers expects BGR; it converts to gray internally.
    corners_list, ids, _ = detector.detectMarkers(image)

    if ids is None:
        return []

    detections = []
    for i, cv_corners in enumerate(corners_list):
        corners = _opencv_corners_to_array(cv_corners)
        bbox = _corners_to_bbox(corners)
        detections.append(
            Detection(
                corners=corners,
                bbox=bbox,
                marker_id=int(ids[i][0]),
                confidence=1.0,  # classical detector has no confidence score
            )
        )

    return detections


def detect_classical_multiscale(
    image: np.ndarray,
    preprocessed: np.ndarray | None = None,
) -> list[Detection]:
    """Run the classical detector on both the raw and CLAHE-enhanced image,
    then merge results.

    WHY BOTH?
    ---------
    CLAHE boosts contrast in dim areas, which helps detect markers in shadows.
    But it can *over-enhance* already-bright areas, occasionally introducing
    ringing artefacts that confuse the thresholding step.  Running on the
    original image catches markers that CLAHE degraded, and vice versa.

    Merging is done by marker ID: if the same ID appears in both result sets,
    we keep the detection with the tighter (smaller area) bounding box, which
    tends to correspond to a more precise corner estimate.

    Args:
        image: Original BGR image.
        preprocessed: CLAHE-enhanced BGR image.  If None, only the raw image
            is processed (falls back to plain detect_classical).

    Returns:
        Merged list of Detection objects, one per unique marker ID.
    """
    raw_dets = detect_classical(image)

    if preprocessed is None:
        return raw_dets

    enh_dets = detect_classical(preprocessed)

    # Merge: keep best (smallest bbox area) detection per ID
    merged: dict[int, Detection] = {}

    for det in raw_dets + enh_dets:
        if det.marker_id is None:
            continue
        mid = det.marker_id
        x0, y0, x1, y1 = det.bbox
        area = (x1 - x0) * (y1 - y0)

        if mid not in merged:
            merged[mid] = det
        else:
            ox0, oy0, ox1, oy1 = merged[mid].bbox
            old_area = (ox1 - ox0) * (oy1 - oy0)
            if area < old_area:
                merged[mid] = det

    return list(merged.values())


# ──────────────────────────────────────────────────────────────────────────────
# Bounding-box utilities (used by corner refinement stage)
# ──────────────────────────────────────────────────────────────────────────────


def expand_bbox(
    bbox: tuple[int, int, int, int],
    img_h: int,
    img_w: int,
    margin: float = 0.20,
) -> tuple[int, int, int, int]:
    """Expand a bounding box by a relative margin on all sides, clamped to
    the image boundaries.

    WHY 20% MARGIN?
    ---------------
    The corner-refinement CNN needs to *see the corners*, not just the
    interior of the marker.  If we crop exactly to the detected bounding box,
    any positional error in the detection will clip off one or more corners.
    The DeepArUco++ paper uses a 20% margin and reports that it reliably
    keeps all four corners in the crop even under moderate detection error.

    Args:
        bbox:   (x_min, y_min, x_max, y_max) tight bounding box.
        img_h:  Image height in pixels (clamp upper bound).
        img_w:  Image width in pixels (clamp upper bound).
        margin: Fractional expansion per side (0.20 = 20%).

    Returns:
        Expanded (x_min, y_min, x_max, y_max), clamped to image bounds.
    """
    x0, y0, x1, y1 = bbox
    dx = int((x1 - x0) * margin)
    dy = int((y1 - y0) * margin)
    return (
        max(0, x0 - dx),
        max(0, y0 - dy),
        min(img_w, x1 + dx),
        min(img_h, y1 + dy),
    )


def crop_detection(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_size: int = 64,
    margin: float = 0.20,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a (possibly expanded) bounding-box region and resize to target_size.

    WHY 64×64?
    ----------
    The corner-refinement CNN (Section 4.2 of DeepArUco++) expects exactly
    64×64 input.  Smaller inputs lose detail; larger inputs waste memory and
    slow inference.  64px is large enough to resolve individual cells of the
    6×6 marker grid even for small markers (~30px wide at native resolution).

    Args:
        image:       Full BGR or grayscale image.
        bbox:        Tight (x_min, y_min, x_max, y_max).
        target_size: Side length to resize the crop to.
        margin:      Fractional margin added before cropping.

    Returns:
        (crop, expanded_bbox) where crop is uint8 shape
        (target_size, target_size, C) and expanded_bbox is the actual pixel
        region that was cropped (needed to map corners back to image coords).
    """
    h, w = image.shape[:2]
    exp_bbox = expand_bbox(bbox, h, w, margin)
    x0, y0, x1, y1 = exp_bbox

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        # Degenerate crop (bbox outside image) — return blank
        if image.ndim == 3:
            crop = np.zeros((target_size, target_size, image.shape[2]), dtype=np.uint8)
        else:
            crop = np.zeros((target_size, target_size), dtype=np.uint8)
        return crop, exp_bbox

    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return crop, exp_bbox


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    image_folder = args.image
    images = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            images.append(img)

    for i, image in enumerate(images):
        detections = detect_classical(image)
        for detection in detections:
            cropped_image = image[
                detection.bbox[1] : detection.bbox[3],
                detection.bbox[0] : detection.bbox[2],
            ]
            plt.subplot(1, len(images), i + 1)
            plt.imshow(cropped_image)
            plt.show()
