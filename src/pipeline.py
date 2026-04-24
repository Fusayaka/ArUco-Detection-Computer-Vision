"""
Pipeline module — stitches all stages together into a single callable.

---------------------------------------------------------------------------
THE BIG PICTURE
---------------------------------------------------------------------------
This module is the "glue" that connects the four individual stages we have
built into one coherent, runnable system:

    Image
      │
      ▼  Stage 1 – preprocess.py
    CLAHE-enhanced image
      │
      ▼  Stage 2 – detect.py
    List of Detection objects  (rough bounding boxes + initial corners)
      │
      ▼  Non-maximum suppression  (de-duplicate overlapping candidates)
    Filtered Detection list
      │
      ▼  Stage 3 – corners.py
    List of refined (4,2) corner arrays  (sub-pixel accurate)
      │
      ▼  Stage 4 – decode.py
    List of DecodeResult objects  (marker_id, hamming, rotation)
      │
      ▼  Post-processing: canonical top-left corner + spam filter
    List of PipelineResult  →  prediction_string for Kaggle CSV

---------------------------------------------------------------------------
KEY DESIGN DECISION: WHY NON-MAXIMUM SUPPRESSION?
---------------------------------------------------------------------------
The Kaggle metric penalises *spam* predictions with λ = 1 per extra
detection.  That means a spurious prediction (wrong ID or duplicate) is
just as bad as a *missed* detection — both subtract 1 from the numerator
without contributing to it.

Sources of duplicates in our pipeline:
  • detect_classical_multiscale runs the detector on both the raw and the
    CLAHE-enhanced image, so the same physical marker can appear twice.
  • Small markers near the detection threshold can produce two overlapping
    quads from slightly different contour paths.
  • Future: YOLO with low NMS threshold.

NMS (Non-Maximum Suppression) removes lower-confidence detections whose
bounding boxes overlap significantly with a higher-confidence one.
We measure overlap with Intersection-over-Union (IoU):

    IoU(A, B) = area(A ∩ B) / area(A ∪ B)

IoU = 0: boxes don't overlap at all.
IoU = 1: boxes are identical.
We suppress any box with IoU > iou_threshold against a higher-confidence box.

---------------------------------------------------------------------------
KEY DESIGN DECISION: CANONICAL TOP-LEFT CORNER
---------------------------------------------------------------------------
The assignment asks for the top-left corner of the *canonical* marker
(i.e., the corner that is top-left when the marker is held in its readable
orientation).

The classical OpenCV detector already corrects for rotation internally,
so corners[0] from detect_classical is always the canonical TL.

But our custom decoder is rotation-agnostic: we warp the spatial quad
(corners ordered TL→TR→BR→BL by image position) and rotate the bit grid
until it matches.  The `rotation` field of DecodeResult tells us how many
90° CCW rotations were applied to the extracted bits.  This maps directly
to which spatial corner is the canonical TL:

    rotation = 0  →  corners[0]  (marker is upright)
    rotation = 1  →  corners[1]  (marker rotated 90° CW in scene)
    rotation = 2  →  corners[2]  (marker rotated 180°)
    rotation = 3  →  corners[3]  (marker rotated 90° CCW in scene)

Proof sketch: if the marker is rotated 90° CW, the canonical TL physically
appears at the TR position.  The warp maps physical TR to the top-right of
the patch, so the patch bit grid looks like the codeword rotated 90° CW.
To match the codeword we rotate the bits 90° CCW (k=1), giving rotation=1.
The canonical TL is therefore the corner that landed at physical TR = corners[1].

---------------------------------------------------------------------------
PIPELINE MODES
---------------------------------------------------------------------------
The pipeline supports two modes, switchable at construction time:

  'classical'  – pure OpenCV: CLAHE + detect_classical_multiscale + cornerSubPix
                 No trained models needed.  Good baseline.  Fast.

  'enhanced'   – classical detection + CNN corner refinement
                 Requires a trained CornerRefinementCNN checkpoint.
                 Better corner accuracy, especially under blur / low contrast.

A future 'yolo' mode (Stage 2 upgrade) will swap in the YOLOv8 detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# from src.preprocess import enhance_image
from src.detect import (
    Detection,
    ARUCO_DICT,
    detect_classical,
    detect_classical_multiscale,
)
from src.corners import CornerRefinementCNN, refine_corners, refine_all
from src.decode import decode_marker, decode_all, MAX_HAMMING


# ──────────────────────────────────────────────────────────────────────────────
# Output data structure
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """One fully decoded marker detected in an image."""

    marker_id: int  # Matched ID in ARUCO_MIP_36H12 (0–249)
    top_left_x: float  # Canonical top-left corner, x pixel coordinate
    top_left_y: float  # Canonical top-left corner, y pixel coordinate
    hamming: int  # Hamming distance of the decode (0 = perfect)
    corners: np.ndarray  # Refined (4,2) corners in image coords


# ──────────────────────────────────────────────────────────────────────────────
# Non-Maximum Suppression
# ──────────────────────────────────────────────────────────────────────────────


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection-over-Union between two axis-aligned bounding boxes.

    Args:
        box_a, box_b: (x_min, y_min, x_max, y_max) tuples.

    Returns:
        IoU in [0, 1].

    WHY IoU?
    It is scale-invariant: a small box that almost perfectly overlaps another
    small box gets the same IoU as two large overlapping boxes.  This makes
    the threshold (e.g., 0.5) meaningful regardless of marker size.
    """
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    # Intersection rectangle
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    inter_w = max(0, ix1 - ix0)
    inter_h = max(0, iy1 - iy0)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def non_maximum_suppression(
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Remove duplicate/overlapping detections, keeping the most confident.

    HOW IT WORKS (greedy NMS):
    1. Sort detections by confidence (highest first).
    2. Iterate through the sorted list.  Each detection is either "kept"
       (added to the output list) or "suppressed".
    3. A detection is suppressed if its IoU with *any already-kept detection*
       exceeds iou_threshold.

    WHY GREEDY?  The optimal NMS (keeping the subset with maximum total score)
    is NP-hard.  The greedy approach runs in O(N²) — fine for N < 100
    detections per image — and produces near-optimal results in practice.

    Args:
        detections:    Unfiltered list from the detect stage.
        iou_threshold: IoU above which two boxes are considered duplicates.

    Returns:
        Filtered list with overlapping/duplicate detections removed.
    """
    if not detections:
        return []

    # Sort by confidence descending (highest confidence = most reliable).
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

    kept: list[Detection] = []
    for candidate in sorted_dets:
        suppressed = False
        for accepted in kept:
            if _iou(candidate.bbox, accepted.bbox) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)

    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Canonical top-left extraction
# ──────────────────────────────────────────────────────────────────────────────


def _canonical_top_left(
    refined_corners: np.ndarray,
    rotation: int,
) -> tuple[float, float]:
    """Return the canonical top-left corner (x, y) in image pixel coordinates.

    The refined_corners array is in spatial order: [TL, TR, BR, BL] by image
    position.  The decode rotation maps which spatial corner is the canonical
    (marker-readable) top-left.  See module docstring for the full derivation.

    Args:
        refined_corners: (4, 2) float32 — output of refine_corners().
        rotation:        decode_result.rotation — 0/1/2/3.

    Returns:
        (x, y) of the canonical top-left corner.
    """
    tl = refined_corners[rotation % 4]
    return float(tl[0]), float(tl[1])


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline class
# ──────────────────────────────────────────────────────────────────────────────


class ArUcoPipeline:
    """End-to-end ArUco marker detection and identification pipeline.

    Usage (no trained models yet — classical mode):
        pipeline = ArUcoPipeline()
        results  = pipeline.run(image)   # image is a BGR np.ndarray

    Usage (with trained corner CNN):
        from src.corners import load_corner_model
        model    = load_corner_model("models/corners/best.pth")
        pipeline = ArUcoPipeline(corner_model=model)
        results  = pipeline.run(image)
    """

    def __init__(
        self,
        corner_model: Optional[CornerRefinementCNN] = None,
        device: str = "cpu",
        max_hamming: int = MAX_HAMMING,
        nms_iou_threshold: float = 0.5,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple = (8, 8),
        use_gradient_correction: bool = False,
    ):
        """
        Args:
            corner_model:
                Trained CornerRefinementCNN.  If None, classical cornerSubPix
                is used instead.  Load with corners.load_corner_model().

            device:
                Torch device for CNN inference ('cpu', 'cuda', 'cuda:0', ...).
                Only relevant when corner_model is not None.

            max_hamming:
                Detections whose best Hamming distance exceeds this are
                discarded as fake markers.  Default = 6 (half the minimum
                inter-codeword distance of 12).  Raising it increases recall
                but allows more false positives.

            nms_iou_threshold:
                IoU threshold for Non-Maximum Suppression.  Boxes with IoU
                above this against a higher-confidence box are removed.
                0.5 is the standard choice (used in PASCAL VOC / COCO).

            clahe_clip_limit / clahe_tile_grid:
                CLAHE parameters forwarded to preprocess.enhance_image().
                See preprocess.py for rationale.

            use_gradient_correction:
                Whether to also apply the coarse gradient correction step
                in preprocess.py before CLAHE.  Useful for images with a
                strong directional light source, but adds a few ms per frame.
        """
        self.corner_model = corner_model
        self.device = device
        self.max_hamming = max_hamming
        self.nms_iou_threshold = nms_iou_threshold
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.use_gradient_correction = use_gradient_correction

    # ── Stage 1: preprocessing ────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE-based illumination normalisation.

        Returns the enhanced BGR image.  The original image is kept for
        warp_marker / cornerSubPix (which can sometimes benefit from the
        original contrast).  We pass the enhanced image to the detector,
        because that's the step most hurt by uneven lighting.
        """
        return enhance_image(
            image,
            clip_limit=self.clahe_clip_limit,
            tile_grid=self.clahe_tile_grid,
            correct_gradient=self.use_gradient_correction,
        )

    # ── Stage 2: detection ────────────────────────────────────────────────────

    def _detect(
        self,
        image_raw: np.ndarray,
        image_enhanced: np.ndarray,
    ) -> list[Detection]:
        """Run multi-scale classical detection on raw + enhanced images.

        WHY PASS BOTH?
        Running on both the original and the CLAHE-enhanced version maximises
        recall: CLAHE helps in dark regions but can over-sharpen bright ones.
        detect_classical_multiscale merges the two result sets, deduplicating
        by ID and keeping the detection with the tightest bounding box.

        The YOLO detector (future Stage 2 upgrade) will replace this method.
        Because the pipeline is structured as a class, swapping the detector
        only requires overriding _detect() — all downstream stages stay the same.
        """
        return detect_classical_multiscale(image_raw, preprocessed=image_enhanced)

    # ── Stage 3: corner refinement ────────────────────────────────────────────

    def _refine(
        self,
        image_raw: np.ndarray,
        detections: list[Detection],
    ) -> list[np.ndarray]:
        """Refine corners for every detection.

        WHY USE THE RAW IMAGE HERE (not the enhanced one)?
        cornerSubPix works best on the original gradient field.  CLAHE can
        introduce gradient artefacts at tile boundaries that confuse the
        sub-pixel solver.  The CNN was also trained on lightly augmented but
        not CLAHE-processed crops, so feeding it the raw image crop is more
        consistent with its training distribution.
        """
        return refine_all(
            image_raw,
            detections,
            model=self.corner_model,
            device=self.device,
        )

    # ── Stage 4: decoding ─────────────────────────────────────────────────────

    def _decode(
        self,
        image_raw: np.ndarray,
        corners_list: list[np.ndarray],
    ) -> list:
        """Decode each set of refined corners into a marker ID.

        WHY USE RAW IMAGE FOR WARP?
        The perspective warp in decode.warp_marker extracts a grayscale patch
        which is then normalised internally (normalize_patch).  Starting from
        the raw image and letting the patch normalisation handle contrast means
        we don't risk double-processing (CLAHE then per-patch normalisation),
        which can sometimes invert the relative ordering of pixels in a dark
        flat region.
        """
        return decode_all(image_raw, corners_list, max_hamming=self.max_hamming)

    # ── Full pipeline run ─────────────────────────────────────────────────────

    def run(self, image: np.ndarray) -> list[PipelineResult]:
        """Run the full detection + identification pipeline on one image.

        Args:
            image: BGR uint8 image (e.g. from cv2.imread).

        Returns:
            List of PipelineResult, one per successfully decoded marker.
            Empty list if no markers found or all detections rejected.
        """
        # Stage 1 — Preprocessing
        enhanced = self._preprocess(image)

        # Stage 2 — Detection
        detections = self._detect(image, enhanced)

        if not detections:
            return []

        # NMS — remove overlapping duplicates before spending time on refinement
        # WHY HERE and not after decoding?  NMS on bounding boxes is cheap
        # (O(N²) pixel comparisons).  Refinement + decoding is expensive
        # (CNN inference, homography).  Filtering early avoids wasted work.
        detections = non_maximum_suppression(detections, self.nms_iou_threshold)

        # Stage 3 — Corner refinement
        corners_list = self._refine(image, detections)

        # Stage 4 — Decoding
        decode_results = self._decode(image, corners_list)

        # Post-processing: assemble final results
        results: list[PipelineResult] = []
        for det, refined_corners, decode_res in zip(
            detections, corners_list, decode_results
        ):
            # Skip markers that didn't decode (None = Hamming > threshold)
            if decode_res is None:
                continue

            # Find the canonical top-left corner using the decode rotation.
            # If the classical detector already gave us an ID (which means
            # ArUco internally corrected for rotation), we can double-check:
            # both should agree.  We always trust our own decoder's rotation
            # for consistency across both classical and YOLO detector modes.
            tl_x, tl_y = _canonical_top_left(refined_corners, decode_res.rotation)

            results.append(
                PipelineResult(
                    marker_id=decode_res.marker_id,
                    top_left_x=tl_x,
                    top_left_y=tl_y,
                    hamming=decode_res.hamming,
                    corners=refined_corners,
                )
            )

        return results

    def run_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[PipelineResult]]:
        """Run the pipeline on a list of images.

        Each image is processed independently.  Results are returned in the
        same order as the input list.

        WHY NOT BATCH THE CNN?
        The corner CNN processes one crop at a time here.  True batching
        (stacking all crops into one tensor) would speed up GPU inference,
        but would require knowing all detections before running any refinement.
        For simplicity we keep the sequential approach; this can be optimised
        later if inference speed becomes a bottleneck.
        """
        return [self.run(img) for img in images]


# ──────────────────────────────────────────────────────────────────────────────
# Kaggle output formatting
# ──────────────────────────────────────────────────────────────────────────────


def format_prediction_string(results: list[PipelineResult]) -> str:
    """Convert a list of PipelineResults to the Kaggle submission string.

    Format: "id x y id x y ..."
    Example: "29 481.785 261.833 102 273.434 321.559"

    If results is empty the prediction string is an empty string, which
    tells the scorer there are no detections in this image.  That is the
    correct behaviour for truly empty images (giving score = 1 if the
    ground truth also has no markers, per the assignment spec).

    WHY ROUND TO 3 DECIMAL PLACES?
    The scorer computes Euclidean distance in pixels.  Three decimal places
    gives sub-pixel precision (0.001 px), which is more than sufficient
    given that the CNN itself has ~1 px accuracy.  More decimal places waste
    bandwidth without improving the score.
    """
    if not results:
        return ""

    parts = []
    for r in results:
        parts.append(f"{r.marker_id} {r.top_left_x:.3f} {r.top_left_y:.3f}")

    return " ".join(parts)
