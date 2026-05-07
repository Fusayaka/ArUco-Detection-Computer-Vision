"""
Decoding module — identifies the marker ID from a set of four refined corners.

---------------------------------------------------------------------------
THE FULL DECODING PIPELINE (this module handles steps 1–4 below)
---------------------------------------------------------------------------

    Refined corners (4 × 2 pixels)
         │
         ▼  step 1: perspective warp
    32 × 32 grayscale patch  (flat, fronto-parallel view of the marker)
         │
         ▼  step 2: normalise + threshold
    6 × 6 binary bit grid    (0 = black cell, 1 = white cell)
         │
         ▼  step 3: try 4 rotations × 250 codewords → Hamming distance
    (marker_id, rotation, hamming_distance)
         │
         ▼  step 4: reject if hamming_distance > threshold
    Decoded ID  or  None  (rejected as fake / misdetection)

---------------------------------------------------------------------------
WHY EACH STEP IS NECESSARY
---------------------------------------------------------------------------

Step 1 — Perspective warp:
    The camera sees the marker at an angle, so its image is a *trapezoid*,
    not a square.  We need to "undo" this distortion to read the bits in
    their canonical grid layout.  A full perspective (homographic) transform
    handles arbitrary tilt — affine transforms only handle rotation/scale/
    shear, which would leave residual distortion for strongly oblique views.

Step 2 — Normalise then threshold:
    Because of lighting variation the same "white" cell can appear as pixel
    value 180 in a dark image and 240 in a bright one.  If we threshold at a
    fixed value (e.g. 128) we get wrong bits in either case.  Normalising the
    patch to [0, 1] first makes the threshold 0.5 lighting-independent.

Step 3 — Try all 4 rotations:
    We detect the marker corners in ArUco convention (TL→TR→BR→BL) but we
    don't know the marker's physical orientation upfront.  A marker rotated
    90° in the scene will produce a bit grid that is also rotated 90°
    relative to the stored codeword.  Trying all 4 rotations and picking the
    minimum Hamming distance implicitly finds the correct orientation.

Step 4 — Hamming distance threshold:
    ARUCO_MIP_36H12 has a minimum *inter-marker* Hamming distance of 12.
    This means two *valid* codewords always differ in at least 12 of the 36
    bits.  So if our minimum distance is ≤ 6 (half of 12), the match is
    unambiguous — any other codeword is at least 6 bits farther away.
    Rejecting detections with distance > MAX_HAMMING weeds out fake markers
    (distractors) and truly misdetected regions.

---------------------------------------------------------------------------
DICTIONARY PRECOMPUTATION
---------------------------------------------------------------------------
At module import time we build a (250, 36) boolean NumPy array where row i
holds the 36-bit codeword for marker ID i.  This is done once and reused for
every decode call — otherwise we'd repeat the same dictionary lookup thousands
of times.  Computing Hamming distance then becomes a single vectorised XOR +
sum operation over the entire table, running in microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# from src.preprocess import normalize_patch


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# The dictionary specified by the assignment.
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36H12)

N_BITS = 6  # The marker encodes a 6×6 grid of bits (36 bits total).
N_CELLS = N_BITS + 2  # 8×8 total cells: 1-cell black border on each side.
CELL_SIZE = 4  # Pixels per cell in the warped patch.
PATCH_SIZE = N_CELLS * CELL_SIZE  # 32 × 32 pixel patch.

# Maximum Hamming distance to accept a decode as valid.  Half the minimum
# inter-codeword distance (12 // 2 = 6) is the theoretical safe threshold.
MAX_HAMMING = 6


# ──────────────────────────────────────────────────────────────────────────────
# One-time dictionary precomputation
# ──────────────────────────────────────────────────────────────────────────────

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

def _build_codeword_table() -> np.ndarray:
    """Build a (250, 36) bool array of all codewords in ARUCO_MIP_36H12.

    HOW:
    We call cv2.aruco.generateImageMarker for every marker ID at the smallest
    useful size (N_CELLS × N_CELLS = 8×8 pixels, one pixel per cell).  The
    inner 6×6 pixels (after discarding the 1-pixel black border) directly
    encode the bit pattern:  white pixel (255) → 1, black pixel (0) → 0.

    WHY generate images instead of reading bytesList directly?
    OpenCV's internal bytesList storage format packs bits into bytes in a way
    that is not publicly documented and has changed between versions.  Rendering
    a tiny marker image is a stable, version-agnostic way to extract bit
    patterns with zero guesswork.
    """
    n_markers = ARUCO_DICT.bytesList.shape[0]
    table = np.zeros((n_markers, N_BITS * N_BITS), dtype=bool)

    for marker_id in range(n_markers):
        # Generate an 8×8 image: 1 px per cell, no outer white padding.
        img = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, N_CELLS)
        # Slice the 6×6 interior, skip the 1-px black border.
        interior = img[1 : N_BITS + 1, 1 : N_BITS + 1]
        # White pixel (255) = bit 1, black pixel (0) = bit 0.
        table[marker_id] = (interior > 128).flatten()

    return table  # shape (250, 36), dtype bool


# Build once at import time — negligible cost (~5 ms).
CODEWORD_TABLE = _build_codeword_table()  # (250, 36) bool


# ──────────────────────────────────────────────────────────────────────────────
# Result data class
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DecodeResult:
    """The result of decoding a single marker patch."""

    marker_id: int  # Matched codeword index (0–249)
    hamming: int  # Minimum Hamming distance found (0 = perfect match)
    rotation: int  # 0/1/2/3 — how many 90° CCW rotations were applied
    # to the extracted bits to reach the matching codeword


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Perspective warp
# ──────────────────────────────────────────────────────────────────────────────


def warp_marker(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """Warp the marker region to a flat PATCH_SIZE × PATCH_SIZE grayscale image.

    HOW:
    cv2.getPerspectiveTransform computes the 3×3 homography matrix H that maps
    the four detected corners to the four corners of a perfect square.  Then
    cv2.warpPerspective applies H to every pixel, producing a fronto-parallel
    ("bird's-eye") view of the marker where all cells appear equally sized.

    WHY corners in this order?
    ArUco convention: corners[0]=TL, [1]=TR, [2]=BR, [3]=BL.
    We map these to the four corners of a PATCH_SIZE square so that the
    warped image has TL at (0,0).  This preserves the canonical bit ordering
    (left-to-right, top-to-bottom) used by the dictionary.

    Args:
        image:   Full BGR or grayscale image.
        corners: (4, 2) float32 array — refined corner pixel coordinates.

    Returns:
        (PATCH_SIZE, PATCH_SIZE) uint8 grayscale image.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Destination: a perfect PATCH_SIZE × PATCH_SIZE square.
    P = float(PATCH_SIZE - 1)
    dst = np.array([[0, 0], [P, 0], [P, P], [0, P]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        gray, H, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_LINEAR
    )
    return warped


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Normalise + extract 6×6 bit grid
# ──────────────────────────────────────────────────────────────────────────────


def extract_bits(warped: np.ndarray) -> np.ndarray:
    """Convert a warped marker patch to a 6×6 binary bit array using Otsu's thresholding.

    HOW:
    The patch is (N_CELLS × CELL_SIZE) × (N_CELLS × CELL_SIZE) = 32×32 pixels.
    It contains 8×8 cells of CELL_SIZE×CELL_SIZE pixels each.  The outer ring
    of cells is the black border; the inner 6×6 are the information-bearing bits.

    We apply Otsu's thresholding to automatically determine the optimal threshold
    value that minimizes intra-class variance. This is more robust than a fixed
    threshold as it adapts to varying lighting conditions without needing explicit
    normalisation.

    For each interior cell (row r, col c), we extract the CELL_SIZE×CELL_SIZE
    pixel block and sample the center pixel from the thresholded binary image.
    Sampling the center pixel is reliable because Otsu's threshold has already
    made a clean binary decision for each pixel.

    WHY Otsu's thresholding?
    Otsu's method automatically computes the threshold that best separates the
    foreground (white bits) from background (black bits) by minimizing within-class
    variance. This eliminates the need for manual normalization and is more
    adaptive to different lighting conditions than fixed-value thresholding.

    Args:
        warped: (PATCH_SIZE, PATCH_SIZE) uint8 grayscale, output of warp_marker.

    Returns:
        (N_BITS, N_BITS) bool array — True = white cell (bit 1).
    """
    # Apply Otsu's thresholding to automatically determine optimal threshold
    _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bits = np.zeros((N_BITS, N_BITS), dtype=bool)
    for r in range(N_BITS):
        for c in range(N_BITS):
            # Interior cell (r, c) starts at pixel row/col (r+1)*CELL_SIZE.
            r0 = (r + 1) * CELL_SIZE
            c0 = (c + 1) * CELL_SIZE
            # Sample the center pixel of the cell from the binary image
            center_r = r0 + CELL_SIZE // 2
            center_c = c0 + CELL_SIZE // 2
            # 255 (white) = bit 1, 0 (black) = bit 0
            bits[r, c] = binary[center_r, center_c] > 128

    return bits


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Hamming distance matching with 4-rotation search
# ──────────────────────────────────────────────────────────────────────────────


def _hamming_to_table(bits_flat: np.ndarray) -> np.ndarray:
    """Compute Hamming distance from bits_flat to every row in CODEWORD_TABLE.

    WHY vectorised XOR?
    A Python loop over 250 codewords would work but is slow.  Broadcasting
    the (36,) query array against the (250, 36) table gives a (250, 36) bool
    difference matrix in one NumPy call; summing each row gives distances.
    This runs in ~5 µs vs ~250 µs for a Python loop — 50× faster.

    Args:
        bits_flat: (36,) bool array (the extracted bit grid, flattened).

    Returns:
        (250,) int array — Hamming distance to each codeword.
    """
    diff = CODEWORD_TABLE ^ bits_flat  # (250, 36) bool
    return diff.sum(axis=1)  # (250,) int


def match_codeword(bits_2d: np.ndarray) -> tuple[int, int, int]:
    """Find the best-matching codeword across all 4 rotations.

    HOW:
    np.rot90(bits_2d, k) rotates the 6×6 grid by k × 90° counterclockwise.
    Rotating the *extracted* bits is equivalent to rotating the physical
    marker: a marker oriented 90° clockwise in the scene produces bits that
    look 90° counterclockwise relative to the stored codeword, so rotating
    the bits 90° CCW aligns them.

    We compute Hamming distances for all 4 rotations simultaneously, then
    find the global minimum.  The rotation index with that minimum tells us
    the marker's physical orientation (useful for downstream pose estimation,
    though the assignment only asks for the top-left corner).

    Args:
        bits_2d: (N_BITS, N_BITS) bool array from extract_bits().

    Returns:
        (marker_id, hamming_distance, rotation_k) where rotation_k is the
        number of 90° CCW rotations applied (0, 1, 2, or 3).
    """
    best_id = -1
    best_dist = 999
    best_rot = 0

    for k in range(4):
        rotated = np.rot90(bits_2d, k=k)  # rotate k × 90° CCW
        distances = _hamming_to_table(rotated.flatten())
        min_dist = int(distances.min())
        min_id = int(distances.argmin())

        if min_dist < best_dist:
            best_dist = min_dist
            best_id = min_id
            best_rot = k

    return best_id, best_dist, best_rot


# ──────────────────────────────────────────────────────────────────────────────
# Steps 1–4 combined: public decoding API
# ──────────────────────────────────────────────────────────────────────────────


def decode_marker(
    image: np.ndarray,
    corners: np.ndarray,
    max_hamming: int = MAX_HAMMING,
) -> Optional[DecodeResult]:
    """Decode the marker ID from a full image and four refined corner coords.

    This is the main entry point for the decoding stage.  It chains:
        warp_marker → extract_bits → match_codeword → threshold check

    Args:
        image:       Full BGR or grayscale image.
        corners:     (4, 2) float32 — refined corner pixel coordinates in
                     ArUco order (TL, TR, BR, BL).
        max_hamming: Reject the detection if the best Hamming distance exceeds
                     this.  Default 6 = half the minimum inter-marker distance.

    Returns:
        DecodeResult if a valid codeword is found within max_hamming, else None.
    """
    # Step 1: warp to flat patch.
    warped = warp_marker(image, corners)

    # Step 2: extract binary bit grid.
    bits_2d = extract_bits(warped)

    # Step 3: find nearest codeword across all 4 rotations.
    marker_id, hamming, rotation = match_codeword(bits_2d)

    # Step 4: reject if the match is too weak (likely a fake marker).
    if hamming > max_hamming:
        return None

    return DecodeResult(marker_id=marker_id, hamming=hamming, rotation=rotation)


def decode_all(
    image: np.ndarray,
    corners_list: list[np.ndarray],
    max_hamming: int = MAX_HAMMING,
) -> list[Optional[DecodeResult]]:
    """Decode every set of corners in a list.

    Args:
        image:        Full BGR image.
        corners_list: List of (4, 2) float32 arrays from the refinement stage.
        max_hamming:  Rejection threshold.

    Returns:
        List of DecodeResult or None, one per entry in corners_list.
        None entries correspond to rejected detections (fake markers / noise).
    """
    return [decode_marker(image, corners, max_hamming) for corners in corners_list]
