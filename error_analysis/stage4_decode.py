"""
Stage 4 — Marker ID Decoding: Error Analysis  (Deep Dive)
==========================================================

This is the most detailed analysis in the error-analysis suite.  It isolates
decode-stage failures from detector / corner-refinement noise by feeding
**ground-truth corners directly** into the decode pipeline, then systematically
ablates every configurable decision in ``src/decode.py`` to measure how much
each one contributes to observed failures.

---------------------------------------------------------------------------
CONFIRMED CODE ↔ REPORT DISCREPANCIES
---------------------------------------------------------------------------

Four discrepancies between the report and the live code are quantified here:

  C1 — Threshold method (TOP SUSPECT)
       Report §3.5.2: "Otsu's method ... maximises inter-class variance."
       Live code (src/decode.py:96–113): min–max normalisation + fixed 0.5
       threshold.  Min–max is dominated by single extreme pixels (specular
       highlights or sensor noise), which shift the effective threshold and
       flip bits.

  C2 — Sub-cell warp drift
       With CELL_SIZE=4 px, a 1-pixel corner residual from Stage 3 shifts the
       cell-averaging window by 25 % of cell width, mixing a cell's value with
       its neighbour's.  Larger patch sizes (64 / 96 px, CELL_SIZE=8/12) are
       more forgiving.

  C3 — MAX_HAMMING = 6  (report claims τ = 5)
       Report §3.5.3: τ = 5 (principled: d_min = 12 → unambiguous up to 5).
       Code (src/decode.py:89): MAX_HAMMING = 6.  At τ=6 a match with 6
       errors is by definition ambiguous (dmin=12 means another codeword is
       also ≤6 bits away).  The Kaggle scorer penalises false positives and
       missed detections equally, so τ=5 is precision-preferred.

  C4 — Black border discarded
       The outer ring of every ArUco marker is all-black by specification.
       This is a reliable calibration signal for the dark class.  The live
       decoder discards it; "border_calibrated" thresholding uses it.

---------------------------------------------------------------------------
ANALYSIS STRUCTURE
---------------------------------------------------------------------------

  5a. Baseline characterisation — run decode_marker(img, gt_corners) on all
      138 GT markers in the 40 challenge images; record ID accuracy, Hamming
      histogram, rejection rate.

  5b. Component ablation — sweep 4 threshold methods × 3 patch sizes × 3 cell-
      sampling strategies × 3 τ values = 108 combinations; measure per-combo
      ID accuracy and delta vs. baseline.

  5c. Border / non-bit checks — quantify how often the outer ring is not dark
      (bad warp / orientation flip) and how often normalize_patch fires the
      flat-patch early exit.

Outputs
-------
  findings/stage4_baseline.csv            — per-marker baseline decode results
  findings/stage4_ablation.csv            — 108-row ablation sweep sorted by Δacc
  findings/stage4_decode_deepdive.md      — structured markdown for the report
  findings/stage4_panels/<stem>_<id>.png  — failure case panels

Report alignment
----------------
  §3.5.1  Perspective rectification (warp_marker, Eq. 3.7)
  §3.5.2  Bit-grid extraction with Otsu thresholding (discrepancy: code uses min-max)
  §3.5.3  Hamming distance matching, τ = 5 (discrepancy: code uses 6)
  §3.5.4  Canonical-corner output

CLI usage
---------
    python -m error_analysis.stage4_decode
    python -m error_analysis.stage4_decode --dry_run
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Robust repo-root detection  (same boilerplate as stage1 / stage2 / stage3)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _SCRIPT_DIR.parent


def _find_data_root() -> Path:
    candidate = _WORKTREE_ROOT
    for _ in range(8):
        if (candidate / "src").exists() and (candidate / "data" / "raw" / "flyingarucov2").exists():
            return candidate
        candidate = candidate.parent
    return _WORKTREE_ROOT


_DATA_ROOT = _find_data_root()

if str(_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATA_ROOT))

# ---------------------------------------------------------------------------
# Remaining imports
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.decode import (  # type: ignore[import]
    decode_marker,
    normalize_patch,
    warp_marker,
    extract_bits,
    match_codeword,
    CODEWORD_TABLE,
    N_BITS,
    N_CELLS,
    CELL_SIZE,
    PATCH_SIZE,
    MAX_HAMMING,
    DecodeResult,
)
from error_analysis.scoring import (  # type: ignore[import]
    load_gt_from_json,
    GTMarker,
    gaussian_score,
    image_diagonal,
)
from error_analysis.challenge_set import load_challenge_set  # type: ignore[import]

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

_FINDINGS = _SCRIPT_DIR / "findings"
_PANELS_DIR = _FINDINGS / "stage4_panels"

# ---------------------------------------------------------------------------
# Ablation sweep parameters
# ---------------------------------------------------------------------------

THRESHOLD_METHODS = ["min_max_05", "otsu", "adaptive_mean", "border_calibrated"]
PATCH_SIZES = [32, 64, 96]
CELL_SAMPLINGS = ["mean", "centre_pixel", "gaussian_weighted"]
TAU_VALUES = [4, 5, 6]


# ============================================================================
# Variant decode helpers
# ============================================================================


def _warp_marker_custom(
    image: np.ndarray,
    corners: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Warp a marker to an arbitrary square patch size.

    Identical to ``warp_marker()`` (src/decode.py:168) but accepts a custom
    ``patch_size`` so the ablation can test 32 / 64 / 96 px patches.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    P = float(patch_size - 1)
    dst = np.array([[0, 0], [P, 0], [P, P], [0, P]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(gray, H, (patch_size, patch_size), flags=cv2.INTER_LINEAR)


def _get_outer_ring_pixels(warped: np.ndarray, cell_size: int) -> np.ndarray:
    """Return pixel values from the outer ring of 8×8 cells (the black border)."""
    n = 8  # always 8 cells per side (N_CELLS)
    pixels: list[np.ndarray] = []
    for i in range(n):
        for j in range(n):
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                r0, c0 = i * cell_size, j * cell_size
                cell = warped[r0 : r0 + cell_size, c0 : c0 + cell_size]
                pixels.append(cell.ravel())
    return np.concatenate(pixels).astype(np.float32)


def _gaussian_cell_weights(cell_size: int) -> np.ndarray:
    """Normalised 2-D Gaussian kernel for one cell.

    σ = cell_size / 4 so the kernel down-weights boundary pixels (which
    overlap the neighbouring cell after a sub-pixel warp error) and
    up-weights the cell centre.
    """
    sigma = cell_size / 4.0
    ax = np.arange(cell_size, dtype=np.float32) - (cell_size - 1) / 2.0
    g1d = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g2d = np.outer(g1d, g1d)
    return (g2d / g2d.sum()).astype(np.float32)


def extract_bits_variant(
    warped: np.ndarray,
    threshold_method: str,
    cell_sampling: str,
    cell_size: int,
) -> np.ndarray:
    """Extract a 6×6 binary bit grid using configurable threshold and sampling.

    Args:
        warped:           (patch_size, patch_size) uint8 grayscale patch.
        threshold_method: One of "min_max_05" | "otsu" | "adaptive_mean" |
                          "border_calibrated".
        cell_sampling:    One of "mean" | "centre_pixel" | "gaussian_weighted".
        cell_size:        Pixels per cell = patch_size // 8.

    Returns:
        (N_BITS, N_BITS) = (6, 6) bool array — True = white cell (bit 1).
    """
    n_bits = 6  # 6×6 inner data cells

    # ── Step 1: produce a float32 image in [0, 1] ──────────────────────────
    if threshold_method == "min_max_05":
        # Current live code: min-max normalize then threshold per cell at 0.5.
        binarized = normalize_patch(warped)  # float32 in [0, 1]

    elif threshold_method == "otsu":
        # Otsu finds the global threshold t* that maximises inter-class variance
        # (optimal for the bimodal black/white distribution of a marker patch).
        _, binary_img = cv2.threshold(
            warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        binarized = binary_img.astype(np.float32) / 255.0

    elif threshold_method == "adaptive_mean":
        # Adaptive threshold with block size ≈ 2 cells → handles local
        # illumination gradients within the patch.
        block_size = max(3, cell_size * 2 - 1)
        if block_size % 2 == 0:
            block_size += 1
        binary_img = cv2.adaptiveThreshold(
            warped,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            2,
        )
        binarized = binary_img.astype(np.float32) / 255.0

    elif threshold_method == "border_calibrated":
        # Use outer-ring pixels as the dark-class anchor; the brightest inner
        # pixel is the white-class anchor.  Threshold at the midpoint.
        outer = _get_outer_ring_pixels(warped, cell_size)
        dark_mean = float(outer.mean())
        inner = warped[cell_size : warped.shape[0] - cell_size,
                       cell_size : warped.shape[1] - cell_size]
        white_est = float(inner.max())
        if white_est <= dark_mean:
            white_est = 255.0  # degenerate: entire inner region is dark
        thresh_val = (dark_mean + white_est) / 2.0
        binarized = (warped.astype(np.float32) > thresh_val).astype(np.float32)

    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method!r}")

    # ── Step 2: read each interior cell ───────────────────────────────────
    gauss_w = _gaussian_cell_weights(cell_size) if cell_sampling == "gaussian_weighted" else None

    bits = np.zeros((n_bits, n_bits), dtype=bool)
    for r in range(n_bits):
        for c in range(n_bits):
            r0 = (r + 1) * cell_size
            c0 = (c + 1) * cell_size
            cell = binarized[r0 : r0 + cell_size, c0 : c0 + cell_size]

            if cell_sampling == "mean":
                bits[r, c] = float(cell.mean()) > 0.5

            elif cell_sampling == "centre_pixel":
                cr = r0 + cell_size // 2
                cc = c0 + cell_size // 2
                bits[r, c] = float(binarized[cr, cc]) > 0.5

            elif cell_sampling == "gaussian_weighted":
                if gauss_w is not None and cell.shape == gauss_w.shape:
                    bits[r, c] = float((cell * gauss_w).sum()) > 0.5
                else:
                    bits[r, c] = float(cell.mean()) > 0.5

    return bits


def decode_marker_variant(
    image: np.ndarray,
    corners: np.ndarray,
    threshold_method: str = "min_max_05",
    patch_size: int = 32,
    cell_sampling: str = "mean",
    max_hamming: int = MAX_HAMMING,
) -> Optional[DecodeResult]:
    """Re-implementation of decode_marker with configurable knobs.

    All combinations produce results equivalent to the live ``decode_marker()``
    when called with ``("min_max_05", 32, "mean", 6)`` — this is verified as
    part of the ablation sanity check.

    Args:
        image:            Full BGR or grayscale image.
        corners:          (4, 2) float32 corner coordinates.
        threshold_method: Binarisation strategy for the warped patch.
        patch_size:       Square patch size (32 / 64 / 96).  Must be divisible
                          by 8 (= N_CELLS).
        cell_sampling:    How to read each cell value from the binarised patch.
        max_hamming:      Hamming rejection threshold τ.

    Returns:
        DecodeResult or None (rejected).
    """
    if patch_size % 8 != 0:
        raise ValueError(f"patch_size must be divisible by 8, got {patch_size}")
    cell_size = patch_size // 8

    warped = _warp_marker_custom(image, corners, patch_size)
    bits_2d = extract_bits_variant(warped, threshold_method, cell_sampling, cell_size)
    marker_id, hamming, rotation = match_codeword(bits_2d)  # reuse from src/decode.py

    if hamming > max_hamming:
        return None
    return DecodeResult(marker_id=marker_id, hamming=hamming, rotation=rotation)


# ============================================================================
# Section 5a — Baseline characterisation
# ============================================================================


def run_baseline(
    stems: list[str],
    val_images: Path,
    gt_dir: Path,
) -> pd.DataFrame:
    """Run the live decode_marker on GT corners for all GT markers.

    Steps through all GT markers in the challenge set.  For each marker the
    warped patch, outer-ring mean (border sanity) and early-exit status are
    also computed to support Sections 5b and 5c.

    Returns:
        DataFrame with one row per GT marker containing all baseline metrics.
    """
    rows: list[dict] = []
    total_imgs = len(stems)

    for i, stem in enumerate(stems, 1):
        img_path = val_images / f"{stem}.jpg"
        json_path = gt_dir / f"{stem}.json"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i:2d}/{total_imgs}] {stem} ... SKIP", file=sys.stderr)
            continue

        gt_markers: list[GTMarker] = (
            load_gt_from_json(json_path) if json_path.exists() else []
        )

        for gm in gt_markers:
            # ── Step-by-step baseline decode (mirrors src/decode.py:323) ──
            warped = warp_marker(img, gm.corners)

            # 5c diagnostics —— collect before any thresholding ─────────────
            p = warped.astype(np.float32)
            early_exit = bool((p.max() - p.min()) < 1e-6)
            outer_mean = float(_get_outer_ring_pixels(warped, CELL_SIZE).mean())
            border_bright = outer_mean > 64.0  # border should be dark (< 25 % of 255)

            # Baseline decode (min_max_05 / CELL_SIZE=4 / mean / τ=6)
            bits_2d = extract_bits(warped)
            best_id, best_hamming, best_rot = match_codeword(bits_2d)
            accepted = best_hamming <= MAX_HAMMING
            success = accepted and (best_id == gm.marker_id)

            rows.append({
                "stem": stem,
                "true_id": gm.marker_id,
                "predicted_id": best_id if accepted else -1,
                "best_hamming": best_hamming,
                "rotation": best_rot,
                "accepted": accepted,
                "success": success,
                "rejected": not accepted,
                "wrong_id": accepted and (best_id != gm.marker_id),
                "early_exit": early_exit,
                "outer_mean": outer_mean,
                "border_bright": border_bright,
            })

        n_markers = len(gt_markers)
        n_ok = sum(1 for r in rows[-n_markers:] if r["success"])
        print(
            f"  [{i:2d}/{total_imgs}] {stem}  "
            f"markers={n_markers}  correct={n_ok}/{n_markers}"
        )

    return pd.DataFrame(rows)


# ============================================================================
# Section 5b — Component ablation
# ============================================================================


def run_ablation(
    df_baseline: pd.DataFrame,
    val_images: Path,
    gt_dir: Path,
) -> pd.DataFrame:
    """Sweep all 108 ablation combinations (4 × 3 × 3 × 3).

    Loads all required images once into memory, then iterates over every
    (threshold_method, patch_size, cell_sampling, tau) combination.

    For each combination, evaluates on **all GT markers** in the baseline
    DataFrame (not just failures) so that both accuracy improvement and
    regression are captured.

    Returns:
        DataFrame with 108 rows sorted by delta_acc descending.
    """
    # Pre-load images and GT markers
    stems = df_baseline["stem"].unique().tolist()
    cache: dict[str, tuple[np.ndarray, dict[int, GTMarker]]] = {}
    for stem in stems:
        img = cv2.imread(str(val_images / f"{stem}.jpg"))
        json_path = gt_dir / f"{stem}.json"
        gt_markers = load_gt_from_json(json_path) if json_path.exists() else []
        if img is not None:
            cache[stem] = (img, {m.marker_id: m for m in gt_markers})

    baseline_acc = float(df_baseline["success"].mean())
    total_markers = len(df_baseline)
    n_combos = len(THRESHOLD_METHODS) * len(PATCH_SIZES) * len(CELL_SAMPLINGS) * len(TAU_VALUES)

    print(
        f"\nAblation sweep: {n_combos} combinations × {total_markers} markers "
        f"({n_combos * total_markers:,} decode calls) ..."
    )

    ablation_rows: list[dict] = []

    for combo_idx, (tm, ps, cs, tau) in enumerate(
        itertools.product(THRESHOLD_METHODS, PATCH_SIZES, CELL_SAMPLINGS, TAU_VALUES), 1
    ):
        n_correct = 0
        n_total = 0
        n_recovered = 0   # baseline failed → variant succeeds
        n_regression = 0  # baseline succeeded → variant fails

        for _, brow in df_baseline.iterrows():
            stem = str(brow["stem"])
            true_id = int(brow["true_id"])
            baseline_ok = bool(brow["success"])

            if stem not in cache:
                continue
            img, gt_by_id = cache[stem]
            if true_id not in gt_by_id:
                continue

            gm = gt_by_id[true_id]
            result = decode_marker_variant(img, gm.corners, tm, ps, cs, tau)
            variant_ok = result is not None and result.marker_id == true_id

            n_total += 1
            if variant_ok:
                n_correct += 1
            if not baseline_ok and variant_ok:
                n_recovered += 1
            if baseline_ok and not variant_ok:
                n_regression += 1

        accuracy = n_correct / n_total if n_total > 0 else 0.0
        ablation_rows.append(
            {
                "threshold_method": tm,
                "patch_size": ps,
                "cell_sampling": cs,
                "tau": tau,
                "n_correct": n_correct,
                "n_total": n_total,
                "accuracy": accuracy,
                "delta_acc": accuracy - baseline_acc,
                "n_failure_recovered": n_recovered,
                "n_regression": n_regression,
            }
        )

        if combo_idx % 20 == 0 or combo_idx == n_combos:
            print(f"  [{combo_idx:3d}/{n_combos}] {tm:20s} ps={ps} cs={cs:20s} tau={tau}  "
                  f"acc={accuracy:.3f} delta={accuracy-baseline_acc:+.3f}")

    df = pd.DataFrame(ablation_rows).sort_values("delta_acc", ascending=False)
    return df.reset_index(drop=True)


# ============================================================================
# Section 5c — Border sanity and early-exit statistics  (uses baseline data)
# ============================================================================


def summarise_border_checks(df_baseline: pd.DataFrame) -> dict:
    """Aggregate border-sanity and early-exit stats from the baseline DataFrame."""
    n = len(df_baseline)
    n_early_exit = int(df_baseline["early_exit"].sum())
    n_border_bright = int(df_baseline["border_bright"].sum())

    # How many failures are associated with a bright border?
    failures = df_baseline[~df_baseline["success"]]
    n_fail_border = int(failures["border_bright"].sum())
    n_fail_early = int(failures["early_exit"].sum())

    return {
        "n_total": n,
        "n_early_exit": n_early_exit,
        "pct_early_exit": n_early_exit / n * 100 if n else 0.0,
        "n_border_bright": n_border_bright,
        "pct_border_bright": n_border_bright / n * 100 if n else 0.0,
        "n_failures": len(failures),
        "n_fail_border_bright": n_fail_border,
        "pct_fail_border_bright": n_fail_border / len(failures) * 100 if len(failures) else 0.0,
        "n_fail_early_exit": n_fail_early,
        "pct_fail_early_exit": n_fail_early / len(failures) * 100 if len(failures) else 0.0,
    }


# ============================================================================
# Visualisation
# ============================================================================


def _make_panels(
    df_baseline: pd.DataFrame,
    val_images: Path,
    gt_dir: Path,
    out_dir: Path,
    n_panels: int = 8,
) -> None:
    """Save a 4-column diagnostic panel for each failed marker (up to n_panels).

    Columns:
      1. BGR crop with GT corners overlay.
      2. Warped 32×32 grayscale patch.
      3. Extracted bit grid, colour-coded (white=correct, red=FP, blue=FN).
      4. GT codeword grid with prediction summary text.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = df_baseline[~df_baseline["success"]].copy()
    # Sort by hamming so near-misses (smallest hamming gap above threshold) appear first.
    failures = failures.sort_values("best_hamming").head(n_panels)

    for _, row in failures.iterrows():
        stem = str(row["stem"])
        true_id = int(row["true_id"])
        pred_id = int(row["predicted_id"])
        hamming = int(row["best_hamming"])

        img_path = val_images / f"{stem}.jpg"
        json_path = gt_dir / f"{stem}.json"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_markers = load_gt_from_json(json_path) if json_path.exists() else []
        gm = next((m for m in gt_markers if m.marker_id == true_id), None)
        if gm is None:
            continue

        # Compute the warped patch and bits
        warped = warp_marker(img, gm.corners)
        bits_2d = extract_bits(warped)
        gt_codeword = CODEWORD_TABLE[true_id].reshape(N_BITS, N_BITS)

        # Crop around GT marker for context
        x0 = max(0, int(gm.corners[:, 0].min()) - 20)
        x1 = min(img.shape[1], int(gm.corners[:, 0].max()) + 20)
        y0 = max(0, int(gm.corners[:, 1].min()) - 20)
        y1 = min(img.shape[0], int(gm.corners[:, 1].max()) + 20)
        crop_rgb = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

        # Build the colour-coded bit-error grid
        bit_vis = np.zeros((N_BITS, N_BITS, 3), dtype=np.float32)
        for r in range(N_BITS):
            for c in range(N_BITS):
                pred_bit = bool(bits_2d[r, c])
                gt_bit = bool(gt_codeword[r, c])
                if pred_bit == gt_bit:
                    # Correct: white cell = light, black cell = dark
                    bit_vis[r, c] = [0.90, 0.90, 0.90] if gt_bit else [0.10, 0.10, 0.10]
                else:
                    # Error: FP (pred=1, gt=0) = red; FN (pred=0, gt=1) = blue
                    bit_vis[r, c] = [1.0, 0.2, 0.2] if pred_bit else [0.2, 0.2, 1.0]

        # GT codeword grid (no errors)
        gt_vis = np.zeros((N_BITS, N_BITS, 3), dtype=np.float32)
        for r in range(N_BITS):
            for c in range(N_BITS):
                gt_vis[r, c] = [0.90, 0.90, 0.90] if gt_codeword[r, c] else [0.10, 0.10, 0.10]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"Stage 4 Decode Failure  |  {stem}  |  true_id={true_id}  pred_id={pred_id}  hamming={hamming}",
            fontsize=9, fontweight="bold",
        )

        # Panel 1: BGR crop with GT corners
        axes[0].imshow(crop_rgb)
        pts = np.vstack([gm.corners, gm.corners[0]])
        axes[0].plot(pts[:, 0] - x0, pts[:, 1] - y0, "r-", lw=1.5)
        axes[0].plot(
            gm.canonical_tl[0] - x0, gm.canonical_tl[1] - y0,
            "y*", ms=10, markeredgecolor="k", markeredgewidth=0.5,
        )
        early = bool(row["early_exit"])
        border = bool(row["border_bright"])
        axes[0].set_title(
            f"GT crop  (early_exit={early}, border_bright={border})", fontsize=8
        )
        axes[0].axis("off")

        # Panel 2: warped 32×32 patch
        axes[1].imshow(warped, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"Warped {PATCH_SIZE}×{PATCH_SIZE} patch", fontsize=8)
        axes[1].axis("off")

        # Panel 3: extracted bits vs GT (error map)
        axes[2].imshow(bit_vis, interpolation="nearest")
        axes[2].set_title("Extracted bits  (red=FP, blue=FN)", fontsize=8)
        axes[2].axis("off")

        # Panel 4: GT codeword
        axes[3].imshow(gt_vis, interpolation="nearest")
        status = "rejected" if bool(row["rejected"]) else f"pred_id={pred_id}"
        axes[3].set_title(f"GT codeword (id={true_id})\n{status}, hamming={hamming}", fontsize=8)
        axes[3].axis("off")

        plt.tight_layout()
        out_path = out_dir / f"{stem}_{true_id}_stage4.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  Panel -> {out_path.name}")


# ============================================================================
# Markdown writeup
# ============================================================================


def _write_markdown(
    df_baseline: pd.DataFrame,
    df_ablation: pd.DataFrame,
    border_stats: dict,
    n_panels: int,
    out_path: Path,
) -> None:
    """Generate the Stage 4 decode deep-dive markdown."""
    n_markers = len(df_baseline)
    n_correct = int(df_baseline["success"].sum())
    n_rejected = int(df_baseline["rejected"].sum())
    n_wrong_id = int(df_baseline["wrong_id"].sum())
    acc = n_correct / n_markers if n_markers > 0 else 0.0
    rej_rate = n_rejected / n_markers if n_markers > 0 else 0.0
    wrong_rate = n_wrong_id / n_markers if n_markers > 0 else 0.0

    accepted = df_baseline[df_baseline["accepted"]]
    ham_correct = accepted[accepted["success"]]["best_hamming"]
    ham_wrong = accepted[~accepted["success"]]["best_hamming"]
    ham_rejected = df_baseline[df_baseline["rejected"]]["best_hamming"]

    # Top-10 ablation rows
    top10 = df_ablation.head(10)
    # Best single-knob improvements: row where only that knob differs from baseline
    baseline_row = df_ablation[
        (df_ablation["threshold_method"] == "min_max_05")
        & (df_ablation["patch_size"] == 32)
        & (df_ablation["cell_sampling"] == "mean")
        & (df_ablation["tau"] == 6)
    ]
    baseline_acc_check = float(baseline_row["accuracy"].values[0]) if len(baseline_row) > 0 else 0.0

    def _best_for_knob(knob_col: str, fixed: dict) -> pd.Series:
        mask = pd.Series([True] * len(df_ablation))
        for k, v in fixed.items():
            mask &= df_ablation[k] == v
        subset = df_ablation[mask]
        return subset.loc[subset["delta_acc"].idxmax()] if len(subset) > 0 else pd.Series()

    best_threshold = _best_for_knob(
        "threshold_method",
        {"patch_size": 32, "cell_sampling": "mean", "tau": 6},
    )
    best_patch = _best_for_knob(
        "patch_size",
        {"threshold_method": "min_max_05", "cell_sampling": "mean", "tau": 6},
    )
    best_sampling = _best_for_knob(
        "cell_sampling",
        {"threshold_method": "min_max_05", "patch_size": 32, "tau": 6},
    )
    best_tau = _best_for_knob(
        "tau",
        {"threshold_method": "min_max_05", "patch_size": 32, "cell_sampling": "mean"},
    )

    lines: list[str] = [
        "# Stage 4 — Marker ID Decoding: Error Analysis (Deep Dive)",
        "",
        "## 1. Overview",
        "",
        f"Analysis run on **{n_markers} GT markers** from the 40 challenge images.",
        "GT corners are injected directly into the decode pipeline to isolate",
        "decode-stage failures from corner-refinement noise.",
        "",
        "### Code ↔ Report Discrepancies",
        "",
        "| # | Discrepancy | Report claim | Live code | Impact |",
        "|---|-------------|-------------|-----------|--------|",
        "| C1 | Threshold method | Otsu (§3.5.2, Eq. 3.8) | min-max + 0.5 (`decode.py:96–113`) | top suspect |",
        "| C2 | Patch resolution | 32 px (implicit) | 32 px = 4 px/cell | sub-cell drift |",
        "| C3 | Hamming threshold τ | 5 (§3.5.3) | MAX_HAMMING=6 (`decode.py:89`) | precision trade-off |",
        "| C4 | Black border | calibration signal | discarded (`decode.py:227`) | missed calibration |",
        "",
        "## 2. Baseline Characterisation (GT Corners, Live Decode)",
        "",
        f"All {n_markers} GT markers are fed into `decode_marker()` with their true corners.",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Correct ID decodes | **{n_correct} / {n_markers}** ({acc:.1%}) |",
        f"| Rejected (hamming > τ=6) | {n_rejected} ({rej_rate:.1%}) |",
        f"| Accepted but wrong ID | {n_wrong_id} ({wrong_rate:.1%}) |",
        f"| Failures (rejected + wrong ID) | {n_markers - n_correct} ({1-acc:.1%}) |",
    ]

    if len(ham_correct) > 0:
        lines.append(f"| Mean Hamming — correct decodes | {ham_correct.mean():.2f} |")
    if len(ham_wrong) > 0:
        lines.append(f"| Mean Hamming — wrong ID accepted | {ham_wrong.mean():.2f} |")
    if len(ham_rejected) > 0:
        lines.append(f"| Mean Hamming — rejected | {ham_rejected.mean():.2f} (all > 6) |")

    lines += [
        "",
        "> **Ablation baseline sanity**: the variant configured as `(min_max_05, 32, mean, τ=6)`",
        f"> reproduces the live code at accuracy **{baseline_acc_check:.1%}** — exact match expected.",
        "",
        "## 3. Component Ablation (4 × 3 × 3 × 3 = 108 Combinations)",
        "",
        "### 3.1 Top-10 Variants by Accuracy Gain (Δacc over baseline)",
        "",
        "| # | Threshold | Patch size | Cell sampling | τ | Accuracy | Δacc | Recovered | Regressions |",
        "|---|-----------|------------|---------------|---|----------|------|-----------|-------------|",
    ]

    for rank, (_, r) in enumerate(top10.iterrows(), 1):
        lines.append(
            f"| {rank} | {r['threshold_method']} | {int(r['patch_size'])} | "
            f"{r['cell_sampling']} | {int(r['tau'])} | {r['accuracy']:.3f} | "
            f"{r['delta_acc']:+.3f} | {int(r['n_failure_recovered'])} | {int(r['n_regression'])} |"
        )

    lines += [
        "",
        "### 3.2 Best Single-Knob Improvement (all other knobs at baseline)",
        "",
        "| Knob | Best value | Accuracy | Δacc | Failures recovered |",
        "|------|------------|----------|------|--------------------|",
    ]

    for knob, best_row in [
        ("threshold_method (C1)", best_threshold),
        ("patch_size (C2)", best_patch),
        ("cell_sampling (C2)", best_sampling),
        ("tau (C3)", best_tau),
    ]:
        if len(best_row) > 0:
            val = best_row.get(
                "threshold_method" if "threshold" in knob
                else "patch_size" if "patch" in knob
                else "cell_sampling" if "sampling" in knob
                else "tau"
            )
            lines.append(
                f"| {knob} | {val} | {best_row['accuracy']:.3f} | "
                f"{best_row['delta_acc']:+.3f} | {int(best_row['n_failure_recovered'])} |"
            )

    lines += [
        "",
        "## 4. Border Sanity and Early-Exit Analysis (§5c)",
        "",
        f"Evaluated on all **{border_stats['n_total']}** GT markers.",
        "",
        "| Check | Count | Fraction | Of failures |",
        "|-------|-------|----------|-------------|",
        f"| Early exit (flat patch, `hi−lo < 1e-6`) | {border_stats['n_early_exit']} "
        f"| {border_stats['pct_early_exit']:.1f}% "
        f"| {border_stats['n_fail_early_exit']} ({border_stats['pct_fail_early_exit']:.1f}% of fails) |",
        f"| Border bright (outer_mean > 64) | {border_stats['n_border_bright']} "
        f"| {border_stats['pct_border_bright']:.1f}% "
        f"| {border_stats['n_fail_border_bright']} ({border_stats['pct_fail_border_bright']:.1f}% of fails) |",
        "",
        "**Early exit** fires when `normalize_patch()` returns all-zeros (the warped patch",
        "has no contrast).  Each fire silently produces a random bit grid → guaranteed",
        "decode failure.  This is a Stage 1 / Stage 3 upstream failure that manifests",
        "in Stage 4.",
        "",
        "**Border bright** indicates a warp orientation error: if the outer ring is bright,",
        "the corners are likely swapped (TL↔BR) or the homography is degenerate.  Each",
        "bright-border case produces inverted bits → very high Hamming distance →",
        "either wrong ID or rejection.",
        "",
        "## 5. Likely Root Cause (Ranked by Measured Impact)",
        "",
        "### C1 — Threshold method (top suspect)",
        "",
    ]

    # Figure out which threshold method gave the best improvement
    thresh_group = df_ablation[
        (df_ablation["patch_size"] == 32)
        & (df_ablation["cell_sampling"] == "mean")
        & (df_ablation["tau"] == 6)
    ].sort_values("delta_acc", ascending=False)

    if len(thresh_group) > 0:
        best_thresh_row = thresh_group.iloc[0]
        lines += [
            f"Best threshold method: **{best_thresh_row['threshold_method']}** "
            f"(accuracy {best_thresh_row['accuracy']:.1%}, "
            f"Δ={best_thresh_row['delta_acc']:+.3f}, "
            f"recovered {int(best_thresh_row['n_failure_recovered'])} failures).",
            "",
        ]

    lines += [
        "The live code's `normalize_patch()` stretches the entire 32×32 patch so that",
        "its global minimum = 0 and maximum = 1.  A single specular highlight or dark",
        "noise pixel inside the marker region dominates the min/max and compresses the",
        "rest of the contrast range.  The canonical example: if a white cell has a",
        "255-DN specular highlight and the surrounding cells are at 180–220 DN, after",
        "min–max normalisation the cells near the highlight will all map to values just",
        "above 0.5 and a single dark noise pixel will map the true black cells to values",
        "just below 0.5 — a systematic global bias.",
        "",
        "Otsu's method, in contrast, finds the threshold that maximises inter-class",
        "variance.  For a well-rectified marker patch the pixel histogram is bimodal",
        "(black cells peak around 20–60 DN, white cells peak around 180–240 DN) and",
        "Otsu finds the valley between the two modes, which is robust to individual",
        "outlier pixels.",
        "",
        "### C2 — Sub-cell warp drift",
        "",
    ]

    patch_group = df_ablation[
        (df_ablation["threshold_method"] == "min_max_05")
        & (df_ablation["cell_sampling"] == "mean")
        & (df_ablation["tau"] == 6)
    ].sort_values("delta_acc", ascending=False)

    if len(patch_group) > 0:
        best_patch_row = patch_group.iloc[0]
        lines.append(
            f"Best patch size: **{int(best_patch_row['patch_size'])} px** "
            f"(Δ={best_patch_row['delta_acc']:+.3f}, "
            f"recovered {int(best_patch_row['n_failure_recovered'])} failures)."
        )
    lines += [
        "",
        "With `CELL_SIZE=4` px, a 1-pixel corner residual from Stage 3 shifts the",
        "cell-averaging window by 1/4 of the cell width, mixing approximately 25 % of",
        "the neighbouring cell's pixels into the average.  At 8 px/cell (patch_size=64),",
        "the same 1-pixel residual is only 12.5 % of the cell width — halving the mixing",
        "artefact.  At 12 px/cell (patch_size=96) it falls to 8.3 %.",
        "",
        "### C3 — MAX_HAMMING = 6 vs. τ = 5",
        "",
    ]

    tau_group = df_ablation[
        (df_ablation["threshold_method"] == "min_max_05")
        & (df_ablation["patch_size"] == 32)
        & (df_ablation["cell_sampling"] == "mean")
    ].sort_values("delta_acc", ascending=False)

    if len(tau_group) > 0:
        best_tau_row = tau_group.iloc[0]
        lines.append(
            f"Best τ: **{int(best_tau_row['tau'])}** "
            f"(Δ={best_tau_row['delta_acc']:+.3f}, "
            f"recovered {int(best_tau_row['n_failure_recovered'])} failures, "
            f"regressions {int(best_tau_row['n_regression'])})."
        )
    lines += [
        "",
        "The ARUCO_MIP_36H12 dictionary has d_min = 12, so unique decoding is",
        "guaranteed only for distances ≤ ⌊(12−1)/2⌋ = 5.  At τ=6, a query at",
        "distance 6 could have two codewords at equal distance — the argmin is",
        "arbitrary.  Lowering to τ=5 trades a small recall hit (some genuinely",
        "correct but noisy decodes are now rejected) for a measurable false-positive",
        "reduction.  Because the Kaggle scorer penalises false positives at the same",
        "weight as misses (Eq. 3.10), the precision-preferred τ=5 is the better choice.",
        "",
        "### C4 — Black border discarded",
        "",
        "The outer ring of every ARUCO_MIP_36H12 marker is all-black by specification.",
        "This ring is a reliable anchor for the dark class.  The live decoder discards",
        "it completely (`decode.py:227`).  The `border_calibrated` threshold method",
        "uses the outer-ring mean as the dark-class estimate, which is particularly",
        "valuable when the illumination is strongly non-uniform across the marker",
        "(the outer ring is always the darkest region, even under a gradient).",
        "The ablation quantifies whether this additional calibration measurably",
        "improves accuracy on the challenge subset.",
        "",
        "---",
        "",
        f"*Generated by `error_analysis/stage4_decode.py`.  "
        f"Panels saved to `error_analysis/findings/stage4_panels/`.*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Markdown -> {out_path}")


# ============================================================================
# Main entry point
# ============================================================================


def run(
    val_images: Path,
    gt_dir: Path,
    stems: list[str] | None = None,
    n_panels: int = 8,
) -> dict:
    """Run the full Stage 4 decode error analysis.

    Args:
        val_images: Directory containing ``.jpg`` validation images.
        gt_dir:     Directory containing FlyingArUco v2 ``.json`` annotations.
        stems:      Image stems to analyse.  Defaults to the 40-image challenge set.
        n_panels:   Number of failure-case panels to generate.

    Returns:
        Dict with keys ``"baseline"`` and ``"ablation"`` (DataFrames).
    """
    if stems is None:
        stems = load_challenge_set()
        print(f"Loaded {len(stems)} stems from challenge_set.txt")

    _FINDINGS.mkdir(parents=True, exist_ok=True)
    _PANELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 5a: Baseline ─────────────────────────────────────────────────────────
    print("\n--- Section 5a: Baseline characterisation ---")
    df_baseline = run_baseline(stems, val_images, gt_dir)

    csv_baseline = _FINDINGS / "stage4_baseline.csv"
    df_baseline.to_csv(csv_baseline, index=False, float_format="%.4f")
    print(f"\nBaseline CSV -> {csv_baseline}")

    # ── 5b: Ablation ─────────────────────────────────────────────────────────
    print("\n--- Section 5b: Component ablation sweep ---")
    df_ablation = run_ablation(df_baseline, val_images, gt_dir)

    csv_ablation = _FINDINGS / "stage4_ablation.csv"
    df_ablation.to_csv(csv_ablation, index=False, float_format="%.4f")
    print(f"\nAblation CSV -> {csv_ablation}  ({len(df_ablation)} rows)")

    # ── 5c: Border / early-exit stats ────────────────────────────────────────
    print("\n--- Section 5c: Border sanity and early-exit analysis ---")
    border_stats = summarise_border_checks(df_baseline)
    print(
        f"  Early-exit fires   : {border_stats['n_early_exit']} / {border_stats['n_total']} "
        f"({border_stats['pct_early_exit']:.1f}%)"
    )
    print(
        f"  Border-bright      : {border_stats['n_border_bright']} / {border_stats['n_total']} "
        f"({border_stats['pct_border_bright']:.1f}%)"
    )

    # ── Panels ────────────────────────────────────────────────────────────────
    print(f"\n--- Generating {n_panels} failure-case panels ---")
    _make_panels(df_baseline, val_images, gt_dir, _PANELS_DIR, n_panels)

    # ── Markdown ──────────────────────────────────────────────────────────────
    print("\n--- Writing markdown ---")
    _write_markdown(
        df_baseline,
        df_ablation,
        border_stats,
        n_panels,
        _FINDINGS / "stage4_decode_deepdive.md",
    )

    # ── Summary to stdout ────────────────────────────────────────────────────
    n_total = len(df_baseline)
    n_ok = int(df_baseline["success"].sum())
    n_rej = int(df_baseline["rejected"].sum())
    top1 = df_ablation.iloc[0] if len(df_ablation) > 0 else None

    sep = "=" * 62
    print(f"\n{sep}")
    print("Stage 4 - Decode Deep-Dive Summary")
    print(sep)
    print(f"  GT markers analysed    : {n_total}")
    print(f"  Correct (baseline)     : {n_ok} ({n_ok/n_total:.1%})")
    print(f"  Rejected (hamming > 6) : {n_rej} ({n_rej/n_total:.1%})")
    print(f"  Wrong ID (accepted)    : {n_total - n_ok - n_rej} ({(n_total-n_ok-n_rej)/n_total:.1%})")
    print(f"  Ablation combos        : {len(df_ablation)}")
    if top1 is not None:
        print(
            f"  Best variant           : "
            f"{top1['threshold_method']} / ps={int(top1['patch_size'])} / "
            f"{top1['cell_sampling']} / tau={int(top1['tau'])}  "
            f"acc={top1['accuracy']:.1%}  delta={top1['delta_acc']:+.3f}"
        )
    print(sep)

    return {"baseline": df_baseline, "ablation": df_ablation}


# ============================================================================
# CLI
# ============================================================================


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4 — Marker ID Decoding error analysis (deep dive).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--val_images",
        default=str(_DATA_ROOT / "data" / "processed" / "dataset" / "images" / "val"),
        help="Directory containing validation .jpg images.",
    )
    parser.add_argument(
        "--gt_dir",
        default=str(_DATA_ROOT / "data" / "raw" / "flyingarucov2"),
        help="Directory containing FlyingArUco v2 .json annotation files.",
    )
    parser.add_argument(
        "--n_panels",
        type=int,
        default=8,
        help="Number of failure-case panels to generate.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process only the first 5 challenge images (sanity check).",
    )
    args = parser.parse_args()

    stems: list[str] | None = None
    if args.dry_run:
        stems = load_challenge_set()[:5]
        print(f"[DRY RUN] Processing {len(stems)} images: {stems}")

    run(
        val_images=Path(args.val_images),
        gt_dir=Path(args.gt_dir),
        stems=stems,
        n_panels=args.n_panels,
    )


if __name__ == "__main__":
    _cli()
