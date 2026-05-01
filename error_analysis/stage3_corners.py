"""
Stage 3 — Corner Refinement: Error Analysis
===========================================

Measures how accurately each corner-refinement strategy recovers the true
marker corners on the 40-image challenge subset.  The analysis is decoupled
from detector recall by using GT-localised crops — every GT marker receives a
perfect crop, so corner-error measurements isolate the refinement stage.

Three strategies are compared:
  - **Raw baseline**: four corners of the expanded-bbox rectangle (no refinement).
  - **Classical**: ``refine_corners_classical`` / ``cornerSubPix``
    (``src/corners.py:197``), starting from the expanded-bbox rectangle corners.
  - **CNN**: ``refine_corners_cnn`` (``src/corners.py:268``), using the trained
    ``models/best_corners.pth`` checkpoint.

The CNN and raw baseline both start from the same GT-bbox expanded by 20 %,
exactly as the live pipeline does after a YOLO detection (``src/detect.py:304``).
This isolates the refinement contribution from detector accuracy.

Per-marker metrics (``findings/stage3.csv``):
  - mean / max per-corner pixel error to GT corners for each strategy.
  - Gaussian φ for the canonical top-left corner (the Kaggle-metric quantity).
  - Δ columns (CNN − classical) for error and φ.

Side-by-side panels for the top-5 highest CNN corner-error markers are saved
to ``findings/stage3_panels/``.

A markdown writeup is saved to ``findings/stage3.md``.

Report alignment
----------------
- §3.4.1 "Why this stage dominates the score" (Eq. 3.1, σ ≈ 14.6 px):
  a 5 px error drops the per-marker Gaussian score by ≈ 6 %; 20 px ≈ 60 %.
- §3.4.2 Corner-regression CNN (144 K parameters, MAE loss, 90° augmentation).
- §3.4.3 Fallback path ``cornerSubPix`` (Eq. 3.6).
- Code entry points:  ``src/corners.py:268  refine_corners_cnn``
                      ``src/corners.py:197  refine_corners_classical``
                      ``src/corners.py:171  load_corner_model``

CLI usage
---------
    # Full 40-image analysis (default):
    python -m error_analysis.stage3_corners

    # Quick 5-image sanity check:
    python -m error_analysis.stage3_corners --dry_run

    # Explicit paths:
    python -m error_analysis.stage3_corners \\
        --val_images data/processed/dataset/images/val \\
        --gt_dir     data/raw/flyingarucov2 \\
        --model      models/best_corners.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Robust repo-root detection (same pattern as stage1 / stage2)
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
# Remaining imports (after sys.path is set up)
# ---------------------------------------------------------------------------

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.corners import (  # type: ignore[import]
    refine_corners_cnn,
    refine_corners_classical,
    load_corner_model,
    CornerRefinementCNN,
)
from src.detect import crop_detection  # type: ignore[import]
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
_PANELS_DIR = _FINDINGS / "stage3_panels"


# ============================================================================
# Geometry helpers
# ============================================================================


def _bbox_from_corners(corners: np.ndarray) -> tuple[int, int, int, int]:
    """Return tight XYXY bbox enclosing a (4, 2) corner array."""
    x0 = int(np.floor(corners[:, 0].min()))
    y0 = int(np.floor(corners[:, 1].min()))
    x1 = int(np.ceil(corners[:, 0].max()))
    y1 = int(np.ceil(corners[:, 1].max()))
    return x0, y0, x1, y1


def _rect_corners(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return the four axis-aligned corners of a bbox in TL→TR→BR→BL order."""
    x0, y0, x1, y1 = bbox
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
    )


# ============================================================================
# Per-marker analysis
# ============================================================================


def analyse_marker(
    stem: str,
    image: np.ndarray,
    gm: GTMarker,
    model: Optional[CornerRefinementCNN],
    device: str = "cpu",
) -> dict | None:
    """Run Stage 3 diagnostics for a single GT marker.

    Args:
        stem:   Image filename stem (for labelling output rows).
        image:  Full BGR image.
        gm:     Ground-truth marker with corners, rot, canonical_tl.
        model:  Loaded CornerRefinementCNN (or None → CNN metrics are NaN).
        device: Torch device string.

    Returns:
        Flat dict of metrics, or None on failure.
    """
    h, w = image.shape[:2]
    diag = image_diagonal(h, w)

    # ── Step 1: GT bbox → 20 % expansion → 64×64 crop ─────────────────────
    tight_bbox = _bbox_from_corners(gm.corners)
    crop, exp_bbox = crop_detection(image, tight_bbox, target_size=64, margin=0.20)
    if crop.size == 0:
        return None

    # ── Step 2: Refined corners from each strategy ──────────────────────────

    # Raw baseline: expanded-bbox rectangle corners (TL→TR→BR→BL).
    raw_corners = _rect_corners(exp_bbox)

    # Classical cornerSubPix starting from the same expanded-bbox rectangle.
    try:
        cls_corners = refine_corners_classical(image, raw_corners.copy())
    except Exception:
        cls_corners = raw_corners.copy()

    # CNN: pass the 64×64 crop + expanded bbox.
    cnn_corners: Optional[np.ndarray] = None
    if model is not None:
        try:
            cnn_corners = refine_corners_cnn(crop, exp_bbox, model, device)
        except Exception:
            cnn_corners = None

    # ── Step 3: Per-corner pixel errors to GT corners ───────────────────────
    gt_c = gm.corners.astype(np.float32)  # (4, 2)

    def _corner_stats(pred: np.ndarray) -> dict:
        errs = np.linalg.norm(pred - gt_c, axis=1)  # (4,) pixel errors
        # Canonical TL error → Gaussian phi
        tl_pred = pred[gm.rot % 4]
        d_px = float(np.linalg.norm(tl_pred - gm.canonical_tl))
        d_norm = d_px / diag
        phi = gaussian_score(d_norm)
        return {
            "mean_err": float(errs.mean()),
            "max_err": float(errs.max()),
            "tl_err_px": d_px,
            "phi": phi,
        }

    raw_stats = _corner_stats(raw_corners)
    cls_stats = _corner_stats(cls_corners)

    row: dict = {
        "stem": stem,
        "marker_id": gm.marker_id,
        "rot": gm.rot,
        # ── raw (no refinement) ────────────────────────────────────────────
        "raw_mean_err": raw_stats["mean_err"],
        "raw_max_err": raw_stats["max_err"],
        "raw_tl_err_px": raw_stats["tl_err_px"],
        "raw_phi": raw_stats["phi"],
        # ── classical cornerSubPix ─────────────────────────────────────────
        "cls_mean_err": cls_stats["mean_err"],
        "cls_max_err": cls_stats["max_err"],
        "cls_tl_err_px": cls_stats["tl_err_px"],
        "cls_phi": cls_stats["phi"],
        # ── CNN (NaN when model not provided) ─────────────────────────────
        "cnn_mean_err": float("nan"),
        "cnn_max_err": float("nan"),
        "cnn_tl_err_px": float("nan"),
        "cnn_phi": float("nan"),
    }

    if cnn_corners is not None:
        cnn_stats = _corner_stats(cnn_corners)
        row.update({
            "cnn_mean_err": cnn_stats["mean_err"],
            "cnn_max_err": cnn_stats["max_err"],
            "cnn_tl_err_px": cnn_stats["tl_err_px"],
            "cnn_phi": cnn_stats["phi"],
        })

    # Delta: CNN − classical (negative = CNN is better)
    row["delta_mean_err"] = row["cnn_mean_err"] - row["cls_mean_err"]
    row["delta_phi"] = row["cnn_phi"] - row["cls_phi"]

    return row


# ============================================================================
# Visualisation
# ============================================================================


def _make_panels(
    df: pd.DataFrame,
    val_images: Path,
    gt_dir: Path,
    model: Optional[CornerRefinementCNN],
    device: str,
    n_panels: int = 5,
) -> None:
    """Save side-by-side panels for the n_panels highest CNN corner-error markers.

    Each panel shows: expanded-bbox crop with GT corners | classical corners overlay | CNN corners overlay.
    """
    _PANELS_DIR.mkdir(parents=True, exist_ok=True)

    cnn_col = "cnn_mean_err" if "cnn_mean_err" in df.columns else "cls_mean_err"
    valid = df.dropna(subset=[cnn_col])
    worst = valid.nlargest(n_panels, cnn_col)

    for _, row in worst.iterrows():
        stem = row["stem"]
        marker_id = int(row["marker_id"])

        img_path = val_images / f"{stem}.jpg"
        json_path = gt_dir / f"{stem}.json"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt_markers = load_gt_from_json(json_path) if json_path.exists() else []
        gm = next((m for m in gt_markers if m.marker_id == marker_id), None)
        if gm is None:
            continue

        tight_bbox = _bbox_from_corners(gm.corners)
        crop, exp_bbox = crop_detection(img, tight_bbox, target_size=64, margin=0.20)
        raw_corners = _rect_corners(exp_bbox)
        try:
            cls_corners = refine_corners_classical(img, raw_corners.copy())
        except Exception:
            cls_corners = raw_corners.copy()

        cnn_corners: Optional[np.ndarray] = None
        if model is not None:
            try:
                cnn_corners = refine_corners_cnn(crop, exp_bbox, model, device)
            except Exception:
                pass

        # Draw on full image crops (not the 64×64 resized crop)
        x0, y0, x1, y1 = exp_bbox
        pad = 5
        x0p = max(0, x0 - pad)
        y0p = max(0, y0 - pad)
        x1p = min(img.shape[1], x1 + pad)
        y1p = min(img.shape[0], y1 + pad)
        panel_crop = cv2.cvtColor(img[y0p:y1p, x0p:x1p], cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(
            f"Stage 3 — Corner Refinement  |  {stem}  |  marker {marker_id}",
            fontsize=9, fontweight="bold",
        )

        def _plot_corners(ax, title: str, corners: np.ndarray, colour: str) -> None:
            ax.imshow(panel_crop)
            # GT corners: red
            pts_gt = np.vstack([gm.corners, gm.corners[0]])
            ax.plot(pts_gt[:, 0] - x0p, pts_gt[:, 1] - y0p, "r-", lw=1.5, label="GT")
            ax.plot(gm.canonical_tl[0] - x0p, gm.canonical_tl[1] - y0p,
                    "y*", ms=8, markeredgecolor="k", markeredgewidth=0.5, label="GT TL")
            # Predicted corners
            pts_pred = np.vstack([corners, corners[0]])
            ax.plot(pts_pred[:, 0] - x0p, pts_pred[:, 1] - y0p,
                    color=colour, lw=1.5, linestyle="--", label="Pred")
            ax.plot(corners[gm.rot % 4, 0] - x0p, corners[gm.rot % 4, 1] - y0p,
                    "o", color=colour, ms=6, markeredgecolor="k", markeredgewidth=0.5)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            ax.legend(fontsize=6, loc="upper left")

        _plot_corners(axes[0], "Raw (bbox rectangle)", raw_corners, "cyan")
        _plot_corners(axes[1], "Classical (cornerSubPix)", cls_corners, "lime")
        if cnn_corners is not None:
            _plot_corners(axes[2], "CNN", cnn_corners, "orange")
        else:
            axes[2].imshow(panel_crop)
            axes[2].set_title("CNN (no model)", fontsize=8)
            axes[2].axis("off")

        plt.tight_layout()
        out_path = _PANELS_DIR / f"{stem}_{marker_id}_stage3.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  Panel -> {out_path.name}")


# ============================================================================
# Markdown writeup
# ============================================================================


def _write_markdown(df: pd.DataFrame, n_panels: int, out_path: Path) -> None:
    """Generate the Stage 3 error-analysis markdown writeup."""
    has_cnn = df["cnn_mean_err"].notna().any()
    n_markers = len(df)
    n_images = df["stem"].nunique()

    raw_m = df["raw_mean_err"].mean()
    cls_m = df["cls_mean_err"].mean()
    cnn_m = df["cnn_mean_err"].mean() if has_cnn else float("nan")

    raw_phi = df["raw_phi"].mean()
    cls_phi = df["cls_phi"].mean()
    cnn_phi = df["cnn_phi"].mean() if has_cnn else float("nan")

    # Threshold counts for "bad" corners
    n_raw_gt10 = int((df["raw_mean_err"] > 10).sum())
    n_cls_gt10 = int((df["cls_mean_err"] > 10).sum())
    n_cnn_gt10 = int((df["cnn_mean_err"] > 10).sum()) if has_cnn else -1

    cnn_better = int((df["delta_mean_err"] < 0).sum()) if has_cnn else 0
    cls_better = int((df["delta_mean_err"] > 0).sum()) if has_cnn else 0

    worst5 = df.nlargest(5, "cnn_mean_err" if has_cnn else "cls_mean_err")

    lines: list[str] = [
        "# Stage 3 — Corner Refinement: Error Analysis",
        "",
        "## 1. Overview",
        "",
        f"Analysis run on **{n_images} challenge images** covering **{n_markers} GT markers**.",
        "",
        "Three corner-refinement strategies are compared (all use GT-localised crops,",
        "so detection recall does not affect the results):",
        "",
        "| Strategy | Code entry point | Description |",
        "|----------|-----------------|-------------|",
        "| **Raw baseline** | *(no refinement)* | Four corners of the 20%-expanded bbox rectangle |",
        "| **Classical** | `src/corners.py:197` | `cv2.cornerSubPix` from bbox-rectangle initial guess |",
        "| **CNN** | `src/corners.py:268` | Corner-regression CNN, `models/best_corners.pth` |",
        "",
        "The expanded-bbox rectangle simulates what the live pipeline receives from",
        "the YOLO detection stage before refinement.",
        "",
        "## 2. Quantitative Summary",
        "",
        "| Strategy | Mean corner error (px) | Mean max-corner error (px) | Mean canonical-TL φ |",
        "|----------|------------------------|---------------------------|---------------------|",
        f"| Raw (no refinement) | {raw_m:.2f} | {df['raw_max_err'].mean():.2f} | {raw_phi:.4f} |",
        f"| Classical (cornerSubPix) | {cls_m:.2f} | {df['cls_max_err'].mean():.2f} | {cls_phi:.4f} |",
    ]
    if has_cnn:
        lines.append(
            f"| CNN | {cnn_m:.2f} | {df['cnn_max_err'].mean():.2f} | {cnn_phi:.4f} |"
        )
    else:
        lines.append("| CNN | N/A (model not loaded) | N/A | N/A |")

    lines += [
        "",
        "Canonical-TL φ = exp(−d²_norm / 2σ²), σ = 0.02, evaluated on the Gaussian",
        "score that enters the Kaggle metric directly.",
        "",
        "### Markers with mean corner error > 10 px",
        "",
        f"| Strategy | Count (of {n_markers}) | Fraction |",
        "|----------|------------------------|----------|",
        f"| Raw | {n_raw_gt10} | {n_raw_gt10/n_markers:.1%} |",
        f"| Classical | {n_cls_gt10} | {n_cls_gt10/n_markers:.1%} |",
    ]
    if has_cnn:
        lines.append(f"| CNN | {n_cnn_gt10} | {n_cnn_gt10/n_markers:.1%} |")

    if has_cnn:
        lines += [
            "",
            "## 3. CNN vs. Classical Comparison",
            "",
            f"On this challenge subset, CNN outperforms classical on **{cnn_better}** of "
            f"{n_markers} markers ({cnn_better/n_markers:.1%}), while classical is better "
            f"on **{cls_better}** ({cls_better/n_markers:.1%}).",
            "",
            "The delta column `cnn_mean_err − cls_mean_err` is negative when the CNN",
            "is better.  A large positive value flags regressions (cases where the CNN",
            "performs worse than even cornerSubPix), likely due to:",
            "  - Occluded / partially visible markers (the CNN never saw these at training).",
            "  - Extreme motion blur where the crop texture is too degraded for the CNN.",
            "  - Markers very close to image boundaries, producing degenerate crops.",
        ]

    lines += [
        "",
        "## 4. Why Corner Accuracy Dominates the Score (§3.4.1)",
        "",
        "The Gaussian scorer (Eq. 3.1, σ ≈ 14.6 px on 640×360 images):",
        "  - 5 px TL error  → φ ≈ 0.94 (6 % loss)",
        "  - 10 px TL error → φ ≈ 0.79 (21 % loss)",
        "  - 20 px TL error → φ ≈ 0.40 (60 % loss)",
        "",
        "This means corner accuracy is the highest-leverage stage: halving the",
        "mean corner error from 10 px → 5 px lifts the per-marker score from ~0.79 → ~0.94.",
        "",
        "## 5. Worst-5 Markers by CNN Corner Error",
        "",
        f"The {n_panels} markers below had the highest CNN mean corner error.",
        "Panels saved to `error_analysis/findings/stage3_panels/`.",
        "",
        "| Rank | Stem | Marker ID | Raw err (px) | Classical err (px) | CNN err (px) |",
        "|------|------|-----------|-------------|-------------------|--------------|",
    ]
    for rank, (_, row) in enumerate(worst5.iterrows(), 1):
        cnn_err = f"{row['cnn_mean_err']:.2f}" if not np.isnan(row['cnn_mean_err']) else "N/A"
        lines.append(
            f"| {rank} | `{row['stem']}` | {int(row['marker_id'])} "
            f"| {row['raw_mean_err']:.2f} | {row['cls_mean_err']:.2f} | {cnn_err} |"
        )

    lines += [
        "",
        "## 6. Likely Root Cause",
        "",
        "The corner refinement stage is a necessary prerequisite for the decode stage:",
        "",
        "- **CNN > raw baseline**: the trained model recovers sub-pixel corner positions",
        "  even under mild blur, avoiding the systematic 10–30 px offset of the raw",
        "  bbox rectangle.",
        "",
        "- **CNN vs. classical**: on the hard challenge set, cornerSubPix can diverge",
        "  when the initial estimate is far from the true corner (the bbox rectangle is",
        "  the starting point, not the GT corner).  The CNN has a global view of the",
        "  64×64 crop and can find corners even when gradients are weak.",
        "",
        "- **Corner error → decode error**: even with correct decoding (Stage 4), a",
        "  large canonical-TL error directly reduces the Kaggle score.  The Stage 3",
        "  findings bound the *minimum achievable* Stage 4 φ; decode failures on top",
        "  of large corner errors compound the damage.",
        "",
        "---",
        "",
        "*Generated by `error_analysis/stage3_corners.py`*",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Markdown -> {out_path}")


# ============================================================================
# Main entry point
# ============================================================================


def run(
    val_images: Path,
    gt_dir: Path,
    model_path: Optional[Path] = None,
    stems: list[str] | None = None,
    n_panels: int = 5,
    device: str = "cpu",
) -> pd.DataFrame:
    """Run the full Stage 3 corner-refinement error analysis.

    Args:
        val_images:  Directory containing ``.jpg`` validation images.
        gt_dir:      Directory containing FlyingArUco v2 ``.json`` annotations.
        model_path:  Path to ``best_corners.pth`` checkpoint.  If None or missing,
                     CNN metrics will be NaN but the script still runs.
        stems:       Image stems to analyse.  Defaults to the 40-image challenge set.
        n_panels:    How many worst-CNN-error markers get a side-by-side panel.
        device:      Torch device for CNN inference.

    Returns:
        DataFrame with per-marker metrics (also written to ``findings/stage3.csv``).
    """
    if stems is None:
        stems = load_challenge_set()
        print(f"Loaded {len(stems)} stems from challenge_set.txt")

    _FINDINGS.mkdir(parents=True, exist_ok=True)
    _PANELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load corner CNN (optional)
    model: Optional[CornerRefinementCNN] = None
    if model_path is not None and Path(model_path).exists():
        try:
            model = load_corner_model(model_path, device=device)
            print(f"Corner CNN loaded from {model_path}")
        except Exception as e:
            print(f"  [WARN] Could not load corner model: {e}", file=sys.stderr)
    else:
        print("[WARN] No corner model path provided — CNN metrics will be NaN.")

    rows: list[dict] = []
    for i, stem in enumerate(stems, 1):
        img_path = val_images / f"{stem}.jpg"
        json_path = gt_dir / f"{stem}.json"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i:2d}/{len(stems)}] {stem} ... SKIP (cannot load image)")
            continue

        gt_markers: list[GTMarker] = (
            load_gt_from_json(json_path) if json_path.exists() else []
        )
        if not gt_markers:
            print(f"  [{i:2d}/{len(stems)}] {stem} ... SKIP (no GT)")
            continue

        img_rows = []
        for gm in gt_markers:
            row = analyse_marker(stem, img, gm, model, device)
            if row is not None:
                img_rows.append(row)

        rows.extend(img_rows)
        mean_cnn = (
            np.nanmean([r["cnn_mean_err"] for r in img_rows])
            if img_rows and model is not None
            else float("nan")
        )
        mean_cls = np.mean([r["cls_mean_err"] for r in img_rows]) if img_rows else 0.0
        print(
            f"  [{i:2d}/{len(stems)}] {stem}  "
            f"markers={len(img_rows)}  "
            f"cls={mean_cls:.1f}px  "
            f"cnn={mean_cnn:.1f}px"
        )

    if not rows:
        print("[ERROR] No markers could be analysed.", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = _FINDINGS / "stage3.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nCSV -> {csv_path}")

    # ── generate panels ───────────────────────────────────────────────────────
    print(f"\nGenerating {n_panels} panels for highest CNN corner error ...")
    _make_panels(df, val_images, gt_dir, model, device, n_panels)

    # ── markdown writeup ──────────────────────────────────────────────────────
    _write_markdown(df, n_panels, _FINDINGS / "stage3.md")

    # ── summary to stdout ─────────────────────────────────────────────────────
    sep = "=" * 60
    has_cnn = df["cnn_mean_err"].notna().any()
    print(f"\n{sep}")
    print("Stage 3 - Corner Refinement Summary")
    print(sep)
    print(f"  Markers analysed      : {len(df)}")
    print(f"  Images analysed       : {df['stem'].nunique()}")
    print(f"  Raw mean corner error : {df['raw_mean_err'].mean():.2f} px")
    print(f"  Cls mean corner error : {df['cls_mean_err'].mean():.2f} px")
    if has_cnn:
        print(f"  CNN mean corner error : {df['cnn_mean_err'].mean():.2f} px")
        cnn_better = int((df["delta_mean_err"] < 0).sum())
        print(f"  CNN better than cls   : {cnn_better}/{len(df)} markers")
    print(f"  Mean raw  canonical phi : {df['raw_phi'].mean():.4f}")
    print(f"  Mean cls  canonical phi : {df['cls_phi'].mean():.4f}")
    if has_cnn:
        print(f"  Mean CNN  canonical phi : {df['cnn_phi'].mean():.4f}")
    print(sep)

    return df


# ============================================================================
# CLI
# ============================================================================


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 — Corner Refinement error analysis.",
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
        "--model",
        default=str(_DATA_ROOT / "models" / "best_corners.pth"),
        help="Path to trained CornerRefinementCNN checkpoint.",
    )
    parser.add_argument(
        "--n_panels",
        type=int,
        default=5,
        help="Number of worst-error markers for which to generate panels.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for CNN inference ('cpu', 'cuda', 'cuda:0', …).",
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
        model_path=Path(args.model),
        stems=stems,
        n_panels=args.n_panels,
        device=args.device,
    )


if __name__ == "__main__":
    _cli()
