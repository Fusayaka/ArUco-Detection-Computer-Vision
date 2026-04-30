"""
Evaluation harness for the ArUco detection pipeline.

Implements the Kaggle Gaussian-distance metric (report Eq. 1 and 9) and
provides helpers to load FlyingArUco v2 ground-truth JSON annotations.

Metric recap (from the report, §3.1):
    φ(d_norm) = exp( -d_norm² / (2 σ²) ),   σ = 0.02
    d_norm = d_px / sqrt(H² + W²)

    Score_img = (1 / (N_gt + N_spam)) * Σ_{j=1}^{N_matched} φ(d_norm,j)

where:
  N_gt      = number of ground-truth markers in the image
  N_spam    = number of predicted markers whose ID does not match any GT
  N_matched = predictions matched to a GT marker by ID
  d_px      = Euclidean distance (predicted canonical TL) to (GT canonical TL)

CLI usage:
    python -m error_analysis.scoring \
        --submission output/submission.csv \
        --gt_dir data/raw/flyingarucov2 \
        [--images data/processed/dataset/images/val]
"""

from __future__ import annotations

import json
import math
import os
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── constants ─────────────────────────────────────────────────────────────────

SIGMA = 0.02  # report §3.1, Eq. (1)


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class GTMarker:
    """One ground-truth marker entry parsed from a FlyingArUco v2 JSON."""
    marker_id: int
    corners: np.ndarray      # (4, 2) float32  — TL, TR, BR, BL in image px
    rot: int                  # canonical rotation index (0-3)
    canonical_tl: np.ndarray  # (2,) float32  — canonical top-left (x, y)


@dataclass
class Prediction:
    """One predicted marker from a submission string."""
    marker_id: int
    x: float
    y: float


@dataclass
class ImageScore:
    """Per-image scoring breakdown."""
    image_stem: str
    n_gt: int
    n_pred: int
    n_matched: int    # GT markers with a matching-ID prediction
    n_spam: int       # predictions whose ID has no GT counterpart
    n_missed: int     # GT markers with no prediction at all
    phi_sum: float    # sum of φ over matched markers
    score: float      # final per-image score in [0, 1]
    per_marker: list[dict] = field(default_factory=list)


# ── core metric ───────────────────────────────────────────────────────────────

def gaussian_score(d_norm: float, sigma: float = SIGMA) -> float:
    """φ(d_norm) = exp(-d_norm² / (2σ²)).  Returns value in [0, 1]."""
    return math.exp(-(d_norm ** 2) / (2 * sigma ** 2))


def image_diagonal(h: int, w: int) -> float:
    return math.sqrt(h ** 2 + w ** 2)


def score_image(
    predictions: list[Prediction],
    gt_markers: list[GTMarker],
    img_h: int,
    img_w: int,
    sigma: float = SIGMA,
) -> ImageScore:
    """Compute the Kaggle per-image score (report Eq. 9).

    Matching is purely by marker ID — a prediction is matched to the GT marker
    with the same ID.  Duplicate predicted IDs count as spam for all but the
    first occurrence.

    Args:
        predictions:  List of Prediction objects for this image.
        gt_markers:   List of GTMarker objects for this image.
        img_h, img_w: Image height and width in pixels.
        sigma:        Gaussian σ parameter.

    Returns:
        ImageScore dataclass with all diagnostics.
    """
    diag = image_diagonal(img_h, img_w)
    gt_by_id = {m.marker_id: m for m in gt_markers}
    gt_ids = set(gt_by_id.keys())

    seen_pred_ids: set[int] = set()
    matched_details: list[dict] = []
    n_spam = 0

    for pred in predictions:
        if pred.marker_id in seen_pred_ids:
            # Duplicate prediction — spam
            n_spam += 1
            continue
        seen_pred_ids.add(pred.marker_id)

        if pred.marker_id not in gt_by_id:
            n_spam += 1
            continue

        gt = gt_by_id[pred.marker_id]
        d_px = math.hypot(pred.x - gt.canonical_tl[0], pred.y - gt.canonical_tl[1])
        d_norm = d_px / diag
        phi = gaussian_score(d_norm, sigma)
        matched_details.append({
            "marker_id": pred.marker_id,
            "d_px": d_px,
            "d_norm": d_norm,
            "phi": phi,
        })

    matched_ids = {d["marker_id"] for d in matched_details}
    n_matched = len(matched_ids)
    n_missed = len(gt_ids - matched_ids)
    phi_sum = sum(d["phi"] for d in matched_details)

    denom = len(gt_markers) + n_spam
    score = phi_sum / denom if denom > 0 else 0.0

    return ImageScore(
        image_stem="",  # filled by caller
        n_gt=len(gt_markers),
        n_pred=len(predictions),
        n_matched=n_matched,
        n_spam=n_spam,
        n_missed=n_missed,
        phi_sum=phi_sum,
        score=score,
        per_marker=matched_details,
    )


# ── GT loading ────────────────────────────────────────────────────────────────

def load_gt_from_json(json_path: str | Path) -> list[GTMarker]:
    """Parse a single FlyingArUco v2 annotation JSON → list of GTMarker.

    JSON schema (from data/raw/flyingarucov2/000000000089.json):
        {
          "markers": [
            { "id": <int>, "corners": [[x,y],[x,y],[x,y],[x,y]], "rot": <int> },
            ...
          ]
        }

    corners[0..3] = TL, TR, BR, BL of the visible marker in image coords.
    rot = number of 90° CCW rotations applied to the canonical marker; the
    canonical top-left is corners[rot % 4].
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    markers = []
    for m in data.get("markers", []):
        corners = np.array(m["corners"], dtype=np.float32)  # (4, 2)
        rot = int(m["rot"])
        canonical_tl = corners[rot % 4]
        markers.append(GTMarker(
            marker_id=int(m["id"]),
            corners=corners,
            rot=rot,
            canonical_tl=canonical_tl,
        ))
    return markers


def load_gt_from_folder(gt_folder: str | Path) -> dict[str, list[GTMarker]]:
    """Load all GT JSONs from a folder.

    Returns a dict mapping image stem (e.g. "000000000089") → list[GTMarker].
    Only files with a corresponding .jpg are included (ignores stray JSONs).
    """
    gt_folder = Path(gt_folder)
    gt: dict[str, list[GTMarker]] = {}

    for json_file in sorted(gt_folder.glob("*.json")):
        stem = json_file.stem
        gt[stem] = load_gt_from_json(json_file)

    return gt


# ── prediction loading ────────────────────────────────────────────────────────

def parse_prediction_string(pred_str: str) -> list[Prediction]:
    """Parse a Kaggle submission prediction string.

    Format: "<id> <x> <y> <id> <x> <y> ..." or empty/whitespace → [].
    """
    tokens = str(pred_str).strip().split()
    if not tokens:
        return []

    preds = []
    i = 0
    while i + 2 < len(tokens):
        try:
            mid = int(tokens[i])
            x = float(tokens[i + 1])
            y = float(tokens[i + 2])
            preds.append(Prediction(marker_id=mid, x=x, y=y))
        except ValueError:
            pass
        i += 3
    return preds


def load_submission(csv_path: str | Path) -> dict[str, list[Prediction]]:
    """Load a Kaggle submission CSV → dict[image_stem → list[Prediction]]."""
    # dtype=str prevents pandas from parsing zero-padded IDs as integers
    # (e.g. "000000000315" would become 315, losing the leading zeros).
    df = pd.read_csv(csv_path, dtype={"image_id": str})
    result: dict[str, list[Prediction]] = {}
    for _, row in df.iterrows():
        stem = str(row["image_id"])
        result[stem] = parse_prediction_string(row.get("prediction_string", ""))
    return result


# ── batch scorer ──────────────────────────────────────────────────────────────

def score_submission(
    submission: dict[str, list[Prediction]],
    gt: dict[str, list[GTMarker]],
    images_dir: str | Path,
    sigma: float = SIGMA,
    image_stems: Optional[list[str]] = None,
) -> tuple[list[ImageScore], float]:
    """Score an entire submission against GT.

    Args:
        submission:   output of load_submission().
        gt:           output of load_gt_from_folder().
        images_dir:   folder containing the .jpg images (needed for H×W).
        sigma:        Gaussian σ.
        image_stems:  if given, score only these stems (useful for subset eval).

    Returns:
        (per_image_scores, mean_score)
    """
    import cv2

    images_dir = Path(images_dir)
    stems = image_stems if image_stems is not None else sorted(gt.keys())

    scores: list[ImageScore] = []
    for stem in stems:
        gt_markers = gt.get(stem, [])
        preds = submission.get(stem, [])

        # Load image dimensions
        img_path = images_dir / f"{stem}.jpg"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img_h, img_w = img.shape[:2]
            else:
                img_h, img_w = 360, 640  # FlyingArUco v2 default
        else:
            img_h, img_w = 360, 640

        s = score_image(preds, gt_markers, img_h, img_w, sigma)
        s.image_stem = stem
        scores.append(s)

    mean_score = float(np.mean([s.score for s in scores])) if scores else 0.0
    return scores, mean_score


def scores_to_dataframe(scores: list[ImageScore]) -> pd.DataFrame:
    """Convert per-image ImageScore list to a tidy DataFrame."""
    rows = []
    for s in scores:
        rows.append({
            "image_stem": s.image_stem,
            "n_gt": s.n_gt,
            "n_pred": s.n_pred,
            "n_matched": s.n_matched,
            "n_spam": s.n_spam,
            "n_missed": s.n_missed,
            "phi_sum": s.phi_sum,
            "score": s.score,
        })
    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Score a Kaggle submission CSV against FlyingArUco v2 GT JSONs."
    )
    parser.add_argument("--submission", required=True, help="Path to submission CSV")
    parser.add_argument("--gt_dir", required=True,
                        help="Folder containing FlyingArUco v2 *.json annotation files")
    parser.add_argument("--images", default=None,
                        help="Folder containing .jpg images (for H×W; defaults to gt_dir)")
    parser.add_argument("--stems", default=None,
                        help="Comma-separated image stems to restrict scoring to (optional)")
    parser.add_argument("--out_csv", default=None,
                        help="If given, save per-image scores to this CSV")
    args = parser.parse_args()

    images_dir = args.images if args.images else args.gt_dir
    stems = [s.strip() for s in args.stems.split(",")] if args.stems else None

    print(f"Loading GT from {args.gt_dir} …")
    gt = load_gt_from_folder(args.gt_dir)
    print(f"  → {len(gt)} images with GT annotations")

    print(f"Loading submission from {args.submission} …")
    submission = load_submission(args.submission)
    print(f"  → {len(submission)} rows")

    print("Scoring …")
    per_image, mean = score_submission(submission, gt, images_dir, image_stems=stems)

    df = scores_to_dataframe(per_image)
    print(f"\n{'='*50}")
    print(f"  Images scored : {len(per_image)}")
    print(f"  Mean score    : {mean:.4f}")
    print(f"  Mean recall   : {df['n_matched'].sum() / max(df['n_gt'].sum(), 1):.4f}")
    print(f"  Total spam    : {df['n_spam'].sum()}")
    print(f"  Total missed  : {df['n_missed'].sum()}")
    print(f"{'='*50}")

    # Worst 10 images
    worst = df.nsmallest(10, "score")
    print("\nWorst 10 images by score:")
    print(worst[["image_stem", "n_gt", "n_matched", "n_spam", "score"]].to_string(index=False))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nPer-image scores saved to {args.out_csv}")


if __name__ == "__main__":
    _cli()
