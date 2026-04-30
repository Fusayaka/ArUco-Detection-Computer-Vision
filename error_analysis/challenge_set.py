"""
Build and persist the 40-image curated challenge subset used across all
error-analysis stages.

Two construction modes
──────────────────────
automatic (default)
    Run the full pipeline on the validation split, score every image against GT,
    and keep the bottom-40 by Gaussian score.  These images are the hardest for
    the pipeline and therefore the most informative for error analysis.

stratified
    Select 10 images from each of four difficulty bins:
        blur     — low Laplacian variance (motion/defocus blur)
        dark     — low mean luminance (underexposure)
        oblique  — high aspect-ratio or foreshortened marker bboxes
        clutter  — images with ≥ 3 GT markers (crowded scenes)
    This mode does not require running inference and is much faster; it gives
    better coverage of distinct failure modes at the cost of not being tied
    to actual pipeline output scores.

The chosen filenames are written to findings/challenge_set.txt (one stem per line,
with a header comment explaining how the set was built).  Every downstream script
reads from that file via load_challenge_set().

CLI usage
──────────
# Build automatically (requires trained model):
python -m error_analysis.challenge_set automatic \
    --val_images  data/processed/dataset/images/val \
    --gt_dir      data/raw/flyingarucov2 \
    --model       models/aruco_best.pt \
    --n           40

# Build by stratified heuristics (no model needed):
python -m error_analysis.challenge_set stratified \
    --val_images  data/processed/dataset/images/val \
    --gt_dir      data/raw/flyingarucov2 \
    --n_per_bin   10

# Just print the existing challenge set:
python -m error_analysis.challenge_set show
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent
FINDINGS_DIR = _REPO_ROOT / "error_analysis" / "findings"
CHALLENGE_SET_FILE = FINDINGS_DIR / "challenge_set.txt"


# ── persistence ───────────────────────────────────────────────────────────────

def load_challenge_set() -> list[str]:
    """Return the list of image stems from findings/challenge_set.txt.

    Raises FileNotFoundError if the file does not exist yet (run the build
    command first).
    """
    if not CHALLENGE_SET_FILE.exists():
        raise FileNotFoundError(
            f"Challenge set not found at {CHALLENGE_SET_FILE}.\n"
            "Run: python -m error_analysis.challenge_set automatic  "
            "(or 'stratified') to build it first."
        )
    stems = []
    for line in CHALLENGE_SET_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            stems.append(line)
    return stems


def save_challenge_set(stems: list[str], method: str, notes: str = "") -> None:
    """Persist stems to CHALLENGE_SET_FILE with a header comment."""
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Challenge set — {len(stems)} images",
        f"# Build method : {method}",
    ]
    if notes:
        lines.append(f"# Notes        : {notes}")
    lines.append("")
    lines.extend(stems)
    CHALLENGE_SET_FILE.write_text("\n".join(lines) + "\n")
    print(f"Challenge set ({len(stems)} images) saved to {CHALLENGE_SET_FILE}")


# ── image-level heuristics ────────────────────────────────────────────────────

def _laplacian_variance(img_bgr: np.ndarray) -> float:
    """Measure of sharpness; low → blurry."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_luminance(img_bgr: np.ndarray) -> float:
    """Mean L* channel value (0–255); low → dark."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].mean())


def _marker_aspect_extremity(gt_markers) -> float:
    """Max aspect-ratio deviation from 1 across all GT marker bboxes.

    A frontal square marker has aspect ratio 1.  Oblique viewpoints produce
    stretched or compressed bboxes, raising this value.
    """
    from error_analysis.scoring import GTMarker
    if not gt_markers:
        return 0.0
    max_dev = 0.0
    for m in gt_markers:
        xs, ys = m.corners[:, 0], m.corners[:, 1]
        w = xs.max() - xs.min()
        h = ys.max() - ys.min()
        if h > 0 and w > 0:
            ratio = w / h
            dev = abs(ratio - 1.0)
            max_dev = max(max_dev, dev)
    return max_dev


# ── stratified builder ────────────────────────────────────────────────────────

def build_challenge_set_stratified(
    val_images_dir: str | Path,
    gt_dir: str | Path,
    n_per_bin: int = 10,
    seed: int = 42,
) -> list[str]:
    """Select n_per_bin images from each of four difficulty bins.

    Bins:
        blur    — bottom quartile of Laplacian variance
        dark    — bottom quartile of mean luminance
        oblique — top quartile of marker aspect-ratio deviation
        clutter — images with the most GT markers (top n_per_bin by count)

    Returns list of image stems (may have overlaps between bins — deduplicated
    at the end, filling gaps with next-best candidates if needed).
    """
    from error_analysis.scoring import load_gt_from_folder

    val_images_dir = Path(val_images_dir)
    gt_data = load_gt_from_folder(gt_dir)

    # Collect image-level stats for every val image that has GT
    stats: list[dict] = []
    print(f"Computing image-level heuristics for val images …")
    jpg_stems = sorted([p.stem for p in val_images_dir.glob("*.jpg")])

    for stem in jpg_stems:
        if stem not in gt_data:
            continue
        img_path = val_images_dir / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        stats.append({
            "stem": stem,
            "blur": _laplacian_variance(img),
            "lum": _mean_luminance(img),
            "oblique": _marker_aspect_extremity(gt_data[stem]),
            "n_markers": len(gt_data[stem]),
        })

    if not stats:
        raise ValueError(f"No val images with GT found in {val_images_dir}")

    print(f"  → {len(stats)} images with GT stats computed")

    rng = random.Random(seed)

    def _bottom_n(key: str, n: int) -> list[str]:
        return [s["stem"] for s in sorted(stats, key=lambda x: x[key])[:n]]

    def _top_n(key: str, n: int) -> list[str]:
        return [s["stem"] for s in sorted(stats, key=lambda x: x[key], reverse=True)[:n]]

    bins = {
        "blur":    _bottom_n("blur", n_per_bin * 3)[:n_per_bin],
        "dark":    _bottom_n("lum",  n_per_bin * 3)[:n_per_bin],
        "oblique": _top_n("oblique", n_per_bin * 3)[:n_per_bin],
        "clutter": _top_n("n_markers", n_per_bin * 3)[:n_per_bin],
    }

    for bin_name, stems_in_bin in bins.items():
        print(f"  Bin '{bin_name}': {len(stems_in_bin)} images selected")

    # Deduplicate while preserving bin order
    seen: set[str] = set()
    final: list[str] = []
    for bin_stems in bins.values():
        for s in bin_stems:
            if s not in seen:
                seen.add(s)
                final.append(s)

    target = n_per_bin * len(bins)
    if len(final) < target:
        # Pad with random remaining images
        remaining = [s["stem"] for s in stats if s["stem"] not in seen]
        rng.shuffle(remaining)
        final.extend(remaining[: target - len(final)])

    notes = f"blur={n_per_bin}, dark={n_per_bin}, oblique={n_per_bin}, clutter={n_per_bin}"
    save_challenge_set(final, method="stratified", notes=notes)
    return final


# ── automatic builder ─────────────────────────────────────────────────────────

def build_challenge_set_automatic(
    val_images_dir: str | Path,
    gt_dir: str | Path,
    model_path: str | Path,
    n: int = 40,
    conf_threshold: float = 0.5,
) -> list[str]:
    """Run the full pipeline on val images, keep the bottom-n by score.

    This is slower (requires the YOLO + corner CNN inference) but directly
    identifies the images where the pipeline actually fails — not just the
    images that look hard by heuristic.
    """
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from error_analysis.scoring import (
        load_gt_from_folder, load_submission, score_submission,
        scores_to_dataframe,
    )
    from src.detector import HybridDetector

    val_images_dir = Path(val_images_dir)
    gt_data = load_gt_from_folder(gt_dir)
    jpg_stems = sorted([p.stem for p in val_images_dir.glob("*.jpg")
                        if p.stem in gt_data])

    print(f"Running pipeline on {len(jpg_stems)} val images …")
    detector = HybridDetector(model_path=str(model_path), conf_threshold=conf_threshold)

    rows = []
    for i, stem in enumerate(jpg_stems):
        img_path = val_images_dir / f"{stem}.jpg"
        pred_str = detector.process_image(str(img_path))
        rows.append({"image_id": stem, "prediction_string": pred_str})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(jpg_stems)} …")

    import pandas as pd
    tmp_csv = FINDINGS_DIR / "_tmp_val_submission.csv"
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tmp_csv, index=False)

    submission = load_submission(tmp_csv)
    scores, mean = score_submission(submission, gt_data, val_images_dir,
                                    image_stems=jpg_stems)
    print(f"Mean val score: {mean:.4f}")

    df = scores_to_dataframe(scores)
    worst_n = df.nsmallest(n, "score")["image_stem"].tolist()

    notes = (f"bottom-{n} by pipeline score on val split; "
             f"mean_val_score={mean:.4f}; model={Path(model_path).name}")
    save_challenge_set(worst_n, method="automatic", notes=notes)
    return worst_n


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Build the 40-image challenge subset for error analysis."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- automatic
    p_auto = sub.add_parser("automatic", help="Run pipeline, pick bottom-N images")
    p_auto.add_argument("--val_images", default="data/processed/dataset/images/val")
    p_auto.add_argument("--gt_dir", default="data/raw/flyingarucov2")
    p_auto.add_argument("--model", default="models/aruco_best.pt")
    p_auto.add_argument("--n", type=int, default=40)
    p_auto.add_argument("--conf", type=float, default=0.5)

    # -- stratified
    p_strat = sub.add_parser("stratified", help="Pick images by heuristic bins")
    p_strat.add_argument("--val_images", default="data/processed/dataset/images/val")
    p_strat.add_argument("--gt_dir", default="data/raw/flyingarucov2")
    p_strat.add_argument("--n_per_bin", type=int, default=10)
    p_strat.add_argument("--seed", type=int, default=42)

    # -- show
    sub.add_parser("show", help="Print existing challenge set")

    args = parser.parse_args()

    if args.mode == "automatic":
        stems = build_challenge_set_automatic(
            val_images_dir=args.val_images,
            gt_dir=args.gt_dir,
            model_path=args.model,
            n=args.n,
            conf_threshold=args.conf,
        )
        print(f"\nChallenge set ({len(stems)} images):")
        for s in stems:
            print(f"  {s}")

    elif args.mode == "stratified":
        stems = build_challenge_set_stratified(
            val_images_dir=args.val_images,
            gt_dir=args.gt_dir,
            n_per_bin=args.n_per_bin,
            seed=args.seed,
        )
        print(f"\nChallenge set ({len(stems)} images):")
        for s in stems:
            print(f"  {s}")

    elif args.mode == "show":
        try:
            stems = load_challenge_set()
            print(f"Challenge set ({len(stems)} images):")
            for s in stems:
                print(f"  {s}")
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)


if __name__ == "__main__":
    _cli()
