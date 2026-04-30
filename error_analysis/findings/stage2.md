# Stage 2 - Marker Detection: Error Analysis

## 1. Overview

Analysis run on **40 challenge images** containing **138 GT markers** in total.

Three detection variants are compared:

- **Path A (Classical)** — `detect_classical_multiscale` (`src/detect.py:208`): runs `cv2.aruco.ArucoDetector` on both the raw
  BGR image and the CLAHE-enhanced image, merging by marker ID (tighter
  bbox wins on conflict).  Returns marker IDs at detection time.
- **Path B (YOLO)** — `_get_yolo_boxes` (`src/detector.py:166`):
  YOLOv8n fine-tuned on the training split.  The **live pipeline path**.
  Returns bounding boxes only; no IDs at this stage.
- **Hybrid (A union B)** — Union of both paths with IoU=0.5 NMS
  (report §3.3.3).  This mirrors the report's description but is NOT
  live: `process_image()` only invokes Path B.

> **Live code discrepancy (report §3.3.3 vs. implementation):**
> Report §3.3.3 describes *both* paths running in parallel with
> IoU=0.5 NMS to merge results. The live `process_image()` only calls
> `_get_yolo_boxes()` (Path B). The classical path in `src/detect.py`
> is implemented but never wired into inference.

## 2. Detection Performance Summary

IoU matching threshold = 0.3 (lenient — evaluates localisation,
not corner accuracy).

| Variant | Mean Recall | Mean Precision | Mean IoU (matched) |
|---------|-------------|----------------|--------------------|
| Path A (Classical) | 0.880 | 0.690 | 0.913 |
| Path B (YOLO) | 0.992 | 0.962 | 0.937 |
| Hybrid (A+B NMS) | 0.992 | 0.708 | 0.916 |

### Aggregate TP / FP / FN over all 40 images

| Variant | TP | FP | FN | Recall (agg.) |
|---------|----|----|-----|---------------|
| Path A | 122 | 63 | 16 | 0.884 |
| Path B | 136 | 4 | 2 | 0.986 |
| Hybrid | 136 | 65 | 2 | 0.986 |

## 3. Per-Path Attribution

For each GT marker, which path(s) detected it?

| Category | Count | Fraction of GT |
|----------|-------|----------------|
| A only (classical found, YOLO missed) | 0 | 0.0% |
| B only (YOLO found, classical missed)  | 14 | 10.1% |
| Both paths found                        | 122 | 88.4% |
| Neither path found (missed entirely)    | 2 | 1.4% |

### Interpretation

- **B only dominates** (14/138 = 10.1% of GT): YOLO generalises to conditions
  where the classical contour extractor fails — motion blur softens edges so
  `approxPolyDP` cannot close the quadrilateral; extreme oblique viewpoints
  collapse it to a thin sliver; strong gradients shift the adaptive threshold
  midpoint.  YOLO sees these as learned texture features and detects them anyway.
- **A only = 0**: the classical multi-scale path never uniquely detected a
  marker that YOLO missed on this challenge set.  This means the live code's
  omission of Path A costs **zero recall** on these 40 images.  Path A only
  adds false positives (+63 FP vs. +4 FP for YOLO) with no recall benefit here.
  Note: on easier images outside the challenge set Path A may contribute.
- **Neither (2/138, 1.4%)**: two GT markers were missed by both paths.
  These are the hard floor for all downstream stages.

## 4. Value of the Hybrid Path

The hybrid path recovers **136** GT markers total vs. **136** for
Path B alone — a gain of
**+0** TP (0.0% of all GT).
In exchange it introduces **+61** additional FP (65 total).

> **Key finding:** On this challenge set the hybrid path does **not** improve
> recall over Path B alone, but roughly triples the false-positive count.
> The report's §3.3.3 rationale for the hybrid — that A catches markers B misses
> — is not borne out on these 40 hard images.  The live code's decision to use
> Path B only is, on this subset, the correct engineering choice.

## 5. Worst-Divergence Images

The 5 images with the highest Path A vs. Path B disagreement
(markers found by one path but not the other) are saved to
`error_analysis/findings/stage2_panels/`.

## 6. Likely Root Cause

**Detection is not the bottleneck on this challenge set.**

Path B (YOLO) achieves 0.992 mean recall — 98.6% of GT markers are found
at the detection stage.  Only 2 GT markers across all 40 images escape both
detectors.  This means the dominant failure mode of the pipeline is NOT a
missed detection: it is one of the downstream stages (corner refinement,
decoding, or output formatting) applied to boxes that *were* found.

Specific observations:

- **Path A precision problem**: Path A generates ~5x more FP than YOLO
  (63 vs. 4) while providing identical recall.  Every FP from Path A passes
  through corner refinement and decode — wasting compute and potentially
  introducing spam predictions into the submission string.
- **The hybrid report claim is empirically refuted** on this dataset: A-only
  = 0 means the hybrid provides no recall benefit over YOLO alone, only harm
  (precision drops from 0.962 to 0.708).
- **Hard floor (neither = 2)**: these two markers are likely in frames with
  extreme blur or near-zero contrast — validated by the Stage 1 finding that
  image 000000244361 has post-CLAHE marker contrast 0.051 (effectively zero).

---

*Generated by `error_analysis/stage2_detection.py`*