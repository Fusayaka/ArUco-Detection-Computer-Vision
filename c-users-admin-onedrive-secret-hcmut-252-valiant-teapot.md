# Error Analysis for the ArUco Detection Pipeline

## Context

The report at `report/main.pdf` documents a five-stage ArUco pipeline (illumination
normalisation → marker detection → corner refinement → decoding → output formatting).
A new section "Error Analysis" is needed: dissect each stage, run quantitative
diagnostics on a fixed evaluation set, surface failure cases, and reason about the
underlying causes — with a particular focus on the marker-decoding stage
(`src/decode.py`), which the user suspects is the weakest link.

User-confirmed parameters:

- **Codebase under analysis:** the **parent repo**
  `C:\Users\Admin\Documents\Code\HCMUTcode\252code\CV\BTL\ArUco-Detection-Computer-Vision\`
  (NOT the simplified worktree). All implementation paths below are relative to that root.
- **Scope:** all five stages, with the **decode stage as the deep-dive**.
- **Evaluation set:** a **curated 40-image challenge subset** drawn from the
  validation split (which has 4-corner ground-truth via the original FlyingArUco JSONs).

The deliverable is a structured set of scripts, figures, and a markdown writeup
that can be lifted directly into `Section X — Error Analysis` of the report.

---

## Pipeline ↔ Code Map (Parent Repo)

| Stage | File | Key entry point |
|------|------|-----------------|
| 1 — Illumination Normalisation | `src/preprocess.py:90` | `enhance_image(img_bgr, correct_gradient=True)` |
| 2A — Classical detection | `src/detect.py:159` | `detect_classical()` (multi-scale via `detect_classical_multiscale`, line 208) |
| 2B — YOLOv8n detection | `src/detector.py:87` | `_get_yolo_boxes()` |
| 2 — Merge / NMS | `src/detector.py:273` | inside `process_image()` (NB: only YOLO active in the live code path) |
| 3 — Corner refinement (CNN) | `src/corners.py:268` | `refine_corners_cnn(crop, exp_bbox, model)` |
| 3 — Corner refinement (fallback) | `src/corners.py:197` | `refine_corners_classical()` |
| 4 — Decode | `src/decode.py:323` | `decode_marker(image, corners, max_hamming=6)` |
| 5 — Output formatting | `src/detector.py:244` | `_format_prediction_string()` |
| 5 — Submission driver | `src/inference.py:7` | `generate_submission()` |
| Weights | `models/aruco_best.pt`, `models/best_corners.pth` | |
| Validation set | `data/processed/dataset/images/val/` (500 imgs) + `data/raw/flyingarucov2/*.json` (4-corner GT) |
| Test set | `data/raw/aruco_data/test/` (500 imgs, no GT — Kaggle holds it) |

### Report ↔ Code Discrepancies (already worth flagging)

These are real divergences between the report's claim and `decode.py` as it stands —
they will become part of the error-analysis findings:

1. **Otsu vs min-max**. Report §3.5.2 says "Otsu's method ... maximises inter-class variance"; code at `src/decode.py:96-113` does **min-max normalisation + fixed threshold at 0.5**. Min-max is brittle to a single bright/dark outlier pixel.
2. **Hamming threshold**. Report §3.5.3 says τ=5 ("strictly tighter than τ = 6"); code at `src/decode.py:89` has `MAX_HAMMING = 6`. At τ=6 a tied match between two codewords becomes plausible (since dmin=12 → ambiguity radius ≥ 6).
3. **Final NMS / spam suppression**. Report §3.6.1 describes IoU=0.5 NMS keeping the lowest-Hamming candidate; the live `process_image()` (line 296-330) builds `results` directly without an NMS pass over duplicates from overlapping YOLO boxes.
4. **Hybrid path A+B**. Report §3.3 describes both the classical OpenCV detector and YOLO running in parallel; live code only calls `_get_yolo_boxes()` (line 292). The classical path exists in `src/detect.py` but isn't wired in.

These are not bugs to fix in this task — they are **observations to document** as part of the error analysis (and they explain failure modes we will measure).

---

## Plan of Work

The work is organised into six steps, each producing a script and a results artefact.
All scripts live under a new top-level folder `error_analysis/` to keep them out of
the production `src/` tree.

```
error_analysis/
├── __init__.py
├── scoring.py            # Kaggle Gaussian-distance metric + GT loader
├── challenge_set.py      # build / load the curated 40-image subset
├── stage1_illumination.py
├── stage2_detection.py
├── stage3_corners.py
├── stage4_decode.py      # the deep-dive
├── stage5_output.py
├── failure_cases.py      # end-to-end per-image score + worst-N case studies
└── findings/             # output figures, CSVs, markdown writeup
    ├── stage1.md         (one .md per stage + a master summary)
    ├── stage2.md
    ├── stage3.md
    ├── stage4_decode_deepdive.md
    ├── stage5.md
    └── summary.md
```

### Step 1 — Build the evaluation harness

**File:** `error_analysis/scoring.py`

- Implement the Kaggle scorer end-to-end (it does not exist locally):
  - `gaussian_score(d_norm, sigma=0.02) = exp(-d_norm**2 / (2*sigma**2))`
  - `score_image(predictions, ground_truth, image_shape)` returning per-image score per Eq. (9) in the report: `(1/(N_gt+N_spam)) * Σ φ(d_norm)`. Match by marker ID; mismatched IDs count as `N_spam`.
  - `load_gt_corners(json_path)` parses the FlyingArUco v2 JSON (the keys for marker ID and 4-corner coordinates were observed in `data/raw/flyingarucov2/000000000089.json`). Output: `dict[image_stem -> list[(marker_id, 4×2 corners, canonical_top_left)]]`.
- `cli`: takes a submission CSV + GT folder, prints per-image and aggregate scores.

**File:** `error_analysis/challenge_set.py`

- Build a 40-image curated subset of the validation split. Two construction modes:
  - **`automatic`** (default): run the full pipeline once on `data/processed/dataset/images/val/`, score every image, keep the bottom-40 by Gaussian score (these will dominate failure cases).
  - **`stratified`**: 10 images each from four bins (severe blur / low light / oblique viewpoint / occlusion-or-clutter), labelled by simple heuristics (Laplacian variance for blur, mean luminance for darkness, marker-bbox aspect-ratio range for oblique viewpoint, marker count for clutter).
- Persist the chosen filenames to `error_analysis/findings/challenge_set.txt` so every later script reads the same 40 images.

### Step 2 — Stage 1: Illumination Normalisation

**File:** `error_analysis/stage1_illumination.py`

For each of the 40 challenge images:

1. Load raw BGR; run `enhance_image(img, correct_gradient=True)`.
2. Compute and log to `findings/stage1.csv`:
   - mean / std luminance, % saturated pixels (L≥250) and % crushed pixels (L≤5), pre and post.
   - per-image **marker-region contrast**: using the GT corners, mask the marker interior and compute `(p99 - p1) / 255` on the L channel.
3. Save side-by-side panels (raw | CLAHE | residual) for the worst-10 by post-CLAHE marker-region contrast → `findings/stage1_panels/`.

**Reasoning to document:** CLAHE only redistributes existing intensity (report §3.2.4); on saturated regions (L=255) marker detail is lost and no preprocessing recovers it. We expect to see a strong correlation between low post-CLAHE marker contrast and downstream decoding failures.

### Step 3 — Stage 2: Marker Detection

**File:** `error_analysis/stage2_detection.py`

For each challenge image:

1. Run **three** detection variants and log GT-bbox recall and false-positive counts:
   - Path A only (`detect_classical_multiscale` from `src/detect.py:208`, on the CLAHE-enhanced image).
   - Path B only (`HybridDetector._get_yolo_boxes`, exactly as the live pipeline does).
   - Hybrid (Path A ∪ Path B with IoU=0.5 NMS keeping max-confidence — implement this locally to mirror what the report describes).
2. For each variant emit:
   - `recall = matched_GT / total_GT`, `precision`, mean IoU of best match.
   - **Per-path attribution table:** of the GT markers that were caught, how many were caught by A only, B only, both. This directly speaks to the report's §3.3.3 claim.
3. Visualise Path A vs Path B differences for 5 most-divergent images.

**Reasoning to document:** if Path B catches markers that Path A misses (e.g. heavy blur), this validates the YOLO path and explains why the classical-only path (the live `cv2.aruco.ArucoDetector`) would fail. Conversely, if Path A and B largely agree on the challenge set, the absence of Path A in `process_image()` is a real but small loss.

### Step 4 — Stage 3: Corner Refinement

**File:** `error_analysis/stage3_corners.py`

Use **GT-localised crops** (so this stage is decoupled from detector recall):

1. For every GT marker in the 40 images, expand the GT bbox by 20 % (matching `train_corners.py:97`), produce a 64×64 crop, and run:
   - `refine_corners_cnn` (`src/corners.py:268`)
   - `refine_corners_classical` cornerSubPix fallback (`src/corners.py:197`)
   - the raw expanded bbox corners (no refinement; baseline)
2. Compute per-corner pixel error to GT, per image and per corner index. Emit:
   - histogram of per-corner errors (CNN vs cornerSubPix vs raw),
   - **Gaussian score** for each refinement strategy (φ for σ ≈ 14.6 px on a 728-px diagonal — Eq. (1) of the report).
3. Identify the top-5 highest-error markers per strategy and save cropped panels.

**Reasoning to document:** report §3.4.1 makes the score explicitly dominated by corner accuracy. We expect cornerSubPix to fail on the harder 5–15 px error regime (it lacks global geometry), while the CNN should hold up better; outliers will reveal failure modes (occlusion, motion blur, marker-on-marker).

### Step 5 — Stage 4: Decode (Deep Dive — the Suspect Stage)

**File:** `error_analysis/stage4_decode.py`

This is the most important step. We separate detector/refinement noise from decode
noise by feeding **GT corners directly** into `decode_marker()`.

**5a. Baseline characterisation**

- For every GT marker in the 40 images, run `decode_marker(img, gt_corners)` and record:
  - reported ID vs true ID → ID accuracy.
  - Hamming distance histogram (split by correct/incorrect).
  - rotation index correctness (does the chosen rotation map to the GT canonical TL?).
- Save the distribution of Hamming distances to `findings/stage4_hamming_hist.png`.
- Identify **all incorrectly-decoded markers** (wrong ID or Hamming > τ).

**5b. Component ablation — the heart of the analysis**

Re-implement `decode_marker` as `decode_marker_variant(...)` accepting these knobs,
and sweep on the failure subset (and a sample of successes):

1. **Threshold method** (the leading hypothesis):
   - `min_max_05` — current behaviour (`src/decode.py:96` + `:249`).
   - `otsu` — `cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)` then majority-vote per cell. **This is what the report claims is in use.**
   - `adaptive_mean` — `cv2.adaptiveThreshold` on the 32×32 patch.
   - `border_calibrated` — use the outer ring (which should be all-black per the dictionary spec) to estimate the dark-class mean, white = 1 − dark; threshold at the midpoint.
2. **Patch resolution**: `PATCH_SIZE ∈ {32, 64, 96}` — at the current 4 px/cell, a 1-pixel sub-cell drift is 25 % of cell width; 8 or 12 px/cell is more forgiving.
3. **Cell sampling**: `mean` (current) vs `centre_pixel` vs `gaussian_weighted` (down-weights cell boundaries to absorb sub-pixel warp drift).
4. **Hamming threshold**: τ ∈ {4, 5, 6} — the report uses 5; the code uses 6.

For each combination, report ID accuracy and decoded-Hamming distribution on the
challenge subset. Emit a results table to `findings/stage4_ablation.csv` and rank
ablations by absolute accuracy gain over the current code.

**5c. Border / non-bit checks**

- After the warp, verify the outer-ring cells are predominantly dark. Cases where
  the outer ring is bright → corners are wrong / orientation flipped. Quantify
  how many failures could be caught by a "border-sanity" rejection.
- Quantify the impact of the `normalize_patch` early-exit branch
  (`src/decode.py:110-112`, returns zeros when `hi-lo < 1e-6`): how often does it
  fire on the challenge set? Each fire silently drops a detection.

**Suspected root causes (to confirm/refute via 5b):**

- **C1 — Threshold sensitivity (top suspect).** Min-max normalisation is dominated by extremes; one specular highlight or one dark sensor pixel inside the marker pulls the global min/max and flips bits. Otsu, which uses class statistics, should be measurably more accurate.
- **C2 — Sub-cell warp drift.** With `CELL_SIZE=4` px, even a 1 px corner residual from Stage 3 shifts the cell-averaging window by 25 % of cell width, mixing in a neighbouring cell's pixels.
- **C3 — `MAX_HAMMING = 6` (off-by-one against the report's τ=5).** A distance of exactly 6 is decoding-ambiguous under dmin=12; lowering to 5 trades a small recall hit for a real precision gain — and Eq. (9) penalises false positives at 1× the weight of false negatives.
- **C4 — Border ignored.** The black border carries no information bits but is exactly the calibration signal a robust decoder needs. Currently discarded.

### Step 6 — Stage 5: Output Formatting and NMS

**File:** `error_analysis/stage5_output.py`

1. Count per-image duplicate detections (same ID emitted twice from overlapping YOLO boxes) on the challenge subset; quantify how many of those would be removed by the IoU=0.5 / lowest-Hamming NMS the report describes.
2. Verify the canonical-corner emission: for each correctly-decoded marker, compare `_canonical_top_left(refined_corners, decode.rotation)` against the GT canonical TL. If `refined_corners` are ordered by spatial position rather than ArUco TL→TR→BR→BL, the rotation index in `decode.rotation` will systematically index the wrong corner.

**Reasoning to document:** even with a perfect ID, a wrong canonical corner shifts the reported (x,y) by up to one marker-edge length, which exceeds 2σ and drops the per-marker score below ~0.4.

### Step 7 — End-to-end failure case studies

**File:** `error_analysis/failure_cases.py`

For the bottom-10 challenge images by full-pipeline score:

- Render a single multi-panel figure per image (raw | CLAHE | YOLO bboxes | refined corners overlay | warped 32×32 patch | extracted bits with codeword overlay | predicted vs GT IDs).
- Auto-attribute each failure to the stage that broke it, using the per-stage diagnostics already collected:
  - **Stage 1 fault** if marker-region contrast post-CLAHE < 0.05.
  - **Stage 2 fault** if no YOLO box overlaps the GT bbox at IoU ≥ 0.3.
  - **Stage 3 fault** if max per-corner pixel error > 10 px when fed GT bbox.
  - **Stage 4 fault** if decode under GT corners returns wrong ID or `None`.
  - **Stage 5 fault** if decode is correct but the canonical TL is mis-indexed.
- Emit `findings/failures.md` with one section per case.

### Step 8 — Master writeup

**File:** `error_analysis/findings/summary.md`

Aggregate the per-stage findings, ablation tables, and failure-case figures into
a single, paste-into-the-report markdown document. Structure mirrors §X.1…§X.5
of the eventual report section: one subsection per pipeline stage, each ending
with a "Likely cause" paragraph drawn from the diagnostics.

---

## Critical Files (Read-Reference)

These are the files to **read** while implementing the analysis (not modify):

- `src/decode.py:80-145` — dictionary build, constants, `normalize_patch`.
- `src/decode.py:168-251` — warp + bit extraction (the deep-dive subject).
- `src/decode.py:278-356` — match + threshold (the τ choice).
- `src/corners.py:171-189` — corner-CNN loader; `:268-311` — CNN inference; `:197-246` — cornerSubPix fallback.
- `src/detect.py:119-259` — classical detection params and multi-scale wrapper.
- `src/detector.py:273-330` — live `process_image()` flow.
- `src/preprocess.py:90-232` — `enhance_image`, `_clahe_lab`, `_correct_gradient`.
- `src/inference.py:7-45` — submission driver.
- `data/raw/flyingarucov2/000000000089.json` — example GT format (read once to confirm field names before writing the GT loader).

## Verification

After implementation:

1. **Sanity-check the scorer.** Run `error_analysis/scoring.py` against the existing `output/submission.csv` if present, or against a hand-crafted "perfect" submission derived from GT — score should be ≈ 1.0.
2. **Run each stage script** on a 5-image dry-run subset before scaling to 40, to catch path/import bugs.
3. **Confirm decode ablation reproduces the live code's accuracy** when the variant is set to `(min_max_05, PATCH_SIZE=32, mean, τ=6)` — must match `decode_marker()` exactly.
4. **Inspect at least 3 of the bottom-10 failure-case figures by hand** to confirm the auto-attribution is correct (i.e. the script isn't silently mis-blaming a stage).
5. **End-state**: `error_analysis/findings/summary.md` exists, contains numbers (not just prose), references the per-stage figures, and has a "Likely root cause" paragraph for each stage. The decode-deep-dive section names which threshold method / patch size / τ value each accounts for the largest accuracy delta in the ablation.
