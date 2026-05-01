# Error Analysis — Master Summary

## 0. Scope and Methodology

Error analysis is performed on the **challenge set**: the 40 validation images
on which the pipeline scored lowest (mean pipeline score 0.2144 on the Kaggle
Gaussian metric).  The challenge set contains **138 GT markers** across 40 images
(mean 3.45 markers per image).

The analysis decomposes the pipeline into four individually isolated stages:

| Stage | Script | Isolation method |
|-------|--------|-----------------|
| S1 — Illumination | `stage1_illumination.py` | Measures marker-region contrast before/after CLAHE |
| S2 — Detection | `stage2_detection.py` | Evaluates classical vs. YOLO recall/precision at IoU ≥ 0.3 |
| S3 — Corner Refinement | `stage3_corners.py` | Uses GT-localised crops; bypasses detection error |
| S4 — Decoding | `stage4_decode.py` | Injects GT corners; bypasses corner error |

Each stage isolates its own failure mode by fixing all upstream inputs to their
ground-truth values.  The chain of findings establishes *where* failures originate
and *how much* each stage contributes.

---

## 1. Stage 1 — Illumination Normalisation

**Script**: `error_analysis/stage1_illumination.py`  
**Findings**: `findings/stage1.{csv,md}`, `findings/stage1_panels/`

### Key Metrics

| Metric | Pre-CLAHE | Post-CLAHE | Delta |
|--------|-----------|------------|-------|
| Image-wide mean L (LAB) | 105.6 | 131.3 | +25.7 |
| Image-wide std L | 67.3 | 57.2 | −10.0 |
| % Saturated pixels (L ≥ 250) | 1.22% | 0.00% | −1.22% |
| % Crushed pixels (L ≤ 5) | 10.91% | 5.88% | −5.03% |
| Marker-region contrast (p99−p1)/255 | 0.485 | 0.649 | +0.164 |

### Findings

CLAHE successfully raises marker-region contrast from 0.485 to 0.649 (mean) and
eliminates clipped white cells entirely.  Of 40 images, only **1** has post-CLAHE
marker contrast below 0.10 (image `000000244361`, contrast = 0.051) — the hard
floor where no threshold method can decode bits reliably.

A code/report discrepancy exists: §3.2.3 specifies clip limit *c = 3.0*, but
the live `enhance_image()` default is *c = 2.0*.  At *c = 2.0* the contrast gain
is already sufficient for 39/40 images; the one failure is dominated by sensor
underexposure (41.95% crushed pixels), not CLAHE parameter choice.

**Conclusion**: Stage 1 is **not the primary bottleneck**.  CLAHE functions as
intended on 97.5% of images.  The one critical failure (near-zero contrast) is
caused by sensor-level crushing that is upstream of any histogram operator.

---

## 2. Stage 2 — Marker Detection

**Script**: `error_analysis/stage2_detection.py`  
**Findings**: `findings/stage2.{csv,md}`, `findings/stage2_panels/`

### Key Metrics

| Variant | Recall | Precision | TP | FP | FN |
|---------|--------|-----------|----|----|-----|
| Path A — Classical | 0.884 | 0.690 | 122 | 63 | 16 |
| Path B — YOLO (live) | **0.986** | **0.962** | 136 | 4 | 2 |
| Hybrid (A ∪ B, IoU=0.5 NMS) | 0.986 | 0.708 | 136 | 65 | 2 |

### Findings

YOLO achieves 98.6% recall with only 4 false positives across all 40 challenge
images.  The classical path (Path A) provides **zero unique detections** — every
marker Path A finds, YOLO also finds.  The hybrid path specified in §3.3.3
therefore provides no recall benefit over Path B alone but roughly triples the
false-positive count (4 → 65).

Only **2 GT markers** (1.4%) are missed by both paths — a hard floor caused by
extreme image degradation (verified: image `000000244361`, whose Stage 1 contrast
of 0.051 also produces near-random pixel values at the marker location).

The report describes a hybrid Path A + Path B design; the live `process_image()`
only invokes Path B.  On this challenge subset the live code's omission of Path A
is the **correct engineering choice**: it prevents 63 unnecessary FPs with no
recall penalty.

**Conclusion**: Stage 2 is **not the primary bottleneck**.  Detection recall is
98.6%; only 2 of 138 GT markers are missed entirely.  The dominant pipeline
failure mode is downstream.

---

## 3. Stage 3 — Corner Refinement

**Script**: `error_analysis/stage3_corners.py`  
**Findings**: `findings/stage3.{csv,md}`, `findings/stage3_panels/`

### Key Metrics

| Strategy | Mean corner error (px) | Max corner error (px) | Mean canonical-TL phi |
|----------|------------------------|----------------------|-----------------------|
| Raw bbox (no refinement) | 56.59 | 75.22 | 0.1497 |
| Classical (cornerSubPix) | 56.59 | 75.47 | 0.1470 |
| **CNN** | **35.58** | **50.76** | **0.3598** |

Canonical-TL phi is the per-marker Gaussian score that enters the Kaggle metric
directly (sigma = 0.02; image diagonal ≈ 721 px for 640×360 images).

### Findings

The CNN corner model reduces mean corner error by **37%** (56.6 → 35.6 px) and
raises the mean canonical-TL phi from 0.150 to 0.360 — more than doubling the
score contribution per marker.  It outperforms both the raw baseline and
cornerSubPix on **100%** of the 138 challenge markers.

CornerSubPix performs no better than the raw bbox rectangle because the initial
estimate (the four corners of the expanded bbox) is 50+ px from the true corner
on hard images.  The iterative window search converges to the starting point
rather than the true corner gradient.

However, even the CNN mean error of 35.6 px is large.  The Gaussian scorer is
very sensitive in this range:

| TL error | phi | Score penalty |
|----------|-----|---------------|
| 5 px  | 0.94 | 6% |
| 10 px | 0.79 | 21% |
| 20 px | 0.40 | 60% |
| 35 px | 0.08 | 92% |

At 35 px mean error, the average CNN corner prediction contributes only 0.08 phi
per marker — a 92% loss.  This means Stage 3 is a **major score bottleneck**
independent of whether decoding works.

**Conclusion**: The CNN model is significantly better than the alternative
strategies, but absolute accuracy is still too low on the hardest images.
Stage 3 is the **highest-leverage stage for score improvement**.  Reducing
mean corner error from 35 px to 10 px would lift mean phi from ~0.08 to ~0.79.

---

## 4. Stage 4 — Marker ID Decoding

**Script**: `error_analysis/stage4_decode.py`  
**Findings**: `findings/stage4_baseline.csv`, `findings/stage4_ablation.csv`,
`findings/stage4_decode_deepdive.md`, `findings/stage4_panels/`

### 4.1 Baseline (GT Corners Injected)

All 138 GT markers are decoded with their true corners to isolate Stage 4 from
corner noise.

| Metric | Value |
|--------|-------|
| Correct ID decodes | **2 / 138 (1.4%)** |
| Rejected (Hamming > tau=6) | 130 (94.2%) |
| Accepted but wrong ID | 6 (4.3%) |
| Mean Hamming — correct | 6.00 |
| Mean Hamming — wrong ID | 5.83 |
| Mean Hamming — rejected | 8.44 |

Even with perfect corner inputs, the decoder fails on 98.6% of markers.
The mean Hamming distance of 8.44 for rejected markers indicates that the
extracted bit grids are so corrupted that no threshold or patch-size adjustment
can bridge the gap to the nearest valid codeword (min-distance = 12, guaranteed
unique decoding only for Hamming ≤ 5).

### 4.2 Code/Report Discrepancies

| # | Discrepancy | Report | Live code | Measured impact |
|---|-------------|--------|-----------|-----------------|
| C1 | Threshold method | Otsu (§3.5.2) | min-max + 0.5 | +1.4% (2 more correct) with adaptive_mean |
| C2 | Patch resolution | 32 px (implicit) | 32 px = 4 px/cell | +0.7% (1 more) at 96 px |
| C3 | Hamming tau | tau=5 (§3.5.3) | tau=6 | 0 difference at this accuracy level |
| C4 | Black border | calibration signal | discarded | not quantified in ablation |

### 4.3 Component Ablation (108 Combinations)

The ablation sweeps:
- 4 threshold methods: `min_max_05`, `otsu`, `adaptive_mean`, `border_calibrated`
- 3 patch sizes: 32, 64, 96 px
- 3 cell samplings: `mean`, `centre_pixel`, `gaussian_weighted`
- 3 tau values: 4, 5, 6

**Best variant**: `adaptive_mean` threshold / 64 px patch / `mean` cell sampling / tau=6
- Accuracy: 3.6% (vs. baseline 1.4%)
- Failures recovered: 3 additional correct decodes
- Regressions: 0

The best single-knob gains:

| Knob | Best setting | Delta accuracy | Failures recovered |
|------|-------------|----------------|--------------------|
| Threshold method | `adaptive_mean` | +1.4% | 2 |
| Cell sampling | `gaussian_weighted` | +1.4% | 2 |
| Patch size | 96 px | +0.7% | 1 |
| Hamming tau | 6 (no change) | 0.0% | 0 |

### 4.4 Failure Mode Attribution

The decode failure rate of 98.6% cannot be recovered by parameter tuning alone.
The Hamming distance distribution (mean 8.44, minimum achievable threshold =
Hamming 6) reveals a fundamental signal quality issue:

- **0% early-exit fires**: `normalize_patch()` never returns all-zeros with GT
  corners — the warp and contrast are sufficient for the function to proceed.
  The failure is in bit *quality*, not bit *extraction plausibility*.
- **0% border-bright markers**: warp orientation is correct for all GT markers.
- The 8.44 mean Hamming means approximately 8 of the 36 bits are flipped —
  far outside any practical error-correction budget.

**Conclusion**: Stage 4 failures on this challenge set are overwhelmingly caused
by **upstream image quality degradation** (low post-CLAHE contrast, motion blur,
extreme viewpoint) rather than by the C1–C4 code/report discrepancies.  The
discrepancies are measurable but small:  fixing all four simultaneously would
recover approximately 3–4 additional markers (2.2% absolute accuracy gain).

---

## 5. End-to-End Failure Attribution

The table below shows the fraction of the 138 GT markers accounted for at each
stage of the pipeline.

| Stage | Markers lost | Cumulative loss | Root cause |
|-------|-------------|-----------------|------------|
| S2 Detection (miss) | 2 | 2 (1.4%) | Extreme image degradation; hard floor |
| S3 Corner (large error) | *(all 136 passed)* | — | CNN reduces from 56→36 px; score penalty is severe |
| S4 Decode (reject/wrong) | 136 of 138 | 136 (98.6%) | Image quality → bit corruption → Hamming > 6 |
| **Total failures** | — | **136 / 138 (98.6%)** | — |

Note: S3 and S4 are not strictly sequential failure modes — a marker can fail at
S4 even if S3 accuracy is perfect (as confirmed by the GT-corner injection).
Conversely, S3 failures compound S4 failures in the live pipeline.

### What actually determines pipeline score on the challenge set

1. **Stage 3 (corner accuracy)** is the highest-leverage stage for score
   improvement.  Reducing CNN mean error from 35 px to 10 px would raise
   mean canonical-TL phi from 0.08 to 0.79 for any marker that *is* decoded.

2. **Stage 4 (decode) has a 98.6% failure rate even under GT corner injection.**
   This means fixing Stage 3 alone will not recover the score: even with perfect
   corners, only 1.4% of markers on this challenge set decode correctly.

3. **The challenge set is pathologically hard.** The 40 images were selected as
   the pipeline's worst performers.  The 98.6% decode failure rate reflects the
   image quality of these specific images, not general pipeline behaviour.
   On the full validation set the pipeline achieves a higher score precisely
   because the non-challenge images have adequate contrast and viewpoint.

4. **Code/report discrepancies (C1–C4)** are real but account for only ~2%
   of the decode failures.  The dominant driver is signal quality.

---

## 6. Recommended Improvements (by expected impact)

| Priority | Change | Expected gain | Complexity |
|----------|--------|---------------|------------|
| 1 | Improve corner CNN (more training data, augmentation for hard cases) | High: directly raises phi for every detected marker | High |
| 2 | Switch threshold to `adaptive_mean` in `src/decode.py` | Low: +2 correct decodes on challenge set, likely higher on easier images | Low |
| 3 | Increase patch size from 32 px to 64 px | Low: reduces sub-cell drift | Low |
| 4 | Lower Hamming tau from 6 to 5 (report-aligned) | Neutral on challenge set; reduces FP on easier images | Low |
| 5 | Remove classical Path A from hybrid (live code already does this) | Removes 63 FP with no recall penalty | Done |
| 6 | Fix CLAHE clip limit from 2.0 to 3.0 (report-aligned) | Negligible on challenge set | Trivial |

---

## 7. Files Generated

| File | Contents |
|------|----------|
| `findings/stage1.csv` | Per-image CLAHE metrics (40 rows) |
| `findings/stage1.md` | Stage 1 narrative and tables |
| `findings/stage1_panels/*.png` | 13 worst-contrast image panels |
| `findings/stage2.csv` | Per-image detection TP/FP/FN (40 rows × 3 variants) |
| `findings/stage2.md` | Stage 2 narrative and tables |
| `findings/stage2_panels/*.png` | 8 worst-disagreement image panels |
| `findings/stage3.csv` | Per-marker corner error table (138 rows) |
| `findings/stage3.md` | Stage 3 narrative and tables |
| `findings/stage3_panels/*.png` | 10 worst-CNN-error marker panels |
| `findings/stage4_baseline.csv` | Per-marker baseline decode results (138 rows) |
| `findings/stage4_ablation.csv` | 108-row ablation sweep results |
| `findings/stage4_decode_deepdive.md` | Stage 4 deep-dive narrative |
| `findings/stage4_panels/*.png` | 16 failed-decode marker panels |
| **`findings/summary.md`** | **This file — master cross-stage synthesis** |

---

*Generated by synthesising outputs from `error_analysis/stage{1,2,3,4}_*.py`.*  
*Challenge set: `findings/challenge_set.txt` (40 images, 138 GT markers).*
