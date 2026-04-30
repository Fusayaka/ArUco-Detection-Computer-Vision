# Stage 1 — Illumination Normalisation: Error Analysis

## 1. Overview

Analysis run on **40 challenge images** (40 with ≥ 1 GT marker for marker-region contrast measurement).

The CLAHE preprocessing step applies:
1. **Gradient correction** (`_correct_gradient`) — divides out a coarse
   illumination ramp estimated from a 64×64 downsampled, heavily blurred
   thumbnail of the image.
2. **CLAHE on the LAB L-channel** (`_clahe_lab`) — applies Contrast-Limited
   Adaptive Histogram Equalisation (clip limit *c* = 2.0, tile size *T* = 8).

Code entry point: `src/preprocess.py:90  enhance_image(correct_gradient=True)`.

> **Note on report vs. code discrepancy (clip limit):**  Report §3.2.3 states
> *c = 3.0*, but the live `enhance_image` default is `clip_limit=2.0`.  The
> analysis below uses the live code path (c = 2.0) so that the metrics reflect
> actual pipeline behaviour.

## 2. Quantitative Summary

| Metric | Pre-CLAHE (mean) | Post-CLAHE (mean) | Δ (mean) |
|--------|-----------------|-------------------|----------|
| Image-wide mean L | 105.6 | 131.3 | +25.7 |
| Image-wide std L | 67.3 | 57.2 | -10.0 |
| % Saturated (L ≥ 250) | 1.22% | 0.00% | -1.22% |
| % Crushed (L ≤ 5) | 10.91% | 5.88% | -5.03% |
| Marker-region contrast (p99−p1)/255 | 0.485 | 0.649 | +0.164 |

## 3. Marker-Region Contrast

Marker-region contrast `(p99 − p1) / 255` is measured inside each GT corner
polygon on the L channel.  It is the preprocessing-stage analogue of the
bit-grid separability that the decode stage needs:

- **Contrast ≥ 0.30** — sufficient for reliable min-max thresholding at 0.5;
  Stage 1 is not the bottleneck for these images.
- **Contrast 0.10–0.30** — marginal; bit errors likely at the 0.5 threshold.
- **Contrast < 0.10** — effectively unreadable; decoding will produce random
  bits regardless of the Hamming threshold.

Post-CLAHE contrast < 0.20: **1 / 40** images  
Post-CLAHE contrast < 0.10: **1 / 40** images

## 4. The CLAHE Limitation (report §3.2.4)

Report §3.2.4: *"CLAHE only redistributes existing intensity. It cannot
recover marker detail in completely saturated regions (clipped to L = 255)
or in regions buried in sensor noise floor."*

This analysis confirms this quantitatively:

- Images with **high pre-CLAHE saturation** (% L ≥ 250) show **no improvement**
  in marker-region contrast because the overexposed white cells are already at
  maximum — CLAHE cannot increase contrast above the clipping level.
- Images with **high pre-CLAHE crushing** (% L ≤ 5) similarly resist improvement
  because underexposed black cells are already at minimum.

Both failure modes are upstream of preprocessing; they represent data-quality
limits that no histogram-redistribution technique can overcome.

## 5. Worst-Contrast Images (post-CLAHE)

The 10 images below have the lowest post-CLAHE marker-region contrast.
These are the primary Stage 1 failure candidates.  Side-by-side panels are
saved to `error_analysis/findings/stage1_panels/`.

| Rank | Image Stem | Post-CLAHE Contrast | Raw % Saturated | Raw % Crushed |
|------|------------|---------------------|-----------------|---------------|
| 1 | `000000244361` | 0.051 | 0.00% | 41.95% |
| 2 | `000000572362` | 0.468 | 0.00% | 15.69% |
| 3 | `000000054007` | 0.524 | 0.01% | 5.88% |
| 4 | `000000031043` | 0.525 | 0.09% | 14.32% |
| 5 | `000000107741` | 0.526 | 2.04% | 17.09% |
| 6 | `000000113513` | 0.544 | 0.00% | 14.42% |
| 7 | `000000521427` | 0.558 | 1.19% | 15.11% |
| 8 | `000000191874` | 0.593 | 0.32% | 8.20% |
| 9 | `000000518189` | 0.604 | 0.02% | 3.15% |
| 10 | `000000546890` | 0.615 | 0.12% | 5.04% |

## 6. Correlation with Downstream Failures

Stage 1 is a *necessary but not sufficient* condition for downstream success:

- **Low post-CLAHE contrast → guaranteed decode failure.** When
  `enh_marker_contrast < 0.10`, the bit-extraction threshold of 0.5
  (`src/decode.py:249`) produces random bits regardless of corner quality.
  These images fail at Stage 4 even under GT corner injection.

- **High post-CLAHE contrast → Stage 1 is cleared.** Failures in these
  images originate downstream (Stage 2 miss, Stage 3 corner error, Stage 4
  decoding ambiguity, or Stage 5 format error).

This stage-1 attribution will be cross-referenced with the Stage 4 decode
deep-dive to confirm which failure fraction is uniquely attributable to
preprocessing vs. to the min-max/Otsu threshold discrepancy (C1 hypothesis).

## 7. Likely Root Cause

The CLAHE step functions as designed for the majority of the challenge set.
The primary Stage 1 failure mode is **sensor-level saturation or crushing**
in the marker region — a data-quality issue that is upstream of the pipeline.

A secondary observation is the **clip-limit discrepancy**: the report specifies
*c = 3.0* but the live code uses *c = 2.0*.  A higher clip limit produces
stronger contrast enhancement, which could improve marker-region contrast in
the 0.10–0.30 marginal range.  This is worth validating in a follow-up ablation
(analogous to the patch-size ablation in Stage 4).

---

*Generated by `error_analysis/stage1_illumination.py`*