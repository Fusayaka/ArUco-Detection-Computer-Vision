# Data Generator Updates

This document summarizes the recent data-generation and dataset-organization changes added to the project.

## What Changed

### 1) Rotation generator added
- File: src/transformation/rotate.py
- Adds random rotation augmentation for images from data/raw/flyingarucov2.
- Applies the same affine transform to JSON marker corners so labels stay aligned with rotated images.
- Writes outputs to data/processed as:
	- rotated_<image_id>.jpg
	- rotated_<image_id>.json

### 2) Blur generator added (with rotation)
- File: src/transformation/blur.py
- Generates augmented images by:
	- random rotation
	- Gaussian blur
- Reuses rotation logic from rotate.py so transformed JSON corners remain correct.
- Writes outputs to data/processed as:
	- rotated_blurred_<image_id>.jpg
	- rotated_blurred_<image_id>.json

### 3) Test folder handling added
- File: src/remove_test_from_raw.py
- Introduces test-folder filtering using data/test.
- Removes raw files that overlap with test image IDs to avoid data leakage.
- Copies the remaining raw files into data/processed.


## Project Structure Changes

New/updated data flow and folders:

data/
- raw/flyingarucov2/         # original training images + json labels
- test/                      # test images used to filter overlaps from raw
- processed/                 # rotated and blurred outputs and/or cleaned training set copy

Code modules involved:
- src/transformation/rotate.py
- src/transformation/blur.py
- src/remove_test_from_raw.py


## ! These step should be run after step 2 in original note:

Run all commands from the project root.


### 1) Remove test overlaps from raw and copy cleaned set

```bash
python src/remove_test_from_raw.py
```

Expected output:
- Files in data/raw/flyingarucov2 that match stems in data/test are removed.
- Remaining files are copied to data/processed.

### 2) Generate rotated data

```bash
python src/transformation/rotate.py
```

Expected output:
- Augmented rotated image/json pairs in data/processed.

### 3) Generate rotated + blurred data

```bash
python src/transformation/blur.py
```

Expected output:
- Augmented rotated-blurred image/json pairs in data/blurred.

## Notes
- Keep image and JSON filenames paired (same stem) so transformations can locate labels.
- If you want reproducible augmentation, set a fixed random seed in rotate.py and blur.py.
