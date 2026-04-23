import os
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
raw_folder = repo_root / "data" / "raw" / "flyingarucov2"
test_folder = repo_root / "data" / "test"
processed_folder = repo_root / "data" / "processed"

# This is copying files from raw to processed
def move_files(src_folder, dst_folder):
    """Copy files from source to destination folder."""
    for filename in os.listdir(src_folder):
        src_path = src_folder / filename
        dst_path = dst_folder / filename
        os.copy(src_path, dst_path)

# Remove duplicate files from raw and move them to processed folders
# Build test stems once to avoid repeated directory scans.
test_image_stems = {
    test_file.stem
    for test_file in test_folder.iterdir()
    if test_file.is_file() and test_file.suffix.lower() == ".jpg"
}
i = 0
for raw_file in raw_folder.iterdir():
    if raw_file.is_file() and raw_file.stem in test_image_stems:
        i += 1
        os.remove(raw_file)
print(f"Removed {i} duplicate files from raw folder.")
        
# Copy remaining files from raw to processed
move_files(raw_folder, processed_folder)