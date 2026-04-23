"""Generate random data what have blur effect and rotated images with corresponding JSON labels for testing purposes."""
from pathlib import Path
import os
from rotate import rotate_image, rotate_json_labels
import cv2
import random
import json

repo_root = Path(__file__).resolve().parents[2]

def apply_blur(image, kernel_size=(3, 3)):
    """Applies Gaussian blur to the input image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def main():
    raw_folder = repo_root / "data" / "raw" / "flyingarucov2"
    output_folder = repo_root / "data" / "processed"
    output_folder.mkdir(parents=True, exist_ok=True)
    images = [f for f in os.listdir(raw_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for image in images:
        # Load image
        image_instance = cv2.imread(str(raw_folder / image))

        # Generate a random angle between 1 and 359
        angle = random.randint(1, 359)
        rotated_image, rotation_matrix = rotate_image(image_instance, angle)

        # Apply blur on the rotated image only.
        rotated_blurred_image = apply_blur(rotated_image)

        # Load corresponding JSON labels
        json_path = repo_root / "data" / "raw" / "flyingarucov2" / f"{image.split('.')[0]}.json"
        with open(json_path, 'r') as f:
            test_json_data = json.load(f)

        rotated_json_data = rotate_json_labels(test_json_data, rotation_matrix)

        # Save one processed output image (rotated + blurred + boxed).
        output_image_path = output_folder / f"rotated_blurred_{image.split('.')[0]}.jpg"

        rotated_json_path = output_folder / f"rotated_blurred_{image.split('.')[0]}.json"
        with open(rotated_json_path, 'w') as f:
            json.dump(rotated_json_data, f, indent=4)
        cv2.imwrite(str(output_image_path), rotated_blurred_image)

        print(f"Saved processed image to {output_image_path} and JSON to {rotated_json_path}")

if __name__ == "__main__":
    main()