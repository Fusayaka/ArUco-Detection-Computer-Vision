"""
blur.py — Gaussian blur utility.
"""
import cv2
import numpy as np


def apply_blur(image: np.ndarray, kernel_size: tuple[int, int] = (3, 3)) -> np.ndarray:
    """
    Applies Gaussian blur to *image*.
 
    Args:
        image:       Input image as a NumPy array.
        kernel_size: Blur kernel dimensions (width, height). Both values
                     must be odd positive integers.
 
    Returns:
        Blurred image as a NumPy array.
    """
    return cv2.GaussianBlur(image, kernel_size, sigmaX=0)

def main():
    from pathlib import Path
    import os
    from rotate import rotate_image, rotate_json_labels
    import random
    import json

    ROOT = Path(__file__).resolve().parents[2]

    raw_folder = ROOT / "data" / "raw" / "flyingarucov2"
    output_folder = ROOT / "data" / "processed"
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
        json_path = ROOT / "data" / "raw" / "flyingarucov2" / f"{image.split('.')[0]}.json"
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