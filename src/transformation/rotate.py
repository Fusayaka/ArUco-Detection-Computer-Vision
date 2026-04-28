"""
rotate.py — Image and label rotation utilities.
"""
 
import copy

import cv2
import numpy as np


def rotate_image(mat, angle):
    """
    Rotates an image by *angle* degrees, expanding the canvas to avoid cropping.
 
    Args:
        mat:   Input image as a NumPy array (H x W x C).
        angle: Clockwise rotation angle in degrees.
 
    Returns:
        rotated_mat:     The rotated image.
        rotation_matrix: The 2x3 affine matrix used for the transform,
                         which can be passed to rotate_json_labels.
    """

    height, width = mat.shape[:2]
    center = (width / 2, height / 2)
 
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
 
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # Shift the rotation origin to the center of the new (expanded) canvas.
    rotation_matrix[0, 2] += bound_w / 2 - center[0]
    rotation_matrix[1, 2] += bound_h / 2 - center[1]
 
    rotated_mat = cv2.warpAffine(mat, rotation_matrix, (bound_w, bound_h))
    return rotated_mat, rotation_matrix



def rotate_json_labels(json_data, rotation_matrix):
    """
    Applies *rotation_matrix* to every marker corner stored in *json_data*.
 
    The function is non-destructive — it returns a deep copy and leaves
    the original dict unchanged.
 
    Args:
        json_data:       Dict with a ``"markers"`` key, each marker having
                         a ``"corners"`` list of [x, y] pairs.
        rotation_matrix: 2x3 affine matrix produced by :func:`rotate_image`.
 
    Returns:
        A new dict identical in structure to *json_data* but with all
        corner coordinates transformed.
    """
    rotated_data = copy.deepcopy(json_data)
    
    for marker in rotated_data.get("markers", []):
        rotated_corners = []
        for x, y in marker["corners"]:
            # Convert corner to homogeneous coordinates [x, y, 1]
            point = np.array([x, y, 1.0], dtype=np.float32)
            
            # Apply rotation matrix transformation
            new_x, new_y = rotation_matrix @ point
            
            # Extract x, y from result [new_x, new_y]
            rotated_corners.append([float(new_x), float(new_y)])
        marker["corners"] = rotated_corners
    
    return rotated_data


def main(debug: bool = False):
    import json
    import random
    from pathlib import Path
    import os

    # Example usage: rotate both image and JSON labels
    ROOT = Path(__file__).resolve().parents[2]
    TEST_DIR_IMAGE_PATH = ROOT / "data" / "raw" / "flyingarucov2"

    output_dir = ROOT / "data" / "processed" / "rotate"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [f for f in os.listdir(TEST_DIR_IMAGE_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image in images:
        # Load image
        image_path = os.path.join(TEST_DIR_IMAGE_PATH, image)
        image_instance = cv2.imread(image_path)

        # Load corresponding JSON labels
        json_path = ROOT / "data" / "raw" / "flyingarucov2" / f"{image.split('.')[0]}.json"
        with open(json_path, 'r') as f:
            test_json_data = json.load(f)

        # Generate a random angle between 1 and 359
        if random.random() > 0.2:
            angle = random.randint(1, 359)
            rotated_image, rotation_matrix = rotate_image(image_instance, angle)
            rotated_json_data = rotate_json_labels(test_json_data, rotation_matrix)
        else:
            rotated_image = image_instance
            rotated_json_data = test_json_data

        rotated_json_data = rotate_json_labels(test_json_data, rotation_matrix)

        # Save rotated image and JSON for verification
        output_image_path = output_dir / f"{image.split('.')[0]}.jpg"
        output_json_path = output_dir / f"{image.split('.')[0]}.json"

        cv2.imwrite(str(output_image_path), rotated_image)
        with open(output_json_path, 'w') as f:
            json.dump(rotated_json_data, f, indent=4)

        if debug:
            print(f"Saved rotated image to {output_image_path} and JSON to {output_json_path}")


if __name__ == "__main__":
    main(False)