import json
import numpy as np
import copy
import random
import os
from pathlib import Path
import cv2

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # Image shape has three dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # The rotation calculates the cosine and sine,
    # taking the absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # Find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # Subtract the old image center (bringing the image
    # back to the origo) and adding the new image
    # center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # Rotate the image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat, rotation_mat



def rotate_json_labels(json_data, rotation_matrix):
    """
    Rotates marker corner coordinates in JSON labels using the same transformation matrix.
    
    Args:
        json_data: Dictionary with 'markers' key containing list of markers
        rotation_matrix: 2x3 affine transformation matrix from cv2.getRotationMatrix2D
    
    Returns:
        Updated JSON data with rotated corner coordinates
    """
    rotated_data = copy.deepcopy(json_data)
    
    if "markers" not in rotated_data:
        return rotated_data
    
    for marker in rotated_data["markers"]:
        
        corners = marker["corners"]
        rotated_corners = []
        
        for corner in corners:
            # Convert corner to homogeneous coordinates [x, y, 1]
            point = np.array([corner[0], corner[1], 1], dtype=np.float32)
            
            # Apply rotation matrix transformation
            rotated_point = rotation_matrix @ point
            
            # Extract x, y from result [new_x, new_y]
            rotated_corners.append([float(rotated_point[0]), float(rotated_point[1])])
        
        marker["corners"] = rotated_corners
    
    return rotated_data


def main():
    # Example usage: rotate both image and JSON labels
    repo_root = Path(__file__).resolve().parents[2]
    test_image_path = repo_root / "data" / "raw" / "flyingarucov2" / "000000000089.jpg"
    test_json_path = repo_root / "data" / "raw" / "flyingarucov2" / "000000000089.json"

    test_image = cv2.imread(str(test_image_path))
    if test_image is None:
        raise FileNotFoundError(f"Could not read image: {test_image_path}")

    # Load JSON labels
    with open(test_json_path, 'r') as f:
        test_json_data = json.load(f)

    for image in range(4):
        angle = random.randint(0, 359)  # Generate a random angle between 0 and 359
        rotated_image, rotation_matrix = rotate_image(test_image, angle)
        rotated_json_data = rotate_json_labels(test_json_data, rotation_matrix)
        # Save rotated image and JSON for verification
        output_image_path = repo_root / "data" / "processed" / f"rotated_{angle}.jpg"
        output_json_path = repo_root / "data" / "processed" / f"rotated_{angle}.json"
        cv2.imwrite(str(output_image_path), rotated_image)
        with open(output_json_path, 'w') as f:
            json.dump(rotated_json_data, f, indent=4)
        print(f"Saved rotated image to {output_image_path} and JSON to {output_json_path}")


if __name__ == "__main__":
    main()