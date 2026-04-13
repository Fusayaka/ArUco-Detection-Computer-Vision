import os
import pandas as pd
import argparse
import sys
from detector import HybridDetector

def generate_submission(test_dir, model_path, output_csv):
    """
    Iterates through the test dataset, applies the hybrid detection pipeline, 
    and generates a submission CSV file compliant with Kaggle requirements.

    Args:
        test_dir (str): Path to the directory containing test images.
        model_path (str): Path to the trained YOLOv8 model weights (.pt).
        output_csv (str): Destination path for the generated submission file.
    """

    if not os.path.exists(test_dir):
        print(f"\n[!] Error: Test directory '{test_dir}' not found.")
        print(f"[*] Please download the dataset from the Kaggle competition page:")
        print(f"    >> https://www.kaggle.com/competitions/aruco-detection-challenge")
        print(f"[*] After downloading, place the images in the correct path or use the --test_dir argument.\n")
        sys.exit(1)
    
    detector = HybridDetector(model_path=model_path, conf_threshold=0.5, padding=20)
    
    result_data = []
    
    print(f"Starting inference on images in {test_dir}...")
    for filename in os.listdir(test_dir):
        if not filename.endswith('.jpg'): continue
            
        img_path = os.path.join(test_dir, filename)
        img_id = os.path.splitext(filename)[0]

        prediction_string = detector.process_image(img_path)
        
        result_data.append({
            "image_id": img_id,
            "prediction_string": prediction_string
        })

    df = pd.DataFrame(result_data)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Success! Submission file saved at {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for ArUco Marker Detection")
    
    parser.add_argument(
        "--test_dir", 
        type=str, 
        default="data/raw/aruco_data/test", 
        help="Path to test images directory"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/best.pt", 
        help="Path to the trained .pt model"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="output/submission.csv", 
        help="Path to save the submission CSV"
    )
    
    args = parser.parse_args()
    
    generate_submission(
        test_dir=args.test_dir, 
        model_path=args.model, 
        output_csv=args.out
    )