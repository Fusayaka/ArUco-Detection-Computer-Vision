import os
import shutil
import argparse
import torch
from ultralytics import YOLO

def train_model(project_path, run_name, output_model_name, cuda_device=None):
    """
    Configures and executes the YOLOv8 training process on the custom ArUco dataset.
    Automatically copies and renames the best weights to the 'models/' directory after training.

    Args:
        project_path (str): Root directory where training experiments are stored 
            (e.g., 'runs/').
        run_name (str): Specific name for this training run session 
            (e.g., 'v0').
        output_model_name (str): The final filename for the best weights 
            within the 'models/' folder (e.g., 'v0.pt').
    """

    base_model_path = os.path.join("models", "yolov8n.pt")

    if not os.path.exists(base_model_path):
        print(f"[*] Base model weights not found at {base_model_path}. Downloading...")
        
        model = YOLO("yolov8n.pt") 
        
        if os.path.exists("yolov8n.pt"):
            shutil.move("yolov8n.pt", base_model_path)
            print(f"Successfully downloaded and relocated base model to: {base_model_path}")
    else:
        print(f"[*] Loading base model from: {base_model_path}")
        model = YOLO(base_model_path)

    print("="*50)
    print(f"Initializing training. Results will be stored at: {project_path}/{run_name}")
    print("-"*50)
    
    device = cuda_device if cuda_device is not None else 0 if torch.cuda.is_available() else "cpu"
    results = model.train(
        data="config/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project=project_path,
        name=run_name,
        workers=0,
        exist_ok=True
    )
    
    best_model_path = os.path.join(project_path, run_name, "weights", "best.pt")
    target_path = os.path.join("models", output_model_name)

    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, target_path)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Best weights saved to: {target_path}")
        print("="*50 + "\n")
    else:
        print(f"Warning: Training finished but weights were not found at: {best_model_path}")
        print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script for ArUco Marker Detection")
    
    parser.add_argument(
        "--project", 
        type=str, 
        default="runs/aruco_training", 
        help="Root directory for storing training experiments"
    )
    parser.add_argument(
        "--run", 
        type=str, 
        default="experiment_v0", 
        help="Sub-directory name for this specific training run"
    )
    parser.add_argument(
        "--save", 
        type=str, 
        default="best.pt", 
        help="Final filename for the best weights in the 'models/' directory"
    )
    
    args = parser.parse_args()

    train_model(
        project_path=args.project, 
        run_name=args.run, 
        output_model_name=args.save,
    )