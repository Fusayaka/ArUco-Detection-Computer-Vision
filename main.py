import argparse
from src.preprocess import convert_to_yolo_file, split_dataset
from src.train import train_model
from src.inference import generate_submission
from src.download import run_download_pipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="ArUco Marker Detection Pipeline - HCMUT Project")
    
    parser.add_argument("--run-all", action="store_true", help="Run the entire pipeline (Download -> Prepare -> Train -> Infer)")
    parser.add_argument("--download", action="store_true", help="Run data download step")
    parser.add_argument("--prepare", action="store_true", help="Run data preparation step")
    parser.add_argument("--train", action="store_true", help="Run model training step")
    parser.add_argument("--infer", action="store_true", help="Run inference on test set")

    # Download args
    parser.add_argument("--source", type=str, choices=["both", "kaggle", "zenodo"], default="both", 
                        help="Select data source to download: 'both' (default), 'kaggle', or 'zenodo'")

    # Preprocess args
    parser.add_argument("--raw_dir", type=str, default="data/raw/flyingarucov2", help="Path to raw dataset")
    parser.add_argument("--out_dir", type=str, default="data/processed/dataset", help="Path to output processed dataset")

    # Train args
    parser.add_argument("--project", type=str, default="runs/aruco_training", help="Project run directory")
    parser.add_argument("--run", type=str, default="v1", help="Run name")
    parser.add_argument("--save", type=str, default="aruco_best.pt", help="Output model name in models/")

    # Inference args
    parser.add_argument("--test_dir", type=str, default="data/raw/aruco_data/test", help="Test images directory")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to trained model weights")
    parser.add_argument("--out", type=str, default="output/submission.csv", help="Output CSV path")

    return parser.parse_args(), parser

def main():
    """Entry point of the application. Routes logic based on parsed commands."""
    args, parser = parse_arguments()

    if not (args.run_all or args.prepare or args.train or args.infer):
        parser.print_help()
        return

    run_download = args.run_all or args.download
    run_prepare = args.run_all or args.prepare
    run_train = args.run_all or args.train
    run_infer = args.run_all or args.infer

    if run_download:
        run_download_pipeline(args.source)

    if run_prepare:
        print("\n" + "="*50)
        print("[*] Starting data preparation pipeline...")
        print(f"    - Input  : {args.raw_dir}")
        print(f"    - Output : {args.out_dir}")
        convert_to_yolo_file(args.raw_dir)
        split_dataset(args.raw_dir, args.out_dir)
        print("[*] Data preparation completed successfully!")

    if run_train:
        print("\n" + "="*50)
        print("[*] Starting training pipeline...")
        train_model(
            project_path=args.project, 
            run_name=args.run, 
            output_model_name=args.save
        )

    if run_infer:
        print("\n" + "="*50)
        print("[*] Starting inference pipeline...")
        generate_submission(
            test_dir=args.test_dir, 
            model_path=args.model, 
            output_csv=args.out
        )
        
    print("\n🎉 ALL SELECTED PIPELINES COMPLETED SUCCESSFULLY!\n")

if __name__ == "__main__":
    main()