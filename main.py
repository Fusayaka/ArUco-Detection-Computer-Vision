import argparse
from src.preprocess import convert_to_yolo_file, split_dataset
from src.train import train_model
from src.inference import generate_submission
from src.download import run_download_pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="ArUco Marker Detection Pipeline - HCMUT Project")

    parser.add_argument("--run-all",  action="store_true", help="Run the entire pipeline (Download -> Prepare -> Train -> Infer)")
    parser.add_argument("--download", action="store_true", help="Download dataset from Zenodo")
    parser.add_argument("--prepare",  action="store_true", help="Run data preparation step")
    parser.add_argument("--train",    action="store_true", help="Run model training step")
    parser.add_argument("--infer",    action="store_true", help="Run inference on test set")

    # Preprocess args
    parser.add_argument("--raw_dir", type=str, default="data/raw/flyingarucov2",
                        help="Path to raw dataset")
    parser.add_argument("--out_dir", type=str, default="data/processed/dataset",
                        help="Path to output processed dataset")

    # Train args
    parser.add_argument("--project", type=str, default="aruco_training",
                        help="Project run directory")
    parser.add_argument("--run",     type=str, default="v1",
                        help="Run name")
    parser.add_argument("--save",    type=str, default="best.pt",
                        help="Output model name in models/")
    parser.add_argument("--cuda-device", type=str, default=None,
                        help="CUDA device index to use for training")

    # Inference args
    parser.add_argument("--test-dir", type=str, default="data/raw/aruco_data/test",
                        help="Test images directory")
    parser.add_argument("--model",    type=str, default="models/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--out",      type=str, default="output/submission.csv",
                        help="Output CSV path")

    return parser.parse_args(), parser


def main():
    """Entry point"""
    args, parser = parse_arguments()

    run_download = args.run_all or args.download
    run_prepare  = args.run_all or args.prepare
    run_train    = args.run_all or args.train
    run_infer    = args.run_all or args.infer

    if not any([run_download, run_prepare, run_train, run_infer]):
        parser.print_help()
        return

    if run_download:
        run_download_pipeline()

    if run_prepare:
        print("\n" + "=" * 50)
        print("[*] Starting data preparation pipeline...")
        print(f"    - Input  : {args.raw_dir}")
        print(f"    - Output : {args.out_dir}")
        convert_to_yolo_file(args.raw_dir)
        split_dataset(args.raw_dir, args.out_dir)
        print("[*] Data preparation completed successfully!")

    if run_train:
        import torch
        if args.cuda_device is not None:
            raw_device = args.cuda_device
            raw_device = raw_device.replace('[', '').replace(']', '').strip()
                
            if ',' in raw_device:
                cuda_device = [int(x.strip()) for x in raw_device.split(',')]
            else:
                try:
                    cuda_device = int(raw_device)
                except ValueError:
                    cuda_device = raw_device
        else:
            cuda_device = 0 if torch.cuda.is_available() else None
 
        print("\n" + "=" * 50)
        print("[*] Starting training pipeline...")
        print(f"    - Device : {'cuda:' + str(cuda_device) if cuda_device is not None else 'None'}")
        train_model(
            project_path=args.project,
            run_name=args.run,
            output_model_name=args.save,
            cuda_device=cuda_device,
        )

    if run_infer:
        print("\n" + "=" * 50)
        print("[*] Starting inference pipeline...")
        generate_submission(
            test_dir=args.test_dir,
            model_path=args.model,
            output_csv=args.out,
        )

    print("\n🎉 ALL SELECTED PIPELINES COMPLETED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()