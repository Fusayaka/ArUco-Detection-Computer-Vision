import argparse
from src.preprocess import convert_to_yolo_file, split_dataset
from src.train import train_model
from src.inference import generate_submission
from src.download import run_download_pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="ArUco Marker Detection Pipeline - HCMUT Project")

    parser.add_argument("--run-all",   action="store_true", help="Run the entire pipeline (Download -> Prepare -> Train -> Infer)")
    parser.add_argument("--download",  action="store_true", help="Download dataset from Zenodo")
    parser.add_argument("--prepare",   action="store_true", help="Run data preparation step")
    parser.add_argument("--train",     action="store_true", help="Run model training step")
    parser.add_argument("--infer",     action="store_true", help="Run inference on test set")

    # Preprocess args
    parser.add_argument("--raw_dir",   type=str, default="data/raw/flyingarucov2",
                        help="Path to raw dataset")
    parser.add_argument("--out_dir",   type=str, default="data/processed/dataset",
                        help="Path to output processed dataset")
    parser.add_argument("--transform", action="store_true",
                        help="Apply rotation + blur augmentation during data preparation")
    parser.add_argument("--angle",     type=float, default=None,
                        help="Fixed rotation angle (degrees) for augmentation. Random per image if omitted.")

    # Train args
    parser.add_argument("--project",     type=str, default="runs/aruco_training",
                        help="Project run directory")
    parser.add_argument("--run",         type=str, default="v1",
                        help="Run name")
    parser.add_argument("--save",        type=str, default="aruco_best.pt",
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


def _resolve_cuda_device(raw: str | None):
    """
    Parses the --cuda-device string into an int, list of ints, or None.
    Falls back to device 0 if CUDA is available and no device was specified.
    """
    import torch
    if raw is not None:
        cleaned = raw.strip("[] ")
        if "," in cleaned:
            return [int(x.strip()) for x in cleaned.split(",")]
        try:
            return int(cleaned)
        except ValueError:
            return cleaned  # pass string as-is (e.g. "cpu")
    return 0 if torch.cuda.is_available() else None


def main():
    """Entry point — routes execution based on the selected pipeline flags."""
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
        print(f"    - Input     : {args.raw_dir}")
        print(f"    - Output    : {args.out_dir}")
        print(f"    - Transform : {'yes (angle=' + str(args.angle) + ')' if args.transform else 'no'}")
        convert_to_yolo_file(args.raw_dir)
        split_dataset(
            src_dir=args.raw_dir,
            base_dir=args.out_dir,
            apply_transform=args.transform,
            angle=args.angle,
        )
        print("[*] Data preparation completed successfully!")

    if run_train:
        cuda_device = _resolve_cuda_device(args.cuda_device)
        print("\n" + "=" * 50)
        print("[*] Starting training pipeline...")
        print(f"    - Device : {'cuda:' + str(cuda_device) if cuda_device is not None else 'cpu'}")
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