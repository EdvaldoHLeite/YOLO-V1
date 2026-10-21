import argparse

from test import test_dataset
from utils import YOLOVOCDataset


def main():
    parser = argparse.ArgumentParser(
        description="YOLO vs R-CNN Object Detection Benchmark"
    )

    # Define command-line arguments
    parser.add_argument(
        "--model",
        type=str,
        # required=True,
        choices=["yolo", "rcnn"],
        help="Choose model architecture",
    )
    parser.add_argument(
        "--mode",
        type=str,
        # required=True,
        choices=["train", "eval"],
        help="Run training or evaluation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for dataloader"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )

    parser.add_argument("--load_dataset", type=bool, default=False)

    parser.add_argument("--test_dataset", action="store_true")

    args = parser.parse_args()

    if args.model == "yolo" and args.mode == "train":
        print("Initializing YOLOv1 Training Pipeline...")
        # import train_yolo from train.py and execute

    elif args.model == "rcnn" and args.mode == "train":
        print("Initializing R-CNN Training Pipeline...")
        # import train_rcnn from train.py and execute

    elif args.mode == "eval":
        print(f"Evaluating {args.model.upper()} on test set...")
        # import evaluation logic from evaluate.py

    elif args.load_dataset:
        print("Prepare to load the dataset")

    elif args.test_dataset:
        test_dataset()


if __name__ == "__main__":
    main()
