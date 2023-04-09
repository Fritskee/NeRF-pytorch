"""main.py."""
import argparse

from config import Config
from inference import inference
from train import train_nerf


def main():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train a NeRF model")
    parser.add_argument(
        "--infer", action="store_true", help="Perform inference with a trained NeRF model"
    )
    args = parser.parse_args()

    if args.train:
        train_nerf(Config)
    elif args.infer:
        inference(Config)
    else:
        print("Please specify either --train or --infer.")


if __name__ == "__main__":
    main()
