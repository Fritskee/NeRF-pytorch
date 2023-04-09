import argparse
from train import train_nerf
from inference import inference
from config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train a NeRF model')
    parser.add_argument('--infer', action='store_true', help='Perform inference with a trained NeRF model')
    args = parser.parse_args()

    if args.train:
        train_nerf(Config)
    elif args.infer:
        inference(Config)
    else:
        print("Please specify either --train or --infer.")

if __name__ == '__main__':
    main()
