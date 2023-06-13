import argparse

import torch
from torch import nn as nn
from model import ResnetClassifier


def predict_single_image(model: torch.nn.Module, image_path: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet_version", type=int, default=101)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    state_dict = torch.load(args.checkpoint_path)
    model = ResnetClassifier(2, 101).load_from_checkpoint(state_dict)
    print(type(model))
    # predict_single_image()
