import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)