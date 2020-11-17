import os

from argparse import ArgumentParser
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from torchvision.datasets.mnist import MNIST
from torchvision import transforms

# ------------
# DATA SETUP
# ------------
pl.seed_everything(1234)
BATCH_SIZE = 32


def main():
    # Init DataLoader from MNIST Dataset
    dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])


if __name__ == '__main__':
    main()
