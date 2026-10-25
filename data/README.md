# Shared MNIST cache

Run Exp2(b) or Exp5 with internet once. Torchvision downloads and extracts the original MNIST dataset into `data/MNIST/raw`. Later runs reuse those files.

The experiments train on a small subset of existing images, but the initial download contains the complete MNIST dataset. Source: https://docs.pytorch.org/vision/2.0/generated/torchvision.datasets.MNIST.html

MNIST is excluded from Git to keep the repository small. For an offline exam computer, copy the downloaded `MNIST` folder here after cloning. The housing dataset is already included separately at `exp2a_regression/Housing.csv`.
