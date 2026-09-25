import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Transform: convert images to tensors and normalize pixel values
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
])

# Download and load MNIST training and test datasets
try:
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
except Exception as e:
    print(f"Error loading MNIST dataset: {e}")
    exit(1)