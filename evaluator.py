import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

test_model = torch.load("/models/test.pth")

transform = transforms.Compose([
    transforms.ToTensor(),
])

try:
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
except Exception as e:
    print(f"Error loading MNIST dataset: {e}")
    exit(1)

test_loader = DataLoader(test_dataset, batch_size=100)

input_size=28**2
hidden_size = 64
num_classes = 10