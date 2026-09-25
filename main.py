import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

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

print(len(train_dataset))
print(len(test_dataset))
print(train_dataset[0][0].shape)
print(train_dataset[0][0].min())
print(train_dataset[0][0].max())
print(train_dataset[0][1])
print(type(train_dataset[0][1]))