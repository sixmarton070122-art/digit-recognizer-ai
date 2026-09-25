from torch.utils.data import random_split, DataLoader
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

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=100, shuffle=True)