from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import model

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
test_loader = DataLoader(test_dataset, batch_size=100)

input_size=28**2
hidden_size = 49
num_classes = 10

model = model.DigitClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

epochs = 60

for epoch in epochs:
    for i in len()





