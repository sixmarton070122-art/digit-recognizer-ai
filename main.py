from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim, nn

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

digit_recognizer = model.DigitClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

epochs = 60
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    for i,(images, targets) in enumerate(train_loader):        
        predictions = model(images)
        loss = criterion(predictions, targets)
