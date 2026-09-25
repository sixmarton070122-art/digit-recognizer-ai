from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim, nn
import torch

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
hidden_size = 64
num_classes = 10

digit_recognizer = model.DigitClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

epochs = 60
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    digit_recognizer.parameters(),
    lr=0.001, # Learning rate
    betas=(0.9, 0.999), # Decay rates for moments
    eps=1e-8, # Numerical stability
    weight_decay=0, # L2 regularization
    amsgrad=False # AMSGrad variant
)

for epoch in range(epochs):
    for i,(images, targets) in enumerate(train_loader):        
        predictions = digit_recognizer(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()