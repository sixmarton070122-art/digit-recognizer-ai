import torch
import model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


input_size=28**2
hidden_size = 64
num_classes = 10

digit_recognizer = model.DigitClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
model_name = "test.pth"
digit_recognizer.load_state_dict(torch.load(f"models/{model_name}"))
digit_recognizer.eval()

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

with torch.no_grad():
    total_correct = 0
    for i, (images, targets) in enumerate(test_loader):
        predictions = digit_recognizer(images)
        predicted_targets = predictions.argmax(dim=1)
        total_correct += (predicted_targets == targets).sum().item()

accuracy = total_correct / len(test_dataset)
print(f"The model \"{model_name}\"s accuracy is: {accuracy:.3f}")