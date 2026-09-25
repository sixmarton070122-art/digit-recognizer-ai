from torch import nn, flatten
import math

class DigitClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.lin1 = nn.Linear(in_features=32*28*28, out_features=hidden_size)
        self.lin2 = nn.Linear(in_features=hidden_size, out_features=16)
        self.output = nn.Linear(in_features=16, out_features=num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = flatten(x, start_dim=1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.output(x)
        return x