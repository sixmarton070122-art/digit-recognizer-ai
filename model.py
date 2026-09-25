from torch import nn

class DigitClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__(self)
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=32, kernel_size=3)
        self.lin1 = nn.Linear(in_features=32, out_features=16)
        self.output = nn.Linear(in_features=16, out_features=num_classes)