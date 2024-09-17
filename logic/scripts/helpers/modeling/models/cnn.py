
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(10, 20, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(20, 40, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(40, 40, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(40, 80, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(80, 80, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2,1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dense = nn.Sequential(
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.double()

    def forward(self, x):
        z = self.conv(x)
        z = z.squeeze()
        result = self.dense(z)
        return result