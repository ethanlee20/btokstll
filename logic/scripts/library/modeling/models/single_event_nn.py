
import torch
from torch import nn


class Single_Event_NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(4, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 44),
        )

        self.double()

    def forward(self, x):
        result = self.layers(x)
        return result