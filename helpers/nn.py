import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=2),  # 64 → 29
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 29 → 13
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 13 → 11
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 11 * 11, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 10, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # flatten all except batch
        x = self.fc(x)
        return x