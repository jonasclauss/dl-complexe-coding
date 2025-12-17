import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


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
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 13 → 11
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 13 → 11
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 13 → 11
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 13 → 11
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 13 → 11
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 10, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # flatten all except batch
        x = self.fc(x)
        torch.softmax(x, dim=1)
        return x
    

class PretrainedResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, freeze_backbone: bool = False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)