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
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 11 → 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 9 → 7
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 7 → 5
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 5 → 3
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 3 → 1
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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class NeuralNetworkMS(NeuralNetwork):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 10, bias=True),
        )

    def forward(self, x):
        x1 = self.conv(x[:, 0:3])
        x2 = self.conv(x[:, 3:6])
        x = torch.cat((x1, x2), dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PretrainedResNet18MS(nn.Module):
    """Pretrained ResNet18-Variante für 13-kanalige MS-Bilder.

    Passt conv1 von 3 auf 13 Kanäle an und initialisiert die zusätzlichen
    Kanäle mit dem Mittel der ImageNet-Filter.
    """

    def __init__(self, num_classes: int = 10, freeze_backbone: bool = False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        base = resnet18(weights=weights)

        # conv1 von 3 auf 13 Eingabekanäle erweitern
        old_conv = base.conv1
        new_conv = nn.Conv2d(
            13,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            old_w = old_conv.weight  # [out, 3, k, k]
            new_w = new_conv.weight  # [out, 13, k, k]
            
            # Initialisiere alle Kanäle mit dem Mittelwert der RGB-Gewichte
            mean_w = old_w.mean(dim=1, keepdim=True)  # [out,1,k,k]
            new_w[:] = mean_w.repeat(1, 13, 1, 1)

            # Kopiere die RGB-Gewichte an die korrekten Positionen für EuroSAT MS
            # EuroSAT MS Channels:
            # 0: B01, 1: B02 (Blue), 2: B03 (Green), 3: B04 (Red), ...
            # ImageNet Weights (RGB): 0: Red, 1: Green, 2: Blue

            # Red: ImageNet 0 -> MS 3
            new_w[:, 3, :, :] = old_w[:, 0, :, :]
            # Green: ImageNet 1 -> MS 2
            new_w[:, 2, :, :] = old_w[:, 1, :, :]
            # Blue: ImageNet 2 -> MS 1
            new_w[:, 1, :, :] = old_w[:, 2, :, :]

        base.conv1 = new_conv
        self.backbone = base

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)