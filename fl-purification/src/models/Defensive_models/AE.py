import torch
import torch.nn as nn

class DetectorIReformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # Conv.Sigmoid 3x3x3
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2),                # AveragePooling 2x2
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # Conv.Sigmoid 3x3x3
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # Conv.Sigmoid 3x3x3
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),                # Upsampling 2x2
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # Conv.Sigmoid 3x3x3
            nn.Sigmoid(),
            nn.Conv2d(1, 1, kernel_size=(3,3), padding=1) # Conv.Sigmoid 3x3x1
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DetectorII(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),   # Conv.Sigmoid 3x3x3
            nn.Sigmoid(),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),   # Conv.Sigmoid 3x3x3
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(3,3), padding=1) # Conv.Sigmoid 3x3x1
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
