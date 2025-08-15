import torch
import torch.nn as nn
import torch.nn.functional as F

class MEDMNIST_CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(MEDMNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, padding=1),   # First Conv
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),  # Second Conv
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),  # Third Conv
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # MaxPool 2x2

            nn.Conv2d(96, 192, kernel_size=3, padding=1), # Fourth Conv
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),# Fifth Conv
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),# Sixth Conv
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # MaxPool 2x2

            nn.Conv2d(192, 192, kernel_size=3, padding=1),# Seventh Conv
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),           # Eighth Conv 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, kernel_size=1),   # Ninth Conv 1x1, output channels = num_classes
            nn.ReLU(inplace=True)
        )
        # Global Average Pooling will be performed in forward()
        
    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2:]) # Global Average Pooling over spatial dims
        x = x.view(x.size(0), -1)
        return x  