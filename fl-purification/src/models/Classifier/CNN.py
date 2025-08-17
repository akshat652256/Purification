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
    

# Model Definition
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        # First Conv Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1x28x28 → Output: 32x28x28
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 32x28x28 → 32x28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x28x28 → 32x14x14
        
        # Second Conv Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64x14x14
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # → 64x7x7
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
    
    def forward(self, x):
        # First Conv Block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Second Conv Block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Softmax applied in loss function
        return x
    
    import torch


class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_CNN, self).__init__()
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
            
        )
        # Global Average Pooling will be performed in forward()
        
    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2:]) # Global Average Pooling over spatial dims
        x = x.view(x.size(0), -1)
        return x  


class EMNIST_CNN(nn.Module):
    """
    # epochs = 100 
    # train acc = 0.9565
    # val acc = 0.9313
    """
    def __init__(self, num_classes=26):  
        super(EMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, padding=1),   # Changed: 3 -> 1 input channels (grayscale)
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
        )
        # Global Average Pooling will be performed in forward()
        
    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[2:]) # Global Average Pooling over spatial dims
        x = x.view(x.size(0), -1)
        return x
