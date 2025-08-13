import torch
import torch.nn as nn
import torch.nn.functional as F

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