# implement a simple CNN model using pytorch
# but I want it to use Trainer api implemneted in trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, ReLU, MaxPool2d, Dropout
from torch.nn import Flatten
from torch.nn import BatchNorm2d

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(16)
        self.conv2 = Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(32)
        self.conv3 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = BatchNorm2d(64)
        self.pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = Linear(64 * 8 * 8, 512)
        self.fc2 = Linear(512, num_classes)
        self.dropout = Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    print("Simple CNN model created.")