# implement a simple CNN model using pytorch
# but I want it to use Trainer api implemneted in trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, ReLU, MaxPool2d, Dropout
from torch.nn import Flatten
from torch.nn import BatchNorm2d

class SimpleCNN(nn.Module):
  def __init__(self,
               input_shape: int,
               hidden_units: int,
               output_shape: int):
    super().__init__()

    self.conv_block1 = nn.Sequential(
        # Create a conv layer
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7, # trick how to calc this - print in forward what happens and used hidd_units * H * W
                  out_features=output_shape)
    )

  def forward(self, x):
    z = self.conv_block1(x)
    z = self.conv_block2(z)
    z = self.classifier(z)
    return z

if __name__ == "__main__":
    print("Simple CNN model created.")

    # Example usage
    input_shape = 1  # Number of input channels (e.g., grayscale image)
    hidden_units = 32  # Number of filters in the convolutional layers
    output_shape = 10  # Number of output classes (e.g., for classification)
    model = SimpleCNN(input_shape, hidden_units, output_shape)
    print(model)
