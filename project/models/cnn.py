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
                 output_shape: int,
                 image_height: int,
                 image_width: int):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        # Run a dummy forward pass to compute the output feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape, image_height, image_width)
            z = self.conv_block1(dummy_input)
            #z = self.conv_block2(z)
            self.flattened_size = z.view(1, -1).shape[1]  # Total features after convs

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, output_shape)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        #x = self.conv_block2(x)
        x = self.classifier(x)
        return x
    
    
    def print_model_summary(self):
        """
        Print a summary of the model architecture and number of parameters.
        """
        print("Model Summary:")
        print(self)
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")