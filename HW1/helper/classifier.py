# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 224, 224]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 112, 3, 1, 1),
            nn.BatchNorm2d(112),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(112, 224, 3, 1, 1),
            nn.BatchNorm2d(224),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(224, 448, 3, 1, 1),
            nn.BatchNorm2d(448),
            nn.ReLU(),
            nn.MaxPool2d(8, 8, 0),
        )

        self.fc_layer_1 = nn.Linear(21952, 1792)
        self.fc_layer_2 = nn.Linear(1792, 224)
        self.fc_layer_3 = nn.Linear(224, 50)

        self.relu = nn.ReLU()
        

    def forward(self, x):
        # input (x): [batch_size, 3, 224, 224]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        
        return x, self.fc_layer_3(self.relu(self.fc_layer_2(self.relu(self.fc_layer_1(x)))))
