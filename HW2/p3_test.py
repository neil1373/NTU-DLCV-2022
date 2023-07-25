import os
import sys
import glob
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
# import constructed model in torchvision
import torchvision.models as models
from tqdm import tqdm
import DigitTestDataset

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

def prediction(data_path, prediction_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = FeatureExtractor().to(device)
    label_predictor = LabelPredictor().to(device)

    test_set = DigitTestDataset.DataProcess(data_path)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    total_num = len(test_set)

    if data_path.find("usps") != -1:    # USPS Dataset
        print(f"USPS Dataset, {total_num} images to inference...")
        feature_extractor.load_state_dict(torch.load("extractor_mnistm_usps.pth"))
        label_predictor.load_state_dict(torch.load("predictor_mnistm_usps.pth"))
        
    elif data_path.find("svhn") != -1:    # SVHN Dataset
        print(f"SVHN Dataset, {total_num} images to inference...")
        feature_extractor.load_state_dict(torch.load("extractor_mnistm_svhn.pth"))
        label_predictor.load_state_dict(torch.load("predictor_mnistm_svhn.pth"))

    result = []
    feature_extractor.eval()
    label_predictor.eval()
    files = sorted(os.listdir(data_path))

    for i, (test_data) in tqdm(enumerate(test_loader)):
        test_data = test_data.cuda()

        class_logits = label_predictor(feature_extractor(test_data))

        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        
        result.append(x)

    result = np.concatenate(result)

    # Generate your submission
    df = pd.DataFrame({'image_name': files, 'label': result})
    df.to_csv(prediction_file,index=False)

def main():
    prediction(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()