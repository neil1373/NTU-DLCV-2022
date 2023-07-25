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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import DigitDataset

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

    digits_datapath = 'hw2_data/digits'
    mnistm_train_set, mnistm_val_set = DigitDataset.DataProcess(os.path.join(digits_datapath, "mnistm"), os.path.join(digits_datapath, "mnistm"))
    usps_train_set, usps_val_set = DigitDataset.DataProcess(os.path.join(digits_datapath, "usps"), os.path.join(digits_datapath, "usps"))
    svhn_train_set, svhn_val_set = DigitDataset.DataProcess(os.path.join(digits_datapath, "svhn"), os.path.join(digits_datapath, "svhn"))
    
    if data_path.find("usps") != -1:    # USPS Dataset
        task = "usps"
        feature_extractor.load_state_dict(torch.load("extractor_mnistm_usps.pth"))
        label_predictor.load_state_dict(torch.load("predictor_mnistm_usps.pth"))
        mnistm_loader = DataLoader(mnistm_val_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
        val_loader = DataLoader(usps_val_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
        total_num = len(usps_val_set)
    elif data_path.find("svhn") != -1:    # SVHN Dataset
        task = "svhn"
        feature_extractor.load_state_dict(torch.load("extractor_mnistm_svhn.pth"))
        label_predictor.load_state_dict(torch.load("predictor_mnistm_svhn.pth"))
        mnistm_loader = DataLoader(mnistm_val_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
        val_loader = DataLoader(svhn_val_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
        total_num = len(svhn_val_set)

    result = []
    feature_extractor.eval()
    label_predictor.eval()
    files = sorted(os.listdir(data_path))

    acc = 0

    features = []
    all_class_labels = []
    all_domain_labels = np.concatenate((np.zeros(len(mnistm_val_set), dtype = int), np.ones(total_num, dtype = int)))

    for i, (mnistm_data, labels) in tqdm(enumerate(mnistm_loader)):
        mnistm_data = mnistm_data.cuda()

        feature = feature_extractor(mnistm_data)

        feature = feature.cpu().detach()
        if not len(features):
            features = feature.numpy()
        else:
            features = np.concatenate((features, feature.numpy()))
        labels = labels.cpu().detach()
        if not len(all_class_labels):
            all_class_labels = labels.numpy()
        else:
            all_class_labels = np.concatenate((all_class_labels, labels.numpy()))

    for i, (test_data, labels) in tqdm(enumerate(val_loader)):
        test_data = test_data.cuda()

        feature = feature_extractor(test_data)
        class_logits = label_predictor(feature)

        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        
        result.append(x)

        acc += sum(pred == truth for pred, truth in zip(x, labels))

        feature = feature.cpu().detach()
        if not len(features):
            features = feature.numpy()
        else:
            features = np.concatenate((features, feature.numpy()))
        labels = labels.cpu().detach()
        if not len(all_class_labels):
            all_class_labels = labels.numpy()
        else:
            all_class_labels = np.concatenate((all_class_labels, labels.numpy()))
    
    result = np.concatenate(result)

    # Generate your submission
    df = pd.DataFrame({'image_name': files, 'label': result})
    df.to_csv(prediction_file,index=False)
    accuracy_rate = (acc / total_num) * 100
    print(f"Overall Accuracy: {acc}/{total_num} = {accuracy_rate:.3f}%")

    print("Start TSNE Analysis...")
    tsne = TSNE(
        n_components=2,
        init="random",
        random_state=5,
        verbose=1,
        n_iter=2000,
    )
    feat_tsne = tsne.fit_transform(features)

    # Visualization
    feat_min, feat_max = feat_tsne.min(0), feat_tsne.max(0)
    feat_norm = (feat_tsne - feat_min) / (feat_max - feat_min)  #Normalize
    print(feat_norm.shape, all_class_labels.shape)
    plt.figure(figsize=(8, 8))
    
    for i in range(feat_norm.shape[0]):
        plt.scatter(feat_norm[i][0], feat_norm[i][1], color=plt.cm.tab10(all_class_labels[i]))
    
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'tsne_{task}_class.png')

    plt.cla()

    plt.figure(figsize=(8, 8))
    
    for i in range(feat_norm.shape[0]):
        plt.scatter(feat_norm[i][0], feat_norm[i][1], color=plt.cm.Set1(all_domain_labels[i]))
    
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'tsne_{task}_domain.png')


def main():
    prediction(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()