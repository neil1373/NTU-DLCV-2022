import os
import glob
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import cv2

class DigitDataset(Dataset):
    def __init__(self, root, datamode="train", transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.datamode = datamode
        self.transform = transform

        # read filenames
        file_list = pd.read_csv(os.path.join(root, f"{self.datamode}.csv"))

        for idx in range(file_list.shape[0]):
            self.filenames.append((os.path.join(root, "data", file_list['image_name'][idx]), file_list['label'][idx]))  # (filename, label) pair
            
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        # image = Image.open(image_fn)
        image = cv2.imread(image_fn)

        if self.transform is not None:
            image = self.transform(image)
			
        return image, label
	
    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

def DataProcess(data_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(40),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
	    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
	])

    mnist_dataset = DigitDataset(root = data_path, datamode = "train", transform = transform)

    return mnist_dataset