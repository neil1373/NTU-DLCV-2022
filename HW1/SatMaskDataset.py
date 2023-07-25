# Import necessary packages.
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
# import constructed model in torchvision
import torchvision.models as models
import os
import glob
# This is for the progress bar.
from tqdm.auto import tqdm
# Evaluation Metric: MIoU
import mean_iou_evaluate

class SatMaskDataset(Dataset):
    def __init__(self, root, transform=None):
        ''' Intialize the dataset '''
        self.images = None
        self.labels = None
        self.filenames = []
        # self.root = root
        self.transform = transform
        
        # read filenames
        files = sorted(glob.glob(os.path.join(root, "*.jpg")))
        for f in files:
            self.filenames.append(f)        # filename for satelite images
        
        self.labels = mean_iou_evaluate.read_masks(root)
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        label = self.labels[index]

        image = Image.open(image_fn)
		
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
			
        return image, label
        
    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len