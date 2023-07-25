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

class DigitTestDataset(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the MNIST/USPS/SVHN dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        file_list = sorted(os.listdir(root))

        for file in file_list:
            self.filenames.append(file)  # (filename, label) pair
            
        # print(self.filenames)
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        # image = Image.open(image_fn)
        image = cv2.imread(os.path.join(self.root, image_fn))

        if self.transform is not None:
            image = self.transform(image)
			
        return image
	
    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def DataProcess(test_path):
	# It is important to do data augmentation in training.
	# However, not every augmentation is useful.
	# Please think about what kind of augmentation is helpful for food recognition.
	# We don't need augmentations in testing and validation.
	# All we need here is to resize the PIL image and transform it into Tensor.
	test_tfm = transforms.Compose([
        transforms.ToPILImage(),
	    # transforms.Resize((128, 128)),
	    # transforms.Resize((224, 224)),
        transforms.Resize((32, 32)),
	    transforms.ToTensor(),
	    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	# Batch size for training, validation, and testing.
	# A greater batch size usually gives a more stable gradient.
	# But the GPU memory is limited, so please adjust it carefully.

	# Construct datasets.
	# The argument "loader" tells how torchvision reads the data.
	# train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
	test_set = DigitTestDataset(root = test_path, transform = test_tfm)

	return test_set