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
        """ Intialize the MNIST/USPS/SVHN dataset """
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


def DataProcess(source_path, target_path):
	# It is important to do data augmentation in training.
	# However, not every augmentation is useful.
	# Please think about what kind of augmentation is helpful for food recognition.
	train_tfm = transforms.Compose([
        transforms.ToPILImage(),
	    # Resize the image into a fixed shape (height = width = 224)
	    # transforms.Resize((160, 160)),
	    # transforms.RandomResizedCrop(128),
	    # transforms.RandomResizedCrop(224),
        transforms.Resize((32, 32)),
	    # You may add some transforms here.
	    transforms.RandomHorizontalFlip(),
	    # transforms.RandomRotation(30),
	    # transforms.RandomAffine(degrees=30, scale=(0.88, 1.2)),
	    # transforms.RandomApply([
	    # 	transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
	    # 	],
	    # 	p = 0.5),
	    transforms.AutoAugment(),
	    transforms.RandomApply([
	    	transforms.GaussianBlur(kernel_size=3),],
	    	p = 0.5),
	    # transforms.RandomPerspective(distortion_scale=0.5, p=0.25),
	    # ToTensor() should be the last one of the transforms.
	    transforms.ToTensor(),
	    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	    # transforms.RandomErasing(),
	])
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
	source_set = DigitDataset(root = source_path, datamode = "train", transform = train_tfm)

	# valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
	target_set = DigitDataset(root = target_path, datamode = "val", transform = test_tfm)
	
	# unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
	
	return source_set, target_set