# Import necessary packages.
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

class LoadDataset(Dataset):
	def __init__(self, root, transform=None):
		""" Intialize the MNIST dataset """
		self.images = None
		self.labels = None
		self.filenames = []
		self.root = root
		self.transform = transform

		# read filenames
		files = sorted(glob.glob(os.path.join(root, "*.png")))
		for f in files:
			label = int(f.split("/")[-1].split("_")[0])
			self.filenames.append((f, label)) # (filename, label) pair
		
		self.len = len(self.filenames)
    
	def __getitem__(self, index):
		""" Get a sample from the dataset """
		image_fn, label = self.filenames[index]
		# image = Image.open(image_fn)
		image = Image.open(image_fn)
		
		if self.transform is not None:
			image = self.transform(image)
			
		return image, label
	
	def __len__(self):
		""" Total number of samples in the dataset """
		return self.len

class LoadTestDataset(Dataset):
	def __init__(self, root, transform=None):
		""" Intialize the MNIST dataset """
		self.images = None
		self.labels = None
		self.filenames = []
		self.root = root
		self.transform = transform

		# read filenames
		files = sorted(glob.glob(os.path.join(root, "*.png")))
		for f in files:
			self.filenames.append(f) # (filename) element
		
		self.len = len(self.filenames)
    
	def __getitem__(self, index):
		""" Get a sample from the dataset """
		image_fn = self.filenames[index]
		# image = Image.open(image_fn)
		image = Image.open(image_fn)
		
		if self.transform is not None:
			image = self.transform(image)
			
		return image, -1
	
	def __len__(self):
		""" Total number of samples in the dataset """
		return self.len

def TrainDataProcess():
	# It is important to do data augmentation in training.
	# However, not every augmentation is useful.
	# Please think about what kind of augmentation is helpful for food recognition.
	train_tfm = transforms.Compose([
	    # Resize the image into a fixed shape (height = width = 224)
	    # transforms.Resize((160, 160)),
	    # transforms.RandomResizedCrop(128),
	    transforms.RandomResizedCrop(224),
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
	    # transforms.Resize((128, 128)),
	    transforms.Resize((224, 224)),
	    transforms.ToTensor(),
	    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	# Batch size for training, validation, and testing.
	# A greater batch size usually gives a more stable gradient.
	# But the GPU memory is limited, so please adjust it carefully.

	# Construct datasets.
	# The argument "loader" tells how torchvision reads the data.
	# train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
	train_set = LoadDataset(root = '../hw1_data/p1_data/train_50', transform = train_tfm)

	# valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
	valid_set = LoadDataset(root = '../hw1_data/p1_data/val_50', transform = test_tfm)
	
	# unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
	
	return train_set, valid_set

def TestDataProcess(test_path):
	# We don't need augmentations in testing and validation.
	# All we need here is to resize the PIL image and transform it into Tensor.
	test_tfm = transforms.Compose([
	    transforms.Resize((224, 224)),
	    # transforms.CenterCrop(224),
	    transforms.ToTensor(),
	    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	# Batch size for training, validation, and testing.
	# A greater batch size usually gives a more stable gradient.
	# But the GPU memory is limited, so please adjust it carefully.

	# Construct datasets.
	# The argument "loader" tells how torchvision reads the data.
	# test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
	test_set = LoadTestDataset(root = test_path, transform = test_tfm)
	
	return test_set
