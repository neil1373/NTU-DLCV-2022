# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import glob

class LoadUnlabeledDataset(Dataset):
	def __init__(self, root, transform=None):
		""" Intialize the MINI-IMAGENET dataset """
		self.images = None
		self.labels = None
		self.filenames = []
		self.root = root
		self.transform = transform

		# read filenames
		files = sorted(glob.glob(os.path.join(root, "*.jpg")))
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
			
		return image
	
	def __len__(self):
		""" Total number of samples in the dataset """
		return self.len
	
class LoadDataset(Dataset):
	def __init__(self, root, transform=None):
		""" Intialize the MINI-IMAGENET dataset WITH LABEL! """
		self.images = None
		self.labels = None
		self.filenames = []
		self.root = root
		self.transform = transform

		# read filenames
		files = sorted(glob.glob(os.path.join(root, "*.png")))
		for f in files:
			# label = int(f.split("/")[-1].split("_")[0])
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
	
def miniDataProcess(labeled = False):
	TRANSFORM_IMG = transforms.Compose([
		transforms.Resize(128),
		transforms.CenterCrop(128),
		transforms.ToTensor(),
		transforms.Normalize(mean =[0.485, 0.456, 0.406],
							 std = [0.229, 0.224, 0.225])
	])
	
	if not labeled:
		train_set = LoadUnlabeledDataset(root = './hw4_data/mini/train', transform = TRANSFORM_IMG)
	else:
		train_set = LoadDataset(root = './hw4_data/mini/train', transform = TRANSFORM_IMG)

	return train_set