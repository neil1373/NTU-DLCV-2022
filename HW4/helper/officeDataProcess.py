# Import necessary packages.
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import glob

class LoadTestDataset(Dataset):
	def __init__(self, root, file_list_data, transform=None):
		""" Intialize the MINI-IMAGENET dataset """
		self.images = None
		self.labels = None
		self.filenames = []
		self.root = root
		self.transform = transform

		# read filenames
		file_list = pd.read_csv(file_list_data)
		
		for idx in range(file_list.shape[0]):
			self.filenames.append(os.path.join(root, file_list['filename'][idx]))  # (filename) pair
        
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
	def __init__(self, root, datamode = "train", transform=None):
		""" Intialize the MINI-IMAGENET dataset WITH LABEL! """
		self.images = None
		self.labels = None
		self.filenames = []
		self.root = root
		self.datamode = datamode
		self.transform = transform

		# read filenames
		# files = sorted(glob.glob(os.path.join(root, "*.png")))
		file_list = pd.read_csv(os.path.join(root, f"{self.datamode}.csv"))
		
		office_label_encoder = LabelEncoder()
		file_list['label'] = office_label_encoder.fit_transform(file_list['label'])

		# Save class encoder for inference
		output = open('office_encoder.pkl', 'wb')
		pickle.dump(office_label_encoder, output)
		output.close()

		for idx in range(file_list.shape[0]):
			self.filenames.append((os.path.join(root, datamode, file_list['filename'][idx]), file_list['label'][idx]))  # (filename, label) pair
        
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
	
def officeDataProcess():
	TRANSFORM_IMG = transforms.Compose([
		transforms.Resize(128),
		transforms.CenterCrop(128),
		transforms.ToTensor(),
		transforms.Normalize(mean =[0.485, 0.456, 0.406],
							 std = [0.229, 0.224, 0.225])
	])
	
	train_set = LoadDataset(root = './hw4_data/office', datamode = "train", transform = TRANSFORM_IMG)
	valid_set = LoadDataset(root = './hw4_data/office', datamode = "val", transform = TRANSFORM_IMG)
	return train_set, valid_set

def officeTestDataProcess(root, file_list_csv):
	TRANSFORM_IMG = transforms.Compose([
		transforms.Resize(128),
		transforms.CenterCrop(128),
		transforms.ToTensor(),
		transforms.Normalize(mean =[0.485, 0.456, 0.406],
							 std = [0.229, 0.224, 0.225])
	])
	
	test_set = LoadTestDataset(root = root, file_list_data = file_list_csv, transform = TRANSFORM_IMG)
	return test_set