# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
# import constructed model in torchvision
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm

def get_pseudo_labels(dataset, model, threshold=0.9):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)
    # Construct dataloader
    batch_size = 25
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    labeled_dataset = []
    # Iterate over the dataset by batches.
    for batch in tqdm(dataloader):
        
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)
        # ---------- TODO ----------
        # Filter the data and construct a new dataset.
        pseudo_probs, pseudo_labels = torch.max(probs, 1)
        
        for i in range(len(batch)):
            if pseudo_probs[i] > threshold:
                labeled_data = (img[i], int(pseudo_labels[i]))
                labeled_dataset.append(labeled_data)
    # # Turn off the eval mode.
    model.train()
    print(f"{len(labeled_dataset):04d} data labeled in Unlabeled Dataset.")
    return labeled_dataset