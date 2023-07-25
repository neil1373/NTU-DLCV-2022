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

from helper.preprocessing import TestDataProcess
from helper.classifier import Classifier

import os
import sys
def main():
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Construct data loaders.
    test_set = TestDataProcess(sys.argv[1])
    batch_size = 25
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.

    # model = Classifier().to(device)
    model = models.resnext101_64x4d()
    # model.head = nn.Linear(768, 50)         # swins, swint
    model.fc = nn.Linear(2048, 50)
    model = model.to(device)

    model_path = 'p1_final_model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Inference model:", model_path)
    # Initialize a list to store the predictions.
    predictions = []

    # Iterate the testing set by batches.
    for batch in tqdm(test_loader, desc = 'Inference'):
        # A batch consists of image data and corresponding labels.
        # But here the variable "labels" is useless since we do not have the ground-truth.
        # If printing out the labels, you will find that it is always 0.
        # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
        # so we have to create fake labels to make it work normally.
        imgs, _ = batch

        # We don't need gradient in testing, and we don't even have labels to compute loss.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    # Get file list in directory.
    files = [f for f in sorted(os.listdir(sys.argv[1]))]
    # acc = 0
    # Save predictions into the file.
    with open(sys.argv[2], "w") as f:

        # The first row must be "Id, Category"
        f.write("filename,label\n")

        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in enumerate(predictions):
            f.write(f"{files[i]},{pred}\n")
            # acc += (pred == int(files[i].split("_")[0]))
    
    # local_path = sys.argv[1].split("/")[-1]
    # if not len(local_path):
        # local_path = sys.argv[1].split("/")[-2]
    
    # print(f"[ Test ] acc on {local_path} = {acc / len(files)}")

if __name__ == '__main__':
    # print(sys.argv)
    # data path: sys.argv[1], output file: sys.argv[2]
    main()
