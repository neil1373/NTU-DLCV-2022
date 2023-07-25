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
from VGG16_models import *
# This is for the progress bar.
import os
import sys
from tqdm.auto import tqdm
# This is for self-defined dataset.
from SatMaskDataset import *
# Evaluation Metric: MIoU
from mean_iou_evaluate import *

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fix random seed for reproducibility
    same_seeds(739)

    train_tfm = transforms.ToTensor()
    test_tfm = transforms.ToTensor()
    
    train_set = SatMaskDataset(root = '../hw1_data/p2_data/train', transform = train_tfm)
    valid_set = SatMaskDataset(root = '../hw1_data/p2_data/validation', transform = test_tfm)
    batch_size = 5

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = models.segmentation.deeplabv3_resnet50(num_classes = 7, weight='DEFAULT')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index = 6)
    optimizer = torch.optim.SGD(model.parameters(), momentum = 0.85, lr=5e-3, weight_decay=5e-4)

    n_epochs = 90

    best_valid_miou = 0

    model_path = 'model_deeplabv3_resnet50.pth'
    '''
    if os.path.exists(model_path):
        print("Loading Checkpoint:", model_path)
        model.load_state_dict(torch.load(model_path))
    '''
    for epoch in range(n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []

        # Iterate the training set by batches.
        for batch in tqdm(train_loader, desc='Training...'):

            imgs, labels = batch
            labels = labels.squeeze(1)

            imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.long)
            logits = model(imgs)['out']

            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
            train_loss.append(loss.item())

        # The average loss of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []

        pred_masks_cpu = []
        labels_cpu = []
        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            with torch.no_grad():
                imgs, labels = batch
                labels = labels.squeeze(1)
                imgs = imgs.to(device)
                labels = labels.to(device, dtype=torch.long)
                logits = model(imgs)['out']

                pred_masks = logits.argmax(dim=1)

                loss = criterion(logits, labels)            
                valid_loss.append(loss.item())

                pred_masks_cpu.append(pred_masks.cpu())
                labels_cpu.append(labels.cpu())
        """    
        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        """

        pred_masks_numpy = torch.cat(pred_masks_cpu).numpy()
        labels_numpy = torch.cat(labels_cpu).numpy()

        valid_miou = mean_iou_score(pred_masks_numpy, labels_numpy)

        if not epoch:
            model_early_path = 'model_deeplabv3_resnet50_early.pth'
            torch.save(model.state_dict(), model_early_path)
            print(f"Save early model with Validation mIoU {valid_miou:.6f}\n")

        if (epoch+1) == 10:
            model_middle_path = 'model_deeplabv3_resnet50_middle.pth'
            torch.save(model.state_dict(), model_middle_path)
            print(f"Save middle model with Validation mIoU {valid_miou:.6f}\n")

        # Save the model with the best validation accuracy
        if valid_miou > best_valid_miou:
            torch.save(model.state_dict(), model_path)
            best_valid_miou = valid_miou
            print(f"Save the current best model with Validation mIoU {best_valid_miou:.6f}\n")

        # The average loss of the validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.6f}, mIoU = {valid_miou:.5f}, best mIoU = {best_valid_miou:.6f}")

if __name__ == '__main__':
    main()