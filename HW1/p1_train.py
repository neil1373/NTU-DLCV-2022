# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
# import constructed model in torchvision
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm

from helper.preprocessing import TrainDataProcess
from helper.classifier import Classifier
from helper.cosine_decay_scheduler import get_cosine_schedule_with_warmup
# from helper.pseudo_labels import get_pseudo_labels

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

    # Construct data loaders.
    train_set, valid_set = TrainDataProcess()
    batch_size = 50
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Initialize a model, and put it on the device specified.
    # model = Classifier().to(device)   # self-defined model
    model = models.resnext50_32x4d(weights = models.ResNeXt50_32X4D_Weights.DEFAULT)
    # model.fc = nn.Linear(512, 50)         # resnet 18, 34
    model.fc = nn.Linear(2048, 50)        # resnet / resnext 50, 101, 152
    # model.fc = nn.Linear(3024, 50)      # regnety16gf
    # model.fc = nn.Linear(2048, 50)      # regnetx16gf
    # model.head = nn.Linear(768, 50)         # swins, swint
    # model.head = nn.Linear(1024, 50)        # swinb
    model = model.to(device)

    print(model)
    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-3)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 20, num_training_steps = 200)
    # The number of training epochs.
    n_epochs = 200
    '''
    # Whether to do semi-supervised learning.
    do_semi = False
    '''
    best_valid_acc = 0
    model_path = 'model_resnext50_32x4d.pth'

    for epoch in range(n_epochs):
        # ---------- TODO ----------
        # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
        # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
        
        '''
        if do_semi and epoch >= 60:
            # Obtain pseudo-labels for unlabeled data using trained model.
            # model_ = Classifier().to(device)
            # model_ = models.resnext50_32x4d()
            # model_.fc = nn.Linear(2048, 50)
            # model_.device = device
            # model_.load_state_dict(torch.load(model_path))

            # pseudo_set = get_pseudo_labels(unlabeled_set, model_)
            pseudo_set = get_pseudo_labels(valid_set, model)

            # Construct a new dataset and a data loader for training.
            # This is used in semi-supervised learning only.
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        '''

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        # Iterate the training set by batches.
        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            
            # Forward the data. (Make sure data and model are on the same device.)ß
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()
            # scheduler.step()
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Save the model with the best validation accuracy
        if valid_acc > best_valid_acc:
            torch.save(model.state_dict(), model_path)
            best_valid_acc = valid_acc
            print(f"Save model with Validation Accuracy {best_valid_acc:.5f}")
            do_semi = True
        else:
            do_semi = False
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}, best acc = {best_valid_acc:.5f}")

if __name__ == '__main__':
    main()