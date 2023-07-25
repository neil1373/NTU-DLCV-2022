# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import DataLoader
# import constructed model in torchvision
import torchvision.models as models

# This is for the progress bar.
from tqdm import tqdm

from helper.officeDataProcess import officeDataProcess
from byol_pytorch import BYOL

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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fix random seed for reproducibility
    same_seeds(739)

    _, valid_set = officeDataProcess()

    valid_loader = DataLoader(valid_set, batch_size = 64, shuffle=True, num_workers=4, pin_memory=True)
    
    resnet = models.resnet50(num_classes=65)

    checkpoint_path = "resnet50_normal.pt"
    resnet.load_state_dict(torch.load(checkpoint_path))

    resnet.to(device)

    print(resnet)
    '''
    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool',
        use_momentum = False       # turn off momentum in the target encoder
    )
    '''

    criterion = nn.CrossEntropyLoss()
    
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    resnet.eval()

    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    valid_progress_bar = tqdm(valid_loader)

    for i, batch in enumerate(valid_progress_bar):
        # A batch consists of image data and corresponding labels.
        images, labels = batch
        
        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = resnet(images.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).sum()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_avg_loss = sum(valid_loss) / len(valid_loss)
    valid_avg_acc = sum(valid_accs) / len(valid_set)
    
    # Print the information.
    print(f"[ Validation | {checkpoint_path} ] | loss = {valid_avg_loss:.6f} | acc = {valid_avg_acc:.6f}")

if __name__ == '__main__':
    main()
