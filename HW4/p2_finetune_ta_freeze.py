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
from qqdm import qqdm

from helper.officeDataProcess import officeDataProcess
from byol_pytorch import BYOL
from tqdm import tqdm
import wandb

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

    train_set, valid_set = officeDataProcess()
    ############# wandb config #############
    wandb.init(project = "DLCV_HW4_P2", name = "Finetune_Office_TA_Freeze", entity="neil1373")
    config = wandb.config
    config.epochs = 30
    config.batchsize = 64
    config.learning_rate = 3e-4
    config.weight_decay = 5e-4
    # config.momentum = args.momentum
    # config.lr_decay_step = args.lr_decay_step
    # config.gamma = args.lr_gamma
    ########################################

    train_loader = DataLoader(train_set, batch_size=config.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    
    resnet = models.resnet50()

    checkpoint_src = "hw4_data/pretrain_model_SL.pt"
    resnet.load_state_dict(torch.load(checkpoint_src))
    checkpoint_dst = "resnet50_finetuned_ta_freeze.pt"

    resnet.fc = nn.Linear(2048, 65)
    resnet.to(device)

    for param in resnet.parameters():
        param.requires_grad = False

    for param in resnet.fc.parameters():
        param.requires_grad = True
    
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
    
    optimizer = torch.optim.Adam(resnet.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay)

    best_valid_avg_acc = 0
    for epoch in range(config.epochs):

        train_loss = []
        train_accs = []

        progress_bar = qqdm(train_loader)

        for i, batch in enumerate(progress_bar):
            images, labels = batch

            images = images.to(device)

            logits = resnet(images)

            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)
            
            progress_bar.set_infos({
                'Train Loss': round(loss.item(), 6),
                'Train Acc': round(acc.item(), 6),
                'Epoch': epoch+1,
                'Step': i+1 + epoch * len(train_loader),
            })
            wandb.log({"Train Finetune Loss": loss.item()})
            wandb.log({"Train Finetune Accuracy": acc})
        
        avg_loss = sum(train_loss) / len(train_loss)
        avg_acc = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:02d}/{config.epochs:02d} ] | loss = {avg_loss:.6f} | acc = {avg_acc:.6f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        resnet.eval()

        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        valid_progress_bar = qqdm(valid_loader)
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
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().sum()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc.item() / len(batch))

            valid_progress_bar.set_infos({
                'Valid Loss': round(loss.item(), 6),
                'Valid Acc': round(acc.item() / len(batch), 6),
                'Epoch': epoch+1,
                'Step': i+1 + epoch * len(valid_loader),
            })
            wandb.log({"Valid Finetune Loss": loss.item()})
            wandb.log({"Valid Finetune Accuracy": (acc.item() / len(batch))})

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_avg_loss = sum(valid_loss) / len(valid_loss)
        valid_avg_acc = sum(valid_accs) / len(valid_set)

        # Save the model with the best validation accuracy
        if valid_avg_acc > best_valid_avg_acc:
            torch.save(resnet.state_dict(), checkpoint_dst)
            best_valid_avg_acc = valid_avg_acc
            print(f"Save model with Validation Accuracy {best_valid_avg_acc:.6f}")
        else:
            pass
        # Print the information.
        print(f"[ Valid | {epoch + 1:02d}/{config.epochs:02d} ] | loss = {valid_avg_loss:.6f} | acc = {valid_avg_acc:.6f}, best acc = {best_valid_avg_acc:.6f}")


    # torch.save(resnet.state_dict(), 'resnet50_backbone.pt')

if __name__ == '__main__':
    main()
