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
from qqdm import qqdm

from helper.DataProcess import miniDataProcess
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

    train_set = miniDataProcess(labeled = False)
    ############# wandb config #############
    wandb.init(project = "DLCV_HW4_P2", name = "Resnet50_Unlabeled", entity="neil1373")
    config = wandb.config
    config.epochs = 300
    config.batchsize = 128
    config.learning_rate = 3e-4
    # config.momentum = args.momentum
    # config.weight_decay = args.weight_decay
    # config.lr_decay_step = args.lr_decay_step
    # config.gamma = args.lr_gamma
    ########################################

    train_loader = DataLoader(train_set, batch_size=config.batchsize, shuffle=True, num_workers=4, pin_memory=True)

    resnet = models.resnet50(weights = "DEFAULT")
    resnet.fc = nn.Linear(2048, 64)

    resnet.to(device)

    print(resnet)

    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool',
        use_momentum = False       # turn off momentum in the target encoder
    )

    optimizer = torch.optim.Adam(learner.parameters(), lr=config.learning_rate)

    best_training_loss = 1000
    for epoch in range(config.epochs):

        losses = []
        progress_bar = qqdm(train_loader)

        for i, batch in enumerate(progress_bar):
            images = batch

            images = images.to(device)

            loss = learner(images)

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

            losses.append(loss.item())

            progress_bar.set_infos({
                'Loss': round(loss.item(), 6),
                'Epoch': epoch+1,
                'Step': i,
            })
            wandb.log({"Train Backbone Loss": loss.item()})
        
        avg_loss = sum(losses) / len(losses)
        print(f"[ Train | {epoch + 1:03d}/{config.epochs:03d} ] loss = {avg_loss:.6f}")

        if (avg_loss < best_training_loss):
            best_training_loss = avg_loss
            torch.save(resnet.state_dict(), 'resnet50_backbone.pt')
            print(f"Saving model with training loss {avg_loss}.")

    torch.save(resnet.state_dict(), 'resnet50_backbone.pt')

if __name__ == '__main__':
    main()