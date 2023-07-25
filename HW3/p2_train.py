# Import necessary packages.
import os
import sys
import random
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from helper.transformer import Transformer
from helper import dataset

from helper.config2 import Config
from helper.engine import *

import wandb
# Evaluation modules
from p2_evaluate import CIDERScore, CLIPScore

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(config):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    same_seeds(739)

    model = Transformer(vocab_size = 18022,
                        d_model = 1024,
                        img_encode_size = 196,
                        enc_ff_dim = 512, 
                        dec_ff_dim = 2048,
                        enc_n_layers = 2, 
                        dec_n_layers = 8,
                        enc_n_heads = 8,
                        dec_n_heads = 8,
                        max_len = 64, 
                        dropout = 0.1,
                        pad_id = 0)
    model.to(device)
    print(model)

    ############ wandb config ##############
    wandb.init(project = "Vision Transformer", name = "ViT Large", entity="neil1373")
    wandb_config = wandb.config
    wandb_config.learning_rate = config.lr
    wandb_config.epochs = config.epochs
    wandb_config.batchsize = config.batch_size
    wandb_config.weight_decay = config.weight_decay
    # config.betas = (0.5, 0.999)
    ########################################

    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    train_set = dataset.build_dataset(config, mode='training')
    valid_set = dataset.build_dataset(config, mode='validation')
    print(f"Training datas: {len(train_set)} images.")
    print(f"Validation datas: {len(valid_set)} images.")

    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(valid_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    train_dataloader = DataLoader(train_set, 
                                    batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    valid_dataloader = DataLoader(valid_set, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

    os.makedirs('model-2', exist_ok=True)

    if os.path.exists(config.last_checkpoint):
        print(f"Loading Checkpoint {config.last_checkpoint}...")
        checkpoint = torch.load(config.last_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

    print("Start Training...")

    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(
            model, criterion, train_dataloader, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")
        wandb.log({"Training Loss": epoch_loss})

        validation_loss = evaluate(model, criterion, valid_dataloader, device)
        print(f"Validation Loss: {validation_loss}")
        wandb.log({"Validation Loss": validation_loss})

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, f"model-2/epoch_{epoch}.pth")

if __name__ == '__main__':
    config = Config()
    main(config)
