import torch
import math
import os
import sys
from tqdm import tqdm

from .utils import NestedTensor


def train_one_epoch(model, criterion, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)

    with tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            images = images.to(device)
            samples = NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs, attns = model(images, caps[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_value = loss.item()
            epoch_loss += loss_value

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            '''
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            '''
            optimizer.step()
            pbar.set_postfix(loss=loss_value)
            pbar.update(1)

    return epoch_loss / total

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)

    with tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            images = images.to(device)
            samples = NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            outputs, attns = model(images, caps[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

            validation_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
        
    return validation_loss / total
