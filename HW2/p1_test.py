import os
import sys
import glob

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import FaceDataset
import DCGAN as dcgan
import SNGAN as sngan
import face_recog

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

def generate(workspace_dir = '.'):
    same_seeds(1373)

    # log_dir = os.path.join(workspace_dir, 'logs')
    # ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
    z_dim = 100

    G = sngan.Generator(z_dim)
    G.load_state_dict(torch.load(os.path.join(workspace_dir, 'Generator.pth')))
    G.eval()
    G.cuda()

    # Generate 1000 images and make a grid to save them.
    n_output = 1000
    z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
    imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(workspace_dir, 'result.png')
    torchvision.utils.save_image(imgs_sample[:32], filename)

    # Save the generated images.
    os.makedirs(sys.argv[1], exist_ok=True)
    for i in range(1000):
        torchvision.utils.save_image(imgs_sample[i], f'{sys.argv[1]}/{i+1:04d}.png')

def main():
    workspace_dir = '.'
    # Generate Images
    generate(workspace_dir)

    # TODO: FID evaluation
    # os.system(f"python3 -m pytorch_fid {sys.argv[1]} hw2_data/face/val")
    # Evaluation metrics: Face recognition Rate
    # acc = face_recog.face_recog(sys.argv[1])
    # print("Face recognition Accuracy: {:.1f}%".format(acc))

if __name__ == '__main__':
    main()
