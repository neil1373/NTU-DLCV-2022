import os
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
from tqdm import tqdm
from qqdm import qqdm

import FaceDataset
import SNGAN2 as sngan
import face_recog
from pytorch_fid import fid_score
# IMPORTANT: wandb package is only available in training code!!
import wandb
from DiffAugment import DiffAugment

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

def main():
    same_seeds(739)

    workspace_dir = '.'
    train_dataset = FaceDataset.get_dataset(os.path.join(workspace_dir, 'hw2_data/face/train'))

    use_dcgan = False
    use_wgan = False
    use_sngan = True
    if use_dcgan:
        project_name = "DLCV_hw2_DCGAN"
    elif use_wgan:
        project_name = "DLCV_hw2_WGAN"
    elif use_sngan:
        project_name = "DLCV_hw2_SNGAN"
    else:
        raise TypeError("set GAN type")
    
    ################ config ################
    wandb.init(project = project_name, name = project_name, entity="neil1373")
    config = wandb.config
    config.learning_rate = 0.0002
    config.epoch = 2000
    config.batchsize = 64
    config.momentum = 0.8
    config.betas = (0.5, 0.999)
    ########################################

    z_dim = 100
    z_sample = Variable(torch.randn(1000, z_dim)).cuda()

    n_critic = 1
    clip_value = 0.01

    log_dir = os.path.join(workspace_dir, 'sngan_2000_logs')
    ckpt_dir = os.path.join(workspace_dir, 'sngan_2000_checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Model
    if use_dcgan:
        G = dcgan.Generator(in_dim = z_dim).cuda()
        D = dcgan.Discriminator(3).cuda()
    elif use_wgan:
        G = wgan.Generator(in_dim = z_dim).cuda()
        D = wgan.Discriminator(3).cuda()
    elif use_sngan:
        G = sngan.Generator(in_dim = z_dim).cuda()
        D = sngan.Discriminator(3).cuda()
    G.load_state_dict(torch.load(os.path.join(ckpt_dir, "Generator.pth")))
    D.load_state_dict(torch.load(os.path.join(ckpt_dir, "Discriminator.pth")))
    G.train()
    D.train()

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=config.learning_rate, betas=config.betas)
    opt_G = torch.optim.Adam(G.parameters(), lr=config.learning_rate, betas=config.betas)

    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True, num_workers=4, pin_memory=True)

    """
    ### Training loop
    We store some pictures regularly to monitor the current performance of the Generator, and regularly record checkpoints.
    """

    steps = 0
    best_fid_value = 100000
    augment_policies = 'color,translation,cutout'
    for e, epoch in enumerate(range(config.epoch)):
        progress_bar = qqdm(train_dataloader)
        for i, data in enumerate(progress_bar):
            imgs = data
            imgs = imgs.cuda()

            bs = imgs.size(0)

            # ============================================
            #  Train D
            # ============================================
            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)

            # Label
            r_label = torch.ones((bs)).cuda()
            f_label = torch.zeros((bs)).cuda()

            # Model forwarding
            r_logit = D(DiffAugment(r_imgs.detach(), policy=augment_policies))
            f_logit = D(DiffAugment(f_imgs.detach(), policy=augment_policies))
            
            # Compute the loss for the discriminator.
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # WGAN Loss
            # loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))
            
            # Model backwarding
            D.zero_grad()
            loss_D.backward()

            # Update the discriminator.
            opt_D.step()
            
            """ Clip weights of discriminator. """
            # for p in D.parameters():
                # p.data.clamp_(-clip_value, clip_value)

            # ============================================
            #  Train G
            # ============================================
            if steps % n_critic == 0:
                # Generate some fake images.
                z = Variable(torch.randn(bs, z_dim)).cuda()
                f_imgs = G(z)

                # Model forwarding
                f_logit = D(DiffAugment(f_imgs.detach(), policy=augment_policies))
                
                # Compute the loss for the generator.
                loss_G = criterion(f_logit, r_label)
                # WGAN Loss
                # loss_G = -torch.mean(D(f_imgs))
                
                # Model backwarding
                G.zero_grad()
                loss_G.backward()

                # Update the generator.
                opt_G.step()
                
            steps += 1
            
            # Set the info of the progress bar
            #   Note that the value of the GAN loss is not directly related to
            #   the quality of the generated images.
            progress_bar.set_infos({
                'Loss_D': round(loss_D.item(), 4),
                'Loss_G': round(loss_G.item(), 4),
                'Epoch': e+1,
                'Step': steps,
            })
            wandb.log({"train_Generator_loss": loss_G.item()})
            wandb.log({"train_Discriminator_loss": loss_D.item()})

        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(log_dir, f'Epoch_{epoch+1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=40)
        print(f'\n | Save some samples to {filename}.')
        
        os.makedirs('sngan_2000_output', exist_ok=True)
        for i in range(1000):
            torchvision.utils.save_image(f_imgs_sample[i], f'sngan_2000_output/{i+1}.jpg')

        evaluation_paths = ['sngan_2000_output', 'hw2_data/face/val']
        fid_value = fid_score.calculate_fid_given_paths(evaluation_paths, 
                                                        batch_size = 50, 
                                                        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu'),
                                                        dims = 2048,
                                                        num_workers = 8)
        wandb.log({"FID_value": fid_value})

        # Show generated images in the jupyter notebook.
        '''
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10,10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        '''

        
        # Evaluating Face Recognition Rate
        face_recog_acc = face_recog.face_recog('sngan_2000_output')
        wandb.log({"Face Recognition Rate": face_recog_acc})
        if fid_value < best_fid_value:
            # Save the checkpoints.
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'Generator.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'Discriminator.pth'))
            best_fid_value = fid_value
            print(f' | Epoch {epoch+1} | Save model with FID Score {fid_value:.6f} and Face Recognition Rate {face_recog_acc:.1f}%.')
        else:
            print(f' | Epoch {epoch+1} | FID Score {fid_value:.6f} and Face Recognition Rate {face_recog_acc:.1f}%.')

        G.train()
    
if __name__ == '__main__':
    main()
