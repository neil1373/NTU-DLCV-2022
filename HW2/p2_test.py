from curses import resize_term
import os
import sys
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader

# Self-defined function
import LoadMNIST
from CondUNet import UNet_conditional
from EMA import EMA

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            img_size = 32
            x = torch.randn((n, 3, img_size, img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                '''
                if i % 120 == 1 or i == 599:
                    sample_0 = x[0]
                    sample_0 = (sample_0.clamp(-1, 1) + 1) / 2
                    sample_0 = (sample_0 * 255).type(torch.uint8)
                    resize_tfm = transforms.Resize(28)
                    sample_grid = torchvision.utils.make_grid(resize_tfm(sample_0), nrow=11)
                    sample_0_ndarr = sample_grid.permute(1, 2, 0).to('cpu').numpy()
                    sample_0_img = Image.fromarray(sample_0_ndarr)
                    sample_0_img.save(f"sample_0_step_{i}.png")
                    print(f"Sample 0 image sample_0_step_{i}.png saved.")
                '''

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

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

def generate(output_path):
    same_seeds(739)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = Diffusion(noise_steps=600, device=device)
    model = UNet_conditional(num_classes=10).to(device)
    model.load_state_dict(torch.load(f"DiffusionUNet.pth"))
    os.makedirs(output_path, exist_ok=True)
    for iter in range(4):
        if iter == 0:
            n_imgs, n_imgs_each = 100, 10
        else:
            n_imgs, n_imgs_each = 300, 30
        labels = torch.ones(n_imgs).long().to(device)
        for i in range(10):
            # print(len(labels[i*(n_imgs_each):(i+1)*(n_imgs_each)]))
            labels[i*(n_imgs_each):(i+1)*(n_imgs_each)] *= i
        sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
        # print(type(sampled_images), sampled_images.shape)
        if iter == 0:
            grid = torchvision.utils.make_grid(sampled_images, nrow=10)
            ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
            img = Image.fromarray(ndarr)
            img.save("p2_report.png")
        resize_tfm = transforms.Resize(28)
        for row in range(10):
            for col in range(n_imgs_each):
                if iter == 0:
                    simple_grid = torchvision.utils.make_grid(resize_tfm(sampled_images[row*n_imgs_each+col]), nrow=1)
                    ndarr = simple_grid.permute(1, 2, 0).to('cpu').numpy()
                    img = Image.fromarray(ndarr)
                    img.save(f"{output_path}/{row}_{col:03d}.png")
                    # torchvision.utils.save_image(resize_tfm(sampled_images[row*n_imgs_each+col]), f"{output_path}/{row}_{col:3d}.png", nrow=10)
                else:
                    simple_grid = torchvision.utils.make_grid(resize_tfm(sampled_images[row*n_imgs_each+col]), nrow=1)
                    ndarr = simple_grid.permute(1, 2, 0).to('cpu').numpy()
                    img = Image.fromarray(ndarr)
                    img.save(f"{output_path}/{row}_{(10+(iter-1)*n_imgs_each+col):03d}.png")
                    # torchvision.utils.save_image(resize_tfm(sampled_images[row*n_imgs_each+col]), f"{output_path}/{row}_{(10+(iter-1)*n_imgs_each+col):3d}.png", nrow=10)




    
def main():
    generate(output_path = sys.argv[1])

if __name__ == '__main__':
    main()