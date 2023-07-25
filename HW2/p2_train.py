import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

# Self-defined function
import LoadMNIST
from CondUNet import UNet_conditional
from EMA import EMA

# IMPORTANT: wandb package is only available in training code!!
import wandb

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
            print(self.img_size)
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
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


def train(config):
    same_seeds(739)
    example_path = "p2_example"
    model_path = "p2_model"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(example_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mnistm_datapath = 'hw2_data/digits/mnistm'
    mnist_dataset = LoadMNIST.DataProcess(mnistm_datapath)
    dataloader = DataLoader(mnist_dataset, batch_size=config.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    model = UNet_conditional(num_classes=config.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    mse = nn.MSELoss()
    diffusion = Diffusion(device=device)
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # load checkpoints and resume
    resume_epoch = 10
    if resume_epoch > 0:
        print(f"Loading checkpoints from Epoch {resume_epoch}...")
        model.load_state_dict(torch.load(os.path.join(model_path, f"{resume_epoch}_ckpt.pth")))
        ema_model.load_state_dict(torch.load(os.path.join(model_path, f"{resume_epoch}_ema_ckpt.pth")))
        optimizer.load_state_dict(torch.load(os.path.join(model_path, f"{resume_epoch}_optim.pth")))

    for epoch in range(resume_epoch, config.epochs):
        print(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            wandb.log({"MSE loss": loss.item()})

        if (epoch + 1) % 1 == 0:
            labels = torch.arange(10).long().to(device)
            # sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            '''
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            '''
            torch.save(model.state_dict(), os.path.join("p2_model", f"{epoch + 1}_ckpt.pth"))
            torch.save(ema_model.state_dict(), os.path.join("p2_model", f"{epoch + 1}_ema_ckpt.pth"))
            torch.save(optimizer.state_dict(), os.path.join("p2_model", f"{epoch + 1}_optim.pth"))


def main():
    '''
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 14
    args.image_size = 64
    args.num_classes = 10
    args.device = "cuda"
    args.lr = 3e-4
    '''
    project = "DLCV_HW2_P2"
    name = "Diffusion U-Net"
    ################ config ################
    wandb.init(project = project, name = name, entity="neil1373")
    config = wandb.config
    config.learning_rate = 0.0003
    config.epochs = 200
    config.batchsize = 64
    config.momentum = 0.8
    config.betas = (0.5, 0.999)
    config.num_classes = 10
    ########################################
    train(config)


if __name__ == '__main__':
    main()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
