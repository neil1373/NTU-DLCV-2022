# Import necessary packages.
import os
import numpy as np
import math
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
from qqdm import qqdm
import wandb

import DigitDataset

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: control the balance of domain adaptation and classification.
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    

    

def main():
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fix random seed for reproducibility
    same_seeds(739)

    ################ config ################
    wandb.init(project = "DLCV_hw2_p3", name = "MNISTM_USPS_resnet34_DANN", entity="neil1373")
    config = wandb.config
    config.learning_rate = 5e-4
    config.epoch = 3000
    config.batchsize = 128
    config.momentum = 0.8
    config.betas = (0.5, 0.999)
    config.weight_decay = 5e-3
    ########################################

    # Construct data loaders.
    digits_datapath = 'hw2_data/digits'
    mnistm_train_set, minstm_val_set = DigitDataset.DataProcess(os.path.join(digits_datapath, "mnistm"), os.path.join(digits_datapath, "mnistm"))
    usps_train_set, usps_val_set = DigitDataset.DataProcess(os.path.join(digits_datapath, "usps"), os.path.join(digits_datapath, "usps"))
    svhn_train_set, svhn_val_set = DigitDataset.DataProcess(os.path.join(digits_datapath, "svhn"), os.path.join(digits_datapath, "svhn"))
    
    source_loader = DataLoader(mnistm_train_set, batch_size=config.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    target_loader = DataLoader(usps_val_set, batch_size=config.batchsize, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize a model, and put it on the device specified.
    # model = Classifier().to(device)   # self-defined model
    # feature_extractor = models.resnet34(weights = 'DEFAULT')
    # model.fc = nn.Linear(512, 50)         # resnet 18, 34
    # model.fc = nn.Linear(2048, 50)        # resnet / resnext 50, 101, 152
    # model.fc = nn.Linear(3024, 50)      # regnety16gf
    # model.fc = nn.Linear(2048, 50)      # regnetx16gf
    # model.head = nn.Linear(768, 50)         # swins, swint
    # model.head = nn.Linear(1024, 50)        # swinb
    # del feature_extractor.fc
    # feature_extractor = feature_extractor().to(device)

    feature_extractor = FeatureExtractor().to(device)
    print(feature_extractor)

    label_predictor = LabelPredictor().to(device)
    domain_classifier = DomainClassifier().to(device)

    # For the classification task, we use cross-entropy as the measurement of performance.
    class_criterion = nn.CrossEntropyLoss()
    # For determining domain, we use Binary cross-entropy with logits loss.
    domain_criterion = nn.BCEWithLogitsLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer_F = torch.optim.Adam(feature_extractor.parameters(), lr=config.learning_rate, betas = config.betas, weight_decay=config.weight_decay)
    optimizer_C = torch.optim.Adam(label_predictor.parameters(), lr=config.learning_rate, betas = config.betas, weight_decay=config.weight_decay)
    optimizer_D = torch.optim.Adam(domain_classifier.parameters(), lr=config.learning_rate, betas = config.betas, weight_decay=config.weight_decay)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 20, num_training_steps = 200)
    
    best_valid_acc = 0
    model_path = 'dann_resnet34_mnistm_usps.pth'
    iters = 0

    for epoch in range(config.epoch):
        current_lambda = ((2 / (1 + math.exp(-10 * epoch / config.epoch))) - 1) * 0.1
        running_D_loss, running_F_loss = 0.0, 0.0
        total_hit, total_num = 0.0, 0.0

        progress_bar = qqdm(zip(source_loader, target_loader))
        for i, ((source_data, source_label), (target_data, _)) in enumerate(progress_bar):

            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()
            
            # Mixed the source data and target data, or it'll mislead the running params
            #   of batch_norm. (runnning mean/var of soucre and target data are different.)
            mixed_data = torch.cat([source_data, target_data], dim=0)
            domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
            # set domain label of source data to be 1.
            domain_label[:source_data.shape[0]] = 1

            # Step 1 : train domain classifier
            feature = feature_extractor(mixed_data)
            # We don't need to train feature extractor in step 1.
            # Thus we detach the feature neuron to avoid backpropgation.
            domain_logits = domain_classifier(feature.detach())
            loss_D = domain_criterion(domain_logits, domain_label)
            running_D_loss+= loss_D.item()
            loss_D.backward()
            optimizer_D.step()

            # Step 2 : train feature extractor and label classifier
            class_logits = label_predictor(feature[:source_data.shape[0]])
            domain_logits = domain_classifier(feature)
            # loss = cross entropy of classification - lamb * domain binary cross entropy.
            #  The reason why using subtraction is similar to generator loss in disciminator of GAN
            loss_F = class_criterion(class_logits, source_label) - current_lambda * domain_criterion(domain_logits, domain_label)
            running_F_loss += loss_F.item()
            loss_F.backward()
            optimizer_F.step()
            optimizer_C.step()

            optimizer_D.zero_grad()
            optimizer_F.zero_grad()
            optimizer_C.zero_grad()

            total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
            total_num += source_data.shape[0]

            iters += 1
            # print(i, end='\r')
            progress_bar.set_infos({
                'Loss_D': round(loss_D.item(), 4),
                'Loss_F': round(loss_F.item(), 4),
                'Current Accuracy': round(total_hit/total_num, 4),
                'Total Iteration': iters,
            })

        train_D_loss, train_F_loss, train_acc = running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num
        wandb.log({"Train Feature Extractor Loss": train_F_loss})
        wandb.log({"Train Domain Classifier Loss": train_D_loss})
        wandb.log({"Train Accuracy": train_acc})

        torch.save(feature_extractor.state_dict(), f'extractor_resnet34_mnistm_usps.pth')
        torch.save(label_predictor.state_dict(), f'predictor_mnistm_usps.pth')

        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))


if __name__ == '__main__':
    main()