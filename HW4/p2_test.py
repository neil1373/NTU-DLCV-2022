# Import necessary packages.
import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import DataLoader
# import constructed model in torchvision
import torchvision.models as models

# This is for the progress bar.
from tqdm import tqdm

from helper.officeDataProcess import officeTestDataProcess
from byol_pytorch import BYOL
from sklearn.preprocessing import LabelEncoder

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

    test_set = officeTestDataProcess(sys.argv[2], sys.argv[1])

    test_loader = DataLoader(test_set, batch_size = 64, shuffle=False, num_workers=4, pin_memory=True)
    
    resnet = models.resnet50(num_classes=65)

    checkpoint_path = "resnet50_finetuned_final.pt"
    resnet.load_state_dict(torch.load(checkpoint_path))

    resnet.to(device)

    # print(resnet)
    '''
    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool',
        use_momentum = False       # turn off momentum in the target encoder
    )
    '''

    # criterion = nn.CrossEntropyLoss()
    
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    resnet.eval()

    # Iterate the validation set by batches.
    test_progress_bar = tqdm(test_loader)

    result = []
    
    for i, batch in enumerate(test_progress_bar):
        # A batch consists of image data and corresponding labels.
        images = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = resnet(images.to(device))

            x = torch.argmax(logits, dim=1).cpu().detach().numpy()
        
            result.append(x)

    result = np.concatenate(result)

    class_pkl_file = open('office_encoder.pkl', 'rb')
    office_label_encoder = pickle.load(class_pkl_file)
    class_pkl_file.close()

    result_string = list(office_label_encoder.inverse_transform(result))

    test_table = pd.read_csv(sys.argv[1])
    test_table["label"] = result_string
    test_table.to_csv(sys.argv[3],index=False)

    # Print the information.
    print(f"Prediction file saved to {sys.argv[3]}")

if __name__ == '__main__':
    main()
