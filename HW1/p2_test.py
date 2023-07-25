# Import necessary packages.
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
# import constructed model in torchvision
import torchvision.models as models
# This is for the progress bar.
import os
import sys
from tqdm.auto import tqdm
# This is for self-defined dataset.
from SatMaskDataset import *
# Evaluation Metric: MIoU
from mean_iou_evaluate import *

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
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fix random seed for reproducibility
    same_seeds(739)

    test_tfm = transforms.ToTensor()
    
    test_set = SatMaskDataset(root = sys.argv[1], transform = test_tfm)
    batch_size = 1

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = models.segmentation.deeplabv3_resnet50(num_classes = 7, weight='DEFAULT')
    model = model.to(device)

    model_path = 'model_deeplabv3_resnet50.pth'
    print("Loading Checkpoint:", model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    best_valid_miou = 0

    pred_masks_cpu = []
    labels_cpu = []

    # fetching file lists and file processing 
    files = [f for f in sorted(os.listdir(sys.argv[1])) if f.endswith(".jpg")]
    
    # create output dir
    if not os.path.isdir(sys.argv[2]):
        os.mkdir(sys.argv[2])

    for batch in tqdm(test_loader, desc='Inference'):
        
        with torch.no_grad():
            imgs, labels = batch
            labels = labels.squeeze(1)
            imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.long)
            logits = model(imgs)['out']

            pred_masks = logits.argmax(dim=1)

            pred_masks_cpu.append(pred_masks.cpu())
            labels_cpu.append(labels.cpu())
    
    pred_masks_numpy = torch.cat(pred_masks_cpu).numpy()
    labels_numpy = torch.cat(labels_cpu).numpy()
    
    test_miou = mean_iou_score(pred_masks_numpy, labels_numpy)
    
    local_path = sys.argv[1].split("/")[-1]
    if not len(local_path):
        local_path = sys.argv[1].split("/")[-2]

    print(f"[ Test ] mIoU score on {local_path} = {test_miou:.6f}\n")
    del model
    # Generate masks
    IMG_SIZE = 512

    cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6: [0, 0, 0],}

    print("Generating masks...\n")

    output_images = np.zeros((pred_masks_numpy.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype='uint8')
    for idx in range(pred_masks_numpy.shape[0]):
        for height in range(IMG_SIZE):
            for width in range(IMG_SIZE):
                output_images[idx, height, width, :] = cls_color[pred_masks_numpy[idx, height, width]]
        image_filename = files[idx].split(".")[0] + ".png"
        image_path = os.path.join(sys.argv[2], image_filename)
        output_image = Image.fromarray(output_images[idx])
        output_image.save(image_path)

    print(f"Masks are saved in {sys.argv[2]}.\n")

    
if __name__ == '__main__':

    # test data path: sys.argv[1], output mask dir: sys.argv[2]
    main()