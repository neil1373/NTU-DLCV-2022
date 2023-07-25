from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from PIL import Image
import numpy as np
import random
import os

from tokenizers import Tokenizer

from .utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 224


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # RandomRotation(),
    # transforms.Lambda(under_max),
    # transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
    transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Lambda(under_max),
    transforms.ToTensor(),
    # transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
    transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
])

class CocoCaption(Dataset):
    
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(self._process(val['image_id'], ann['images']), val['caption']) for val in ann['annotations']]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)
        self.tokenizer = Tokenizer.from_file('./hw3_data/caption_tokenizer.json')
        self.tokenizer.enable_padding(length=64)
        self.max_length = max_length + 1

    def _process(self, image_id, images):
        # val = str(image_id).zfill(12)
        for val in images:
            if val['id'] == image_id:
                return val['file_name']
        assert False

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id)).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode(caption)

        caption = np.array(caption_encoded.ids)
        cap_mask = (1 - np.array(caption_encoded.attention_mask)).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask

def build_dataset(config, mode='training'):
    
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'images', 'train')
        train_file = os.path.join(config.dir, 'train.json')
        data = CocoCaption(root=train_dir, ann=read_json(train_file),
                            max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training')
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.dir, 'images', 'val')
        val_file = os.path.join(config.dir, 'val.json')
        data = CocoCaption(root=val_dir, ann=read_json(val_file),
                            max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
