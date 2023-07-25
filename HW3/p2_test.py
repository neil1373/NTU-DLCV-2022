import os
import sys
import json
import torch

from tokenizers import Tokenizer

from PIL import Image

from helper import dataset
from helper.config import Config
from helper.transformer import Transformer
from helper import dataset

config = Config()
# load model checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

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

# for i in range(1):  # config.epoch
checkpoint_path = "ViT_finetuned.pth"
if checkpoint_path is None:
    raise NotImplementedError('No model to chose from!')
else:
    if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
    print(f"Loading Checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)

tokenizer = Tokenizer.from_file('caption_tokenizer.json')

# start_token = Tokenizer.convert_tokens_to_ids(Tokenizer._cls_token)
# end_token = Tokenizer.convert_tokens_to_ids(Tokenizer._sep_token)
start_token = 2


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

@torch.no_grad()
def evaluate(image, caption):
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        image, caption = image.to(device), caption.to(device)
        predictions, attns = model(image, caption)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        
        if predicted_id[0] == 3 or (predicted_id[0] == 5 and caption[:, i] == 5):

            if (caption[:, i] != 13):
                caption[:, i+1] = 13
            return caption
        
        caption[:, i+1] = predicted_id[0]
        # cap_mask[:, i+1] = False

    return caption

images_path = sorted(os.listdir(sys.argv[1]))

count = 0
output_dict = {}
for image_path in images_path:
    # image_path = images_path[0]
    # print(image_path)
    image = Image.open(os.path.join(sys.argv[1], image_path)).convert("RGB")
    image = dataset.val_transform(image)
    image = image.unsqueeze(0)

    output = evaluate(image, caption)
    # print(output)
    # print(output[0])
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    # result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(count, result.capitalize())
    output_dict[image_path.split('.')[0]] = result.capitalize()
    count += 1

with open(sys.argv[2], 'w') as f:
    json.dump(output_dict, f)