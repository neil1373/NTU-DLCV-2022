import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import clip

def eval(data_path, class_dict_file, prediction_file):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # load class list
    with open(class_dict_file, newline='') as file:
        class_dict = json.load(file)
    class_list = []
    for cls in class_dict:    
        class_list.append(class_dict[cls])
    
    # Prepare the input images
    image_list = sorted(os.listdir(data_path))
    result = []
    correct = 0
    for image_file in tqdm(image_list, desc = 'Inference'):
        image = Image.open(os.path.join(data_path, image_file))
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat([clip.tokenize(f"This is a {c}.") for c in class_list]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
        value, index = values.detach().cpu().numpy()[0], indices.detach().cpu().numpy()[0]
        result.append(index)
        correct += (index == int(image_file.split('_')[0]))
        '''
        # Print the result
        print("\nTop predictions:\n")
        for value, index in zip(values, indices):
            print(f"{class_list[index]:>16s}: {100 * value.item():.2f}%")
        '''
    
    print(f"Accuracy: {correct / len(image_list) * 100}%.")

    # Generate your submission
    result = np.array(result)    
    df = pd.DataFrame({'filename': image_list, 'label': result})
    df.to_csv(prediction_file,index=False) 

def main():
    eval(sys.argv[1], sys.argv[2], sys.argv[3])

if __name__ == '__main__':
    main()