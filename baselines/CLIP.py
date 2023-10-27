import torch
import clip
from PIL import Image
from datasets import load_dataset
import numpy as np
import random
from matplotlib import pyplot as plt
import json
from tqdm import tqdm


dataset = load_dataset('facebook/winoground', use_auth_token='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')['test']
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def img_score(tag=None):
    if tag is not None and tag != 'No Tag':
        f = open('/user_data/junruz/11777Proj/new_tag_assignments.json')
        tags = json.load(f)
        data_ids = [int(key) for key, value in tags.items() if (tag in value)]
        f.close()
    elif tag == 'No Tag':
        f = open('/user_data/junruz/11777Proj/new_tag_assignments.json')
        tags = json.load(f)
        data_ids = [int(key) for key, value in tags.items() if len(value) == 0]
        f.close()
    else:
        data_ids = np.arange(400)
    
    score = []
    for id in tqdm(data_ids):
        id = int(id)
        caption_0 = dataset[id]['caption_0']
        caption_1 = dataset[id]['caption_1']
        img_0 = dataset[id]['image_0']
        img_1 = dataset[id]['image_1']

        with torch.no_grad():
            text_0 = clip.tokenize(caption_0).to(device)
            text_1 = clip.tokenize(caption_1).to(device)
            image_0 = preprocess(img_0)
            image_1 = preprocess(img_1)

            images = [image_0, image_1]
            img_input = torch.stack(images).to(device)
            
            logits_per_img_0, logits_per_text_0 = model(img_input, text_0)
            probs_0 = logits_per_text_0.softmax(dim=-1).cpu().numpy().squeeze()
            logits_per_img_1, logits_per_text_1 = model(img_input, text_1)
            probs_1 = logits_per_text_1.softmax(dim=-1).cpu().numpy().squeeze()
        
        if probs_0[0] > probs_0[1] and probs_1[1] > probs_1[0]:
            score.append(1)
        else:
            score.append(0)
        
    return np.mean(score), score


def txt_score(tag=None):
    if tag is not None and tag != 'No Tag':
        f = open('/user_data/junruz/11777Proj/new_tag_assignments.json')
        tags = json.load(f)
        data_ids = [int(key) for key, value in tags.items() if (tag in value)]
        f.close()
    elif tag == 'No Tag':
        f = open('/user_data/junruz/11777Proj/new_tag_assignments.json')
        tags = json.load(f)
        data_ids = [int(key) for key, value in tags.items() if len(value) == 0]
        f.close()
    else:
        data_ids = np.arange(400)
    
    score = []
    for id in tqdm(data_ids):
        id = int(id)
        caption_0 = dataset[id]['caption_0']
        caption_1 = dataset[id]['caption_1']
        img_0 = dataset[id]['image_0']
        img_1 = dataset[id]['image_1']

        with torch.no_grad():
            image_0 = preprocess(img_0).unsqueeze(0).to(device)
            image_1 = preprocess(img_1).unsqueeze(0).to(device)

            text_input = clip.tokenize([caption_0, caption_1]).to(device)
            
            logits_per_img_0, logits_per_text_0 = model(image_0, text_input)
            probs_0 = logits_per_img_0.softmax(dim=-1).cpu().numpy().squeeze()
            logits_per_img_1, logits_per_text_1 = model(image_1, text_input)
            probs_1 = logits_per_img_1.softmax(dim=-1).cpu().numpy().squeeze()
        
        if probs_0[0] > probs_0[1] and probs_1[1] > probs_1[0]:
            score.append(1)
        else:
            score.append(0)
        
    return np.mean(score), np.array(score)




    


if __name__ == '__main__':
    image_score, img_scores = img_score()
    text_score, txt_scores = txt_score()
    group_score = np.mean(img_scores * txt_scores)
    print(f'text score:{text_score} image score: {image_score} group score: {group_score}')

    pass


# image = preprocess(Image.open("/user_data/junruz/11777Proj/Raj_dnn.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["person with stick", "person"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]