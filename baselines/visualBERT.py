from transformers import AutoTokenizer, VisualBertForVisualReasoning
import torch
from PIL import Image
from datasets import load_dataset
import numpy as np
import random
from matplotlib import pyplot as plt
import json
from tqdm import tqdm
import clip


dataset = load_dataset('facebook/winoground', use_auth_token='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')['test']
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def visualbert_score(image, label):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2")

    text = "Who is eating the apple?"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        image_features = model.encode_image(image)
    visual_embeds = get_visual_embeddings(image).unsqueeze(0)
    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

    inputs.update(
        {
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        }
    )

    labels = torch.tensor(label).unsqueeze(0)  # Batch size 1, Num choices 2

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    scores = outputs.logits

    return scores


def img_score(tag):
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