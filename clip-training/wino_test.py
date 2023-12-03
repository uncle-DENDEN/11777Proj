import torch
import os
from PIL import Image
from datasets import load_dataset
import numpy as np
import random
from matplotlib import pyplot as plt
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from model.model import CLIP
from utils.simple_tokenizer import SimpleTokenizer
from utils import set_seed, mkdir, setup_logger, load_config_file


MODEL_CONFIG_PATH = '/user_data/junruz/11777Proj/clip-training/model/model_config.yaml'
# checkpoint_path = '/user_data/junruz/11777Proj/clip-training/saved_checkpoints/baseline/checkpoint_49_16200.pt'
checkpoint_path = '/user_data/junruz/11777Proj/clip-training/saved_checkpoints/auxilary/checkpoint_49_16200.pt'

COCO_MEAN = (0.4711, 0.4475, 0.4080)
COCO_STD = (0.2398, 0.2348, 0.2398)
SALIENCY_MEAN = (0.1663, )
SALIENCY_STD = (0.3104, )
DEPTH_MEAN = (0.5225, 0.2541, 0.2476)
DEPTH_STD = (0.3443, 0.2542, 0.1338)

dataset = load_dataset('facebook/winoground', use_auth_token='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')['test']
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
wino_dir = '/user_data/junruz/11777Proj/wino'

model_config = load_config_file(MODEL_CONFIG_PATH)

# Image transform and text tokenizer
preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)),
    ])

tokenizer = SimpleTokenizer()

# creating RN50 CLIP model
model_params = dict(model_config.RN50)
model_params['vision_layers'] = tuple(model_params['vision_layers'])
model_params['vision_patch_size'] = None
model = CLIP(**model_params)

# loading trained weights
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


def tokenize(tokenizer, text, context_length=77):
        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + tokenizer.encode(text) + [eot_token]
        result = torch.zeros(context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result


def aux_scores(tag=None):
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
    
    h, w = 224, 224
    pre_img = Compose([
        Resize((h, w), interpolation=Image.BICUBIC),
        CenterCrop((h, w)),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(COCO_MEAN, COCO_STD)
    ])

    pre_saliency = Compose([
        Resize((h, w), interpolation=Image.BICUBIC),
        CenterCrop((h, w)),
        ToTensor(),
        Normalize(SALIENCY_MEAN, SALIENCY_STD)
    ])

    pre_depth = Compose([
        Resize((h, w), interpolation=Image.BICUBIC),
        CenterCrop((h, w)),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(DEPTH_MEAN, DEPTH_STD)
    ])

    scores = []
    for id in tqdm(data_ids):
    # for id in [6]:
        id = int(id)
        caption_0 = dataset[id]['caption_0']
        caption_1 = dataset[id]['caption_1']
        img_0 = Image.open(os.path.join(wino_dir, 'img', f'{id}_0.png'))
        img_1 = Image.open(os.path.join(wino_dir, 'img', f'{id}_1.png'))
        depth_0 = Image.open(os.path.join(wino_dir, 'depth', f'{id}_0.png'))
        depth_1 = Image.open(os.path.join(wino_dir, 'depth', f'{id}_1.png'))
        saliency_0 = Image.open(os.path.join(wino_dir, 'saliency', f'{id}_0.png'))
        saliency_1 = Image.open(os.path.join(wino_dir, 'saliency', f'{id}_1.png'))

        with torch.no_grad():
            text_0 = tokenize(tokenizer, caption_0).to(device)
            text_1 = tokenize(tokenizer, caption_1).to(device)
            image_0 = pre_img(img_0)
            image_1 = pre_img(img_1)
            depth_0 = pre_depth(depth_0)
            depth_1 = pre_depth(depth_1)
            saliency_0 = pre_saliency(saliency_0)
            saliency_1 = pre_saliency(saliency_1)

            img_input_0 = torch.cat([image_0, saliency_0, depth_0], 0)
            img_input_1 = torch.cat([image_1, saliency_1, depth_1], 0)

            img_inputs = [img_input_0, img_input_1]
            img_input = torch.stack(img_inputs).to(device)
            texts = [text_0, text_1]
            text_input = torch.stack(texts).to(device)
            img_features, text_features = model(img_input, text_input)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()

            logits_per_img = logit_scale * img_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ img_features.t()

            logits_per_img = logits_per_img.cpu().numpy()
            logits_per_text = logits_per_text.cpu().numpy()
        
        img_score = 1 if logits_per_text[0, 0] > logits_per_text[0, 1] and logits_per_text[1, 1] > logits_per_text[1, 0] else 0
        text_score = 1 if logits_per_img[0, 0] > logits_per_img[0, 1] and logits_per_img[1, 1] > logits_per_img[1, 0] else 0
        group_score = img_score * text_score
        scores.append([text_score, img_score, group_score])
    scores = np.array(scores)
    
    return np.mean(scores, axis=0)


def scores(tag=None):
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
    
    scores = []
    for id in tqdm(data_ids):
    # for id in [6]:
        id = int(id)
        caption_0 = dataset[id]['caption_0']
        caption_1 = dataset[id]['caption_1']
        img_0 = dataset[id]['image_0']
        img_1 = dataset[id]['image_1']

        with torch.no_grad():
            text_0 = tokenize(tokenizer, caption_0).to(device)
            text_1 = tokenize(tokenizer, caption_1).to(device)
            image_0 = preprocess(img_0)
            image_1 = preprocess(img_1)

            images = [image_0, image_1]
            img_input = torch.stack(images).to(device)
            texts = [text_0, text_1]
            text_input = torch.stack(texts).to(device)
            img_features, text_features = model(img_input, text_input)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()

            logits_per_img = logit_scale * img_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ img_features.t()

            logits_per_img = logits_per_img.cpu().numpy()
            logits_per_text = logits_per_text.cpu().numpy()
        
        img_score = 1 if logits_per_text[0, 0] > logits_per_text[0, 1] and logits_per_text[1, 1] > logits_per_text[1, 0] else 0
        text_score = 1 if logits_per_img[0, 0] > logits_per_img[0, 1] and logits_per_img[1, 1] > logits_per_img[1, 0] else 0
        group_score = img_score * text_score
        scores.append([text_score, img_score, group_score])
    scores = np.array(scores)
    
    return np.mean(scores, axis=0)


if __name__ == '__main__':
    # score = scores()
    aux_score = aux_scores()
    print(aux_score)