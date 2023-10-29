from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from datautils import *
import torch
import torch.nn as nn
from tqdm import trange
from PIL import Image
import json


def itm_score(image:Image, caption):
    img = vis_processors["eval"](image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)
    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    itm_scores = torch.softmax(itm_output, dim=1)
    return itm_scores[:, 1].item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = winoground(convert_to_tensor=False)
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)

# forward pass
scores_all = []
attr_all = []
for i in trange(len(ds)):
    # prepare inputs
    scores = dict()
    attr = dict()

    # four pair encoding
    i1, i2, c1, c2, t, sec_t, collapsed_t = ds[i]
    
    # image and text score
    s_i1c1 = itm_score(i1, c1)
    s_i1c2 = itm_score(i1, c2)
    s_i2c1 = itm_score(i2, c1)
    s_i2c2 = itm_score(i2, c2)
    scores['image'] = 1 if (s_i1c1 > s_i2c1) & (s_i2c2 > s_i1c2) else 0
    scores['text'] = 1 if (s_i1c1 > s_i1c2) & (s_i2c2 > s_i2c1) else 0
    scores['group'] = 1 if bool(scores['image']) & bool(scores['text']) else 0
    scores_all.append(scores)
    
    # additional attr
    attr['tag'] = t
    attr['secondary_tag'] = sec_t
    attr['collapsed_tag'] = collapsed_t
    attr_all.append(attr)

img_scores = [dic['image'] for dic in scores_all]
text_scores = [dic['text'] for dic in scores_all]
group_scores = [dic['group'] for dic in scores_all]
print(f'mean img score: {np.mean(img_scores)}; mean text score: {np.mean(text_scores)}; mean group score: {np.mean(group_scores)}')

file = open('new_tag_assignments.json')
new_tags = json.load(file)

for tag in ['Non Minimal', 'Ambiguously Correct', 'Visually Difficult', 'Unusual Image', 'Unusual Text', 'Complex Reasoning']:
    tag_idx = [int(key) for key, val in new_tags.items() if tag in val]
    img_scores = [scores_all[i]['image'] for i in tag_idx]
    text_scores = [scores_all[i]['text'] for i in tag_idx]
    group_scores = [scores_all[i]['group'] for i in tag_idx]
    print(f'{tag}: mean img score: {np.mean(img_scores)}; mean text score: {np.mean(text_scores)}; mean group score: {np.mean(group_scores)}')

# no tag
tag_idx = [int(key) for key, val in new_tags.items() if len(val) == 0]
img_scores = [scores_all[i]['image'] for i in tag_idx]
text_scores = [scores_all[i]['text'] for i in tag_idx]
group_scores = [scores_all[i]['group'] for i in tag_idx]
print(f'No Tag: mean img score: {np.mean(img_scores)}; mean text score: {np.mean(text_scores)}; mean group score: {np.mean(group_scores)}')
