from transformers import ViltProcessor, ViltForImageAndTextRetrieval
from winoground import winoground
import torch
import torch.nn as nn
from tqdm import trange
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = winoground(convert_to_tensor=False)

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")
model = model.to(device)

# forward pass
scores_all = []
attr_all = []
for i in trange(len(ds)):
    # prepare inputs
    scores = dict()
    attr = dict()

    # four pair encoding
    i1, i2, c1, c2, t, sec_t, collapsed_t = ds[i]
    i1c1 = processor(i1, c1, return_tensors="pt").to(device)
    i1c2 = processor(i1, c2, return_tensors="pt").to(device)
    i2c1 = processor(i2, c1, return_tensors="pt").to(device)
    i2c2 = processor(i2, c2, return_tensors="pt").to(device)
    outputs_i1c1 = model(**i1c1)
    outputs_i1c2 = model(**i1c2)
    outputs_i2c1 = model(**i2c1)
    outputs_i2c2 = model(**i2c2)
    
    # image and text score
    logits_i1c1 = outputs_i1c1.logits[0, :].item()
    logits_i1c2 = outputs_i1c2.logits[0, :].item()
    logits_i2c1 = outputs_i2c1.logits[0, :].item()
    logits_i2c2 = outputs_i2c2.logits[0, :].item()
    scores['image'] = 1 if (logits_i1c1 > logits_i2c1) & (logits_i2c2 > logits_i1c2) else 0
    scores['text'] = 1 if (logits_i1c1 > logits_i1c2) & (logits_i2c2 > logits_i2c1) else 0
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
