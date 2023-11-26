# conda activate shap (rampage)
import shap
from datautils import winoground
import torch
import numpy as np
from PIL import Image
import os, copy, sys
import math, json
import random
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


def custom_masker(mask, x):
    """
    Shap relevant function. Defines the masking function so the shap computation
    can 'know' how the model prediction looks like when some tokens are masked.
    """
    masked_X = x.clone()
    mask = torch.tensor(mask).unsqueeze(0)
    masked_X[~mask] = 0  # ~mask !!! to zero
    # never mask out CLS and SEP tokens (makes no sense for the model to work without them)
    masked_X[0, 0] = 49406
    masked_X[0, text_length_tok-1] = 49407
    return masked_X


def get_model_prediction(x):
    """
    Shap relevant function. Predict the model output for all combinations of masked tokens.
    """
    with torch.no_grad():
        # split up the input_ids and the image_token_ids from x (containing both appended)
        input_ids = torch.tensor(x[:, :inputs.input_ids.shape[1]])
        masked_image_token_ids = torch.tensor(x[:, inputs.input_ids.shape[1]:])

        # select / mask features and normalized boxes from masked_image_token_ids
        result = np.zeros(input_ids.shape[0])

        row_cols = 224 // patch_size # 224 / 32 = 7

        # call the model for each "new image" generated with masked features
        for i in range(input_ids.shape[0]):
            # here the actual masking of CLIP is happening. The custom masker only specified which patches to mask, but no actual masking has happened
            masked_inputs = copy.deepcopy(inputs)  # initialize the thing
            masked_inputs['input_ids'] = input_ids[i].unsqueeze(0)

            # pathify the image
            # torch.Size([1, 3, 224, 224]) image size CLIP
            for k in range(masked_image_token_ids[i].shape[0]):
                if masked_image_token_ids[i][k] == 0:  # should be zero
                    m = k // row_cols
                    n = k % row_cols
                    masked_inputs["pixel_values"][:, :, m *
                        patch_size:(m+1)*patch_size, n*patch_size:(n+1)*patch_size] = 0 # torch.rand(3, patch_size, patch_size)  # np.random.rand()
            
            outputs = model(**masked_inputs)
            # CLIP does not work with probabilities, because these are computed with softmax among choices (which I do not have here)
            # this is the image-text similarity score
            result[i] = outputs.logits_per_image
    return result


def compute_mm_score(text_length, shap_values):
    """ Compute Multimodality Score. (80% textual, 20% visual, possibly: 0% knowledge). """
    text_contrib = np.abs(shap_values.values[0, 0, :text_length]).sum()
    image_contrib = np.abs(shap_values.values[0, 0, text_length:]).sum()
    text_score = text_contrib / (text_contrib + image_contrib)
    # image_score = image_contrib / (text_contrib + image_contrib) # is just 1 - text_score in the two modalities case
    return text_score


def load_models():
    """ Load models and model components. """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def get_colormap(values):
    minima = min(values)
    maxima = max(values)

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys)

    return mapper
    

model, processor = load_models()
ds = winoground()
i1, i2, c1, c2, t, sec_t, collapsed_t = ds[23]  # find a wrong sample

# shap values need one sentence for transformer
inputs = processor(text=c1, images=i2, return_tensors="pt", padding=True)
model_prediction = model(**inputs).logits_per_image[0,0].item()

text_length_tok = inputs.input_ids.shape[1]
p = int(math.ceil(np.sqrt(text_length_tok)))
patch_size = 224 // p
image_token_ids = torch.tensor(range(1, p**2+1)).unsqueeze(0)  # (inputs.pixel_values.shape[-1] // patch_size)**2 +1
# make a cobination between tokens and pixel_values (transform to patches first)
X = torch.cat((inputs.input_ids, image_token_ids), 1).unsqueeze(1)

# create an explainer with model and image masker
explainer = shap.Explainer(get_model_prediction, custom_masker, silent=True)
shap_values = explainer(X)
shap_val_abs = np.abs(shap_values.values)
mm_score = compute_mm_score(text_length_tok, shap_values)

# visualize patch importance
img = i2.unsqueeze(0).numpy()  # 1, 3, 224, 224
row_cols = 224 // patch_size  # 224 / 32 = 7
mask = np.zeros_like(img)
image_val = shap_val_abs.flatten()[text_length_tok:]
# image_val = np.random.rand(9)
mapper = get_colormap(list(image_val))
for k in range(image_val.shape[0]): 
    m = k // row_cols
    n = k % row_cols
    v = image_val[k]
    c = np.array(mapper.to_rgba(v)[:-1]).reshape(1, 3, 1, 1)
    mask[:, :, m*patch_size: (m+1)*patch_size, n*patch_size: (n+1)*patch_size] = c
mask = np.moveaxis(mask, 1, -1).squeeze()  # 224, 224, 3
mask = (mask * 255).astype(np.int64)
img = np.moveaxis(img, 1, -1).squeeze().astype(np.int64)

fig = plt.figure(tight_layout=True, figsize=(16, 20))
gs = gridspec.GridSpec(5, 4)

ax1 = fig.add_subplot(gs[:-1, :])
ax1.imshow(img, alpha=1.0)
ax1.imshow(mask, alpha=.5)
# fig.savefig('image_i1c1.jpg')

# visualize text importance
tok = processor.tokenizer
strs = tok.convert_ids_to_tokens(inputs.input_ids.flatten())
text_val = shap_val_abs.flatten()[:text_length_tok]
# text_val = np.random.rand(6)

ax2 = fig.add_subplot(gs[-1, :])
ax2.plot(text_val)
ax2.set_xticks(np.arange(len(text_val)), strs, rotation=45, ha='right', fontsize=20)
ax2.set_xticklabels(strs, rotation=45, ha='right')
fig.savefig('sample_23_i2c1.png')
