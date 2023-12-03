from PIL import Image
from datasets import load_dataset
import os
import numpy as np
from tqdm import tqdm


dataset = load_dataset('facebook/winoground', use_auth_token='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')['test']

data_ids = np.arange(400)

scores = []
save_dir = '/user_data/junruz/11777Proj/wino/img'
for id in tqdm(data_ids):
    id = int(id)
    caption_0 = dataset[id]['caption_0']
    caption_1 = dataset[id]['caption_1']
    img_0 = dataset[id]['image_0']
    img_1 = dataset[id]['image_1']
    path_0 = os.path.join(save_dir,f'{id}_0.png')
    path_1 = os.path.join(save_dir,f'{id}_1.png')
    img_0.save(path_0)
    img_1.save(path_1)