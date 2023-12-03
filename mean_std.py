import torch
import numpy as np
import os
import skimage.io as io
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = sorted(os.listdir(img_dir))
        self.transform = Compose([Resize((224, 224)), ToTensor()])
        
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        img_path = img_path.replace('.jpg', '-midas_v21_small_256.png')
        img_path = img_path.replace('saliency', 'depth')
        img = Image.open(img_path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_list)
    
img_dir = '/user_data/junruz/coco/saliency/train2014'
dataset = MyDataset(img_dir)
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    shuffle=False
    )

mean = np.zeros(3)
std = np.zeros(3)
nb_samples = 0.

for img in tqdm(dataset):
    img = img.view(img.size(0), -1)
    mean += torch.mean(img, axis=1).numpy()
    std += torch.var(img, axis=1).numpy()
    nb_samples += 1

# for data in loader:
#     data = data / 1.
#     batch_samples = data.size(0)
#     data = data.view(batch_samples, data.size(1), -1)
#     mean += torch.sum(torch.mean(data, axis=2), axis=0)
#     std += torch.sum(torch.var(data, axis=2), axis=0)
#     nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
std = np.sqrt(std)
print(f'mean: {mean}')
print(f'std: {std}')