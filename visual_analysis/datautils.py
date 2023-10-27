from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torchvision.transforms as transforms 


class winoground(Dataset):
    def __init__(self, h, w) -> None:
        super().__init__()
        auth_token = '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'
        self._winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)['test']
        self.transforms = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((h, w))
        ])
    
    def __getitem__(self, index):
        sample = self._winoground[index]
        i1 = sample['image_0'].convert("RGB")
        i2 = sample['image_1'].convert("RGB")
        i1 = self.transforms(i1).float()
        i2 = self.transforms(i2).float()
        c1 = sample['caption_0']
        c2 = sample['caption_1']
        return i1, i2, c1, c2, sample['tag'], sample['secondary_tag'], sample['collapsed_tag']
    
    def __len__(self):
        return len(self._winoground)


def get_loader(height, width, batch_size, num_workers, pin_memory=True, drop_last=True, shuffle=True):
    return DataLoader(winoground(h=height, w=width), 
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      shuffle=shuffle,
                      drop_last=drop_last)
