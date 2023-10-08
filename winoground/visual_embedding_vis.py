from datautils import get_loader
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# param
height = 224
width = 224
batch_size = 16
num_workers = 0
pin_memory = True
shuffle = True
drop_last=True

# models
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
del model._modules['fc']
encoder = nn.Sequential(*list(model._modules.values()))
encoder.to(_device)
loader = get_loader(height, width, batch_size, num_workers, 
                    pin_memory=pin_memory, shuffle=shuffle, drop_last=drop_last)

# get embeddings
e1s, e2s = [], []
c1s, c2s = [], []
tags, sc_tags, co_tags = [], [], []
for idx, (i1, i2, c1, c2, tag, sc_tag, co_tag) in enumerate(loader):
    # sent to gpu
    i1 = i1.to(_device)
    i2 = i2.to(_device)
    # record embeddings
    e1s.append(encoder(i1).squeeze())
    e2s.append(encoder(i2).squeeze())
    c1s += c1
    c2s += c2
    tags += tag
    sc_tags += sc_tag
    co_tags += co_tag

e1s = torch.cat(e1s, 0)
e2s = torch.cat(e2s, 0)

# t-sne distance


