# dataloader here
from torch.utils.data import Dataset

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from omegaconf import OmegaConf
import os.path as op
import random
import os

from utils import load_from_yaml_file, read_json, load_config_file


import torch
import numpy as np


COCO_MEAN = (0.4711, 0.4475, 0.4080)
COCO_STD = (0.2398, 0.2348, 0.2398)
SALIENCY_MEAN = (0.1663, )
SALIENCY_STD = (0.3104, )
DEPTH_MEAN = (0.4225, 0.4012, 0.3659)
DEPTH_STD = (0.2681, 0.2635, 0.2763)


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])


def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_id_to_img_path[img_id] = file_name
    
    return img_id_to_img_path

def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for caption_info in annotations['annotations']:
        img_id = caption_info['image_id']
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []
        
        caption = caption_info['caption']
        img_id_to_captions[img_id].append(caption)
    
    return img_id_to_captions

class CLIP_COCO_dataset(Dataset):
    """CLIP_COCO_dataset. To train CLIP on COCO-Captions."""

    def __init__(self, config, text_tokenizer, context_length=77, input_resolution=224):
        
        super(CLIP_COCO_dataset, self).__init__()

        self.config = config

        annotation_file = self.config.train_annotation_file
        # print("annotation_file : ", annotation_file)
        annotations = read_json(annotation_file)

        self.img_id_to_filename = get_img_id_to_img_path(annotations)
        # print("img_id_to_filename : ", self.img_id_to_filename)

        self.img_id_to_captions = get_img_id_to_captions(annotations)

        self.img_ids = list(self.img_id_to_filename.keys())
        # print("total image ids = ", len(self.img_ids))

        self.img_dir = self.config.train_img_dir
        # print("img dir : ", self.img_dir)

        self.transform = _transform(input_resolution)
        self._tokenizer = text_tokenizer
        self.context_length = context_length


    def tokenize(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # randomly pick one caption from the image captions
        text = random.choice(self.img_id_to_captions[img_id])

        img_filename = self.img_id_to_filename[img_id]

        img_path = op.join(self.img_dir, img_filename)
        img = Image.open(img_path)
        img_input = self.transform(img)
        text_input = self.tokenize(text)

        return img_input, text_input


class Coco_Aug(Dataset):
    def __init__(self, text_tokenizer, context_length=77, h=224, w=224) -> None:
        super().__init__()
        self.img_dir = '/user_data/weifanw/coco/train2014/train2014'
        self.saliency_map_dir = '/user_data/junruz/coco/saliency/train2014'
        self.depth_map_dir = '/user_data/junruz/coco/depth/train2014'
        
        # get annotations
        annotation_file = '/user_data/weifanw/coco/annotations/captions_train2014.json'
        ann =  read_json(annotation_file)
        self.ann = ann['annotations']
        self.image_meta = ann['images']

        # tokonizer
        self._tokenizer = text_tokenizer
        self.context_length = context_length

        # transforms
        self.tfs_img = Compose([
            Resize((h, w), interpolation=Image.BICUBIC),
            CenterCrop((h, w)),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(COCO_MEAN, COCO_STD)
        ])
        self.tfs_saliency = Compose([
            Resize((h, w), interpolation=Image.BICUBIC),
            CenterCrop((h, w)),
            ToTensor(),
            Normalize(SALIENCY_MEAN, SALIENCY_STD)
        ])
        self.tfs_depth = Compose([
            Resize((h, w), interpolation=Image.BICUBIC),
            CenterCrop((h, w)),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(DEPTH_MEAN, DEPTH_STD)
        ])
        
        # self._coco = CocoDetection(root, annotation_file, transform=tfs)

    def tokenize(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result
    
    def get_augmented_image(self, img, saliency_map, depth_map):
        img = self.tfs_img(img)
        saliency_map = self.tfs_saliency(saliency_map)
        depth_map = self.tfs_depth(depth_map)
        return torch.cat([img, saliency_map, depth_map], 0)  # C, H, W

        
    def __getitem__(self, index):
        image_id = self.image_meta[index]['id']
        img_file_name = self.image_meta[index]['file_name']
        depth_map_name = img_file_name.split('.')[0] + '-midas_v21_small_256.png'

        # get caption
        id_list = np.array([ann_['image_id'] for ann_ in self.ann])
        idxs = np.where(id_list == image_id)[0]
        caps = [self.ann[int(idx)]['caption'] for idx in idxs]
        cap = random.choice(caps)
        text_input = self.tokenize(cap)

        # get aug image
        img = Image.open(os.path.join(self.img_dir, img_file_name))
        saliency_map = Image.open(os.path.join(self.saliency_map_dir, img_file_name))
        depth_map = Image.open(os.path.join(self.depth_map_dir, depth_map_name))
        image_input = self.get_augmented_image(img, saliency_map, depth_map)

        return image_input, text_input
    
    def __len__(self):
        return len(self.image_meta)
