import os
import cv2
import pickle
import more_itertools as mi
import albumentations
from pathlib import Path

import torch

import torch.nn.functional as F

import numpy as np

from collections import Counter

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

from tqdm import tqdm

from configs import *


class ImageDataset(Dataset):
    def __init__(self, img, transform):
        self.path = img
        self.imgs = []
        for f in os.listdir(self.path):
            f_lower = f.lower()
            if f_lower.endswith(".jpg") or f_lower.endswith(".jpeg") or f_lower.endswith(".png"):
                self.imgs.append(f)
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.path, self.imgs[idx])).convert("RGB")
        image = self.transform(image)
        return image
    

class PixivFacesDataset(Dataset):
    def __init__(self, path, max_cap_len, transform, multihot=False):
        self.path = path
        self.max_cap_len = max_cap_len
        self.imgs = []
        self.tags = []
        for f in Path(self.path).glob('**/*.png'):
            self.imgs.append(f)
            self.tags.append(f.with_suffix('.txt'))
        self.transform = transform
        self.multihot = multihot
        tok2idx_path = os.path.join(self.path, "tok2idx.pkl")
        idx2tok_path = os.path.join(self.path, "idx2tok.pkl")
        size=256
        rotate_pre_cropper = albumentations.Crop(x_min=0, y_min=0, x_max=int(size*1.90),y_max=int(size*1.90))
        rotater = albumentations.augmentations.geometric.Rotate(limit=10, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_CONSTANT,always_apply=True,value=(255,255,255))
        rotate_cropper = albumentations.CenterCrop(height=int(size*1.90),width=int(size*1.90))
        cropper = albumentations.RandomResizedCrop(height=size,width=size, scale=(0.95,1.00), ratio=(1.0,1.0),interpolation=cv2.INTER_AREA)
        self.preprocessor = albumentations.Compose([rotate_pre_cropper, rotater, rotate_cropper, cropper],keypoint_params=albumentations.KeypointParams(format='xy',check_each_transform=False))

        if not os.path.exists(tok2idx_path) or not os.path.exists(idx2tok_path):
            print("Building vocab")
            vocab = set()
            for tag_file in tqdm(self.tags):
                caption = open(os.path.join(self.path, tag_file), "r").read()
                for token in (caption.strip().split()):
                    vocab.add(token)
            vocab = list(vocab)
            vocab.insert(0, "<pad>")
            idx = range(len(vocab))
            self.tok2idx = dict(zip(vocab, idx))
            self.idx2tok = dict(zip(idx, vocab))
            with open(tok2idx_path, 'wb') as tok2idx_file:
                pickle.dump(self.tok2idx, tok2idx_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(idx2tok_path, 'wb') as idx2tok_file:
                pickle.dump(self.idx2tok, idx2tok_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading vocab")
            with open(tok2idx_path, 'rb') as tok2idx_file:
               self.tok2idx = pickle.load(tok2idx_file)
            with open(idx2tok_path, 'rb') as idx2tok_file:
               self.idx2tok = pickle.load(idx2tok_file)
               
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, idx):
        tags = self.tags[idx].read_text()
        tags_split = tags.strip().split()
        tags_xy=list(mi.chunked(list(map(int,tags_split[:56])),2))
        #if not self.multihot:
        #    tags = torch.tensor(self.caption2tokens(tags)).long()
        #else:
        #    tags = self.tags2multihot(tags).float()
        #img = Image.open(os.path.join(self.path, self.imgs[idx])).convert("RGB")
        #print(tags_xy)
        #if np.min(tags_xy) < 0:
        #    print(self.imgs[idx])
        img,tags_xy = self.preprocess_image(str(self.imgs[idx]),tags_xy)
        #print(tags_xy)
        tags_split[:56] = map(str,np.round(tags_xy).flatten().astype(int))
        tags = ' '.join(tags_split)
        tags = torch.tensor(self.caption2tokens(tags)).long()
        #if self.transform is not None:
        #    img = self.transform(img)
        return img, tags

    def caption2tokens(self, caption):
        tags_idxs = [self.tok2idx[t] for t in caption.strip().split()][:self.max_cap_len]
        tags_idxs.extend([0] * (self.max_cap_len - len(tags_idxs)))
        return np.array(tags_idxs)
        
    def tokens2captions(self, idx):
        captions = []
        for i in idx:
            if i > 0:
                captions.append(self.idx2tok[i])
        return " ".join(captions)

    def tags2multihot(self, tags):
        tags_idxs = [self.tok2idx[t] for t in tags.strip().split()]
        tags_idxs = torch.tensor(tags_idxs)
        tags_idxs = F.one_hot(tags_idxs, num_classes=len(self.tok2idx))
        tags_idxs = torch.sum(tags_idxs, axis=0)
        return tags_idxs
        
    def multihot2tags(self, multihot):
        tags = []
        for i, x in enumerate(multihot.squeeze().long()):
            if x:
                tags.append(self.idx2tok[i])
        return " ".join(tags)

    def preprocess_image(self, image_path, keypoints):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        transformed=self.preprocessor(image=image,keypoints=keypoints)
        keypoints = np.round(transformed['keypoints'])
        image = transformed['image']
        #image = self.cropper(image=image)['image']

        if self.transform is not None:
            image = self.transform(image)
        #image = (image/127.5 - 1.0).astype(np.float32)
        return image,keypoints


if __name__ == "__main__":
    print("Test Dataloader")
    #dataset = PixivFacesDataset(DATASET_ROOT, DALLE_TEXT_SEQ_LEN, None)
    #dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=8)
    #img_batch, tag_batch = next(iter(dataloader))
    #print("Image batch has shape:", img_batch.shape)
    #print("Tag batch has shape:", tag_batch.shape)
    #print("First tags translates to:\n", dataset.tokens2captions(tag_batch[0]))
    for p in Path('danbooru2020_filter_face_nobg_anno_clean').glob('**/*.txt'):
        tags = p.read_text()
        tags_split = tags.strip().split()
        tags_xy=list(mi.chunked(list(map(int,tags_split[:56])),2))
        #if np.min(tags_xy) < 0:
        #    print(p)
        if np.max(tags_xy) > 380:
            print(p)