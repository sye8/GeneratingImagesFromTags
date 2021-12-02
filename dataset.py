import os
import pickle

import torch

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
    def __init__(self, path, max_cap_len):
        self.path = path
        self.max_cap_len = max_cap_len
        self.imgs = []
        self.tags = []
        for f in os.listdir(self.path):
            f_lower = f.lower()
            if f_lower.endswith(".jpg"):
                self.imgs.append(f)
                self.tags.append(".".join(f.split(".")[:-1]) + ".txt")
        self.transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        tok2idx_path = os.path.join(self.path, "tok2idx.pkl")
        idx2tok_path = os.path.join(self.path, "idx2tok.pkl")
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
        tags = open(os.path.join(self.path, self.tags[idx]), "r").read()
        tags = torch.tensor(self.caption2tokens(tags)).long()
        img = Image.open(os.path.join(self.path, self.imgs[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img.float(), tags

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


if __name__ == "__main__":
    print("Test Dataloader")
    dataset = PixivFacesDataset(DATASET_ROOT, DALLE_TEXT_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=8)
    img_batch, tag_batch = next(iter(dataloader))
    print("Image batch has shape:", img_batch.shape)
    print("Tag batch has shape:", tag_batch.shape)
    print("First tags translates to:\n", dataset.multihot2tags(tag_batch[0]))
