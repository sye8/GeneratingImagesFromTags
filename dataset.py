import os
import re
import pickle

import spacy

import torch

import json

import numpy as np

from itertools import chain

from pathlib import Path

from collections import Counter
#os.environ["ARGOS_DEVICE_TYPE"]="cuda"
#from argostranslate import translate

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from PIL import Image

from tqdm import tqdm

from pycocotools.coco import COCO

from configs import *


def get_COCO_captions(root, img, ann, transform, max_cap_len):
    if not os.path.isdir(root):
        raise ValueError(root + " is a file or doesn't exist")
    img = os.path.join(root, img)
    ann = os.path.join(root, ann)
    dict_path = os.path.join(root, "dictionary")
    return CocoDataset(img, ann, transform, dict_path, max_cap_len)
        
def get_COCO_images(root, img, transform):
    if not os.path.isdir(root):
        raise ValueError(root + " is a file or doesn't exist")
    img = os.path.join(root, img)
    return ImageDataset(img, transform)

def get_COCO_images_2014_train(coco_root, transform):
    return get_COCO_images(coco_root, "train2014", transform)
    
def get_COCO_images_2014_val(coco_root, transform):
    return get_COCO_images(coco_root, "val2014", transform)

def get_COCO_captions_2014_train(coco_root, transform, max_cap_len):
    return get_COCO_captions(coco_root, "train2014",  os.path.join("annotations", "captions_train2014.json"), transform, max_cap_len)
    
def get_COCO_captions_2014_val(coco_root, transform, max_cap_len):
    return get_COCO_captions(coco_root, "val2014",  os.path.join("annotations", "captions_val2014.json"), transform, max_cap_len)


def get_rainbow_captions(root, transform, max_cap_len):
    if not os.path.isdir(root):
        raise ValueError(root + " is a file or doesn't exist")
    dict_path = os.path.join(root, "dictionary")
    return RainbowDataset(os.path.join(root, 'rainbow'),transform, dict_path, max_cap_len)

def get_rainbow_captions_2014_train(rainbow_root, transform, max_cap_len):
    return get_rainbow_captions(rainbow_root, transform, max_cap_len)

def get_pixiv_captions(root, transform, max_cap_len):
    if not os.path.isdir(root):
        raise ValueError(root + " is a file or doesn't exist")
    dict_path = os.path.join(root, "dictionary")
    return PixivDataset(os.path.join(root),transform, dict_path, max_cap_len)

def get_pixiv_captions_2021_train(rainbow_root, transform, max_cap_len):
    return get_pixiv_captions(rainbow_root, transform, max_cap_len)

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
        

class ImageSubDataset(ImageDataset):
    def __init__(self, img, samples, transform):
        super().__init__(img, transform)
        with open(samples, 'rb') as pkl:
            self.imgs = pickle.load(pkl)
    

class CocoDataset(Dataset):
    def __init__(self, img, ann, transform, dict_path, max_cap_len):
        self.img = img
        self.coco = COCO(ann)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.max_cap_len = max_cap_len
        self.nlp = spacy.load("en_core_web_sm")

        if not os.path.exists(dict_path):
            os.makedirs(dict_path)
            
        tok2idx_path = os.path.join(dict_path, "tok2idx.pkl")
        idx2tok_path = os.path.join(dict_path, "idx2tok.pkl")
        freq_path = os.path.join(dict_path, "freq.pkl")
        if not os.path.exists(tok2idx_path) or not os.path.exists(idx2tok_path):
            print("Building vocab")
            vocab = Counter()
            for i in tqdm(self.ids):
                caption = str(self.coco.anns[i]['caption']).lower()
                for token in self.nlp(caption):
                    if (token.pos_ == "NOUN" or token.pos_ == "PROPN") and len(token) > 2:
                        vocab[token.text] += 1
            vocab = vocab.most_common(VOCAB_SIZE-2)
            self.freq = dict(vocab)
            vocab = [e[0] for e in vocab]
            vocab.insert(0, "<pad>")
            vocab.insert(1, "<unk>")
            idx = range(len(vocab))
            self.tok2idx = dict(zip(vocab, idx))
            self.idx2tok = dict(zip(idx, vocab))
            with open(tok2idx_path, 'wb') as tok2idx_file:
                pickle.dump(self.tok2idx, tok2idx_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(idx2tok_path, 'wb') as idx2tok_file:
                pickle.dump(self.idx2tok, idx2tok_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(freq_path, 'wb') as freq_file:
                pickle.dump(self.freq, freq_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading vocab")
            with open(tok2idx_path, 'rb') as tok2idx_file:
                self.tok2idx = pickle.load(tok2idx_file)
            with open(idx2tok_path, 'rb') as idx2tok_file:
                self.idx2tok = pickle.load(idx2tok_file)
            with open(freq_path, 'rb') as freq_file:
                self.freq = pickle.load(freq_file)
        
    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = str(self.coco.anns[ann_id]['caption']).lower()
        img_id = self.coco.anns[ann_id]['image_id']
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        
        image = Image.open(os.path.join(self.img, img_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        return image.float(), torch.tensor(self.caption2tokens(caption)).long()

    def __len__(self):
        return len(self.ids)
        
    def caption2tokens(self, caption):
        tags = Counter()
        for token in self.nlp(caption):
            if token.text in self.freq:
                tags[token.text] = self.freq[token.text]
        tags = [self.tok2idx[e[0]] for e in tags.most_common(self.max_cap_len)]
        tags.extend([0] * (self.max_cap_len - len(tags)))
        return np.array(tags)
        
    def tokens2captions(self, idx):
        captions = []
        for i in idx:
            if i > 0:
                captions.append(self.idx2tok[i])
        return " ".join(captions)

class CocoSubDataset(Dataset):
    def __init__(self, img, ann, dict_path):
        with open(img, 'rb') as pkl:
            self.img = pickle.load(pkl)
        with open(ann, 'rb') as pkl:
            self.ann = pickle.load(pkl)
        tok2idx_path = os.path.join(dict_path, "tok2idx.pkl")
        idx2tok_path = os.path.join(dict_path, "idx2tok.pkl")
        print("Loading vocab")
        with open(tok2idx_path, 'rb') as tok2idx_file:
            self.tok2idx = pickle.load(tok2idx_file)
        with open(idx2tok_path, 'rb') as idx2tok_file:
            self.idx2tok = pickle.load(idx2tok_file)
    
    def __len__(self):
        return len(self.img)
        
    def __getitem__(self, index):
        return self.img[index], self.ann[index]

class RainbowDataset(Dataset):
    def __init__(self, img, transform, dict_path, max_cap_len):
        self.img = img
        #self.coco = COCO(ann)
        self.ids = list([str(p) for p in Path(img).glob('*.png')])
        self.transform = transform
        self.max_cap_len = max_cap_len
        self.nlp = spacy.load("en_core_web_sm")

        if not os.path.exists(dict_path):
            os.makedirs(dict_path)
            
        tok2idx_path = os.path.join(dict_path, "tok2idx.pkl")
        idx2tok_path = os.path.join(dict_path, "idx2tok.pkl")
        freq_path = os.path.join(dict_path, "freq.pkl")
        if not os.path.exists(tok2idx_path) or not os.path.exists(idx2tok_path):
            print("Building vocab")
            vocab = Counter()
            for i in tqdm(self.ids):
                caption = Path(i).stem.replace('_',' ')
                for token in self.nlp(caption):
                    if (token.pos_ == "NOUN" or token.pos_ == "PROPN" or token.pos_ == "ADJ") and len(token) > 2:
                        vocab[token.text] += 1
            vocab = vocab.most_common(VOCAB_SIZE-2)
            self.freq = dict(vocab)
            vocab = [e[0] for e in vocab]
            vocab.insert(0, "<pad>")
            vocab.insert(1, "<unk>")
            idx = range(len(vocab))
            self.tok2idx = dict(zip(vocab, idx))
            self.idx2tok = dict(zip(idx, vocab))
            with open(tok2idx_path, 'wb') as tok2idx_file:
                pickle.dump(self.tok2idx, tok2idx_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(idx2tok_path, 'wb') as idx2tok_file:
                pickle.dump(self.idx2tok, idx2tok_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(freq_path, 'wb') as freq_file:
                pickle.dump(self.freq, freq_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading vocab")
            with open(tok2idx_path, 'rb') as tok2idx_file:
                self.tok2idx = pickle.load(tok2idx_file)
            with open(idx2tok_path, 'rb') as idx2tok_file:
                self.idx2tok = pickle.load(idx2tok_file)
            with open(freq_path, 'rb') as freq_file:
                self.freq = pickle.load(freq_file)
        
    def __getitem__(self, index):
        ann_id = self.ids[index]
        #caption = str(self.coco.anns[ann_id]['caption']).lower()
        caption = Path(ann_id).stem.replace('_',' ')
        #img_id = self.coco.anns[ann_id]['image_id']
        img_path = ann_id
        
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        return image.float(), torch.tensor(self.caption2tokens(caption)).long()

    def __len__(self):
        return len(self.ids)
        
    def caption2tokens(self, caption):
        tags = Counter()
        for token in self.nlp(caption):
            if token.text in self.freq:
                tags[token.text] = self.freq[token.text]
        tags = [self.tok2idx[e[0]] for e in tags.most_common(self.max_cap_len)]
        tags.extend([0] * (self.max_cap_len - len(tags)))
        return np.array(tags)
        
    def tokens2captions(self, idx):
        captions = []
        for i in idx:
            if i > 0:
                captions.append(self.idx2tok[i])
        return " ".join(captions)


class PixivDataset(Dataset):
    def __init__(self, img, transform, dict_path, max_cap_len):
        self.img = img
        self.pixiv = json.loads((Path(img)/'valid_img.json').read_text())
        self.transform = transform
        self.max_cap_len = max_cap_len
        #self.nlp = spacy.load("en_core_web_sm")
        #installed_languages=translate.load_installed_languages()
        #elf.tran = installed_languages[1].get_translation(installed_languages[0])
        self.enlu = {}

        if not os.path.exists(dict_path):
            os.makedirs(dict_path)
            
        tok2idx_path = os.path.join(dict_path, "tok2idx.pkl")
        idx2tok_path = os.path.join(dict_path, "idx2tok.pkl")
        freq_path = os.path.join(dict_path, "freq.pkl")
        enlu_path = os.path.join(dict_path, 'enlu.pkl')
        if not os.path.exists(tok2idx_path) or not os.path.exists(idx2tok_path) or not os.path.exists(enlu_path):
            print("Building vocab, and english loopup")
            vocab = Counter()
            for i in tqdm(self.pixiv):
                likes = ['popular'] if i[1]>1000 else []
                image = Path(img)/Path(i[2]).name
                self.enlu[image.name]=[]
                if image.exists: 
                    caption = i[3]
                    #for token in chain(*[self.nlp(self.tran.translate(c)) for c in caption]):
                    #    if (token.pos_ == "NOUN" or token.pos_ == "PROPN" or token.pos_ == "ADJ") and len(token) > 2:
                    #        vocab[token.text] += 1
                    #        self.enlu[image.name].append(token.text)
                    for token in caption:
                        if 'users' not in token and 'Project' not in token:
                            if not re.search('C\d+',token):
                                vocab[token] += 1
                                self.enlu[image.name].append(token)
                self.enlu[image.name]=likes+self.enlu[image.name]
            self.ids = list(self.enlu.keys())
            vocab = vocab.most_common(VOCAB_SIZE-3)
            self.freq = dict(vocab)
            vocab = [e[0] for e in vocab]
            vocab.insert(0, "<pad>")
            vocab.insert(1, "<unk>")
            vocab.insert(2, "popular")
            idx = range(len(vocab))
            self.tok2idx = dict(zip(vocab, idx))
            self.idx2tok = dict(zip(idx, vocab))
            with open(tok2idx_path, 'wb') as tok2idx_file:
                pickle.dump(self.tok2idx, tok2idx_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(idx2tok_path, 'wb') as idx2tok_file:
                pickle.dump(self.idx2tok, idx2tok_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(freq_path, 'wb') as freq_file:
                pickle.dump(self.freq, freq_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(enlu_path, 'wb') as enlu_file:
                pickle.dump(self.enlu, enlu_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading vocab")
            with open(tok2idx_path, 'rb') as tok2idx_file:
                self.tok2idx = pickle.load(tok2idx_file)
            with open(idx2tok_path, 'rb') as idx2tok_file:
                self.idx2tok = pickle.load(idx2tok_file)
            with open(freq_path, 'rb') as freq_file:
                self.freq = pickle.load(freq_file)
            with open(enlu_path, 'rb') as enlu_file:
                self.enlu = pickle.load(enlu_file)
            self.ids = list(self.enlu.keys())
        
    def __getitem__(self, index):
        ann_id = self.ids[index]
        #caption = str(self.coco.anns[ann_id]['caption']).lower()
        caption = self.enlu[ann_id]
        #img_id = self.coco.anns[ann_id]['image_id']
        img_path = Path(self.img)/'img_clean'/ann_id
        
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        return image.float(), torch.tensor(self.caption2tokens(caption)).long()

    def __len__(self):
        return len(self.ids)
        
    def caption2tokens(self, caption):
        tags = Counter()
        for token in caption:
            if token in self.freq:
                tags[token] = self.freq[token]
        tags = [self.tok2idx[e[0]] for e in tags.most_common(self.max_cap_len)]
        tags.extend([0] * (self.max_cap_len - len(tags)))
        return np.array(tags)
        
    def tokens2captions(self, idx):
        captions = []
        for i in idx:
            if i > 0:
                captions.append(self.idx2tok[i])
        return " ".join(captions)
        
class PixivFacesDataset(Dataset):
    def __init__(self, use_pickle=False):
        self.path = "CLEAN_PIXIV"
        self.use_pickle = use_pickle
        self.max_cap_len = 80
        self.imgs = []
        self.tags = []
        if self.use_pickle:
            self.embs = []
        for f in os.listdir(self.path):
            f_lower = f.lower()
            if f_lower.endswith(".jpg"):
                self.imgs.append(f)
                self.tags.append(".".join(f.split(".")[:-1]) + ".txt")
                if self.use_pickle:
                    self.embs.append(".".join(f.split(".")[:-1]) + ".pkl")
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
        if self.use_pickle:
            with open(os.path.join(self.path, self.embs[idx]), 'rb') as pkl:
                return torch.tensor(pickle.load(pkl)), tags
        else:
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
    t = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    t = transforms.Compose(t)

    train_set = get_pixiv_captions_2021_train("PIXIV", t, DALLE_TEXT_SEQ_LEN)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    img_batch, tag_batch = next(iter(train_loader))
    print()
    print("Test COCO tags dataset and dataloader")
    print("Dataset has length", len(train_set))
    print("Image batch shape")
    print(img_batch.shape)
    print()
    print("Tag batch with shape", tag_batch.shape)
    print(tag_batch)
    print()
    print('\n'.join([train_set.tokens2captions(t.masked_select(t != 0).tolist()) for t in tag_batch]))

    # train_set = get_rainbow_captions_2014_train("rainbow", t, DALLE_TEXT_SEQ_LEN)
    # train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    # img_batch, tag_batch = next(iter(train_loader))
    # print()
    # print("Test COCO tags dataset and dataloader")
    # print("Dataset has length", len(train_set))
    # print("Image batch shape")
    # print(img_batch.shape)
    # print()
    # print("Tag batch with shape", tag_batch.shape)
    # print(tag_batch)
    # print()
    

    # pixiv_image_set = ImageDataset(os.path.join("Pixiv", "img_clean"), t)
    # pixiv_image_loader = DataLoader(pixiv_image_set, batch_size=8, shuffle=True)
    # img_batch = next(iter(pixiv_image_loader))
    # print()
    # print("Test Pixiv image set and dataloader")
    # print("Dataset has length", len(pixiv_image_set))
    # print("Image batch shape")
    # print(img_batch.shape)a
    # print()
    
    # pixiv_image_subset = ImageSubDataset(os.path.join("Pixiv", "img_clean"), os.path.join("datasubsets", "Pixiv_img_subset_10000.pkl"), t)
    # pixiv_image_subset_loader = DataLoader(pixiv_image_subset, batch_size=8, shuffle=True)
    # img_batch = next(iter(pixiv_image_subset_loader))
    # print()
    # print("Test Pixiv image subset and dataloader")
    # print("Dataset has length", len(pixiv_image_subset))
    # print("Image batch shape")
    # print(img_batch.shape)
    # print()
    
    # coco_image_set = get_COCO_images_2014_train("MS-COCO", t)
    # coco_image_loader = DataLoader(coco_image_set, batch_size=8, shuffle=True)
    # img_batch = next(iter(coco_image_loader))
    # print()
    # print("Test COCO image set and dataloader")
    # print("Dataset has length", len(coco_image_set))
    # print("Image batch shape")
    # print(img_batch.shape)
    # print()
    
    # coco_image_subset = ImageSubDataset(os.path.join("MS-COCO", "train2014"), os.path.join("datasubsets", "COCO_img_subset_10000.pkl"), t)
    # coco_image_subset_loader = DataLoader(coco_image_subset, batch_size=8, shuffle=True)
    # img_batch = next(iter(coco_image_subset_loader))
    # print()
    # print("Test COCO image subset and dataloader")
    # print("Dataset has length", len(coco_image_subset))
    # print("Image batch shape")
    # print(img_batch.shape)
    # print()

    # train_set = get_COCO_captions_2014_train("MS-COCO", t, DALLE_TEXT_SEQ_LEN)
    # train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    # img_batch, tag_batch = next(iter(train_loader))
    # print()
    # print("Test COCO tags dataset and dataloader")
    # print("Dataset has length", len(train_set))
    # print("Image batch shape")
    # print(img_batch.shape)
    # print()
    # print("Tag batch with shape", tag_batch.shape)
    # print(tag_batch)
    # print()
    
    # train_set = CocoSubDataset(
    #     os.path.join("MS-COCO", "subsets", "4800", "COCO_img_cap_subset_4800_imgtensor.pkl"),
    #     os.path.join("MS-COCO", "subsets", "4800", "COCO_img_cap_subset_4800_tagtensor.pkl"),
    #     os.path.join("MS-COCO", "dictionary"))
    # train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    # img_batch, tag_batch = next(iter(train_loader))
    # print()
    # print("Test COCO tags sub dataset and dataloader")
    # print("Dataset has length", len(train_set))
    # print("Image batch shape")
    # print(img_batch.shape)
    # print()
    # print("Tag batch with shape", tag_batch.shape)
    # print(tag_batch)
    # print()

    tags = input("Your custom set of tags here: ")
    idxs = train_set.caption2tokens(tags)
    print("Indices:", idxs)
    tags = train_set.tokens2captions(idxs)
    print("Tags:", tags)
