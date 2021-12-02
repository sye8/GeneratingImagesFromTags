import argparse
from PIL import Image
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt

import torch

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms

from tqdm import tqdm

from dalle_pytorch import DiscreteVAE, DALLE
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch.tokenizer import tokenizer

from dataset import *


BATCH_SIZE = 4

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('dvae_ckpt_path', type=str)
    parser.add_argument('dalle_ckpt_path', type=str)
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using cpu")
        device = torch.device("cpu")

    vae = DiscreteVAE(
        image_size = IMAGE_SIZE,
        num_layers = DVAE_NUM_LAYERS,
        num_tokens = DVAE_NUM_TOKENS,
        codebook_dim = CODEBOOK_DIM,
        hidden_dim = DVAE_HIDDEN_DIM,
        num_resnet_blocks = DVAE_NUM_RESNET_BLOCKS,
    ).to(device)
    vae.load_state_dict(torch.load(args.dvae_ckpt_path, map_location=torch.device(device)))
    vae = vae.to(device)
    dalle = DALLE(
        vae = vae,                  
        dim = CODEBOOK_DIM,
        num_text_tokens = VOCAB_SIZE,    
        text_seq_len = DALLE_TEXT_SEQ_LEN,         
        depth = DALLE_DEPTH,                 
        heads = DALLE_HEADS,                 
        dim_head = DALLE_DIM_HEAD,             
        attn_dropout = 0.1,         
        ff_dropout = 0.1,
        loss_img_weight = 7,
        attn_types = ('conv_like','full'),
        shift_tokens = False,
        rotary_emb = False         
    ).to(device)
    dalle.load_state_dict(torch.load(args.dalle_ckpt_path, map_location=torch.device(device)))
    dalle = dalle.to(device)

    train_set = PixivFacesDataset(DATASET_ROOT, DALLE_TEXT_SEQ_LEN)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    token_batch,img_batch = next(iter(train_loader))
    token_batch,img_batch = map(lambda t: t.cuda(), (token_batch,img_batch))

    with torch.no_grad():
        img_batch_decode = dalle.generate_images(token_batch, filter_thres = 0.90)

        text_batch_decode=[]
        for token in token_batch:
            token_list = token.masked_select(token != 0).tolist()
            decoded_text = tokenizer.decode(token_list)
            text_batch_decode.append(decoded_text)

        images, recons = map(lambda t: t.detach().cpu(), (img_batch, img_batch_decode))
        images = make_grid(images.float(), nrow = 1, normalize = True, value_range = (0, 1))
        recons = make_grid(recons.float(), nrow = 1, normalize = True, value_range = (-1, 1))

        f, ax = plt.subplots(1,2,figsize=(8,8))
        f.suptitle('\n'.join(text_batch_decode), fontname="MS Gothic")
        ax[0].axis('off')
        ax[0].imshow(images.permute(1,2,0))
        ax[1].axis('off')
        ax[1].imshow(recons.permute(1,2,0))
        plt.show()
