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
    #parser.add_argument('--load', type=Path, default=['dalle_params.pt','dalle_weights.pt','vae_params.pt','vae_weights.pt'], nargs='+')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using cpu")
        device = torch.device("cpu")

    print("Using actual images and dVAE")
    #vae_params = torch.load("model/vae_params.pt")
    vae = DiscreteVAE(
        image_size = IMAGE_SIZE,
        num_layers = DVAE_NUM_LAYERS,
        num_tokens = DVAE_NUM_TOKENS,
        codebook_dim = CODEBOOK_DIM,
        hidden_dim = DVAE_HIDDEN_DIM,
        num_resnet_blocks = DVAE_NUM_RESNET_BLOCKS,
    ).to(device)
    vae_weights = torch.load("model/vae_weights.pt")
    vae.load_state_dict(vae_weights)
    vae = vae.to(device)
    #dalle_params = torch.load("model/dalle_params.pt")
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
    dalle_weights = torch.load("model/dalle_weights.pt")
    dalle.load_state_dict(dalle_weights)
    dalle = dalle.to(device)

    #transform = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    #transform = transforms.Compose(transform)

    train_set = TextImageDataset('CLEAN_PIXIV',truncate_captions=True,tokenizer=tokenizer,text_len=80)
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

        #img_batch_decode -= img_batch_decode.min(1, keepdim=True)[0]
        #img_batch_decode /= img_batch_decode.max(1, keepdim=True)[0]
        matplotlib.use('qt5agg')
        f, ax = plt.subplots(1,2,figsize=(8,8))
        f.suptitle('\n'.join(text_batch_decode), fontname="MS Gothic")
        ax[0].axis('off')
        ax[0].imshow(images.permute(1,2,0))
        ax[1].axis('off')
        ax[1].imshow(recons.permute(1,2,0))

        plt.show()

