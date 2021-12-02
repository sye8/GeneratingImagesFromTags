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

from dataset import *
from configs import *


BATCH_SIZE = 25
NUM_ROWS = 5
COCO_ROOT = "MS-COCO"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('--img_subset', type = str)
    parser.add_argument('--cap_subset', type = str)
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
        num_resnet_blocks = DVAE_NUM_RESNET_BLOCKS
    ).to(device)

    transform = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    transform = transforms.Compose(transform)

    if args.img_subset is not None and args.cap_subset is not None:
        train_set = CocoSubDataset(args.img_subset, args.cap_subset, os.path.join(COCO_ROOT, "dictionary"))
    else:
        train_set = get_COCO_captions_2014_train(COCO_ROOT, transform, DALLE_TEXT_SEQ_LEN)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


    dalle = DALLE(
        dim = CODEBOOK_DIM,
        vae = vae,                  
        num_text_tokens = VOCAB_SIZE,
        text_seq_len = DALLE_TEXT_SEQ_LEN,
        depth = DALLE_DEPTH,                 
        heads = DALLE_HEADS,                 
        dim_head = DALLE_DIM_HEAD,             
        attn_dropout = DALLE_ATTN_DROPOUT,         
        ff_dropout = DALLE_FF_DROPOUT            
    ).to(device)

    if not os.path.isfile(args.ckpt_path):
        print(args.ckpt_path, "is not a file or doesn't exist!")
        quit()
    dalle.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device(device)))

    img_batch,token_batch = next(iter(train_loader))
    img_batch,token_batch = map(lambda t: t.to(device), (img_batch,token_batch))

    with torch.no_grad():
        
        img_batch_decode = dalle.generate_images(token_batch, filter_thres = 0.9)

        images, recons = map(lambda t: t.detach().cpu(), (img_batch, img_batch_decode))
        images, recons = map(lambda t: make_grid(t[1].float(), nrow = NUM_ROWS, normalize = True, value_range = tuple(np.array([[0, 1],[-1,1]])[t[0],:])), enumerate([images, recons]))

        f, ax = plt.subplots(1,2,figsize=(64,32))
        ax[0].axis('off')
        ax[0].imshow(images.permute(1,2,0))
        ax[1].axis('off')
        ax[1].imshow(recons.permute(1,2,0))
        plt.tight_layout()
        plt.savefig("figure.png")
