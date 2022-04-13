import argparse
from PIL import Image
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt

import torch

from einops import repeat

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms

from tqdm import tqdm

import sys
sys.path.insert(0,'c:/Users/thapp/work/dalle_pytorch')
sys.path.insert(0,'c:/Users/thapp/work/taming-transformers')
from dalle_pytorch import VQGanVAE, DALLE, CLIP

from dataset import *

torch.manual_seed(125)
BATCH_SIZE = 64

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dalle_ckpt_path', type=str, default='checkpoints/202203170118/dalle_temp.ckpt')
    parser.add_argument('--clip_ckpt_path', type=str, default='checkpoints_clip/202203191420/clip_temp.ckpt')
    parser.add_argument('--dvae_ckpt_path', type = str, default = 'checkpoints/2022-03-07T17-21-57_custom_vqgan/testtube/version_0/checkpoints/epoch=715-step=579200.ckpt')
    parser.add_argument('--dvae_ckpt_conf', type = str, default = 'checkpoints/2022-03-07T17-21-57_custom_vqgan/configs/2022-03-07T17-21-57-project.yaml')
    parser.add_argument('--text', type = str, default = '')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using cpu")
        device = torch.device("cpu")

    vae = VQGanVAE(args.dvae_ckpt_path,args.dvae_ckpt_conf)
    
    dalle = DALLE(
        vae = vae,                  
        dim = CODEBOOK_DIM,
        num_text_tokens = VOCAB_SIZE,    
        text_seq_len = DALLE_TEXT_SEQ_LEN,         
        depth = DALLE_DEPTH,                 
        heads = DALLE_HEADS,                 
        dim_head = DALLE_DIM_HEAD,             
        attn_dropout = 0.0,         
        ff_dropout = 0.0,
        loss_img_weight = 7,
        attn_types = ('full','full'),
        shift_tokens = False,
        rotary_emb = False         
    )
    dalle.load_state_dict(torch.load(args.dalle_ckpt_path, map_location='cpu'))
    dalle = dalle.to(device)

    clip = CLIP(        
        dim_text = CODEBOOK_DIM,
        dim_image = CODEBOOK_DIM,
        dim_latent = CLIP_DIM_LATENT,
        num_text_tokens = VOCAB_SIZE,
        text_enc_depth = DALLE_DEPTH,
        text_seq_len = DALLE_TEXT_SEQ_LEN,
        text_heads = DALLE_HEADS,
        num_visual_tokens = DVAE_HIDDEN_DIM,
        visual_enc_depth = CLIP_VISUAL_ENC_DEPTH,
        visual_image_size = IMAGE_SIZE,
        visual_patch_size = CLIP_PATCH_SIZE,
        visual_heads = CLIP_HEADS
        )
    clip.load_state_dict(torch.load(args.clip_ckpt_path, map_location='cpu'))
    clip = clip.to(device)

    transform = [
                 transforms.ToTensor(),
                ]
    transform = transforms.Compose(transform)

    train_set = PixivFacesDataset(DATASET_ROOT, DALLE_TEXT_SEQ_LEN, transform)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, drop_last=True)

    out_dir='synthesize_images'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i, (imgs, tags) in enumerate(tqdm(train_loader)):
        img_batch, token_batch = (imgs, tags)

        token_batch=token_batch[:,:56].cuda()
        token_batch, gen_texts = dalle.generate_texts(None, text=token_batch, filter_thres = 0.9)
        token_batch[token_batch>=len(train_set.idx2tok)]=0

        token_batch = repeat(token_batch, '() n -> b n', b = BATCH_SIZE)
        img_batch, token_batch = map(lambda t: t.cuda(), (img_batch, token_batch))

        text_batch_decode=[]
        for token in token_batch:
            token_list = token.masked_select(token != 0).tolist()
            decoded_text = train_set.tokens2captions(token_list)
            text_batch_decode.append(decoded_text)
            break
        
        with torch.no_grad():
            img_batch_decode = dalle.generate_images(token_batch, filter_thres = 0.90, clip=clip)

            images, recons = map(lambda t: t.detach().cpu(), (img_batch, img_batch_decode[0]))
            #print(img_batch_decode[1])
            rank_top=torch.argsort(img_batch_decode[1], descending=True)[:4]
            #rank_bot=torch.argsort(torch.abs(img_batch_decode[1]), descending=True)[:4]
            #recons=recons[[*rank_top,*rank_bot],...]
            #recons=recons[rank_top,...]
            for j in rank_top:
                save_image(recons[j], Path(out_dir)/f'{i}_{j.item()}.png')
            #images = make_grid(images.float(), nrow = 1, normalize = True, value_range = (0, 1))
            #recons = make_grid(recons.float(), nrow = 4, normalize = True, value_range = (0, 1))

            #f, ax = plt.subplots(2,1,figsize=(8,4))
            #f.suptitle(text_batch_decode[0], y=1.0, fontsize=8)
            #ax[0].axis('off')
            #ax[0].imshow(images.permute(1,2,0))
            #ax[1].axis('off')
            #ax[1].imshow(recons.permute(1,2,0))
            #plt.show()
