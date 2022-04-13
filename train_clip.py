import argparse
from collections import OrderedDict

import torch
import math

from datetime import datetime
from statistics import mean

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid
from torch.cuda.amp import autocast,GradScaler

import sys

sys.path.insert(0,'c:/Users/thapp/work/dalle_pytorch')
sys.path.insert(0,'c:/Users/thapp/work/taming-transformers')
from dalle_pytorch import VQGanVAE, DALLE, CLIP

from tqdm import tqdm

from dataset import *
from configs import *

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--epochs', type = int, default = 500)
    parser.add_argument('--save_step', type = int, default = 5)
    parser.add_argument('--clip_ckpt_path', type = str, default = None)
    parser.add_argument('--clip_ckpt_epoch', type = int, default=0)

    train_group = parser.add_argument_group('Training Settings')
    train_group.add_argument('--learning_rate', type = float, default = 1e-4, help = 'learning rate')
    train_group.add_argument('--lr_decay_rate', type = float, default = 0.5, help = 'learning rate decay')
    train_group.add_argument('--lr_min', type = float, default = 1e-8, help = 'learning rate min')
    train_group.add_argument('--num_images_save', type = int, default = 4, help = 'number of images to save')
    train_group.add_argument('--batch_size', type = int, default = 64, help = 'batch_size')

    dvae_group = parser.add_argument_group('dVAE Settings')
    dvae_group.add_argument('--num_image_tokens', type = int, default = DVAE_NUM_TOKENS, help = 'number of image tokens')
    dvae_group.add_argument('--num_layers', type = int, default = DVAE_NUM_LAYERS, help = 'number of layers (should be 3 or above)')
    dvae_group.add_argument('--num_resnet_blocks', type = int, default = DVAE_NUM_RESNET_BLOCKS, help = 'number of residual net blocks')
    dvae_group.add_argument('--emb_dim', type = int, default = CODEBOOK_DIM, help = 'embedding dimension')
    dvae_group.add_argument('--hidden_dim', type = int, default = DVAE_HIDDEN_DIM, help = 'hidden dimension')

    dalle_group = parser.add_argument_group('DALL-E Settings')
    dalle_group.add_argument('--dalle_depth', type = int, default = DALLE_DEPTH, help = 'should aim to be 64')
    dalle_group.add_argument('--dalle_heads', type = int, default = DALLE_HEADS, help = 'attention heads')
    dalle_group.add_argument('--dalle_dim_head', type = int, default = DALLE_DIM_HEAD, help = 'attention head dimension')
    dalle_group.add_argument('--dalle_attn_dropout', type = float, default = DALLE_ATTN_DROPOUT, help = 'attention dropout')
    dalle_group.add_argument('--dalle_ff_dropout', type = float, default = DALLE_FF_DROPOUT, help = 'feedforward dropout')
    dalle_group.add_argument('--dalle_loss_img_weight', type = float, default = 7, help = 'image weight scaling')
    
    args = parser.parse_args()

    NUM_EPOCHS = args.epochs
    SAVE_STEP = args.save_step
    LEARNING_RATE = args.learning_rate
    LR_DECAY_RATE = args.lr_decay_rate
    LR_MIN = args.lr_min
    BATCH_SIZE = args.batch_size

    NUM_IMAGES_SAVE = args.num_images_save

    num_layers = args.num_layers
    num_image_tokens = args.num_image_tokens
    emb_dim = args.emb_dim
    hidden_dim = args.hidden_dim
    num_resnet_blocks = args.num_resnet_blocks
    
    dalle_depth = args.dalle_depth
    dalle_heads = args.dalle_heads
    dalle_dim_head = args.dalle_dim_head
    dalle_attn_dropout = args.dalle_attn_dropout
    dalle_ff_dropout = args.dalle_ff_dropout
    dalle_loss_img_weight = args.dalle_loss_img_weight
    transform = [
                 transforms.ToTensor(),
                ]
    transform = transforms.Compose(transform)
    
    train_set = PixivFacesDataset(DATASET_ROOT, DALLE_TEXT_SEQ_LEN, transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, persistent_workers=True, num_workers=8)

    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using cpu")
        device = torch.device("cpu")

    clip_params = {
        "dim_text" : emb_dim,
        "dim_image" : emb_dim,
        "dim_latent" : CLIP_DIM_LATENT,
        "num_text_tokens" : VOCAB_SIZE,
        "text_enc_depth" : dalle_depth,
        "text_seq_len" : DALLE_TEXT_SEQ_LEN,
        "text_heads" : dalle_heads,
        "num_visual_tokens" : DVAE_HIDDEN_DIM,
        "visual_enc_depth" : CLIP_VISUAL_ENC_DEPTH,
        "visual_image_size" : IMAGE_SIZE,
        "visual_patch_size" : CLIP_PATCH_SIZE,
        "visual_heads" : CLIP_HEADS
    }

    clip = CLIP(**clip_params)

    clip_params_str = ""
    for k in clip_params:
        clip_params_str += k + ": " + str(clip_params[k]) + "\n"
    print("CLIP params:")
    print(clip_params_str)

    ckpts_dir = os.path.join(SAVE_PATH, datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(ckpts_dir,exist_ok=True)
    if args.clip_ckpt_path is not None and args.clip_ckpt_epoch is not None:
        if not os.path.isfile(args.clip_ckpt_path):
            print(args.clip_ckpt_path, "is not a file or doesn't exist!")
            quit()
        START_EPOCH = args.clip_ckpt_epoch
        clip_state=torch.load(args.clip_ckpt_path)
        clip.load_state_dict(clip_state,strict=False)
    else:
        with open(os.path.join(ckpts_dir, "config.txt"), "w") as f:
            f.write(clip_params_str)
        START_EPOCH = 0

    clip = clip.to(device)

    #opt = AdamW(group_weight(dalle), lr = LEARNING_RATE, betas = (0.9, 0.96), eps = 1e-08, weight_decay = 4.5e-2, amsgrad = False)
    opt = Adam(get_trainable_params(clip), lr = LEARNING_RATE)

    sched = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=LR_DECAY_RATE,
        patience=50,
        cooldown=50,
        min_lr=LR_MIN,
        verbose=True,
    )

    global_step = 0
    grad_acc = 10
    writer = SummaryWriter()

    for epoch in range(START_EPOCH, NUM_EPOCHS):

        epoch_loss = 0
        accum_loss = 0
        step_loss = 0
        num_steps_accum = 0
        num_steps_sched = 0
        #opt.zero_grad()

        for i, (imgs, tags) in enumerate(tqdm(train_loader)):
            tags, imgs = map(lambda t: t.to(device), (tags, imgs))
            loss = clip(tags, imgs, return_loss=True)
            (loss/grad_acc).backward()
            opt.step()
            opt.zero_grad()
            
            epoch_loss += loss.item()    
            accum_loss += loss.item()   
            step_loss += loss.item()
            num_steps_accum += 1
            num_steps_sched += 1
            
            if (i+1) % grad_acc == 0:
                writer.add_scalar("Avg Accmulated Loss/train", accum_loss/num_steps_accum, global_step)
                writer.add_scalar("lr", opt.param_groups[0]['lr'], global_step)
                clip_grad_norm_(clip.parameters(), 0.5)
                #accum_loss.backward(), does not save vram?

                accum_loss = 0
                num_steps_accum = 0

                if (i+1) % 100 == 0:
                    sched.step(step_loss/num_steps_sched)
                    step_loss = 0
                    num_steps_sched = 0
                    if (i+1) % 200 == 0:
                        k = NUM_IMAGES_SAVE
                        try:
                            with torch.no_grad():
                                print("Saving checkpoint", 'clip_temp.ckpt', "to", ckpts_dir)
                                torch.save(clip.state_dict(), os.path.join(ckpts_dir, 'clip_temp.ckpt'), _use_new_zipfile_serialization=False)
                        except RuntimeError:
                            print("Failed to save sample to tensorboard!")
                            pass
            global_step += 1

        #opt.step() throw away last graph, it was erroring out
        opt.zero_grad()

        print("Done epoch", epoch + 1, "with mean epoch loss", epoch_loss/len(train_loader))
        
        #if (epoch + 1) % SAVE_STEP == 0:
        #    ckpt_name = "dalle_" + str(epoch+1) + ".ckpt"
        #else:
        #    ckpt_name = "dalle_last.ckpt"
        #print("Saving checkpoint", ckpt_name, "to", ckpts_dir)
        #torch.save(dalle.state_dict(), os.path.join(ckpts_dir, ckpt_name), _use_new_zipfile_serialization=False)

