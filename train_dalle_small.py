import argparse

import torch
import math

from datetime import datetime
from statistics import mean

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

import sys

from dalle_pytorch import DiscreteVAE, DALLE

from tqdm import tqdm

from dataset import *
from configs import *


if __name__ == '__main__':

    COCO_ROOT = "rainbow"
    SAVE_PATH = "ckpts"

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--dvae_ckpt_path', type = str, default='ckpts/202111141726/dVAE_64.ckpt')
    parser.add_argument('--epochs', type = int, default = 64)
    parser.add_argument('--save_step', type = int, default = 1)
    parser.add_argument('--dalle_ckpt_path', type = str)
    parser.add_argument('--dalle_ckpt_epoch', type = int)
    parser.add_argument('--img_subset', type = str)
    parser.add_argument('--cap_subset', type = str)

    train_group = parser.add_argument_group('Training Settings')
    train_group.add_argument('--learning_rate', type = float, default = 1e-3, help = 'learning rate')
    train_group.add_argument('--num_images_save', type = int, default = 4, help = 'number of images to save')
    train_group.add_argument('--batch_size', type = int, default = 16, help = 'batch_size')

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
    
    args = parser.parse_args()

    NUM_EPOCHS = args.epochs
    SAVE_STEP = args.save_step
    LEARNING_RATE = args.learning_rate
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

    transform = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    transform = transforms.Compose(transform)
    
    if args.img_subset is not None and args.cap_subset is not None:
        train_set = CocoSubDataset(args.img_subset, args.cap_subset)
    else:
        train_set = get_rainbow_captions_2014_train(COCO_ROOT, transform, DALLE_TEXT_SEQ_LEN)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using cpu")
        device = torch.device("cpu")

    vae = DiscreteVAE(
        image_size = IMAGE_SIZE,
        num_layers = num_layers,
        num_tokens = num_image_tokens,
        codebook_dim = emb_dim,
        hidden_dim   = hidden_dim,
        num_resnet_blocks = num_resnet_blocks
    )

    if not os.path.isfile(args.dvae_ckpt_path):
        print(args.dvae_ckpt_path, "is not a file or doesn't exist!")
        quit()
    vae.load_state_dict(torch.load(args.dvae_ckpt_path, map_location=torch.device(device)))

    vae = vae.to(device)

    dalle = DALLE(
        dim = emb_dim,
        vae = vae,                  
        num_text_tokens = VOCAB_SIZE,    
        text_seq_len = DALLE_TEXT_SEQ_LEN,
        depth = dalle_depth,
        heads = dalle_heads,
        dim_head = dalle_dim_head,
        attn_dropout = dalle_attn_dropout,
        ff_dropout = dalle_ff_dropout
    )
    
    dalle_params = {
        "dim": emb_dim,
        "vae": vae,
        "num_text_tokens": VOCAB_SIZE,
        "text_seq_len": DALLE_TEXT_SEQ_LEN,
        "depth": dalle_depth,
        "heads": dalle_heads,
        "dim_head": dalle_dim_head,
        "attn_dropout": dalle_attn_dropout,
        "ff_dropout": dalle_ff_dropout
    }
    dalle_params_str = ""
    for k in dalle_params:
        dalle_params_str += k + ": " + str(dalle_params[k]) + "\n"
    print("DALL-E params:")
    print(dalle_params_str)

    if args.dalle_ckpt_path is not None and args.dalle_ckpt_epoch is not None:
        if not os.path.isfile(args.dalle_ckpt_path):
            print(args.dalle_ckpt_path, "is not a file or doesn't exist!")
            quit()
        ckpts_dir = os.path.dirname(args.dalle_ckpt_path)
        START_EPOCH = args.dalle_ckpt_epoch
        dalle.load_state_dict(torch.load(args.dalle_ckpt_path, map_location=torch.device(device)))
    else:
        ckpts_dir = os.path.join(SAVE_PATH, datetime.now().strftime("%Y%m%d%H%M"))
        if not os.path.exists(ckpts_dir):
            os.makedirs(ckpts_dir)
        with open(os.path.join(ckpts_dir, "config.txt"), "w") as f:
            f.write(dalle_params_str)
        START_EPOCH = 0

    dalle = dalle.to(device)

    opt = Adam(dalle.parameters(), lr = LEARNING_RATE)
    sched = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=10,
        cooldown=10,
        min_lr=1e-6,
        verbose=True,
    )

    global_step = 0
    writer = SummaryWriter()

    for epoch in range(START_EPOCH, NUM_EPOCHS):

        epoch_losses = []

        for i, (imgs, tags) in enumerate(tqdm(train_loader)):
            tags, imgs = map(lambda t: t.to(device), (tags, imgs))
            loss = dalle(tags, imgs, return_loss=True)
            loss.backward()
            clip_grad_norm_(dalle.parameters(), 0.5)
            epoch_losses.append(loss.item())
            opt.step()
            opt.zero_grad()
            
            if i % 10 == 9:
                writer.add_scalar("Loss/train", epoch_losses[-1], global_step)
                writer.add_scalar("lr", opt.param_groups[0]['lr'], global_step)

                if i % 100 == 99: #quick fix for tensorboard not updating images?
                    sched.step(torch.mean(loss))
                    if i % 5000 == 4999:
                        k = NUM_IMAGES_SAVE
                        try:
                            sample_text = tags[:1]
                            token_list = sample_text.masked_select(sample_text != 0).tolist()
                            decoded_text = train_set.tokens2captions(token_list)
                            images = dalle.generate_images(tags[:1], filter_thres=0.9)  # topk sampling at 0.9

                            images = images[:k].detach().cpu()
                            images = make_grid(images.float(), nrow = int(math.sqrt(k)), normalize = True, range = (-1,1))

                            writer.add_image('original images', images, global_step)
                            writer.add_text('codebook_indices',decoded_text, global_step)
                            writer.flush()
                        except RuntimeError:
                            pass
            global_step += 1
                    
        print("Done epoch", epoch + 1, "with mean loss", mean(epoch_losses))
        
        if (epoch + 1) % SAVE_STEP == 0:
            ckpt_name = "dalle_" + str(epoch+1) + ".ckpt"
        else:
            ckpt_name = "dalle_last.ckpt"
        print("Saving checkpoint", ckpt_name, "to", ckpts_dir)
        torch.save(dalle.state_dict(), os.path.join(ckpts_dir, ckpt_name), _use_new_zipfile_serialization=False)

