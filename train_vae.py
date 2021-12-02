import argparse
from collections import OrderedDict

import torch

import math

import os

import numpy as np
from datetime import datetime

from torch.utils.data import DataLoader

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast,GradScaler

from dalle_pytorch import DiscreteVAE

from statistics import mean
from tqdm import tqdm

from dataset import *
from configs import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--epochs', type = int, default = 64)
    parser.add_argument('--save_step', type = int, default = 4)
    parser.add_argument('--ckpt_path', type = str)
    parser.add_argument('--ckpt_epoch', type = int)

    train_group = parser.add_argument_group('Training settings')
    train_group.add_argument('--learning_rate', type = float, default = 1e-4, help = 'learning rate')
    train_group.add_argument('--lr_decay_rate', type = float, default = 0.9, help = 'learning rate decay')
    train_group.add_argument('--lr_min', type = float, default = 1.25e-6, help = 'learning rate min')
    train_group.add_argument('--starting_temp', type = float, default = 0.5, help = 'starting temperature')
    train_group.add_argument('--temp_min', type = float, default = 1/16, help = 'minimum temperature to anneal to')
    train_group.add_argument('--temp_anneal_rate', type = float, default = 1e-4, help = 'temperature annealing rate')
    train_group.add_argument('--starting_beta', type = float, default = 0, help = 'starting beta')
    train_group.add_argument('--beta_max', type = float, default = 0, help = 'maximum beta to anneal to')
    train_group.add_argument('--beta_anneal_rate', type = float, default = 1e-6, help = 'beta annealing rate')
    train_group.add_argument('--num_images_save', type = int, default = 4, help = 'number of images to save to tensorboard log')
    train_group.add_argument('--batch_size', type = int, default = 64, help = 'batch_size')
    train_group.add_argument('--use_adamw', action = 'store_true', help = 'use open-ai adamW for schduler')

    model_group = parser.add_argument_group('Model settings')
    model_group.add_argument('--num_tokens', type = int, default = DVAE_NUM_TOKENS, help = 'number of image tokens')
    model_group.add_argument('--num_layers', type = int, default = DVAE_NUM_LAYERS, help = 'number of layers (should be 3 or above)')
    model_group.add_argument('--num_resnet_blocks', type = int, default = DVAE_NUM_RESNET_BLOCKS, help = 'number of residual net blocks')
    model_group.add_argument('--emb_dim', type = int, default = CODEBOOK_DIM, help = 'embedding dimension')
    model_group.add_argument('--hidden_dim', type = int, default = DVAE_HIDDEN_DIM, help = 'hidden dimension')
    
    args = parser.parse_args()

    NUM_EPOCHS = args.epochs
    SAVE_STEP = args.save_step
    LEARNING_RATE = args.learning_rate
    LR_DECAY_RATE = args.lr_decay_rate
    LR_MIN = args.lr_min
    BATCH_SIZE = args.batch_size

    STARTING_TEMP = args.starting_temp
    TEMP_MIN = args.temp_min
    TEMP_ANNEAL_RATE = args.temp_anneal_rate  
 
    STARTING_BETA = args.starting_beta
    BETA_MAX = args.beta_max
    BETA_ANNEAL_RATE = args.beta_anneal_rate  

    num_layers = args.num_layers
    num_tokens = args.num_tokens
    emb_dim = args.emb_dim
    hidden_dim = args.hidden_dim
    num_resnet_blocks = args.num_resnet_blocks

    NUM_IMAGES_SAVE = args.num_images_save

    print("Num epochs:", NUM_EPOCHS)
    print("Save step:", SAVE_STEP)

    transform = [transforms.RandomResizedCrop(IMAGE_SIZE,scale=(1,1)),
                 transforms.ToTensor(),
                 transforms.ColorJitter(brightness=0.025, contrast=0.025, saturation=0.2, hue=0),
                 ]
    transform = transforms.Compose(transform)

    train_set = ImageDataset(DATASET_ROOT, transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)

    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using cpu")
        device = torch.device("cpu")

    vae = DiscreteVAE(
        image_size = IMAGE_SIZE,
        num_layers = num_layers,
        num_tokens = num_tokens,
        codebook_dim = emb_dim,
        hidden_dim   = hidden_dim,
        num_resnet_blocks = num_resnet_blocks,
        smooth_l1_loss = True,
        kl_div_loss_weight = 0,
    )

    vae_params = {
        "num_layers": num_layers,
        "num_tokens": num_tokens,
        "codebook_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "num_resnet_blocks": num_resnet_blocks,
        "kl_div_weight": BETA_MAX
    }
    vae_params_str = ""
    for k in vae_params:
        vae_params_str += k + ": " + str(vae_params[k]) + "\n"
    print("dVAE params:")
    print(vae_params_str)

    if args.ckpt_path is not None and args.ckpt_epoch is not None:
        if not os.path.isfile(args.ckpt_path):
            print(args.ckpt_path, "is not a file or doesn't exist!")
            quit()
        ckpts_dir = os.path.dirname(args.ckpt_path)
        START_EPOCH = args.ckpt_epoch
        dvae_state=torch.load(args.ckpt_path, map_location=torch.device(device))
        vae.load_state_dict(dvae_state, strict=False)
    else:
        ckpts_dir = os.path.join(SAVE_PATH, datetime.now().strftime("%Y%m%d%H%M"))
        if not os.path.exists(ckpts_dir):
            os.makedirs(ckpts_dir)
        with open(os.path.join(ckpts_dir, "config.txt"), "w") as f:
            f.write(vae_params_str)
        START_EPOCH = 0

    vae = vae.to(device)

    if not args.use_adam:
        opt = AdamW(vae.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    else:
        opt = Adam(vae.parameters(), lr = LEARNING_RATE)

    sched = ReduceLROnPlateau(
         opt,
         mode="min",
         factor=LR_DECAY_RATE,
         patience=5,
         cooldown=5,
         min_lr=LR_MIN,
         verbose=True,
     )

    global_step = 0

    temp = STARTING_TEMP

    writer = SummaryWriter()
    scaler = GradScaler()

    for epoch in range(START_EPOCH, NUM_EPOCHS):

        epoch_losses = []
        step_losses = []

        for i,images in enumerate(tqdm(train_loader)):
            images = images.to(device)
            opt.zero_grad()
            with autocast():
                loss, recons = vae(images, return_loss = True, return_recons = True)
            scaler.scale(loss).backward()
            epoch_losses.append(loss.item())
            step_losses.append(loss.item())
            scaler.step(opt)
            scaler.update()

            if i % 10 == 0:
                writer.add_scalar("Loss/train", epoch_losses[-1], global_step)

                if i % 100 == 0:
                    k = NUM_IMAGES_SAVE
                    with torch.no_grad():
                        codes = vae.get_codebook_indices(images[:k])
                        hard_recons = vae.decode(codes)

                    images, recons = map(lambda t: t[:k], (images, recons))
                    images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
                    images, recons, hard_recons = map(lambda t: make_grid(t[1].float(), nrow = int(math.sqrt(k)), normalize = True, value_range = tuple(np.array([[-1, 1],[-1,1],[-1,1]])[t[0],:])), enumerate([images, recons, hard_recons]))

                    writer.add_image('original images', images, global_step)
                    writer.add_image('reconstructions', recons, global_step)
                    writer.add_image('hard reconstructions', hard_recons, global_step)
                    writer.add_histogram('codebook_indices',codes, global_step)
    
                    # temperature anneal
                    if TEMP_ANNEAL_RATE * global_step < 1:
                        temp = (1+math.cos(math.pi * TEMP_ANNEAL_RATE * global_step))/2*(STARTING_TEMP-TEMP_MIN)+TEMP_MIN
                    else:
                        temp = TEMP_MIN

                    if BETA_ANNEAL_RATE * global_step < 1:
                        beta = (1+math.cos(math.pi * BETA_ANNEAL_RATE * global_step))/2*(STARTING_BETA-BETA_MAX)+BETA_MAX
                    else:
                        beta = BETA_MAX

                    lr = opt.param_groups[0]['lr']
                    sched.step(mean(step_losses))
                    step_losses = []
                        
                    vae.temperature = temp
                    vae.kl_div_loss_weight = beta

                    writer.add_scalar("temp", temp, global_step)
                    writer.add_scalar("beta", beta, global_step)
                    writer.add_scalar("lr", lr, global_step)

                    writer.flush()         
            global_step += 1

        print("Done epoch", epoch + 1, "with mean loss", mean(epoch_losses))

        if (epoch+1) % SAVE_STEP == 0:
            ckpt_name = "dVAE_" + '_'.join([str(tag) for tag in [
            epoch+1,
            int(math.log(lr)),
            str(temp)[:6],
            DVAE_NUM_TOKENS,
            DVAE_NUM_LAYERS,
            DVAE_NUM_RESNET_BLOCKS,
            CODEBOOK_DIM,
            DVAE_HIDDEN_DIM]]) + ".ckpt"
        else:
            ckpt_name = "dVAE_" + '_'.join([str(tag) for tag in [
            'latest',
            DVAE_NUM_TOKENS,
            DVAE_NUM_LAYERS,
            DVAE_NUM_RESNET_BLOCKS,
            CODEBOOK_DIM,
            DVAE_HIDDEN_DIM]]) + ".ckpt"
            
        print("Saving checkpoint", ckpt_name, "to", ckpts_dir)

        torch.save(vae.state_dict(), os.path.join(ckpts_dir, ckpt_name))
