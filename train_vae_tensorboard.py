import argparse

import torch

import math

import os

import numpy as np

from datetime import datetime

from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from dalle_pytorch import DiscreteVAE

from statistics import mean

from tqdm import tqdm

#from coco_dataset import *
from configs import *

BATCH_SIZE = 16

COCO_ROOT = "PIXIV"
SAVE_PATH = "ckpts"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--epochs', type=int, default=256)
    parser.add_argument('--save_step', type=int, default=4)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--ckpt_epoch', type=int, default=0)

    train_group = parser.add_argument_group('Training settings')
    train_group.add_argument('--learning_rate', type = float, default = 1e-4, help = 'learning rate')
    train_group.add_argument('--lr_decay_rate', type = float, default = 0.98, help = 'learning rate decay')
    train_group.add_argument('--lr_min', type = float, default = 1.25e-6, help = 'learning rate min')
    train_group.add_argument('--starting_temp', type = float, default = 1, help = 'starting temperature')
    train_group.add_argument('--temp_min', type = float, default = 0.0625, help = 'minimum temperature to anneal to')
    train_group.add_argument('--starting_beta', type = float, default = 0, help = 'starting beta')
    train_group.add_argument('--beta_max', type = float, default = 1/192, help = 'maximum beta to anneal to')
    train_group.add_argument('--temp_anneal_rate', type = float, default = 1e-8, help = 'temperature annealing rate')
    train_group.add_argument('--beta_anneal_rate', type = float, default = 1e-6, help = 'beta annealing rate')
    train_group.add_argument('--num_images_save', type = int, default = 4, help = 'number of images to save')

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

    transform = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    transform = transforms.Compose(transform)

    train_set = datasets.ImageFolder(COCO_ROOT, transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, persistent_workers=True, num_workers=8)

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
        kl_div_loss_weight = 0
    )

    if args.ckpt_path is not None and args.ckpt_epoch is not None:
        if not os.path.isfile(args.ckpt_path):
            print(args.ckpt_path, "is not a file or doesn't exist!")
            quit()
        ckpts_dir = os.path.dirname(args.ckpt_path)
        START_EPOCH = args.ckpt_epoch
        vae.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device(device)))
    else:
        ckpts_dir = os.path.join(SAVE_PATH, datetime.now().strftime("%Y%m%d%H%M"))
        if not os.path.exists(ckpts_dir):
            os.makedirs(ckpts_dir)
        START_EPOCH = 0

    vae = vae.to(device)

    #opt = Adam([{'params':vae.parameters(), 'initial_lr':LEARNING_RATE}], lr = LEARNING_RATE)
    #opt = Adam(vae.parameters(), lr = LEARNING_RATE)
    opt = AdamW(vae.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    sched = ExponentialLR(optimizer = opt, gamma = LR_DECAY_RATE)#, last_epoch = START_EPOCH)

    global_step = 0
    temp = STARTING_TEMP

    writer = SummaryWriter()

    for epoch in range(START_EPOCH, NUM_EPOCHS):

        epoch_loss = []

        for i,(images,_) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            opt.zero_grad()
            loss, recons = vae(images, return_loss = True, return_recons = True)
            loss.backward()
            epoch_loss.append(loss.item())
            opt.step()
 
            if i % 10 == 9:
                writer.add_scalar("Loss/train", epoch_loss[-1], global_step)
 
                if i % 100 == 99: #quick fix for tensorboard not updating images?
                    k = NUM_IMAGES_SAVE
                    with torch.no_grad():
                        codes = vae.get_codebook_indices(images[:k])
                        hard_recons = vae.decode(codes)

                    images, recons = map(lambda t: t[:k], (images, recons))
                    images, recons, hard_recons, codes = map(lambda t: t.detach().cpu(), (images, recons, hard_recons, codes))
                    images, recons, hard_recons = map(lambda t: make_grid(t[1].float(), nrow = int(math.sqrt(k)), normalize = True, range = tuple(np.array([[0, 1],[-1,1],[-1,1]])[t[0],:])), enumerate([images, recons, hard_recons]))

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

                    lr = sched.get_last_lr()[0]
                    if lr > LR_MIN:
                        sched.step()

                    vae.temperature = temp
                    vae.kl_div_loss_weight = beta

                    writer.add_scalar("temp", temp, global_step)
                    writer.add_scalar("beta", beta, global_step)
                    writer.add_scalar("lr", lr, global_step)

                    writer.flush()

            global_step += 1

        print("Done epoch", epoch, "with total loss", mean(epoch_loss))

        if epoch % SAVE_STEP == 0:
            ckpt_name = "dVAE_" + '_'.join([str(tag) for tag in [
            epoch,
            int(math.log(sched.get_last_lr()[0])),
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
