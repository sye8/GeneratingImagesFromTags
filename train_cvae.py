import argparse
import math

import torch

import torch.nn.functional as F

from datetime import datetime
from statistics import mean

from torch import einsum
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from einops import rearrange

from dalle_pytorch import DiscreteVAE

from tqdm import tqdm

from dataset import *
from cvae import *


def tags_to_multihot(tags):
    tags = F.one_hot(tags, num_classes=VOCAB_SIZE)
    tags = torch.sum(tags, axis=1)
    return tags


if __name__ == '__main__':
    SAVE_PATH = "ckpts"

    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--epochs', type = int, default = 1000)
    parser.add_argument('--save_step', type = int, default = 100)
    parser.add_argument('--use_attn', action='store_true', default=False)
    parser.add_argument('--use_embs', action='store_true', default=False)
    
    train_group = parser.add_argument_group('Training Settings')
    train_group.add_argument('--learning_rate', type = float, default = 1e-3, help = 'Learning Rate')
    train_group.add_argument('--kl_weight', type = int, default = 1.0, help = 'KL divergence weight')
    train_group.add_argument('--batch_size', type = int, default = 16, help = 'Batch Size')
    
    args = parser.parse_args()
    
    NUM_EPOCHS = args.epochs
    SAVE_STEP = args.save_step
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    KL_WEIGHT = args.kl_weight
    
    NUM_IMAGES_SAVE = 4
    
    dataset = PixivFacesDataset(args.use_embs)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
    
    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using cpu")
        device = torch.device("cpu")
    
    vae = None
    if not args.use_embs:
        print("Using actual images and dVAE")
        vae_params = torch.load("vae_params.pt")
        vae = DiscreteVAE(**vae_params)
        vae_weights = torch.load("vae_weights.pt", map_location=torch.device(device))
        vae.load_state_dict(vae_weights)
        vae = vae.to(device)
        # Set vae no grad
        for param in vae.parameters():
            param.requires_grad = False
    
    if args.use_attn:
        print("Using attention")
        cvae = CVAE_SelfAttn(vae)
    else:
        cvae = CVAE(vae)
    cvae = cvae.to(device)
    
    ckpts_dir = os.path.join(SAVE_PATH, "cvae_faces_" + datetime.now().strftime("%Y%m%d%H%M"))
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)
    START_EPOCH = 0
    
    opt = Adam(cvae.parameters(), lr = LEARNING_RATE)
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
        epoch_recs = []
        epoch_kl = []
        
        for i, (inputs, tags) in enumerate(tqdm(dataloader)):
            inputs, tags = map(lambda t: t.to(device), (inputs, tags))
            
            if not args.use_embs:
                inputs = vae(inputs, return_logits=True)
            
            result, mu, log_var = cvae(inputs, tags)
            
            loss, recons_loss, kld_loss = cvae.loss_function(result, inputs, mu, log_var, KL_WEIGHT)
            
            loss.backward()
            clip_grad_norm_(cvae.parameters(), 0.5)
            epoch_losses.append(loss.item())
            epoch_recs.append(recons_loss.item())
            epoch_kl.append(kld_loss.item())
            opt.step()
            opt.zero_grad()

            if global_step % 10 == 9:
                writer.add_scalar("Loss/train", epoch_losses[-1], global_step)
                writer.add_scalar("Rec/train", epoch_recs[-1], global_step)
                writer.add_scalar("KL/train", epoch_kl[-1], global_step)
                writer.add_scalar("lr", opt.param_groups[0]['lr'], global_step)
                
                if global_step % 50 == 49:
                    sched.step(torch.mean(loss))
                    # Since no dVAE if loaded if using embeddings, cannot sample images
                    if not args.use_embs:
                        # Sample
                        sample_tags = tags[0]
                        embedding = cvae.sample(NUM_IMAGES_SAVE, device, sample_tags.repeat(NUM_IMAGES_SAVE, 1).float())
                        # Img Recon
                        images = vae.decoder(embedding)
                        images = make_grid(images.float(), nrow = int(math.sqrt(NUM_IMAGES_SAVE)), normalize = True, value_range = (-1,1)).detach().cpu()
                        writer.add_image('generated images', images, global_step)
                        # Tags
                        tags_str = dataset.multihot2tags(sample_tags)
                        writer.add_text('tags', tags_str, global_step)
                        writer.flush()
                    
            global_step += 1
        
        print("Done epoch", epoch + 1, "with mean loss", mean(epoch_losses), "mean rec loss", mean(epoch_recs), "mean KL", mean(epoch_kl))
    
        if (epoch + 1) % SAVE_STEP == 0:
            ckpt_name = "cvae_face" + str(epoch+1) + ".ckpt"
        else:
            ckpt_name = "cvae_last.ckpt"
        print("Saving checkpoint", ckpt_name, "to", ckpts_dir)
        torch.save(cvae.state_dict(), os.path.join(ckpts_dir, ckpt_name), _use_new_zipfile_serialization=False)
