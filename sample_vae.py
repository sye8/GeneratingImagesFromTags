import argparse
import matplotlib

import matplotlib.pyplot as plt
from math import sqrt
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from dalle_pytorch import DiscreteVAE

from dataset import *
from configs import *


NUM_SAMPLES = 1
BATCH_SIZE = 4

parser = argparse.ArgumentParser(description="")
parser.add_argument('ckpt_path', type=str)
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
)

if not os.path.isfile(args.ckpt_path):
    print(args.ckpt_path, "is not a file or doesn't exist!")
    quit()
vae.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device(device)))
    
vae = vae.to(device)

transform = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
transform = transforms.Compose(transform)

train_set = ImageDataset(DATASET_ROOT, transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

for _ in range(NUM_SAMPLES):
    img_batch = torch.Tensor(next(iter(train_loader))).to(device)

    with torch.no_grad():
        img_codes = vae.get_codebook_indices(img_batch)
        img_batch_decode = vae.decode(img_codes)
        images, recons = map(lambda t: t.detach().cpu(), (img_batch, img_batch_decode))
        images, recons = map(lambda t: make_grid(t[1].float(), nrow = int(sqrt(BATCH_SIZE)), normalize = True, value_range = tuple(np.array([[0, 1],[-1,1]])[t[0],:])), enumerate([images, recons]))

        _, ax = plt.subplots(1,2,figsize=(8,8))
        ax[0].axis('off')
        ax[0].imshow(images.permute(1,2,0))
        ax[1].axis('off')
        ax[1].imshow(recons.permute(1,2,0))
        plt.show()

