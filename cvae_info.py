import torch

from torchinfo import summary

from dalle_pytorch import DiscreteVAE

from cvae import CVAE


if torch.cuda.is_available():
    print("Using cuda")
    device = torch.device("cuda")
else:
    print("Using cpu")
    device = torch.device("cpu")

vae_params = torch.load("vae_params.pt", map_location=torch.device(device))
vae_weights = torch.load("vae_weights.pt", map_location=torch.device(device))
vae = DiscreteVAE(**vae_params)
vae.load_state_dict(vae_weights)

cvae = CVAE(vae)

BATCH_SIZE = 8

summary(cvae, ((BATCH_SIZE, 8192, 16, 16), (BATCH_SIZE, 3933)))
