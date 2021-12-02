import torch

from torchinfo import summary

from dalle_pytorch import DiscreteVAE

from configs import *

device = torch.device("cpu")
    
vae = DiscreteVAE(
    image_size = IMAGE_SIZE,
    num_layers = DVAE_NUM_LAYERS,
    num_tokens = DVAE_NUM_TOKENS,
    codebook_dim = CODEBOOK_DIM,
    hidden_dim = DVAE_HIDDEN_DIM,
    num_resnet_blocks = DVAE_NUM_RESNET_BLOCKS,
#    layer_hidden_dim_scale = DVAE_LAYER_HIDDEN_DIM_SCALE
)

BATCH_SIZE = 1

print("Batch size:", BATCH_SIZE)
print("num_layers:", DVAE_NUM_LAYERS)
print("num_tokens:", DVAE_NUM_TOKENS)
print("codebook_dim:", CODEBOOK_DIM)
print("hidden_dim:", DVAE_HIDDEN_DIM)
print("num_resnet_blocks:", DVAE_NUM_RESNET_BLOCKS)

summary(vae, (BATCH_SIZE,3,IMAGE_SIZE,IMAGE_SIZE))
