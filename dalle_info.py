import torch

from torchinfo import summary

from dalle_pytorch import DiscreteVAE, DALLE

from configs import *


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

dalle = DALLE(
    dim = CODEBOOK_DIM,
    vae = vae,
    num_text_tokens = VOCAB_SIZE,
    text_seq_len = DALLE_TEXT_SEQ_LEN,
    depth = DALLE_DEPTH,
    heads = DALLE_HEADS,
    dim_head = DALLE_DIM_HEAD,
    attn_dropout = DALLE_ATTN_DROPOUT,
    ff_dropout = DALLE_FF_DROPOUT,
    attn_types = ('axial_row'),
    shift_tokens = False,
    rotary_emb = False      
)

BATCH_SIZE = 1

print("Batch size:", BATCH_SIZE)

print("num_layers:", DVAE_NUM_LAYERS)
print("num_tokens:", DVAE_NUM_TOKENS)
print("codebook_dim:", CODEBOOK_DIM)
print("hidden_dim:", DVAE_HIDDEN_DIM)
print("num_resnet_blocks:", DVAE_NUM_RESNET_BLOCKS)

print("dalle dim:", CODEBOOK_DIM)
print("num_text_tokens:", VOCAB_SIZE)
print("text_seq_len:", DALLE_TEXT_SEQ_LEN)
print("depth:", DALLE_DEPTH)
print("heads:", DALLE_HEADS)
print("dim_head:", DALLE_DIM_HEAD)

summary(dalle, ((BATCH_SIZE, DALLE_TEXT_SEQ_LEN), (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)), dtypes=(torch.long, torch.float))
