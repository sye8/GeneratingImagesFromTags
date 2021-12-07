import math

import torch

from einops import rearrange

from torch import nn
from torch.nn import functional as F


class CVAE(nn.Module):
    def __init__(self, dvae=None, latent_dim=1024, tag_dim=3933):
        super().__init__()
        
        self.dvae = dvae
        
        self.latent_dim = latent_dim
        
        self.embed_tags = nn.Linear(tag_dim, 16 * 16)
        self.embed_data = nn.Conv2d(512, 512, kernel_size=1)
        
        conv_channels = [512+1, 1024, 2048, 4096]
        modules = []
        for i in range(1, len(conv_channels)):
            modules.append(nn.Sequential(
                nn.Conv2d(conv_channels[i-1], conv_channels[i], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(conv_channels[i]),
                nn.LeakyReLU()
            ))
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(4096 * 2 * 2, latent_dim)
        self.fc_var = nn.Linear(4096 * 2 * 2, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim + tag_dim, 4096 * 2 * 2)

        modules = []
        conv_channels.reverse()
        conv_channels[-1] -= 1
        for i in range(1, len(conv_channels)):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(conv_channels[i-1], conv_channels[i], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(conv_channels[i]),
                nn.LeakyReLU()
            ))
        modules.append(
            nn.Conv2d(conv_channels[-1], conv_channels[-1], kernel_size=1)
        )
        self.decoder = nn.Sequential(*modules)
    
    def encode(self, input):
        result = self.encoder(input).flatten(start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.reshape(-1, 4096, 2, 2)
        result = self.decoder(result)
        return result
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def logits_to_embedding(self, input):
        # Inputs are dVAE logits
        # Convert logits to codebook indices
        input = input.argmax(dim = 1).flatten(1)
        # Get embedding from codebook
        input = self.dvae.codebook(input)
        # Rearrange for convolution
        input = rearrange(input, 'b (h w) d -> b d h w', h = 16, w = 16)
        return input
    
    def forward(self, input, tags):
        embedded_tags = self.embed_tags(tags)
        embedded_tags = embedded_tags.reshape(-1, 16, 16).unsqueeze(1)
        if self.dvae:
            input = self.logits_to_embedding(input)
        embedded_input = self.embed_data(input)
        x = torch.cat([embedded_input, embedded_tags], dim = 1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, tags], dim = 1)
        return self.decode(z), mu, log_var
    
    def loss_function(self, recons, input, mu, log_var, kld_weight):
        if self.dvae:
            input = self.logits_to_embedding(input)
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return loss, recons_loss, kld_loss
        
    def sample(self, num_samples, device, tags):
        tags = tags.float()
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(device)
        z = torch.cat([z, tags], dim=1)
        samples = self.decode(z)
        return samples


# https://arxiv.org/pdf/1805.08318.pdf
class SelfAttn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # x
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()
        # f,g,h: C' x C; v: C x C'; C' = C // 8
        # Part of eq. 2
        self.f = nn.Conv1d(out_channels, out_channels // 8, kernel_size=1)
        self.g = nn.Conv1d(out_channels, out_channels // 8, kernel_size=1)
        self.h = nn.Conv1d(out_channels, out_channels // 8, kernel_size=1)
        self.v = nn.Conv1d(out_channels // 8, out_channels, kernel_size=1)
        # Gamma is a learnable pamarameter and is initialized to 0
        # Part of eq. 3
        self.gamma = nn.Parameter(torch.tensor([0.]))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        # Eq. 1
        s = torch.bmm(self.f(x).transpose(1,2), self.g(x))
        beta = F.softmax(s, dim=1)
        # Eq. 2
        o = self.v(torch.bmm(self.h(x), beta))
        # Eq. 3
        y = self.gamma * o + x
        y = y.reshape(y.shape[0], y.shape[1], int(math.sqrt(y.shape[2])), int(math.sqrt(y.shape[2])))
        return y


class CVAE_SelfAttn(CVAE):
    def __init__(self, dvae=None, latent_dim=1024, tag_dim=3933):
        super().__init__()
        
        self.dvae = dvae
        
        self.latent_dim = latent_dim
        
        self.embed_tags = nn.Linear(tag_dim, 16*16)
        self.embed_data = nn.Conv2d(512, 512, kernel_size=1)
        
        conv_channels = [512+1, 1024, 2048, 4096]
        modules = []
        for i in range(1, len(conv_channels)):
            modules.append(nn.Sequential(
                SelfAttn(conv_channels[i-1], conv_channels[i]),
            ))
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(4096*2*2, latent_dim)
        self.fc_var = nn.Linear(4096*2*2, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim + tag_dim, 4096*2*2)

        modules = []
        conv_channels.reverse()
        conv_channels[-1] -= 1
        for i in range(1, len(conv_channels)):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(conv_channels[i-1], conv_channels[i], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(conv_channels[i]),
                nn.LeakyReLU()
            ))
        self.decoder = nn.Sequential(*modules)
