import torch
from torch import nn


def fourier_act(x):
    output = torch.zeros_like(x)
    output[..., 0::2] = torch.sin(x[..., 0::2])
    output[..., 1::2] = torch.cos(x[..., 1::2])
    return output


class FourierAct(torch.nn.Module):
    def __init__(self, cat=False):
        super().__init__()
        self.cat = cat

    def forward(self, x):
        output = torch.zeros_like(x)
        output[..., 0::2] = torch.sin(x[..., 0::2])
        output[..., 1::2] = torch.cos(x[..., 1::2])
        if self.cat:
            return torch.cat([x, output], dim=-1)
        return output


class FourierMLP(nn.Module):
    def __init__(self, dim=1024, expansion_factor=4):
        super().__init__()
        hidden_dim = dim * expansion_factor

        self.net = torch.nn.Sequential(
            nn.Linear(dim, hidden_dim),
            FourierAct(cat=True),  # This doubles the dimension because of cat=True!
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # Fix input dim here
            FourierAct(),
            nn.Linear(hidden_dim, hidden_dim),
            FourierAct(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def mueller_hash(x):
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = (x >> 16) ^ x
    return x


class VQVAE(nn.Module):
    def __init__(self, dim=1024, expansion_factor=4):
        super().__init__()
        self.dim = dim
        self.ffn = FourierMLP(dim, expansion_factor)
        # TODO: maybe it is better to explicitly decompose the embedding table
        # into one table per hash, and combine for decoding as well?
        self.emb = nn.Embedding(num_embeddings=4096, embedding_dim=dim)

    def latent(self, x):
        distances = torch.cdist(x, self.emb.weight)
        min_indices = distances.argmin(dim=-1)
        quantized = torch.zeros_like(x)
        for i in range(1, 4):
            select = mueller_hash(
                min_indices + i * self.emb.weight.size(0)
            ) % self.emb.weight.size(0)
            quantized += self.emb(select) / 3

        return quantized

    def encode(self, x):
        return self.latent(self.ffn(x))

    def decode(self, x):
        logits = x @ self.emb.weight.t()
        return self.emb((logits.argmax(dim=-1)))
