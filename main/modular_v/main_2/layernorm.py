import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # trainable neural network weights
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased = False means that the variance is calculated with the sample statistics if it was true the population statistics would be calculated
        normed = (x - mean) / torch.sqrt(var + self.eps) # avoid division by zero
        return self.scale * normed + self.shift # give the netowrk the ability to learn the best scale and shift for the data