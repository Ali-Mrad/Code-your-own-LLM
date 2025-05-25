import torch
import torch.nn as nn
from main_2.gelu import GELU

class FeedForward(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        emb = cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb, 4 * emb),
            GELU(),
            nn.Linear(4 * emb, emb),
        )

    def forward(self, x):
        return self.layers(x)