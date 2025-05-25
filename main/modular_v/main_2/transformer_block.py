import torch
import torch.nn as nn
from main_2.attention import MultiHeadAttention
from main_2.feedforward import FeedForward
from main_2.layernorm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"]) # we have two layer normalization in the model in order to learn the parameters of the model at the respective layers position
        self.norm2 = LayerNorm(cfg["emb_dim"]) # 2 disctinct layer normalization layers modules 
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x    