import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # separate Q, K, V projections 
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out, bias=True)
        self.dropout = nn.Dropout(dropout)

        # mask registered once, will be moved to the right device each forward
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool(), persistent=False)

    def forward(self, x):
        
        B, L, _ = x.shape # Shape: (batch_size, num_tokens, d_in)
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        # reshape to (B, n_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:L, :L]
        
        # Compute self-attention with a causal mask
        attn_scores = q @ k.transpose(2, 3)  # Dot product for each head

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / q.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ v).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(B, L, self.d_out) # contiguous() is needed to reshape the tensor and does the concatenation (view is the reshape function in pytorch) that does an in-place operation, reshaping the tensor without copying the data)
        context_vec = self.out_proj(context_vec) # this is an optional projection

        return context_vec