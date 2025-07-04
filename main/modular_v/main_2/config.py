GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers , number of transformer blocks
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": True       # Query-Key-Value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

def get_config(name: str = "gpt2-small (124M)") -> dict:
    """
    Return a full config dict, merging the baseline with the requested size.
    """
    cfg = GPT_CONFIG_124M.copy()
    if name in model_configs:
        cfg.update(model_configs[name])
    else:
        raise ValueError(f"Unknown model name: {name}")
    return cfg