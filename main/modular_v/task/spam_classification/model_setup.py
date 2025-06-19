import torch.nn as nn
from main_2.config  import get_config
from main_2.model   import GPTModel
#from main_2.weights import download_weights, load_into
from main_2.load_from_safetensors import download_weights,load_weights_into_gpt


def build_spam_model_simple(
    model_size: str = "small",
    checkpoint: str | None = None,
    device: str = "cpu",
):
    
    cfg_name = {
        "small":  "gpt2-small (124M)",
        "medium": "gpt2-medium (355M)",
        "large":  "gpt2-large (774M)",
        "xl":     "gpt2-xl (1558M)",
    }[model_size]
    cfg   = get_config(cfg_name)
    model = GPTModel(cfg).to(device)

    
    ckpt = checkpoint or download_weights(model_size)
    load_weights_into_gpt(model, ckpt) 

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model.out_head = nn.Linear(cfg["emb_dim"], 2)

    for p in model.trf_blocks[-1].parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True


    return model