from main_2.config  import get_config
from main_2.model   import GPTModel
from main_2.load_from_safetensors import download_weights, load_weights_into_gpt
import torch
from pathlib import Path


def build_sft_model(ckpt_path: str, model_size="small", device="cpu"):
    cfg_name = {"small":"gpt2-small (124M)", "medium":"gpt2-medium (355M)",
                "large":"gpt2-large (774M)", "xl":"gpt2-xl (1558M)"}[model_size]
    cfg = get_config(cfg_name)
    model = GPTModel(cfg).to(device)

    state_dict = download_weights(model_size)
    load_weights_into_gpt(model, state_dict)

    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading fine-tuned model from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)

    model.eval()
    return model