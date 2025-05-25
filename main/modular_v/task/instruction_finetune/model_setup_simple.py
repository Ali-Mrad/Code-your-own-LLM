from main_2.config  import get_config
from main_2.model   import GPTModel
from main_2.weights import load_into


def build_sft_model(ckpt_path: str, model_size="small", device="cpu"):
    cfg_name = {"small":"gpt2-small (124M)", "medium":"gpt2-medium (355M)",
                "large":"gpt2-large (774M)", "xl":"gpt2-xl (1558M)"}[model_size]
    cfg = get_config(cfg_name)
    model = GPTModel(cfg).to(device)

    load_into(model, ckpt_path, device=device, strict=False)

    model.eval()
    return model