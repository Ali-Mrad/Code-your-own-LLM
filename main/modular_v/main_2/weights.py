import urllib.request, os, pathlib, torch
from safetensors.torch import load_file

HF_BASE = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/"

ALLOWED = {
    "small":  "gpt2-small-124M.pth",
    "medium": "gpt2-medium-355M.pth",
    "large":  "gpt2-large-774M.pth",
    "xl":     "gpt2-xl-1558M.pth",
}

def download_weights(model_size: str = "small", dst_dir: str | os.PathLike = "models") -> pathlib.Path:
    """
    Download *only* the raw PyTorch checkpoint if it is not present locally.
    """
    key = model_size.lower()
    if key not in ALLOWED:
        raise ValueError(f"model_size must be one of {list(ALLOWED)}")

    file_name = ALLOWED[key]
    dst_dir   = pathlib.Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    local_fp  = dst_dir / file_name

    if local_fp.exists():
        print(f"weights already downloaded → {local_fp}")
        return local_fp

    url = f"{HF_BASE}{file_name}"
    print(f" downloading {file_name} …")
    urllib.request.urlretrieve(url, local_fp)
    print(f"saved to {local_fp}")
    return local_fp


def load_into(
    model,
    weight_path: str | os.PathLike,
    device: str = "cpu",
    strict: bool = True,
):
   
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model