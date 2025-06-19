
from safetensors.torch import load_file
import torch
import os, pathlib, urllib.request
'''
URL_DIR = {
  "gpt2-small (124M)": "gpt2",         # works ok
  "gpt2-medium (355M)": "gpt2-medium", # this file seems to have issues via `generate`
  "gpt2-large (774M)": "gpt2-large",   # works ok
  "gpt2-xl (1558M)": "gpt2-xl"         # works ok
}


def download_safetensors(model_name):
    url = f"https://huggingface.co/openai-community/{URL_DIR[model_name]}/resolve/main/model.safetensors"
    output_file = f"models/model-{URL_DIR[model_name]}.safetensors"
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(output_file):
        print(f"Downloading {output_file} ...")
        urllib.request.urlretrieve(url, output_file)
    return load_file(output_file)
'''

HF_BASE = "https://huggingface.co/openai-community/"
ALLOWED = {
    "small":  "gpt2",
    "medium": "gpt2-medium",
    "large":  "gpt2-large",
    "xl":     "gpt2-xl"
}


def download_weights(model_size: str = "small", dst_dir: str | os.PathLike = "models"):
    """
    Download GPT-2 safetensors weights and return state_dict.
    """
    key = model_size.lower()
    if key not in ALLOWED:
        raise ValueError(f"model_size must be one of {list(ALLOWED)}")

    model_name = ALLOWED[key]
    file_name  = f"model-{model_name}.safetensors"
    dst_dir    = pathlib.Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    local_fp   = dst_dir / file_name

    if not local_fp.exists():
        url = f"{HF_BASE}{model_name}/resolve/main/model.safetensors"
        print(f" Downloading {file_name} ...")
        urllib.request.urlretrieve(url, local_fp)
        print(f" Saved to {local_fp}")
    else:
        print(f" Weights already downloaded → {local_fp}")

    return load_file(local_fp)


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.detach())

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe.weight"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte.weight"])
    for b in range(len(gpt.trf_blocks)):
        q_w, k_w, v_w = torch.chunk(params[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = torch.chunk(params[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params[f"h.{b}.attn.c_proj.bias"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params[f"h.{b}.mlp.c_proj.bias"])

        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params[f"h.{b}.ln_2.bias"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["ln_f.bias"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte.weight"])