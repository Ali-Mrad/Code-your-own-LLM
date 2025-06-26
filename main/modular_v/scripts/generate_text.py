import argparse, sys, torch, tiktoken, pathlib
from main_2.config     import get_config
from main_2.model      import GPTModel
from main_2.load_from_safetensors import download_weights, load_weights_into_gpt
from main_2.generation import generate
from main_2.train      import text_to_token_ids, token_ids_to_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=["small","medium","large","xl"], default="small")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to GPT-2 checkpoint (.pth)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Prompt text; if omitted → interactive")
    p.add_argument("--tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    return p.parse_args()

def main():
    args = parse_args()
    cfg_name = {
        "small":  "gpt2-small (124M)",
        "medium": "gpt2-medium (355M)",
        "large":  "gpt2-large (774M)",
        "xl":     "gpt2-xl (1558M)"
    }[args.size]
    cfg       = get_config(cfg_name)
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(cfg).to(args.device)
    state_dict = download_weights(args.size)  
    load_weights_into_gpt(model, state_dict)  
    model.eval()
    torch.manual_seed(123)

    prompt = args.prompt or input("Enter prompt: ")
    idx    = text_to_token_ids(prompt, tokenizer).to(args.device)
    out    = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.tokens,
        context_size=cfg["context_length"],
        temperature=args.temperature,
        top_k=args.top_k
    )
    print("\nOutput:\n", token_ids_to_text(out, tokenizer))

if __name__ == "__main__":
    main()
