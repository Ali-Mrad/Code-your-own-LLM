import argparse, torch, tiktoken, pathlib, sys
from main_2.config   import get_config
from main_2.model    import GPTModel
from main_2.weights  import download_weights, load_into
from main_2.generation import generate
from main_2.train import text_to_token_ids, token_ids_to_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=["small", "medium", "large", "xl"],
                   default="small", help="GPT-2 variant")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to an existing .pth ")
    p.add_argument("--prompt", type=str, default=None,
                   help="Prompt to start generation; if omitted → interactive")
    p.add_argument("--tokens", type=int, default=50,
                   help="How many new tokens to generate")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_name = (f"gpt2-{args.size} (124M)" if args.size == "small" else
                f"gpt2-{args.size} (355M)" if args.size == "medium" else
                f"gpt2-{args.size} (774M)" if args.size == "large" else
                f"gpt2-{args.size} (1558M)")
    cfg = get_config(cfg_name)
    tokenizer = tiktoken.get_encoding("gpt2")

    ckpt_path = pathlib.Path(args.ckpt) if args.ckpt else download_weights(args.size)

    model = GPTModel(cfg).to(args.device)
    load_into(model, ckpt_path, device=args.device, strict=False)
    model.eval()

    prompt_text = args.prompt
    if prompt_text is None:
        try:
            prompt_text = input("Enter prompt: ")
        except KeyboardInterrupt:
            sys.exit(0)

    prompt_ids = text_to_token_ids("Every effort moves you", tokenizer).to(args.device),
    out_ids = generate(
        model=model,
        idx=prompt_ids,
        max_new_tokens=args.tokens,
        context_size=cfg["context_length"],
        top_k=args.top_k,
        temperature=args.temperature,
        
    )
    print("Output text:\n", token_ids_to_text(out_ids, tokenizer))

if __name__ == "__main__":
    main()
