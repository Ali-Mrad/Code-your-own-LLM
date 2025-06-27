import torch
import tiktoken
from pathlib import Path
from main_2.config import get_config
from main_2.model import GPTModel
from main_2.load_from_safetensors import load_weights_into_gpt, download_weights
from task.instruction_finetune.inference import generate_response


MODEL_SAVE_PATH = Path("models/gpt2-small124M-sft.pth")

def load_instruction_model(device="cpu"):
    cfg = get_config("gpt2-small (124M)") # or "gpt2-small (124M)", "gpt2-large (774M)", "gpt2-xl (1558M)",  "gpt2-medium (355M)"
    model = GPTModel(cfg).to(device)

    weights_path = download_weights("small") # medium
    load_weights_into_gpt(model, weights_path)

    
    if MODEL_SAVE_PATH.exists():
        print(f"Loading fine-tuned model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.eval()
    else:
        raise FileNotFoundError(f"Fine-tuned model not found at {MODEL_SAVE_PATH}")
    
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = load_instruction_model(device=device)

    print("\nWelcome! Type your instruction (Ctrl+C to quit)\n")
    while True:
        try:
            input_text  = input("Input: ").strip()

            response = generate_response(
                model,input_text, device=device
            )
            print("\nModel response:")
            print(response)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n Session ended.")
            break
