import torch
import tiktoken
from pathlib import Path
from task.spam_classification.model_setup import build_spam_model_simple
from task.spam_classification.data import prepare_dataset
from task.spam_classification.dataloader import get_dataloaders
from task.spam_classification.inference import classify_review

MODEL_SAVE_PATH = Path("models/review_classifier.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tok = tiktoken.get_encoding("gpt2")
prepare_dataset("data/sms_spam")  
_, _, _, max_len = get_dataloaders(tok, data_root="data/sms_spam", batch_size=1)

model = build_spam_model_simple(device=device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()


while True:
    user_input = input("\n Enter your message (or type 'exit' to quit):\n> ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print(" Exiting.")
        break

    label = classify_review(user_input, model, tok, device, max_len)
    print("Classification:", label)
