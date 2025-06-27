import time, json, torch, tiktoken
from pathlib import Path
from tqdm import tqdm

from task.instruction_finetune.data       import download_and_load_file, split_data, format_input
from task.instruction_finetune.dataloader import create_dataloaders
from task.instruction_finetune.model_setup_simple import build_sft_model
from task.instruction_finetune.training   import train_instruction_model
from task.instruction_finetune.inference  import generate_response
from main_2.generation import generate
from main_2.train    import text_to_token_ids, token_ids_to_text

FILE_PATH           = "data/instruction-data.json"
URL                 = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
MODEL_SAVE_PATH     = Path("models/gpt2-small124M-sft.pth") # 
LOAD_EXISTING_MODEL = False     # False train the model again
GENERATE_FULL_TEST_SET = False  # True to generate responses for the full test set
BATCH_SIZE          = 8
NUM_EPOCHS          = 2
EVAL_FREQ           = 5
EVAL_ITER           = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# data
data = download_and_load_file(FILE_PATH, URL)
train_data, val_data, test_data = split_data(data)
tok = tiktoken.get_encoding("gpt2")
train_loader, val_loader, test_loader = create_dataloaders(
    train_data, val_data, test_data, tok, BATCH_SIZE, device
)

# model
model = build_sft_model(MODEL_SAVE_PATH, model_size="small", device=device) # medium
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

if LOAD_EXISTING_MODEL and MODEL_SAVE_PATH.exists():
    print("Loading existing model →", MODEL_SAVE_PATH)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print("Model loaded. Skipping training.")
else:
    print("Training new model…")
    torch.manual_seed(123)
    start_ctx = format_input(val_data[0])
    t0 = time.time()

    train_instruction_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=NUM_EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER,
        start_context=start_ctx, tokenizer=tok
    )

    print(f"Training done in {(time.time()-t0)/60:.2f} min.")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved ", MODEL_SAVE_PATH)

# quick demo
print("\n### Demo ###\n")
print(generate_response(model, "List three benefits of regular exercise.", device=device))
print("\n### --------\n")

# generate full test-set
if GENERATE_FULL_TEST_SET:
    print("Generating responses for test set …")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        prompt = format_input(entry)
        idx    = text_to_token_ids(prompt, tok).to(device)
        out_ids = generate(
            model, idx, max_new_tokens=256,
            context_size=model.cfg["context_length"], eos_id=50256
        )
        full = token_ids_to_text(out_ids, tok)
        response = full[len(prompt):].replace("### Response:", "").strip()
        test_data[i]["model_response"] = response

    with open("instruction-data-with-response.json", "w") as f:
        json.dump(test_data, f, indent=4)
    print("Saved responses instruction-data-with-response.json")
