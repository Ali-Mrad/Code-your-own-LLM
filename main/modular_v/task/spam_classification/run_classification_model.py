import time, torch, tiktoken, json
from pathlib import Path
from task.spam_classification.data        import prepare_dataset
from task.spam_classification.dataloader  import get_dataloaders
from task.spam_classification.model_setup import build_spam_model_simple
from task.spam_classification.training    import train_classifier_simple
from task.spam_classification.evaluation  import calc_accuracy_loader
from task.spam_classification.inference  import classify_review


MODEL_SAVE_PATH      = Path("models/review_classifier.pth")
LOAD_EXISTING_MODEL  = True        # change to False to retrain the model
BATCH_SIZE           = 8
NUM_EPOCHS           = 2
EVAL_FREQ            = 50         
EVAL_ITER            = 5          


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device:", device)

prepare_dataset("data/sms_spam")
tok = tiktoken.get_encoding("gpt2")
train_loader, val_loader, test_loader, max_len = get_dataloaders(
    tok, data_root="data/sms_spam", batch_size=BATCH_SIZE, device=device
)

# model
model = build_spam_model_simple(device=device)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5
)

if LOAD_EXISTING_MODEL and MODEL_SAVE_PATH.exists():
    print("Loading existing model", MODEL_SAVE_PATH)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print("Model loaded. Skipping training.")
else:
    print("Training new model…")
    torch.manual_seed(123)
    t0 = time.time()

    train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=NUM_EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER
    )

    print(f"Training done in {(time.time()-t0)/60:.2f} min.")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved ", MODEL_SAVE_PATH)


test_acc = calc_accuracy_loader(test_loader, model, device)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

print("\nClassifying a sample review:\n")
print(classify_review("your the winner of our lottery! send us your credit card details to claim your prize and win 3000 dollars in cash", model, tok, device,max_len))
print(classify_review("Hey, just wanted to check if we're still on for dinner tonight? Let me know!", model, tok, device,max_len))

