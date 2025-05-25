import torch

def classify_review(text, model, tokenizer, device, max_length, pad_token_id=50_256):

    model.eval()

    ids = tokenizer.encode(text)[: max_length]
    ids += [pad_token_id] * (max_length - len(ids))
    inp = torch.tensor(ids, device=device).unsqueeze(0)

    logits = model(inp)[:, -1, :]               # last-token logits
    label  = torch.argmax(logits, dim=-1).item()

    return "spam" if label == 1 else "not spam"
