import torch, matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from main_2.generation import generate 
import tiktoken as tk



def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (inp, tgt) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(inp, tgt, model, device).item()

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, eval_iter)
    model.train()
    return train_loss, val_loss


#ENC = tk.get_encoding("gpt2")
tokenizer = tk.get_encoding("gpt2")

def text_to_token_ids(text,tokenizer):
    return torch.tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"})).unsqueeze(0)


def token_ids_to_text(token_ids,tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(model,tokenizer, device, start_context):
    model.eval()
    ctx_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        out_ids = generate_text_simple(model, encoded, 50, ctx_size)
    print(token_ids_to_text(out_ids,tokenizer).replace("\n", " "))
    model.train()


def train_model_simple(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device: str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,                      
):
    
    train_losses, val_losses, tokens_seen = [], [], []
    total_tokens, global_step = 0, -1

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            total_tokens += input_batch.numel()
            global_step += 1

            # periodic evaluation
            if global_step % eval_freq == 0:
                tr_loss, va_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(tr_loss)
                val_losses.append(va_loss)
                tokens_seen.append(total_tokens)
                print(
                    f"Epoch {epoch+1} | step {global_step:06d} — "
                    f"train {tr_loss:.3f} | val {va_loss:.3f}"
                )

        # text sample after each epoch
        generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_context=start_context,
        )

    return train_losses, val_losses, tokens_seen


def plot_losses(tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    epochs = list(range(1, len(train_losses) + 1))
    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses,  "-.", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    plt.tight_layout()
    plt.savefig("loss-plot.pdf")
    plt.show()
