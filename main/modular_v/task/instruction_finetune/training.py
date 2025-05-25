import torch
from main_2.train import train_model_simple        

def train_instruction_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter,
        start_context, tokenizer
    )

    
    return train_losses, val_losses, tokens_seen