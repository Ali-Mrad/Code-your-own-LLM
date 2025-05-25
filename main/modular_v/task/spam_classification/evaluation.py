
import torch
from torch.utils.data import DataLoader



def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]                # last-token logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
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


@torch.no_grad()
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct, total = 0, 0
    for i, (inp, tgt) in enumerate(data_loader):
        if num_batches and i >= num_batches:
            break
        inp, tgt = inp.to(device), tgt.to(device)
        preds = torch.argmax(model(inp)[:, -1, :], dim=-1)
        correct += (preds == tgt).sum().item()
        total   += tgt.size(0)
    model.train()
    return correct / total


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, eval_iter)
    model.train()
    return train_loss, val_loss
