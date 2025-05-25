import torch
from .evaluation import (
    evaluate_model,
    calc_accuracy_loader,
    calc_loss_batch,
)


def train_classifier_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for inp, tgt in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inp, tgt, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += inp.size(0)   # nb messages (pas tokens)
            global_step   += 1

            if global_step % eval_freq == 0:
                tr, va = evaluate_model(model, train_loader, val_loader,
                                        device, eval_iter)
                train_losses.append(tr)
                val_losses.append(va)
                print(f"Ep {epoch+1} (step {global_step:06d}) — "
                      f"train {tr:.3f} | val {va:.3f}")

        tr_acc = calc_accuracy_loader(train_loader, model, device, eval_iter)
        va_acc = calc_accuracy_loader(val_loader,   model, device, eval_iter)
        print(f"Accuracy — train {tr_acc*100:.2f}% | val {va_acc*100:.2f}%")
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

    return train_losses, val_losses, train_accs, val_accs, examples_seen
