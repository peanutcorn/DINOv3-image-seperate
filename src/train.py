"""One supervised training loop for linear heads and CNN fine-tuning."""

import copy
import time
import torch
from src.models import training_mode
from src.utils import synchronize


def fit(
    model, train_loader, validation_loader, device, epochs, lr, weight_decay, patience
):
    """Train on train data and restore lowest-validation-loss parameters."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_loss, bad_epochs = float("inf"), 0
    best_state, history = None, []
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(epochs):
        training_mode(model)
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for inputs, targets in validation_loader:
                val_loss += criterion(
                    model(inputs.to(device)), targets.to(device)
                ).item() * len(targets)
        val_loss /= len(validation_loader.dataset)
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=total_loss / len(train_loader.dataset),
                val_loss=val_loss,
            )
        )
        print(f"Epoch {epoch + 1}/{epochs}: validation loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError(
            "No valid training epoch; check epochs and non-finite losses"
        )
    model.load_state_dict(best_state)
    synchronize(device)
    return time.perf_counter() - started, history
