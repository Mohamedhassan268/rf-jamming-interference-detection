"""Small, explicit supervised training loop with validation-only selection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class TrainingResult:
    """States and history produced by a completed training run."""

    best_state: dict[str, torch.Tensor]
    last_state: dict[str, torch.Tensor]
    best_epoch: int
    history: list[dict[str, float | int]]


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for samples, labels, _metadata in loader:
            samples = samples.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(samples)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total_examples += batch_size
    if total_examples == 0:
        raise ValueError("A training or validation loader contained zero examples.")
    return total_loss / total_examples, total_correct / total_examples


def train_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    training_config: dict[str, Any],
    device: torch.device,
) -> TrainingResult:
    """Train with cross-entropy and select only by validation loss."""
    optimizer_name = str(training_config.get("optimizer", "adam")).lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}; currently supported: adam.")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and early_stopping_patience must be positive.")
    model.to(device)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_accuracy = _run_epoch(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} train_accuracy={train_accuracy:.4f} "
            f"validation_loss={validation_loss:.6f} validation_accuracy={validation_accuracy:.4f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training produced no best model state.")
    return TrainingResult(
        best_state=best_state,
        last_state=deepcopy(model.state_dict()),
        best_epoch=best_epoch,
        history=history,
    )

