from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochStats:
    loss: float
    accuracy: float


def _batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["image"].to(device, non_blocking=True)
    targets = batch["target"].to(device, dtype=torch.long, non_blocking=True)
    return inputs, targets


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochStats:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for batch in loader:
        inputs, targets = _batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch_size

    if total == 0:
        raise RuntimeError("O DataLoader de treino não forneceu amostras.")
    return EpochStats(loss=loss_sum / total, accuracy=correct / total)


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochStats:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for batch in loader:
        inputs, targets = _batch_to_device(batch, device)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch_size

    if total == 0:
        raise RuntimeError("O DataLoader de avaliação não forneceu amostras.")
    return EpochStats(loss=loss_sum / total, accuracy=correct / total)


def fit_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int,
    patience: int,
    min_delta: float,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
) -> tuple[dict[str, list[float]], int, float]:
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rate": [],
        "epoch_seconds": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        started_at = time.perf_counter()

        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = evaluate_one_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_stats.loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        elapsed = time.perf_counter() - started_at

        history["train_loss"].append(train_stats.loss)
        history["val_loss"].append(val_stats.loss)
        history["train_accuracy"].append(train_stats.accuracy)
        history["val_accuracy"].append(val_stats.accuracy)
        history["learning_rate"].append(current_lr)
        history["epoch_seconds"].append(elapsed)

        improved = val_stats.loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_stats.loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "validation_loss": best_val_loss,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metadata": checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_stats.loss:.4f} | "
            f"val_loss={val_stats.loss:.4f} | "
            f"train_acc={train_stats.accuracy:.4f} | "
            f"val_acc={val_stats.accuracy:.4f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping: a validação não melhorou por {patience} épocas.")
            break

    model.load_state_dict(best_state)
    return history, best_epoch, best_val_loss


@torch.inference_mode()
def predict_model(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int], list[dict[str, str]]]:
    model.eval()
    total_loss = 0.0
    total = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    metadata_all: list[dict[str, str]] = []

    for batch in loader:
        inputs, targets = _batch_to_device(batch, device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total += batch_size

        targets_all.extend(targets.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())

        for sample_id, path in zip(batch["sample_id"], batch["path"], strict=True):
            metadata_all.append({"sample_id": str(sample_id), "path": str(path)})

    if total == 0:
        raise RuntimeError("O DataLoader de teste não forneceu amostras.")

    return total_loss / total, targets_all, predictions_all, metadata_all
