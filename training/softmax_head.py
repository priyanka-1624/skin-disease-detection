from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class SoftmaxClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass(slots=True)
class SoftmaxTrainConfig:
    batch_size: int
    learning_rate: float
    optimizer_name: str
    min_epochs: int
    max_epochs: int
    patience: int
    weight_decay: float = 1e-4
    momentum: float = 0.9


@dataclass(slots=True)
class SoftmaxTrainResult:
    best_val_loss: float
    best_val_acc: float
    best_epoch: int
    csv_log_path: Path
    checkpoint_path: Path
    resume_state_path: Path
    history_epochs: list[int]
    history_train_loss: list[float]
    history_train_acc: list[float]
    history_val_loss: list[float]
    history_val_acc: list[float]
    model: SoftmaxClassifier


def _build_optimizer(config: SoftmaxTrainConfig, model: nn.Module) -> Optimizer:
    if config.optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    total = 0
    correct = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += float(loss.item()) * x_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y_batch).sum().item())
        total += int(y_batch.size(0))

    return running_loss / max(total, 1), correct / max(total, 1)


def train_softmax_with_resume(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    config: SoftmaxTrainConfig,
    device: torch.device,
    best_model_path: Path,
    resume_checkpoint_path: Path,
    csv_log_path: Path,
) -> SoftmaxTrainResult:
    train_loader = DataLoader(
        FeatureDataset(train_features, train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        FeatureDataset(val_features, val_labels),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = SoftmaxClassifier(in_features=train_features.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(config, model)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    wait = 0

    history_epochs: list[int] = []
    history_train_loss: list[float] = []
    history_train_acc: list[float] = []
    history_val_loss: list[float] = []
    history_val_acc: list[float] = []

    if resume_checkpoint_path.exists():
        state = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

        start_epoch = int(state["epoch"]) + 1
        best_val_loss = float(state["best_val_loss"])
        best_val_acc = float(state["best_val_acc"])
        best_epoch = int(state["best_epoch"])
        wait = int(state["wait"])

        history_epochs = list(state["history_epochs"])
        history_train_loss = list(state["history_train_loss"])
        history_train_acc = list(state["history_train_acc"])
        history_val_loss = list(state["history_val_loss"])
        history_val_acc = list(state["history_val_acc"])

    if not csv_log_path.exists() or start_epoch == 1:
        with csv_log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    for epoch in range(start_epoch, config.max_epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, optimizer=None)

        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])

        history_epochs.append(epoch)
        history_train_loss.append(train_loss)
        history_train_acc.append(train_acc)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)

        with csv_log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{lr:.8f}",
            ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "wait": wait,
                "history_epochs": history_epochs,
                "history_train_loss": history_train_loss,
                "history_train_acc": history_train_acc,
                "history_val_loss": history_val_loss,
                "history_val_acc": history_val_acc,
            },
            resume_checkpoint_path,
        )

        if epoch >= config.min_epochs and wait >= config.patience:
            break

    if not best_model_path.exists():
        raise RuntimeError("Softmax training failed to produce best model checkpoint")

    best_state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_state)

    return SoftmaxTrainResult(
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        csv_log_path=csv_log_path,
        checkpoint_path=best_model_path,
        resume_state_path=resume_checkpoint_path,
        history_epochs=history_epochs,
        history_train_loss=history_train_loss,
        history_train_acc=history_train_acc,
        history_val_loss=history_val_loss,
        history_val_acc=history_val_acc,
        model=model,
    )


def predict_softmax(
    model: SoftmaxClassifier,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start : start + batch_size].astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        predictions.append(preds)
        probabilities.append(probs)

    return np.concatenate(predictions, axis=0), np.concatenate(probabilities, axis=0)
