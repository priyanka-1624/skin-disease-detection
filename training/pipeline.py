from __future__ import annotations

import csv
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset.manager import DatasetManifest, DermnetManager
from optimization.pca_eigen import EigenPCA
from optimization.swan import SwanFeatureSelector, SwanSelectionResult, save_swan_result
from training.config import Paths, TrainingConfig, ensure_dirs
from training.metrics import EvalMetrics, compute_metrics
# plot_training_curves removed; plotting disabled
from training.softmax_head import (
    SoftmaxTrainConfig,
    SoftmaxTrainResult,
    predict_softmax,
    train_softmax_with_resume,
)
from training.transformer import BackboneSpec, create_classifier_backbone, create_feature_extractor, load_backbone_for_features


@dataclass(slots=True)
class BackboneTrialConfig:
    batch_size: int
    learning_rate: float
    optimizer_name: str
    min_epochs: int
    max_epochs: int
    patience: int
    weight_decay: float = 1e-4
    momentum: float = 0.9


@dataclass(slots=True)
class BackboneTrialResult:
    trial_id: str
    best_val_loss: float
    best_val_acc: float
    best_epoch: int
    best_model_path: Path
    resume_path: Path
    csv_log_path: Path
    history_epochs: list[int]
    history_train_loss: list[float]
    history_train_acc: list[float]
    history_val_loss: list[float]
    history_val_acc: list[float]


@dataclass(slots=True)
class PipelineArtifacts:
    backbone_path: Path
    softmax_path: Path
    pca_path: Path
    swan_path: Path
    class_map_path: Path
    metrics_path: Path
    metadata_path: Path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. CPU training is disabled.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    free_mem, total_mem = torch.cuda.mem_get_info()

    free_gb = free_mem / (1024**3)
    total_gb = total_mem / (1024**3)

    print(f"GPU: {gpu_name}")
    print(f"GPU Memory - Free: {free_gb:.2f} GB | Total: {total_gb:.2f} GB")

    if free_gb < 2.0:
        raise RuntimeError("Insufficient free GPU memory to start full training.")

    return device


def _image_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_tf, eval_tf


def _build_loaders(
    paths: Paths,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, DataLoader], list[str]]:
    train_tf, eval_tf = _image_transforms(image_size)

    train_ds = datasets.ImageFolder(str(paths.dataset_splits / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(paths.dataset_splits / "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(paths.dataset_splits / "test"), transform=eval_tf)

    class_names = train_ds.classes
    if len(class_names) not in (6, 7):
        raise RuntimeError(f"Expected exactly 6 or 7 classes, found {len(class_names)}")

    if class_names != val_ds.classes or class_names != test_ds.classes:
        raise RuntimeError("Class order mismatch across train/val/test splits")

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    }

    return loaders, class_names


def _build_feature_loaders(
    paths: Paths,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, DataLoader], list[str]]:
    _, eval_tf = _image_transforms(image_size)

    train_ds = datasets.ImageFolder(str(paths.dataset_splits / "train"), transform=eval_tf)
    val_ds = datasets.ImageFolder(str(paths.dataset_splits / "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(paths.dataset_splits / "test"), transform=eval_tf)

    class_names = train_ds.classes

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    }

    return loaders, class_names


def _build_optimizer(model: nn.Module, config: BackboneTrialConfig) -> Optimizer:
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


def _load_history_from_csv(csv_path: Path) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    if not csv_path.exists():
        return [], [], [], [], []

    epochs: list[int] = []
    train_losses: list[float] = []
    train_accs: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                epochs.append(int(row["epoch"]))
                train_losses.append(float(row["train_loss"]))
                train_accs.append(float(row["train_acc"]))
                val_losses.append(float(row["val_loss"]))
                val_accs.append(float(row["val_acc"]))
            except (KeyError, TypeError, ValueError):
                continue

    return epochs, train_losses, train_accs, val_losses, val_accs


def _run_image_epoch(
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

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    return running_loss / max(total, 1), correct / max(total, 1)


def _train_backbone_trial(
    trial_id: str,
    loaders: dict[str, DataLoader],
    num_classes: int,
    config: BackboneTrialConfig,
    device: torch.device,
    paths: Paths,
    backbone_spec: BackboneSpec,
) -> BackboneTrialResult:
    model = create_classifier_backbone(num_classes=num_classes, spec=backbone_spec).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, config)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    trial_dir = paths.logs_checkpoints / f"backbone_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = trial_dir / "best_model.pth"
    resume_path = trial_dir / "resume_state.pt"
    csv_log_path = paths.logs_root / f"backbone_trial_{trial_id}.csv"

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

    if resume_path.exists():
        state = torch.load(resume_path, map_location=device)
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
        train_loss, train_acc = _run_image_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = _run_image_epoch(model, loaders["val"], criterion, device, optimizer=None)

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
            resume_path,
        )

        if epoch >= config.min_epochs and wait >= config.patience:
            break

    if not best_model_path.exists():
        raise RuntimeError(f"Backbone trial {trial_id} did not produce best model")

    return BackboneTrialResult(
        trial_id=trial_id,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        best_model_path=best_model_path,
        resume_path=resume_path,
        csv_log_path=csv_log_path,
        history_epochs=history_epochs,
        history_train_loss=history_train_loss,
        history_train_acc=history_train_acc,
        history_val_loss=history_val_loss,
        history_val_acc=history_val_acc,
    )


def _extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            vectors = model(images).detach().cpu().numpy()
            features.append(vectors)
            labels.append(y.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def _evaluate_softmax_model(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    batch_size: int,
    device: torch.device,
) -> EvalMetrics:
    preds, _ = predict_softmax(model=model, features=features, batch_size=batch_size, device=device)
    return compute_metrics(labels, preds, class_names)


def _tune_softmax(
    prefix: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    cfg: TrainingConfig,
    device: torch.device,
    paths: Paths,
    forced_config: SoftmaxTrainConfig | None = None,
) -> tuple[SoftmaxTrainResult, SoftmaxTrainConfig]:
    trial_configs: list[SoftmaxTrainConfig] = []

    if forced_config is not None:
        trial_configs = [forced_config]
    else:
        for lr in cfg.head_lr_candidates:
            for batch_size in cfg.head_batch_size_candidates:
                for optimizer_name in cfg.head_optimizer_candidates:
                    trial_configs.append(
                        SoftmaxTrainConfig(
                            batch_size=batch_size,
                            learning_rate=lr,
                            optimizer_name=optimizer_name,
                            min_epochs=cfg.min_epochs,
                            max_epochs=cfg.max_epochs,
                            patience=cfg.patience,
                        )
                    )

    best_result: SoftmaxTrainResult | None = None
    best_config: SoftmaxTrainConfig | None = None

    for index, trial_config in enumerate(trial_configs, start=1):
        trial_id = f"{prefix}_{index:02d}"
        best_model_path = paths.logs_checkpoints / f"{trial_id}_best.pth"
        resume_path = paths.logs_checkpoints / f"{trial_id}_resume.pt"
        csv_path = paths.logs_root / f"{trial_id}.csv"

        result = train_softmax_with_resume(
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            num_classes=num_classes,
            config=trial_config,
            device=device,
            best_model_path=best_model_path,
            resume_checkpoint_path=resume_path,
            csv_log_path=csv_path,
        )

        if best_result is None:
            best_result = result
            best_config = trial_config
        else:
            if result.best_val_acc > best_result.best_val_acc:
                best_result = result
                best_config = trial_config
            elif result.best_val_acc == best_result.best_val_acc and result.best_val_loss < best_result.best_val_loss:
                best_result = result
                best_config = trial_config

    if best_result is None or best_config is None:
        raise RuntimeError("Softmax hyperparameter tuning failed")

    return best_result, best_config


def run_full_training(backbone_name: str = "vit_base_patch16_224") -> PipelineArtifacts:
    cfg = TrainingConfig()
    paths = Paths()
    ensure_dirs(paths)
    _seed_everything(cfg.seed)

    device = _require_cuda()

    dataset_manager = DermnetManager(paths=paths, cfg=cfg)
    manifest: DatasetManifest = dataset_manager.prepare(force_rebuild=False)

    if len(manifest.selected_classes) not in (6, 7):
        raise RuntimeError("Prepared class count is not 6 or 7")

    backbone_spec = BackboneSpec(model_name=backbone_name, pretrained=True)

    reuse_checkpoint_env = os.getenv("REUSE_BACKBONE_CHECKPOINT", "").strip()
    if reuse_checkpoint_env:
        checkpoint_path = Path(reuse_checkpoint_env)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (paths.root / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise RuntimeError(f"Reuse backbone checkpoint not found: {checkpoint_path}")

        reuse_csv_env = os.getenv("REUSE_BACKBONE_CSV", "").strip()
        history_csv_path = paths.logs_root / "backbone_reused.csv"
        if reuse_csv_env:
            history_csv_path = Path(reuse_csv_env)
            if not history_csv_path.is_absolute():
                history_csv_path = (paths.root / history_csv_path).resolve()

        (
            history_epochs,
            history_train_loss,
            history_train_acc,
            history_val_loss,
            history_val_acc,
        ) = _load_history_from_csv(history_csv_path)

        if history_val_loss:
            best_idx = int(np.argmin(np.asarray(history_val_loss)))
            best_val_loss = float(history_val_loss[best_idx])
            best_val_acc = float(history_val_acc[best_idx])
            best_epoch = int(history_epochs[best_idx])
        else:
            best_val_loss = float("inf")
            best_val_acc = 0.0
            best_epoch = 0

        reuse_batch_size = int(os.getenv("REUSE_BACKBONE_BATCH_SIZE", str(cfg.batch_size_candidates[0])))
        reuse_lr = float(os.getenv("REUSE_BACKBONE_LR", str(cfg.lr_candidates[0])))
        reuse_optimizer = os.getenv("REUSE_BACKBONE_OPTIMIZER", cfg.optimizer_candidates[0])

        _, class_names = _build_loaders(
            paths=paths,
            image_size=cfg.image_size,
            batch_size=reuse_batch_size,
            num_workers=cfg.num_workers,
        )

        best_backbone_cfg = BackboneTrialConfig(
            batch_size=reuse_batch_size,
            learning_rate=reuse_lr,
            optimizer_name=reuse_optimizer,
            min_epochs=cfg.min_epochs,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
        )
        best_backbone_result = BackboneTrialResult(
            trial_id="reused",
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            best_model_path=checkpoint_path,
            resume_path=checkpoint_path.parent / "resume_state.pt",
            csv_log_path=history_csv_path,
            history_epochs=history_epochs,
            history_train_loss=history_train_loss,
            history_train_acc=history_train_acc,
            history_val_loss=history_val_loss,
            history_val_acc=history_val_acc,
        )
    else:
        backbone_trials: list[tuple[BackboneTrialResult, BackboneTrialConfig]] = []
        trial_counter = 0

        for lr in cfg.lr_candidates:
            for batch_size in cfg.batch_size_candidates:
                for optimizer_name in cfg.optimizer_candidates:
                    trial_counter += 1
                    trial_id = f"t{trial_counter:02d}"

                    loaders, class_names = _build_loaders(
                        paths=paths,
                        image_size=cfg.image_size,
                        batch_size=batch_size,
                        num_workers=cfg.num_workers,
                    )

                    trial_cfg = BackboneTrialConfig(
                        batch_size=batch_size,
                        learning_rate=lr,
                        optimizer_name=optimizer_name,
                        min_epochs=cfg.min_epochs,
                        max_epochs=cfg.max_epochs,
                        patience=cfg.patience,
                    )

                    trial_result = _train_backbone_trial(
                        trial_id=trial_id,
                        loaders=loaders,
                        num_classes=len(class_names),
                        config=trial_cfg,
                        device=device,
                        paths=paths,
                        backbone_spec=backbone_spec,
                    )

                    backbone_trials.append((trial_result, trial_cfg))

        if not backbone_trials:
            raise RuntimeError("No backbone trials executed")

        backbone_trials.sort(
            key=lambda item: (item[0].best_val_acc, -item[0].best_val_loss),
            reverse=True,
        )
        best_backbone_result, best_backbone_cfg = backbone_trials[0]

    backbone_best_path = paths.logs_artifacts / "backbone_best.pth"
    shutil.copy2(best_backbone_result.best_model_path, backbone_best_path)

    # history_epochs collected but plotting disabled
    # if best_backbone_result.history_epochs:
    #     plot_training_curves(
    #         epochs=best_backbone_result.history_epochs,
    #         train_losses=best_backbone_result.history_train_loss,
    #         val_losses=best_backbone_result.history_val_loss,
    #         train_accs=best_backbone_result.history_train_acc,
    #         val_accs=best_backbone_result.history_val_acc,
    #         output_path=paths.logs_plots / "backbone_training_curves.png",
    #         title_prefix="Backbone",
    #     )

    best_backbone_hparams = {
        "batch_size": best_backbone_cfg.batch_size,
        "learning_rate": best_backbone_cfg.learning_rate,
        "optimizer": best_backbone_cfg.optimizer_name,
        "best_val_acc": best_backbone_result.best_val_acc,
        "best_val_loss": best_backbone_result.best_val_loss,
        "best_epoch": best_backbone_result.best_epoch,
    }

    feature_loaders, class_names = _build_feature_loaders(
        paths=paths,
        image_size=cfg.image_size,
        batch_size=best_backbone_cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    classifier_state = torch.load(backbone_best_path, map_location="cpu")
    feature_extractor = create_feature_extractor(backbone_spec)
    load_backbone_for_features(feature_extractor, classifier_state)
    feature_extractor = feature_extractor.to(device)

    train_features, train_labels = _extract_features(feature_extractor, feature_loaders["train"], device)
    val_features, val_labels = _extract_features(feature_extractor, feature_loaders["val"], device)
    test_features, test_labels = _extract_features(feature_extractor, feature_loaders["test"], device)

    pca_components = min(cfg.pca_components, train_features.shape[1], max(2, train_features.shape[0] - 1))
    pca = EigenPCA(n_components=pca_components)
    train_pca = pca.fit_transform(train_features)
    val_pca = pca.transform(val_features)
    test_pca = pca.transform(test_features)

    baseline_result, baseline_head_cfg = _tune_softmax(
        prefix="baseline",
        train_features=train_pca,
        train_labels=train_labels,
        val_features=val_pca,
        val_labels=val_labels,
        num_classes=len(class_names),
        cfg=cfg,
        device=device,
        paths=paths,
    )

    baseline_val_metrics = _evaluate_softmax_model(
        model=baseline_result.model,
        features=val_pca,
        labels=val_labels,
        class_names=class_names,
        batch_size=baseline_head_cfg.batch_size,
        device=device,
    )

    best_swan_result: SwanSelectionResult | None = None
    best_swan_train_result: SoftmaxTrainResult | None = None
    best_swan_val_metrics: EvalMetrics | None = None

    for ratio in cfg.swan_feature_ratios:
        for seed_offset in [0, 17, 41]:
            selector = SwanFeatureSelector(
                population_size=cfg.swan_population,
                iterations=cfg.swan_iterations,
                feature_ratio=ratio,
                seed=cfg.seed + seed_offset,
            )
            swan_result = selector.optimize(train_pca, train_labels)

            train_opt = SwanFeatureSelector.transform(train_pca, swan_result.indices, swan_result.weights)
            val_opt = SwanFeatureSelector.transform(val_pca, swan_result.indices, swan_result.weights)

            forced = SoftmaxTrainConfig(
                batch_size=baseline_head_cfg.batch_size,
                learning_rate=baseline_head_cfg.learning_rate,
                optimizer_name=baseline_head_cfg.optimizer_name,
                min_epochs=cfg.min_epochs,
                max_epochs=cfg.max_epochs,
                patience=cfg.patience,
            )

            trial_prefix = f"swan_r{int(ratio*100)}_s{seed_offset}"
            swan_train_result, _ = _tune_softmax(
                prefix=trial_prefix,
                train_features=train_opt,
                train_labels=train_labels,
                val_features=val_opt,
                val_labels=val_labels,
                num_classes=len(class_names),
                cfg=cfg,
                device=device,
                paths=paths,
                forced_config=forced,
            )

            swan_val_metrics = _evaluate_softmax_model(
                model=swan_train_result.model,
                features=val_opt,
                labels=val_labels,
                class_names=class_names,
                batch_size=forced.batch_size,
                device=device,
            )

            if best_swan_train_result is None or swan_val_metrics.accuracy > best_swan_val_metrics.accuracy:
                best_swan_result = swan_result
                best_swan_train_result = swan_train_result
                best_swan_val_metrics = swan_val_metrics

    if best_swan_result is None or best_swan_train_result is None or best_swan_val_metrics is None:
        raise RuntimeError("Swan optimization stage did not produce a result")

    if best_swan_val_metrics.accuracy <= baseline_val_metrics.accuracy:
        print(
            "Warning: Swan optimization did not exceed baseline validation accuracy. "
            "Continuing with best Swan result for deployment artifacts."
        )

    swan_path = paths.logs_artifacts / "swan_result.npz"
    save_swan_result(swan_path, best_swan_result)

    pca_path = paths.logs_artifacts / "pca_state.npz"
    pca.save(pca_path)

    softmax_best_path = paths.logs_artifacts / "softmax_head_best.pth"
    shutil.copy2(best_swan_train_result.checkpoint_path, softmax_best_path)

    # plotting for softmax stage disabled
    # plot_training_curves(
    #     epochs=best_swan_train_result.history_epochs,
    #     train_losses=best_swan_train_result.history_train_loss,
    #     val_losses=best_swan_train_result.history_val_loss,
    #     train_accs=best_swan_train_result.history_train_acc,
    #     val_accs=best_swan_train_result.history_val_acc,
    #     output_path=paths.logs_plots / "softmax_training_curves.png",
    #     title_prefix="Softmax",
    # )

    val_opt_best = SwanFeatureSelector.transform(val_pca, best_swan_result.indices, best_swan_result.weights)
    test_opt_best = SwanFeatureSelector.transform(test_pca, best_swan_result.indices, best_swan_result.weights)

    final_val_metrics = _evaluate_softmax_model(
        model=best_swan_train_result.model,
        features=val_opt_best,
        labels=val_labels,
        class_names=class_names,
        batch_size=baseline_head_cfg.batch_size,
        device=device,
    )
    final_test_metrics = _evaluate_softmax_model(
        model=best_swan_train_result.model,
        features=test_opt_best,
        labels=test_labels,
        class_names=class_names,
        batch_size=baseline_head_cfg.batch_size,
        device=device,
    )

    class_map = {name: index for index, name in enumerate(class_names)}
    class_map_path = paths.logs_artifacts / "class_to_idx.json"
    class_map_path.write_text(json.dumps(class_map, indent=2), encoding="utf-8")

    metadata: dict[str, Any] = {
        "backbone_name": backbone_name,
        "image_size": cfg.image_size,
        "classes": class_names,
        "backbone_hyperparams": best_backbone_hparams,
        "softmax_hyperparams": {
            "batch_size": baseline_head_cfg.batch_size,
            "learning_rate": baseline_head_cfg.learning_rate,
            "optimizer": baseline_head_cfg.optimizer_name,
        },
        "manifest": asdict(manifest),
        "baseline_val_accuracy": baseline_val_metrics.accuracy,
        "swan_val_accuracy": final_val_metrics.accuracy,
    }

    metadata_path = paths.logs_artifacts / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_payload = {
        "validation": asdict(final_val_metrics),
        "test": asdict(final_test_metrics),
        "baseline_validation": asdict(baseline_val_metrics),
        "swan_best_fitness": best_swan_result.best_fitness,
        "swan_feature_ratio": best_swan_result.feature_ratio,
        "swan_selected_features": int(len(best_swan_result.indices)),
    }

    metrics_path = paths.logs_artifacts / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Validation Metrics")
    print(json.dumps(asdict(final_val_metrics), indent=2))
    print("Test Metrics")
    print(json.dumps(asdict(final_test_metrics), indent=2))

    return PipelineArtifacts(
        backbone_path=backbone_best_path,
        softmax_path=softmax_best_path,
        pca_path=pca_path,
        swan_path=swan_path,
        class_map_path=class_map_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
    )


if __name__ == "__main__":
    run_full_training()
