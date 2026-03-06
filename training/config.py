from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Paths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def kaggle_json(self) -> Path:
        return self.root / "kaggle.json"

    @property
    def dataset_root(self) -> Path:
        return self.root / "dataset"

    @property
    def dataset_raw(self) -> Path:
        return self.dataset_root / "raw"

    @property
    def dataset_clean(self) -> Path:
        return self.dataset_root / "clean"

    @property
    def dataset_processed(self) -> Path:
        return self.dataset_root / "processed"

    @property
    def dataset_splits(self) -> Path:
        return self.dataset_root / "splits"

    @property
    def logs_root(self) -> Path:
        return self.root / "logs"

    @property
    def logs_artifacts(self) -> Path:
        return self.logs_root / "artifacts"

    @property
    def logs_checkpoints(self) -> Path:
        return self.logs_root / "checkpoints"

    @property
    def logs_plots(self) -> Path:
        return self.logs_root / "plots"


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 42
    image_size: int = 224
    num_workers: int = 4
    class_count: int = 7
    min_images_per_class: int = 500
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    min_epochs: int = 50
    max_epochs: int = 300
    patience: int = 18

    lr_candidates: tuple[float, ...] = (3e-5, 1e-4, 2e-4)
    batch_size_candidates: tuple[int, ...] = (8, 12)
    optimizer_candidates: tuple[str, ...] = ("adamw", "sgd")

    head_lr_candidates: tuple[float, ...] = (1e-3, 3e-3, 5e-3)
    head_batch_size_candidates: tuple[int, ...] = (64, 128)
    head_optimizer_candidates: tuple[str, ...] = ("adamw", "sgd")

    pca_components: int = 256
    swan_population: int = 30
    swan_iterations: int = 80
    swan_feature_ratios: tuple[float, ...] = (0.45, 0.55, 0.65)


def ensure_dirs(paths: Paths) -> None:
    for directory in [
        paths.dataset_raw,
        paths.dataset_clean,
        paths.dataset_processed,
        paths.dataset_splits,
        paths.logs_root,
        paths.logs_artifacts,
        paths.logs_checkpoints,
        paths.logs_plots,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
