from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# plotting imports are not needed when curves are disabled
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    # classification_report,
    # confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass(slots=True)
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    # confusion_matrix: list[list[int]]
    # classification_report: dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> EvalMetrics:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    # report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)

    return EvalMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        # confusion_matrix=confusion_matrix(y_true, y_pred).astype(int).tolist(),
        # classification_report=report,
    )


def plot_training_curves(
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    output_path: Path,
    title_prefix: str,
) -> None:
    # plotting functionality disabled per user request; keep signature for compatibility
    # if needed, recreate images using matplotlib and similar code above.
    pass