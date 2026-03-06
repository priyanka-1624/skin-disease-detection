from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class PCAState:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray


class EigenPCA:
    """PCA via covariance eigendecomposition (no sklearn PCA usage)."""

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = int(n_components)
        self.state: PCAState | None = None

    def fit(self, x: np.ndarray) -> "EigenPCA":
        if x.ndim != 2:
            raise ValueError("Input must be 2D [n_samples, n_features]")

        n_samples, n_features = x.shape
        if n_samples < 2:
            raise ValueError("Need at least 2 samples for covariance")

        mean = x.mean(axis=0)
        centered = x - mean

        covariance = (centered.T @ centered) / (n_samples - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        k = min(self.n_components, n_features)
        top_eigenvalues = eigenvalues[:k]
        top_components = eigenvectors[:, :k].T

        total_variance = float(np.sum(np.maximum(eigenvalues, 0.0)))
        if total_variance <= 0:
            ratio = np.zeros_like(top_eigenvalues, dtype=np.float64)
        else:
            ratio = np.maximum(top_eigenvalues, 0.0) / total_variance

        self.state = PCAState(
            mean=mean.astype(np.float32),
            components=top_components.astype(np.float32),
            eigenvalues=top_eigenvalues.astype(np.float32),
            explained_variance_ratio=ratio.astype(np.float32),
        )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("PCA is not fitted")
        centered = x - self.state.mean
        return centered @ self.state.components.T

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def save(self, path: Path) -> None:
        if self.state is None:
            raise RuntimeError("Cannot save PCA before fitting")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.state.mean,
            components=self.state.components,
            eigenvalues=self.state.eigenvalues,
            explained_variance_ratio=self.state.explained_variance_ratio,
            n_components=np.array([self.n_components], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "EigenPCA":
        data = np.load(path)
        n_components = int(data["n_components"][0])
        instance = cls(n_components=n_components)
        instance.state = PCAState(
            mean=data["mean"],
            components=data["components"],
            eigenvalues=data["eigenvalues"],
            explained_variance_ratio=data["explained_variance_ratio"],
        )
        return instance
