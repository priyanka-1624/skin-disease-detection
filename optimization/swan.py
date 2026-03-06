from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class SwanSelectionResult:
    indices: np.ndarray
    weights: np.ndarray
    best_fitness: float
    history: list[float]
    feature_ratio: float


class SwanFeatureSelector:
    """
    Swan Optimization Algorithm for feature selection.

    Each swan position is a continuous feature score vector in [0, 1]^d.
    Top-k score indices are selected as active features, where k = round(feature_ratio * d).
    Fitness maximizes class separability via Fisher ratio in selected weighted space.
    """

    def __init__(
        self,
        population_size: int,
        iterations: int,
        feature_ratio: float,
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1,
        seed: int = 42,
    ) -> None:
        if not 0.0 < feature_ratio < 1.0:
            raise ValueError("feature_ratio must be in (0, 1)")
        self.population_size = population_size
        self.iterations = iterations
        self.feature_ratio = feature_ratio
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seed = seed

    def optimize(self, features: np.ndarray, labels: np.ndarray) -> SwanSelectionResult:
        if features.ndim != 2:
            raise ValueError("features must be 2D")

        rng = np.random.default_rng(self.seed)
        n_features = features.shape[1]
        k = max(2, int(round(self.feature_ratio * n_features)))

        population = rng.uniform(0.15, 1.0, size=(self.population_size, n_features)).astype(np.float64)
        fitness = np.array([self._fitness(features, labels, p, k) for p in population], dtype=np.float64)

        best_idx = int(np.argmax(fitness))
        best_pos = population[best_idx].copy()
        best_fit = float(fitness[best_idx])
        history = [best_fit]

        for iteration in range(self.iterations):
            order = np.argsort(fitness)
            elite = population[order[-max(2, self.population_size // 4) :]]
            elite_center = elite.mean(axis=0)
            worst = population[order[0]]

            decay = 1.0 - (iteration / max(1, self.iterations))

            for i in range(self.population_size):
                pos = population[i]
                r1 = rng.random(n_features)
                r2 = rng.random(n_features)
                r3 = rng.random(n_features)

                glide = self.alpha * r1 * (best_pos - pos)
                flock = self.beta * r2 * (elite_center - pos)
                repel = self.gamma * r3 * (pos - worst)

                candidate = np.clip(pos + decay * (glide + flock + repel), 0.0, 1.0)

                if rng.random() < 0.2:
                    candidate = np.clip(candidate + rng.normal(0.0, 0.035, size=n_features), 0.0, 1.0)

                candidate_fit = self._fitness(features, labels, candidate, k)
                if candidate_fit >= fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fit

            epoch_best_idx = int(np.argmax(fitness))
            epoch_best_fit = float(fitness[epoch_best_idx])
            if epoch_best_fit > best_fit:
                best_fit = epoch_best_fit
                best_pos = population[epoch_best_idx].copy()

            history.append(best_fit)

        indices = np.argsort(best_pos)[::-1][:k]
        selected = best_pos[indices]
        l1 = np.linalg.norm(selected, ord=1)
        if l1 <= 0:
            weights = np.ones_like(selected, dtype=np.float32)
        else:
            weights = (selected / l1 * len(selected)).astype(np.float32)

        return SwanSelectionResult(
            indices=indices.astype(np.int32),
            weights=weights,
            best_fitness=best_fit,
            history=history,
            feature_ratio=self.feature_ratio,
        )

    @staticmethod
    def transform(features: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
        selected = features[:, indices]
        return selected * weights

    @staticmethod
    def _fitness(features: np.ndarray, labels: np.ndarray, position: np.ndarray, k: int) -> float:
        indices = np.argsort(position)[::-1][:k]
        weights = position[indices]
        weighted = features[:, indices] * weights

        global_mean = weighted.mean(axis=0)
        classes = np.unique(labels)

        between = 0.0
        within = 0.0

        for cls in classes:
            cls_samples = weighted[labels == cls]
            if cls_samples.size == 0:
                continue
            cls_mean = cls_samples.mean(axis=0)
            n_cls = cls_samples.shape[0]
            between += n_cls * float(np.sum((cls_mean - global_mean) ** 2))
            within += float(np.sum((cls_samples - cls_mean) ** 2))

        sparsity_bonus = k / max(1, features.shape[1])
        return between / (within + 1e-9) + 0.05 * sparsity_bonus


def save_swan_result(path: Path, result: SwanSelectionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=result.indices,
        weights=result.weights,
        best_fitness=np.array([result.best_fitness], dtype=np.float64),
        history=np.array(result.history, dtype=np.float64),
        feature_ratio=np.array([result.feature_ratio], dtype=np.float64),
    )


def load_swan_result(path: Path) -> SwanSelectionResult:
    data = np.load(path)
    return SwanSelectionResult(
        indices=data["indices"],
        weights=data["weights"],
        best_fitness=float(data["best_fitness"][0]),
        history=data["history"].tolist(),
        feature_ratio=float(data["feature_ratio"][0]),
    )
