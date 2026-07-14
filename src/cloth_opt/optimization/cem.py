from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CEMConfig:
    population_size: int = 24
    iterations: int = 8
    elite_fraction: float = 0.25
    initial_std_fraction: float = 0.20
    smoothing: float = 0.20
    minimum_std_fraction: float = 0.01
    seed: int = 0


@dataclass
class CEMResult:
    best_parameters: np.ndarray
    best_loss: float
    history: list[dict[str, float | int | list[float]]]


def optimize_cem(
    objective: Callable[[np.ndarray], float],
    initial_parameters: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    config: CEMConfig,
    iteration_callback: Callable[[dict[str, float | int | list[float]]], None] | None = None,
) -> CEMResult:
    initial_parameters = np.asarray(initial_parameters, dtype=np.float64)
    lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
    upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
    if not (initial_parameters.shape == lower_bounds.shape == upper_bounds.shape):
        raise ValueError("initial parameters and bounds must have identical shapes")
    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("every lower bound must be smaller than its upper bound")
    if config.population_size < 2 or config.iterations < 1:
        raise ValueError("CEM population_size >= 2 and iterations >= 1 are required")
    if not 0.0 < config.elite_fraction <= 1.0:
        raise ValueError("CEM elite_fraction must be in (0, 1]")
    if not 0.0 <= config.smoothing < 1.0:
        raise ValueError("CEM smoothing must be in [0, 1)")

    rng = np.random.default_rng(config.seed)
    span = upper_bounds - lower_bounds
    mean = np.clip(initial_parameters, lower_bounds, upper_bounds)
    std = span * config.initial_std_fraction
    minimum_std = span * config.minimum_std_fraction
    elite_count = max(1, int(round(config.population_size * config.elite_fraction)))
    best_parameters = mean.copy()
    best_loss = float(objective(best_parameters))
    history: list[dict[str, float | int | list[float]]] = []

    for iteration in range(config.iterations):
        samples = rng.normal(mean, std, size=(config.population_size, len(mean)))
        samples = np.clip(samples, lower_bounds, upper_bounds)
        # Retain the current mean so an iteration cannot lose its incumbent.
        samples[0] = mean
        losses = np.asarray([float(objective(sample)) for sample in samples])
        order = np.argsort(losses)
        elites = samples[order[:elite_count]]
        elite_mean = elites.mean(axis=0)
        elite_std = elites.std(axis=0)
        mean = config.smoothing * mean + (1.0 - config.smoothing) * elite_mean
        std = np.maximum(
            config.smoothing * std + (1.0 - config.smoothing) * elite_std,
            minimum_std,
        )

        if losses[order[0]] < best_loss:
            best_loss = float(losses[order[0]])
            best_parameters = samples[order[0]].copy()
        record: dict[str, float | int | list[float]] = {
            "iteration": iteration,
            "best_loss": best_loss,
            "iteration_best_loss": float(losses[order[0]]),
            "mean_loss": float(losses.mean()),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
        history.append(record)
        if iteration_callback is not None:
            iteration_callback(record)

    return CEMResult(best_parameters, best_loss, history)
