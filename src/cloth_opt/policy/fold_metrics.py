from typing import Callable

import numpy as np


def evaluate_fold_trajectory(
    trajectory_positions: np.ndarray,
    final_velocities: np.ndarray,
    targets: np.ndarray,
    moving: np.ndarray,
    stationary: np.ndarray,
    width: int,
    height: int,
    spacing: float,
    control_dt: float,
    layer_gap: float,
    grid_index: Callable[[int, int], int],
) -> dict[str, float | bool]:
    """Size-normalized proxies for animation quality and physical integrity."""

    final_positions = trajectory_positions[-1]
    desired = final_positions[stationary].copy()
    desired[:, 1] += layer_gap
    alignment = float(np.linalg.norm(final_positions[moving] - desired, axis=1).mean())

    edge_pairs: list[tuple[int, int]] = []
    for row in range(height):
        for column in range(width):
            index = grid_index(row, column)
            if column + 1 < width:
                edge_pairs.append((index, grid_index(row, column + 1)))
            if row + 1 < height:
                edge_pairs.append((index, grid_index(row + 1, column)))
    edges = np.asarray(edge_pairs, dtype=np.int64)
    edge_vectors = trajectory_positions[:, edges[:, 0]] - trajectory_positions[:, edges[:, 1]]
    stretch = np.abs(np.linalg.norm(edge_vectors, axis=2) / spacing - 1.0)
    mean_stretch = float(stretch.mean())
    max_stretch = float(stretch.max())

    if len(targets) >= 2:
        target_velocity = np.diff(targets, axis=0) / control_dt
        # Integral of squared commanded speed, averaged over controlled vertices.
        control_effort = float(
            np.square(target_velocity).sum(axis=2).sum() * control_dt / targets.shape[1]
        )
    else:
        control_effort = 0.0

    if len(targets) >= 3:
        second_difference = targets[2:] - 2.0 * targets[1:-1] + targets[:-2]
        smoothness = float(np.square(second_difference).mean())
    else:
        smoothness = 0.0

    penetration = float(
        np.maximum(
            final_positions[stationary, 1] + layer_gap - final_positions[moving, 1],
            0.0,
        ).mean()
    )
    speeds = np.linalg.norm(final_velocities, axis=1)
    mean_speed = float(speeds.mean())
    max_speed = float(speeds.max())

    extent = max((width - 1) * spacing, (height - 1) * spacing)
    alignment_score = float(np.exp(-alignment / max(0.08 * extent, 1e-8)))
    distortion_score = float(np.exp(-max_stretch / 0.25))
    smoothness_score = float(
        np.exp(-np.sqrt(smoothness) / max(0.01 * extent, 1e-8))
    )
    intersection_score = float(
        np.exp(-penetration / max(layer_gap, 0.01, 1e-8))
    )
    aesthetic_quality = 100.0 * (
        0.35 * alignment_score
        + 0.30 * distortion_score
        + 0.20 * smoothness_score
        + 0.15 * intersection_score
    )

    return {
        "alignment_error": alignment,
        "control_effort_proxy": control_effort,
        "trajectory_mean_stretch": mean_stretch,
        "trajectory_max_stretch": max_stretch,
        "target_smoothness": smoothness,
        "layer_penetration_proxy": penetration,
        "terminal_mean_speed": mean_speed,
        "terminal_max_speed": max_speed,
        "aesthetic_quality": float(aesthetic_quality),
        "alignment_score": alignment_score,
        "distortion_score": distortion_score,
        "smoothness_score": smoothness_score,
        "intersection_score": intersection_score,
    }
