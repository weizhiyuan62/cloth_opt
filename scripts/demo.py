"""Hydra-managed, position-control-only ClothOpt rollout."""

import json
import logging
from pathlib import Path
import shutil
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from cloth_opt.sim import (
    ClothAction,
    ClothEnv,
    ClothEnvConfig,
    SceneConfig,
    SingleCameraRenderer,
    frames_to_video,
)


logger = logging.getLogger(__name__)


def interpolate_waypoints(waypoints: np.ndarray, frames_per_segment: int) -> np.ndarray:
    if frames_per_segment <= 0:
        raise ValueError("trajectory.frames_per_segment must be positive")
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) < 2:
        raise ValueError("trajectory.offset_waypoints must have shape (N, 3), N >= 2")

    segments = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        alpha = np.linspace(0.0, 1.0, frames_per_segment, endpoint=False)[:, None]
        segments.append(start[None, :] * (1.0 - alpha) + end[None, :] * alpha)
    segments.append(waypoints[-1:])
    return np.concatenate(segments, axis=0)


def get_controlled_indices(env: ClothEnv, edge: str) -> np.ndarray:
    scene = env.config.scene
    if edge == "top":
        coordinates = [(0, column) for column in range(scene.width)]
    elif edge == "bottom":
        coordinates = [(scene.height - 1, column) for column in range(scene.width)]
    elif edge == "left":
        coordinates = [(row, 0) for row in range(scene.height)]
    elif edge == "right":
        coordinates = [(row, scene.width - 1) for row in range(scene.height)]
    else:
        raise ValueError(f"unsupported controlled edge: {edge}")
    return np.asarray([env.engine.grid_index(row, column) for row, column in coordinates], dtype=np.int64)


def make_env_config(cfg: DictConfig) -> ClothEnvConfig:
    scene_values = OmegaConf.to_container(cfg.env.scene, resolve=True)
    assert isinstance(scene_values, dict)
    return ClothEnvConfig(
        scene=SceneConfig(**scene_values),
        n_substeps=int(cfg.env.n_substeps),
        reset_height=float(cfg.env.reset_height),
        pin_corners=bool(cfg.env.pin_corners),
    )


def make_renderer(cfg: DictConfig, env: ClothEnv) -> SingleCameraRenderer | None:
    if not cfg.render.enabled:
        return None
    if cfg.render.backend != "matplotlib":
        raise ValueError(f"unsupported render backend: {cfg.render.backend}")

    scene = env.config.scene
    extent = max((scene.width - 1) * scene.spacing, (scene.height - 1) * scene.spacing)
    return SingleCameraRenderer(
        env.engine.mesh.triangles,
        bounds=((-0.4, extent + 0.4), (-0.4, extent + 0.4), (0.0, 1.2)),
        elevation=float(cfg.render.camera.elevation),
        azimuth=float(cfg.render.camera.azimuth),
    )


@hydra.main(config_path="../configs", config_name="demo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.seed))
    run_dir = Path.cwd()
    logger.info("run directory: %s", run_dir)
    logger.info("seed: %d", cfg.seed)
    OmegaConf.save(cfg, run_dir / ".hydra" / "resolved.yaml", resolve=True)

    env = ClothEnv(make_env_config(cfg))
    observation = env.reset()
    controlled_indices = get_controlled_indices(env, str(cfg.trajectory.controlled_edge))
    initial_targets = observation["positions"][controlled_indices].copy()
    waypoints = np.asarray(cfg.trajectory.offset_waypoints, dtype=np.float64)
    offsets = interpolate_waypoints(waypoints, int(cfg.trajectory.frames_per_segment))

    frame_dir = run_dir / "frames"
    renderer = make_renderer(cfg, env)
    positions_all = [observation["positions"].copy()]
    velocities_all = [observation["velocities"].copy()]
    targets_all = []
    times = [observation["time"]]
    action_records: list[dict[str, Any]] = []

    try:
        for frame_index, offset in enumerate(offsets):
            targets = initial_targets + offset
            action = ClothAction(
                mode="position",
                vertex_indices=controlled_indices,
                values=targets,
                params={
                    "gain": float(cfg.trajectory.position_gain),
                    "max_force": float(cfg.trajectory.max_force),
                },
            )
            observation = env.step(action)
            positions_all.append(observation["positions"].copy())
            velocities_all.append(observation["velocities"].copy())
            targets_all.append(targets.copy())
            times.append(observation["time"])
            action_records.append(action.to_dict())

            if renderer is not None:
                renderer.save_frame(
                    observation["positions"],
                    controlled_indices,
                    targets,
                    frame_dir / f"frame_{frame_index:05d}.png",
                    title=f"Position control  t={observation['time']:.3f}s",
                )
            if frame_index % 20 == 0 or frame_index == len(offsets) - 1:
                logger.info("rollout frame %d/%d", frame_index + 1, len(offsets))

        if cfg.save.final_mesh:
            env.export_mesh(run_dir / "final.obj")
    finally:
        if renderer is not None:
            renderer.close()
        env.close()

    positions_array = np.asarray(positions_all)
    velocities_array = np.asarray(velocities_all)
    targets_array = np.asarray(targets_all)
    if cfg.save.trajectory:
        np.savez_compressed(
            run_dir / "trajectory.npz",
            time=np.asarray(times),
            positions=positions_array,
            velocities=velocities_array,
            target_positions=targets_array,
            controlled_indices=controlled_indices,
        )
    if cfg.save.actions:
        (run_dir / "actions.json").write_text(json.dumps(action_records, indent=2), encoding="utf-8")

    metrics = {
        "rollout_frames": len(offsets),
        "simulation_time": float(times[-1]),
        "final_center": positions_array[-1].mean(axis=0).tolist(),
        "final_max_speed": float(np.linalg.norm(velocities_array[-1], axis=1).max()),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if renderer is not None:
        frames_to_video(frame_dir, run_dir / "trajectory.mp4", int(cfg.render.fps))
        if not cfg.render.keep_frames:
            shutil.rmtree(frame_dir)

    logger.info("controlled vertices: %s", controlled_indices.tolist())
    logger.info("metrics: %s", metrics)
    logger.info("rollout complete: %s", run_dir)


if __name__ == "__main__":
    main()
