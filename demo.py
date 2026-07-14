"""Single-rollout, position-control-only ClothOpt example.

This mirrors the useful core of FoldVLA's rollout workflow: create one run
directory, save its resolved configuration and actions, execute a rollout,
store numerical trajectories, render a fixed camera, and encode an MP4.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil

import numpy as np

from cloth_opt import ClothAction, ClothEnv, ClothEnvConfig, SceneConfig
from cloth_opt.render import SingleCameraRenderer, frames_to_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/position_demo.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true", help="skip frame rendering and MP4 encoding")
    parser.add_argument("--keep-frames", action="store_true", help="keep PNG files after encoding")
    return parser.parse_args()


def interpolate_waypoints(waypoints: np.ndarray, frames_per_segment: int) -> np.ndarray:
    """Piecewise-linear targets, excluding duplicate segment endpoints."""

    segments = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        alpha = np.linspace(0.0, 1.0, frames_per_segment, endpoint=False)[:, None]
        segments.append(start[None, :] * (1.0 - alpha) + end[None, :] * alpha)
    segments.append(waypoints[-1:])
    return np.concatenate(segments, axis=0)


def main() -> None:
    args = parse_args()
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    scene = SceneConfig(**raw_config["scene"])
    env_config = ClothEnvConfig(
        scene=scene,
        n_substeps=int(raw_config["n_substeps"]),
        reset_height=float(raw_config["reset_height"]),
        pin_corners=bool(raw_config["pin_corners"]),
    )
    output_dir = args.output_dir or Path("outputs") / dt.datetime.now().strftime("position_demo_%Y%m%d_%H%M%S")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    frame_dir = output_dir / "frames"
    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(json.dumps(raw_config, indent=2), encoding="utf-8")

    env = ClothEnv(env_config)
    observation = env.reset()

    # Control the complete bottom edge. Each vertex keeps its relative X/Z
    # position while the shared offset follows the trajectory below.
    controlled_indices = np.array(
        [env.engine.grid_index(scene.height - 1, column) for column in range(scene.width)],
        dtype=np.int64,
    )
    initial_targets = observation["positions"][controlled_indices].copy()
    offset_waypoints = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.00, 0.20, 0.15],
            [0.20, 0.25, 0.15],
            [-0.15, 0.18, -0.10],
            [0.00, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    offsets = interpolate_waypoints(offset_waypoints, int(raw_config["frames_per_segment"]))

    renderer = None
    if not args.no_video:
        extent = max((scene.width - 1) * scene.spacing, (scene.height - 1) * scene.spacing)
        renderer = SingleCameraRenderer(
            env.engine.mesh.triangles,
            bounds=((-0.4, extent + 0.4), (-0.4, extent + 0.4), (0.0, 1.2)),
            elevation=float(raw_config["camera"]["elevation"]),
            azimuth=float(raw_config["camera"]["azimuth"]),
        )

    positions_all = [observation["positions"].copy()]
    velocities_all = [observation["velocities"].copy()]
    targets_all = []
    times = [observation["time"]]
    action_records = []

    try:
        for frame_index, offset in enumerate(offsets):
            targets = initial_targets + offset
            action = ClothAction(
                mode="position",
                vertex_indices=controlled_indices,
                values=targets,
                params={
                    "gain": float(raw_config["position_gain"]),
                    "max_force": float(raw_config["max_force"]),
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
        env.export_mesh(output_dir / "final.obj")
    finally:
        if renderer is not None:
            renderer.close()
        env.close()

    np.savez_compressed(
        output_dir / "trajectory.npz",
        time=np.asarray(times),
        positions=np.asarray(positions_all),
        velocities=np.asarray(velocities_all),
        target_positions=np.asarray(targets_all),
        controlled_indices=controlled_indices,
    )
    (output_dir / "actions.json").write_text(json.dumps(action_records, indent=2), encoding="utf-8")
    if renderer is not None:
        frames_to_video(frame_dir, output_dir / "trajectory.mp4", int(raw_config["video_fps"]))
        if not args.keep_frames:
            shutil.rmtree(frame_dir)

    print(f"rollout frames: {len(offsets)}")
    print(f"controlled vertices: {controlled_indices.tolist()}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
