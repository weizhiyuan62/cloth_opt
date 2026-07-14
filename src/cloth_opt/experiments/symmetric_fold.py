from pathlib import Path
import shutil

from omegaconf import DictConfig, OmegaConf

from cloth_opt.engine import SceneConfig
from cloth_opt.env import ClothEnvConfig
from cloth_opt.render import SingleCameraRenderer, frames_to_video
from cloth_opt.tasks import (
    SymmetricFoldParameters,
    SymmetricFoldResult,
    SymmetricFoldTaskConfig,
)


def make_env_config(cfg: DictConfig) -> ClothEnvConfig:
    scene_values = OmegaConf.to_container(cfg.env.scene, resolve=True)
    assert isinstance(scene_values, dict)
    return ClothEnvConfig(
        scene=SceneConfig(**scene_values),
        n_substeps=int(cfg.env.n_substeps),
        reset_height=float(cfg.env.reset_height),
        pin_corners=bool(cfg.env.pin_corners),
    )


def make_fold_task_config(traj_cfg: DictConfig) -> SymmetricFoldTaskConfig:
    objective = OmegaConf.to_container(traj_cfg.objective, resolve=True)
    assert isinstance(objective, dict)
    return SymmetricFoldTaskConfig(
        controlled_edge=str(traj_cfg.task.controlled_edge),
        anchor_edge=str(traj_cfg.task.anchor_edge),
        initial_settle_frames=int(traj_cfg.task.initial_settle_frames),
        final_settle_frames=int(traj_cfg.task.final_settle_frames),
        **objective,
    )


def make_fold_parameters(cfg: DictConfig) -> SymmetricFoldParameters:
    values = OmegaConf.to_container(cfg.traj_cfg.rules, resolve=True)
    assert isinstance(values, dict)
    return SymmetricFoldParameters.from_mapping(values)


def render_fold_result(
    result: SymmetricFoldResult,
    env_config: ClothEnvConfig,
    render_cfg: DictConfig,
    output_dir: str | Path,
) -> None:
    if not render_cfg.enabled:
        return
    if result.positions is None or result.target_positions is None or result.phases is None:
        raise ValueError("rendering requires a recorded rollout")
    if render_cfg.backend != "matplotlib":
        raise ValueError(f"unsupported render backend: {render_cfg.backend}")

    output_dir = Path(output_dir)
    frame_dir = output_dir / "frames"
    scene = env_config.scene
    extent = max((scene.width - 1) * scene.spacing, (scene.height - 1) * scene.spacing)
    renderer = SingleCameraRenderer(
        result.triangles,
        bounds=((-0.4, extent + 0.4), (-0.4, extent + 0.4), (0.0, 1.2)),
        elevation=float(render_cfg.camera.elevation),
        azimuth=float(render_cfg.camera.azimuth),
    )
    try:
        for frame_index, (positions, targets, phase) in enumerate(
            zip(result.positions[1:], result.target_positions, result.phases)
        ):
            renderer.save_frame(
                positions,
                result.controlled_indices,
                targets,
                frame_dir / f"frame_{frame_index:05d}.png",
                title=f"Symmetric fold: {phase}",
            )
    finally:
        renderer.close()
    frames_to_video(frame_dir, output_dir / "trajectory.mp4", int(render_cfg.fps))
    if not render_cfg.keep_frames:
        shutil.rmtree(frame_dir)
