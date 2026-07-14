from pathlib import Path
import shutil

from omegaconf import DictConfig, OmegaConf

from cloth_opt.sim import ClothEnvConfig, SingleCameraRenderer, frames_to_video
from .rule_policy import (
    SymmetricFoldParameters,
    SymmetricFoldResult,
    SymmetricFoldPolicyConfig,
)


def make_symmetric_policy_config(policy_cfg: DictConfig) -> SymmetricFoldPolicyConfig:
    objective = OmegaConf.to_container(policy_cfg.objective, resolve=True)
    assert isinstance(objective, dict)
    return SymmetricFoldPolicyConfig(
        controlled_edge=str(policy_cfg.setup.controlled_edge),
        anchor_edge=str(policy_cfg.setup.anchor_edge),
        initial_settle_frames=int(policy_cfg.setup.initial_settle_frames),
        final_settle_frames=int(policy_cfg.setup.final_settle_frames),
        **objective,
    )


def make_fold_parameters(cfg: DictConfig) -> SymmetricFoldParameters:
    values = OmegaConf.to_container(cfg.policy.params, resolve=True)
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
