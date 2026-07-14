from pathlib import Path
import shutil

from omegaconf import DictConfig, OmegaConf

from cloth_opt.sim import ClothEnvConfig, SingleCameraRenderer, frames_to_video
from .rule_policy import (
    DiagonalFoldParameters,
    DiagonalFoldResult,
    DiagonalFoldPolicyConfig,
)

def make_diagonal_policy_config(policy_cfg: DictConfig) -> DiagonalFoldPolicyConfig:
    objective = OmegaConf.to_container(policy_cfg.objective, resolve=True)
    assert isinstance(objective, dict)
    return DiagonalFoldPolicyConfig(
        fold_diagonal=str(policy_cfg.setup.fold_diagonal),
        controlled_corner=str(policy_cfg.setup.controlled_corner),
        pin_fold_endpoints=bool(policy_cfg.setup.pin_fold_endpoints),
        initial_settle_frames=int(policy_cfg.setup.initial_settle_frames),
        final_settle_frames=int(policy_cfg.setup.final_settle_frames),
        **objective,
    )


def make_diagonal_parameters(cfg: DictConfig) -> DiagonalFoldParameters:
    values = OmegaConf.to_container(cfg.policy.params, resolve=True)
    assert isinstance(values, dict)
    return DiagonalFoldParameters.from_mapping(values)


def render_diagonal_result(
    result: DiagonalFoldResult,
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
        for frame_index, (positions, target, phase) in enumerate(
            zip(result.positions[1:], result.target_positions, result.phases)
        ):
            renderer.save_frame(
                positions,
                result.controlled_indices,
                target,
                frame_dir / f"frame_{frame_index:05d}.png",
                title=f"Diagonal fold: {phase}",
            )
    finally:
        renderer.close()
    frames_to_video(frame_dir, output_dir / "trajectory.mp4", int(render_cfg.fps))
    if not render_cfg.keep_frames:
        shutil.rmtree(frame_dir)
