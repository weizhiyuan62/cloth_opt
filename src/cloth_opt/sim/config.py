from omegaconf import DictConfig, OmegaConf

from .engine import SceneConfig
from .env import ClothEnvConfig


def make_env_config(cfg: DictConfig) -> ClothEnvConfig:
    """Build the simulator config from the Hydra ``env`` group."""

    scene_values = OmegaConf.to_container(cfg.env.scene, resolve=True)
    assert isinstance(scene_values, dict)
    return ClothEnvConfig(
        scene=SceneConfig(**scene_values),
        n_substeps=int(cfg.env.n_substeps),
        reset_height=float(cfg.env.reset_height),
        pin_corners=bool(cfg.env.pin_corners),
    )
