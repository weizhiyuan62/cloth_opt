"""Run and evaluate the surface-constrained diagonal-fold state machine."""

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from cloth_opt.policy.diagonal_fold import (
    DiagonalFoldPolicy,
    make_diagonal_parameters,
    make_diagonal_policy_config,
    render_diagonal_result,
)
from cloth_opt.sim import make_env_config


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="diagonal_fold", version_base="1.3")
def main(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.seed))
    run_dir = Path.cwd()
    OmegaConf.save(cfg, run_dir / ".hydra" / "resolved.yaml", resolve=True)

    env_config = make_env_config(cfg)
    policy_config = make_diagonal_policy_config(cfg.policy)
    parameters = make_diagonal_parameters(cfg)
    policy = DiagonalFoldPolicy(env_config, policy_config)

    logger.info(
        "running diagonal-fold baseline: diagonal=%s, controlled_corner=%s, parameters=%s",
        policy_config.fold_diagonal,
        policy_config.controlled_corner,
        parameters,
    )
    result = policy.rollout(parameters, record=True)
    result.save(run_dir)
    render_diagonal_result(result, env_config, cfg.render, run_dir)

    logger.info("controlled moving-boundary vertices: %d", len(result.controlled_indices))
    logger.info("loss: %.6f", result.loss)
    logger.info("metrics: %s", result.metrics)
    logger.info("saved rollout: %s", run_dir)


if __name__ == "__main__":
    main()
