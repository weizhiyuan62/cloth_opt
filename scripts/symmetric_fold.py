"""Run and evaluate the hand-initialized symmetric-fold state machine."""

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from cloth_opt.policy.symmetric_fold import (
    SymmetricFoldPolicy,
    make_fold_parameters,
    make_symmetric_policy_config,
    render_fold_result,
)
from cloth_opt.sim import make_env_config


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="symmetric_fold", version_base="1.3")
def main(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.seed))
    run_dir = Path.cwd()
    OmegaConf.save(cfg, run_dir / ".hydra" / "resolved.yaml", resolve=True)

    env_config = make_env_config(cfg)
    policy = SymmetricFoldPolicy(env_config, make_symmetric_policy_config(cfg.policy))
    parameters = make_fold_parameters(cfg)
    logger.info("running symmetric-fold baseline with %s", parameters)
    result = policy.rollout(parameters, record=True)
    result.save(run_dir)
    render_fold_result(result, env_config, cfg.render, run_dir)

    logger.info("loss: %.6f", result.loss)
    logger.info("metrics: %s", result.metrics)
    logger.info("saved rollout: %s", run_dir)


if __name__ == "__main__":
    main()
