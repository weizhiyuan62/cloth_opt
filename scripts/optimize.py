"""Optimize symmetric-fold phase parameters with the cross-entropy method."""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from cloth_opt.policy.symmetric_fold import (
    SymmetricFoldParameters,
    SymmetricFoldPolicy,
    make_fold_parameters,
    make_symmetric_policy_config,
    render_fold_result,
)
from cloth_opt.optimization import CEMConfig, optimize_cem
from cloth_opt.sim import make_env_config


logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="optimize", version_base="1.3")
def main(cfg: DictConfig) -> None:
    np.random.seed(int(cfg.seed))
    run_dir = Path.cwd()
    OmegaConf.save(cfg, run_dir / ".hydra" / "resolved.yaml", resolve=True)

    env_config = make_env_config(cfg)
    policy = SymmetricFoldPolicy(env_config, make_symmetric_policy_config(cfg.policy))
    initial = make_fold_parameters(cfg)
    parameter_names = list(cfg.optimizer.parameter_names)
    initial_vector = np.asarray([getattr(initial, name) for name in parameter_names], dtype=np.float64)
    lower_bounds = np.asarray(cfg.optimizer.lower_bounds, dtype=np.float64)
    upper_bounds = np.asarray(cfg.optimizer.upper_bounds, dtype=np.float64)

    def decode(vector: np.ndarray) -> SymmetricFoldParameters:
        values = {name: value for name, value in zip(parameter_names, vector)}
        return SymmetricFoldParameters.from_mapping(values)

    evaluation_count = 0

    def objective(vector: np.ndarray) -> float:
        nonlocal evaluation_count
        evaluation_count += 1
        try:
            result = policy.rollout(decode(vector), record=False)
            loss = result.loss
            if not np.isfinite(loss):
                logger.warning("rollout %d returned non-finite loss", evaluation_count)
                loss = 1e6
        except Exception:
            logger.exception("rollout %d failed", evaluation_count)
            loss = 1e6
        logger.info("evaluation=%d loss=%.6f", evaluation_count, loss)
        return loss

    cem_config = CEMConfig(
        population_size=int(cfg.optimizer.population_size),
        iterations=int(cfg.optimizer.iterations),
        elite_fraction=float(cfg.optimizer.elite_fraction),
        initial_std_fraction=float(cfg.optimizer.initial_std_fraction),
        smoothing=float(cfg.optimizer.smoothing),
        minimum_std_fraction=float(cfg.optimizer.minimum_std_fraction),
        seed=int(cfg.optimizer.seed),
    )

    def on_iteration(record: dict) -> None:
        logger.info(
            "iteration=%d best=%.6f iteration_best=%.6f mean=%.6f",
            record["iteration"], record["best_loss"],
            record["iteration_best_loss"], record["mean_loss"],
        )
        (run_dir / "optimization_history.json").write_text(
            json.dumps(history_snapshot + [record], indent=2), encoding="utf-8"
        )
        history_snapshot.append(record)

    history_snapshot: list[dict] = []
    optimization = optimize_cem(
        objective, initial_vector, lower_bounds, upper_bounds, cem_config, on_iteration
    )
    best_parameters = decode(optimization.best_parameters)
    best_result = policy.rollout(best_parameters, record=True)
    best_dir = run_dir / "best"
    best_result.save(best_dir)
    render_fold_result(best_result, env_config, cfg.render, best_dir)
    (run_dir / "optimization_summary.json").write_text(
        json.dumps(
            {
                "parameter_names": parameter_names,
                "best_vector": optimization.best_parameters.tolist(),
                "best_loss_during_search": optimization.best_loss,
                "best_metrics_rerun": best_result.metrics,
                "evaluations": evaluation_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("best parameters: %s", best_parameters)
    logger.info("best metrics: %s", best_result.metrics)
    logger.info("optimization complete: %s", run_dir)


if __name__ == "__main__":
    main()
