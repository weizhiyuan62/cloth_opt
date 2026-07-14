"""FoldVLA-style rollout demo using the ClothOpt C++ engine from Python."""

import argparse
import json
from pathlib import Path

from cloth_opt import ClothEnv, ClothEnvConfig, SceneConfig
from cloth_opt.policy import DemoControlPolicy


def load_config(path: Path) -> tuple[ClothEnvConfig, Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scene = SceneConfig(**raw["scene"])
    env_config = ClothEnvConfig(
        scene=scene,
        n_substeps=raw["n_substeps"],
        reset_height=raw["reset_height"],
        pin_corners=raw["pin_corners"],
    )
    return env_config, Path(raw["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sim_cloth.json"))
    args = parser.parse_args()

    env_config, output_dir = load_config(args.config)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = ClothEnv(env_config)
    policy = DemoControlPolicy(env_config.scene.width, env_config.scene.height)

    observation = env.reset()
    policy.reset()
    actions: list[dict] = []
    trajectory: list[dict] = []

    while True:
        result = policy.get_action(observation)
        if result is None:
            break
        action, action_info = result
        observation = env.step(action)
        actions.append({"action": action.to_dict(), "info": action_info})
        trajectory.append({
            "time": observation["time"],
            "center": observation["positions"].mean(axis=0).tolist(),
            "max_speed": float((observation["velocities"] ** 2).sum(axis=1).max() ** 0.5),
        })
        print(
            f"step={len(actions):02d} control={action.mode:10s} "
            f"time={observation['time']:.3f} center={trajectory[-1]['center']}"
        )

    env.export_mesh(output_dir / "final.obj")
    (output_dir / "actions.json").write_text(json.dumps(actions, indent=2), encoding="utf-8")
    (output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    (output_dir / "env_config.json").write_text(
        json.dumps(env.get_env_info(), indent=2), encoding="utf-8"
    )
    env.close()
    policy.close()
    print(f"saved rollout to {output_dir}")


if __name__ == "__main__":
    main()
