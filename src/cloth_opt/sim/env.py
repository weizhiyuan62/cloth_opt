from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .action import ClothAction
from .engine import ClothOptEngine, SceneConfig


@dataclass
class ClothEnvConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    n_substeps: int = 25
    reset_height: float = 0.5
    pin_corners: bool = False


class ClothEnv:
    """Policy-facing reset/step API following FoldVLA's environment pattern."""

    def __init__(self, config: ClothEnvConfig):
        if config.n_substeps <= 0:
            raise ValueError("n_substeps must be positive")
        self.config = config
        self.engine = ClothOptEngine()
        self.time = 0.0

    def reset(self) -> dict[str, Any]:
        cfg = self.config
        self.engine.set_scene(cfg.scene)
        positions = self.engine.get_cloth_positions()
        positions[:, 1] = cfg.reset_height
        self.engine.set_cloth_positions(positions)
        self.engine.set_cloth_velocities(np.zeros_like(positions))
        if cfg.pin_corners:
            self.engine.mesh.pin_corners()
        self.time = 0.0
        return self.get_observation()

    def _values_nx3(self, action: ClothAction) -> np.ndarray:
        if action.values is None:
            raise ValueError(f"{action.mode} action requires values")
        values = np.asarray(action.values, dtype=np.float64)
        if values.shape == (3,) and len(action.vertex_indices) == 1:
            values = values.reshape(1, 3)
        expected = (len(action.vertex_indices), 3)
        if values.shape != expected:
            raise ValueError(f"{action.mode} values must have shape {expected}, got {values.shape}")
        return values

    def apply_action(self, action: ClothAction) -> None:
        controller = self.engine.controller
        if action.replace_controls:
            controller.remove_all_controls()
        if action.mode == "clear":
            return

        indices = action.vertex_indices.tolist()
        p = action.params
        if action.mode in {"position", "velocity", "force"}:
            values = self._values_nx3(action)
            for index, value in zip(indices, values):
                if action.mode == "position":
                    controller.add_position_control(
                        index, value, float(p.get("gain", 1000.0)), float(p.get("max_force", 100.0))
                    )
                elif action.mode == "velocity":
                    controller.add_velocity_control(
                        index, value, float(p.get("gain", 100.0)), float(p.get("max_force", 50.0))
                    )
                else:
                    controller.add_force_control(index, value)
        elif action.mode == "trajectory":
            if len(indices) != 1 or action.values is None:
                raise ValueError("trajectory requires one vertex and waypoint values")
            controller.set_trajectory(indices[0], action.values, p["times"], bool(p.get("loop", False)))
        elif action.mode == "circular":
            if len(indices) != 1 or action.values is None:
                raise ValueError("circular requires one vertex and a center value")
            controller.add_circular_motion(
                indices[0], action.values.reshape(3), float(p["radius"]), float(p["frequency"]),
                np.asarray(p.get("axis", [0.0, 1.0, 0.0]), dtype=np.float64),
            )
        elif action.mode == "sinusoidal":
            if len(indices) != 1 or action.values is None:
                raise ValueError("sinusoidal requires one vertex and a center value")
            controller.add_sinusoidal_motion(
                indices[0], action.values.reshape(3),
                np.asarray(p["amplitude"], dtype=np.float64), float(p["frequency"]),
            )
        elif action.mode == "wind":
            controller.add_wind_force(
                indices, np.asarray(p["direction"], dtype=np.float64),
                float(p["strength"]), float(p.get("turbulence", 0.0)),
            )
        else:
            raise ValueError(f"unsupported action mode: {action.mode}")

    def step(self, action: ClothAction | None) -> dict[str, Any]:
        if action is not None:
            self.apply_action(action)
        self.engine.step(self.config.n_substeps)
        self.time += self.config.scene.dt * self.config.n_substeps
        return self.get_observation()

    def get_observation(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "positions": self.engine.get_cloth_positions(),
            "velocities": self.engine.get_cloth_velocities(),
            "pinned": self.engine.get_cloth_pin_flags(),
        }

    def export_mesh(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        positions = self.engine.get_cloth_positions()
        triangles = self.engine.mesh.triangles
        with path.open("w", encoding="utf-8") as file:
            for x, y, z in positions:
                file.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")
            for i, j, k in triangles:
                file.write(f"f {i + 1} {j + 1} {k + 1}\n")

    def get_env_info(self) -> dict[str, Any]:
        return asdict(self.config)

    def close(self) -> None:
        self.engine.close()
