from typing import Any

import numpy as np

from cloth_opt.action import ClothAction
from cloth_opt.policy.base import BasePolicy


class DemoControlPolicy(BasePolicy):
    """Cycles through every control primitive exposed by the C++ controller."""

    control_names = ("position", "velocity", "force", "trajectory", "circular", "sinusoidal", "wind")

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.step_index = 0

    def reset(self) -> None:
        self.step_index = 0

    def _index(self, row: int, column: int) -> int:
        return row * self.width + column

    def get_action(self, observation: dict[str, Any]) -> tuple[ClothAction, dict[str, Any]] | None:
        if self.step_index >= len(self.control_names):
            return None
        mode = self.control_names[self.step_index]
        self.step_index += 1
        positions = observation["positions"]
        center_index = self._index(self.height - 1, self.width // 2)
        center = positions[center_index].copy()

        if mode == "position":
            action = ClothAction(mode, [center_index], center + [0.0, 0.2, 0.2],
                                 {"gain": 800.0, "max_force": 100.0})
        elif mode == "velocity":
            action = ClothAction(mode, [center_index], [[0.0, 0.3, 0.0]],
                                 {"gain": 50.0, "max_force": 50.0})
        elif mode == "force":
            action = ClothAction(mode, [center_index], [[0.0, 8.0, 0.0]])
        elif mode == "trajectory":
            waypoints = np.stack([center, center + [0.15, 0.2, 0.0], center + [0.0, 0.25, 0.2]])
            action = ClothAction(mode, [center_index], waypoints,
                                 {"times": [0.0, 0.05, 0.1], "loop": True})
        elif mode == "circular":
            action = ClothAction(mode, [center_index], center,
                                 {"radius": 0.1, "frequency": 1.0, "axis": [0.0, 1.0, 0.0]})
        elif mode == "sinusoidal":
            action = ClothAction(mode, [center_index], center,
                                 {"amplitude": [0.1, 0.1, 0.0], "frequency": 1.0})
        else:
            indices = np.arange(self.width * self.height)
            action = ClothAction(mode, indices, params={
                "direction": [1.0, 0.2, 0.0], "strength": 0.5, "turbulence": 0.0,
            })

        return action, {"control": mode}
