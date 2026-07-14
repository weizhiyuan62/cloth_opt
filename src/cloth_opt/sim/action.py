from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


ControlMode = Literal[
    "position",
    "velocity",
    "force",
    "trajectory",
    "circular",
    "sinusoidal",
    "wind",
    "clear",
]


@dataclass
class ClothAction:
    """Engine-independent action consumed by :class:`ClothEnv`."""

    mode: ControlMode
    vertex_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    values: np.ndarray | None = None
    params: dict[str, Any] = field(default_factory=dict)
    replace_controls: bool = True

    def __post_init__(self) -> None:
        self.vertex_indices = np.asarray(self.vertex_indices, dtype=np.int64).reshape(-1)
        if self.values is not None:
            self.values = np.asarray(self.values, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "vertex_indices": self.vertex_indices.tolist(),
            "values": None if self.values is None else self.values.tolist(),
            "params": self.params,
            "replace_controls": self.replace_controls,
        }
