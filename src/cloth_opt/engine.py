from dataclasses import dataclass

import numpy as np

from . import _core


@dataclass
class SceneConfig:
    width: int = 10
    height: int = 10
    spacing: float = 0.1
    dt: float = 0.002
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    mass: float = 1.0
    stiffness: float = 800.0
    bending_stiffness: float = 20.0
    damping: float = 0.9
    friction: float = 0.8


class ClothOptEngine:
    """FoldVLA-style engine facade backed by the ClothOpt C++ core."""

    def __init__(self) -> None:
        self.mesh = _core.ClothMesh()
        self.controller = _core.ClothController()
        self.integrator = _core.SemiImplicitEulerIntegrator()
        self.config: SceneConfig | None = None

    def set_scene(self, config: SceneConfig) -> None:
        if config.width <= 1 or config.height <= 1:
            raise ValueError("cloth width and height must both be greater than one")
        if config.dt <= 0.0 or config.spacing <= 0.0:
            raise ValueError("dt and spacing must be positive")

        self.config = config
        self.mesh.create_grid(config.width, config.height, config.spacing)
        props = self.mesh.properties
        props.gravity = np.asarray(config.gravity, dtype=np.float64)
        props.mass = config.mass
        props.stiffness = config.stiffness
        props.bending_stiffness = config.bending_stiffness
        props.damping = config.damping
        props.friction = config.friction
        self.mesh.properties = props
        self.controller = _core.ClothController()

    def _require_scene(self) -> SceneConfig:
        if self.config is None:
            raise RuntimeError("call set_scene() before using the engine")
        return self.config

    def get_cloth_positions(self) -> np.ndarray:
        self._require_scene()
        return self.mesh.positions

    def set_cloth_positions(self, positions: np.ndarray, indices: np.ndarray | None = None) -> None:
        current = self.get_cloth_positions()
        if indices is None:
            current = np.asarray(positions, dtype=np.float64)
        else:
            current[np.asarray(indices, dtype=np.int64)] = np.asarray(positions, dtype=np.float64)
        self.mesh.positions = current

    def get_cloth_velocities(self) -> np.ndarray:
        self._require_scene()
        return self.mesh.velocities

    def set_cloth_velocities(self, velocities: np.ndarray, indices: np.ndarray | None = None) -> None:
        current = self.get_cloth_velocities()
        if indices is None:
            current = np.asarray(velocities, dtype=np.float64)
        else:
            current[np.asarray(indices, dtype=np.int64)] = np.asarray(velocities, dtype=np.float64)
        self.mesh.velocities = current

    def get_cloth_pin_flags(self) -> np.ndarray:
        self._require_scene()
        return np.asarray(self.mesh.pinned, dtype=bool)

    def set_cloth_pin_flags(self, flags: np.ndarray, indices: np.ndarray | None = None) -> None:
        current = self.get_cloth_pin_flags()
        if indices is None:
            current = np.asarray(flags, dtype=bool)
        else:
            current[np.asarray(indices, dtype=np.int64)] = np.asarray(flags, dtype=bool)
        self.mesh.pinned = current.tolist()

    def add_sphere(self, center: np.ndarray, radius: float) -> None:
        self._require_scene()
        self.mesh.add_sphere(np.asarray(center, dtype=np.float64), float(radius))

    def clear_spheres(self) -> None:
        self.mesh.clear_spheres()

    def grid_index(self, row: int, column: int) -> int:
        cfg = self._require_scene()
        if not (0 <= row < cfg.height and 0 <= column < cfg.width):
            raise IndexError((row, column))
        return self.mesh.grid_index(row, column)

    def step(self, substeps: int = 1) -> None:
        cfg = self._require_scene()
        _core.simulate(self.mesh, self.controller, self.integrator, cfg.dt, substeps)

    def close(self) -> None:
        self.controller.remove_all_controls()
