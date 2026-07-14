from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from cloth_opt.action import ClothAction
from cloth_opt.env import ClothEnv, ClothEnvConfig


class FoldPhase(str, Enum):
    INITIAL_SETTLE = "initial_settle"
    LIFT = "lift"
    TRANSFER = "transfer"
    PLACE = "place"
    RELEASE = "release"
    FINAL_SETTLE = "final_settle"
    DONE = "done"


@dataclass(frozen=True)
class SymmetricFoldParameters:
    lift_height: float = 0.20
    arc_height: float = 0.20
    landing_offset: float = 0.0
    place_clearance: float = 0.04
    layer_gap: float = 0.01
    lift_frames: int = 20
    transfer_frames: int = 50
    place_frames: int = 20
    position_gain: float = 800.0
    max_force: float = 100.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SymmetricFoldParameters":
        converted = dict(values)
        for key in ("lift_frames", "transfer_frames", "place_frames"):
            converted[key] = max(1, int(round(float(converted[key]))))
        for key in (
            "lift_height",
            "arc_height",
            "landing_offset",
            "place_clearance",
            "layer_gap",
            "position_gain",
            "max_force",
        ):
            converted[key] = float(converted[key])
        return cls(**converted)


@dataclass(frozen=True)
class SymmetricFoldTaskConfig:
    controlled_edge: str = "bottom"
    anchor_edge: str = "top"
    initial_settle_frames: int = 20
    final_settle_frames: int = 40
    alignment_weight: float = 10.0
    stretch_weight: float = 1.0
    terminal_velocity_weight: float = 0.5
    smoothness_weight: float = 0.05
    success_alignment: float = 0.08
    success_max_speed: float = 0.10

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SymmetricFoldTaskConfig":
        return cls(**dict(values))


@dataclass
class SymmetricFoldResult:
    parameters: SymmetricFoldParameters
    metrics: dict[str, float | bool]
    controlled_indices: np.ndarray
    anchor_indices: np.ndarray
    final_positions: np.ndarray
    final_velocities: np.ndarray
    triangles: np.ndarray
    time: np.ndarray | None = None
    positions: np.ndarray | None = None
    velocities: np.ndarray | None = None
    target_positions: np.ndarray | None = None
    phases: list[str] | None = None
    actions: list[dict[str, Any] | None] | None = None

    @property
    def loss(self) -> float:
        return float(self.metrics["loss"])

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "parameters.json").write_text(
            json.dumps(asdict(self.parameters), indent=2), encoding="utf-8"
        )
        (directory / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2), encoding="utf-8"
        )
        if self.actions is not None:
            (directory / "actions.json").write_text(
                json.dumps(self.actions, indent=2), encoding="utf-8"
            )
        if self.positions is not None:
            np.savez_compressed(
                directory / "trajectory.npz",
                time=self.time,
                positions=self.positions,
                velocities=self.velocities,
                target_positions=self.target_positions,
                phases=np.asarray(self.phases),
                controlled_indices=self.controlled_indices,
                anchor_indices=self.anchor_indices,
            )
        with (directory / "final.obj").open("w", encoding="utf-8") as file:
            for x, y, z in self.final_positions:
                file.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")
            for i, j, k in self.triangles:
                file.write(f"f {i + 1} {j + 1} {k + 1}\n")


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class SymmetricFoldStateMachine:
    def __init__(
        self,
        start_targets: np.ndarray,
        destination_targets: np.ndarray,
        controlled_indices: np.ndarray,
        parameters: SymmetricFoldParameters,
        task_config: SymmetricFoldTaskConfig,
    ) -> None:
        self.start_targets = start_targets.copy()
        self.destination_targets = destination_targets.copy()
        self.controlled_indices = controlled_indices.copy()
        self.parameters = parameters
        self.task_config = task_config
        self.phase = FoldPhase.INITIAL_SETTLE
        self.phase_frame = 0

        p = parameters
        self.lift_targets = self.start_targets + np.array([0.0, p.lift_height, 0.0])
        self.transfer_end = self.destination_targets.copy()
        self.transfer_end[:, 1] += p.place_clearance
        self.transfer_end[:, 2] += p.landing_offset
        self.place_targets = self.destination_targets.copy()
        self.place_targets[:, 1] += p.layer_gap
        self.place_targets[:, 2] += p.landing_offset

    @property
    def done(self) -> bool:
        return self.phase == FoldPhase.DONE

    def _duration(self) -> int:
        p, c = self.parameters, self.task_config
        return {
            FoldPhase.INITIAL_SETTLE: c.initial_settle_frames,
            FoldPhase.LIFT: p.lift_frames,
            FoldPhase.TRANSFER: p.transfer_frames,
            FoldPhase.PLACE: p.place_frames,
            FoldPhase.RELEASE: 1,
            FoldPhase.FINAL_SETTLE: c.final_settle_frames,
        }[self.phase]

    def _advance(self) -> None:
        order = [
            FoldPhase.INITIAL_SETTLE,
            FoldPhase.LIFT,
            FoldPhase.TRANSFER,
            FoldPhase.PLACE,
            FoldPhase.RELEASE,
            FoldPhase.FINAL_SETTLE,
            FoldPhase.DONE,
        ]
        self.phase = order[order.index(self.phase) + 1]
        self.phase_frame = 0

    def next_action(self) -> tuple[ClothAction | None, FoldPhase, np.ndarray]:
        if self.done:
            raise RuntimeError("state machine already completed")

        phase = self.phase
        duration = max(1, self._duration())
        progress = _smoothstep((self.phase_frame + 1) / duration)
        p = self.parameters

        if phase == FoldPhase.INITIAL_SETTLE:
            action = None
            targets = self.start_targets
        elif phase == FoldPhase.LIFT:
            targets = self.start_targets * (1.0 - progress) + self.lift_targets * progress
            action = self._position_action(targets)
        elif phase == FoldPhase.TRANSFER:
            targets = self.lift_targets * (1.0 - progress) + self.transfer_end * progress
            targets[:, 1] += np.sin(np.pi * progress) * p.arc_height
            action = self._position_action(targets)
        elif phase == FoldPhase.PLACE:
            targets = self.transfer_end * (1.0 - progress) + self.place_targets * progress
            action = self._position_action(targets)
        elif phase == FoldPhase.RELEASE:
            targets = self.place_targets
            action = ClothAction(mode="clear")
        elif phase == FoldPhase.FINAL_SETTLE:
            targets = self.place_targets
            action = None
        else:
            raise AssertionError(phase)

        self.phase_frame += 1
        if self.phase_frame >= duration:
            self._advance()
        return action, phase, targets.copy()

    def _position_action(self, targets: np.ndarray) -> ClothAction:
        return ClothAction(
            mode="position",
            vertex_indices=self.controlled_indices,
            values=targets,
            params={
                "gain": self.parameters.position_gain,
                "max_force": self.parameters.max_force,
            },
        )


class SymmetricFoldTask:
    def __init__(
        self,
        env_config: ClothEnvConfig,
        task_config: SymmetricFoldTaskConfig,
    ) -> None:
        self.env_config = env_config
        self.task_config = task_config

    def _edge_indices(self, env: ClothEnv, edge: str) -> np.ndarray:
        scene = env.config.scene
        if edge == "top":
            coordinates = [(0, column) for column in range(scene.width)]
        elif edge == "bottom":
            coordinates = [(scene.height - 1, column) for column in range(scene.width)]
        elif edge == "left":
            coordinates = [(row, 0) for row in range(scene.height)]
        elif edge == "right":
            coordinates = [(row, scene.width - 1) for row in range(scene.height)]
        else:
            raise ValueError(f"unsupported edge: {edge}")
        return np.asarray([env.engine.grid_index(r, c) for r, c in coordinates], dtype=np.int64)

    def _fold_correspondence(self, env: ClothEnv) -> tuple[np.ndarray, np.ndarray]:
        scene = env.config.scene
        if self.task_config.controlled_edge not in {"top", "bottom"}:
            raise NotImplementedError("symmetric fold currently supports top/bottom folds")
        half = scene.height // 2
        moving_rows = range(half, scene.height) if self.task_config.controlled_edge == "bottom" else range(half)
        moving, stationary = [], []
        for row in moving_rows:
            partner_row = scene.height - 1 - row
            for column in range(scene.width):
                moving.append(env.engine.grid_index(row, column))
                stationary.append(env.engine.grid_index(partner_row, column))
        return np.asarray(moving), np.asarray(stationary)

    def rollout(
        self,
        parameters: SymmetricFoldParameters,
        record: bool = False,
        frame_callback: Callable[[int, FoldPhase, dict[str, Any], np.ndarray, np.ndarray], None] | None = None,
    ) -> SymmetricFoldResult:
        env = ClothEnv(self.env_config)
        observation = env.reset()
        controlled = self._edge_indices(env, self.task_config.controlled_edge)
        anchor = self._edge_indices(env, self.task_config.anchor_edge)
        pin_flags = env.engine.get_cloth_pin_flags()
        pin_flags[anchor] = True
        env.engine.set_cloth_pin_flags(pin_flags)

        initial_positions = observation["positions"].copy()
        start_targets = initial_positions[controlled]
        destination = initial_positions[anchor]
        if len(destination) != len(controlled):
            raise ValueError("controlled and anchor edges must have equal vertex counts")
        machine = SymmetricFoldStateMachine(
            start_targets, destination, controlled, parameters, self.task_config
        )

        positions_all = [observation["positions"].copy()] if record else None
        velocities_all = [observation["velocities"].copy()] if record else None
        times = [observation["time"]] if record else None
        targets_all: list[np.ndarray] = []
        phases: list[str] = []
        actions: list[dict[str, Any] | None] | None = [] if record else None
        frame_index = 0

        try:
            while not machine.done:
                action, phase, targets = machine.next_action()
                observation = env.step(action)
                targets_all.append(targets)
                phases.append(phase.value)
                if record:
                    positions_all.append(observation["positions"].copy())
                    velocities_all.append(observation["velocities"].copy())
                    times.append(observation["time"])
                    actions.append(None if action is None else action.to_dict())
                if frame_callback is not None:
                    frame_callback(frame_index, phase, observation, controlled, targets)
                frame_index += 1

            final_positions = observation["positions"].copy()
            final_velocities = observation["velocities"].copy()
            triangles = env.engine.mesh.triangles.copy()
            metrics = self._evaluate(
                env, final_positions, final_velocities, np.asarray(targets_all), parameters
            )
        finally:
            env.close()

        return SymmetricFoldResult(
            parameters=parameters,
            metrics=metrics,
            controlled_indices=controlled,
            anchor_indices=anchor,
            final_positions=final_positions,
            final_velocities=final_velocities,
            triangles=triangles,
            time=None if times is None else np.asarray(times),
            positions=None if positions_all is None else np.asarray(positions_all),
            velocities=None if velocities_all is None else np.asarray(velocities_all),
            target_positions=np.asarray(targets_all) if record else None,
            phases=phases if record else None,
            actions=actions,
        )

    def _evaluate(
        self,
        env: ClothEnv,
        positions: np.ndarray,
        velocities: np.ndarray,
        targets: np.ndarray,
        parameters: SymmetricFoldParameters,
    ) -> dict[str, float | bool]:
        moving, stationary = self._fold_correspondence(env)
        desired = positions[stationary].copy()
        desired[:, 1] += parameters.layer_gap
        alignment = float(np.linalg.norm(positions[moving] - desired, axis=1).mean())

        scene = env.config.scene
        stretch_values = []
        for row in range(scene.height):
            for column in range(scene.width):
                index = env.engine.grid_index(row, column)
                if column + 1 < scene.width:
                    neighbor = env.engine.grid_index(row, column + 1)
                    stretch_values.append(abs(np.linalg.norm(positions[index] - positions[neighbor]) / scene.spacing - 1.0))
                if row + 1 < scene.height:
                    neighbor = env.engine.grid_index(row + 1, column)
                    stretch_values.append(abs(np.linalg.norm(positions[index] - positions[neighbor]) / scene.spacing - 1.0))
        stretch = float(np.mean(stretch_values))
        mean_speed = float(np.linalg.norm(velocities, axis=1).mean())
        max_speed = float(np.linalg.norm(velocities, axis=1).max())

        if len(targets) >= 3:
            second_difference = targets[2:] - 2.0 * targets[1:-1] + targets[:-2]
            smoothness = float(np.square(second_difference).mean())
        else:
            smoothness = 0.0

        c = self.task_config
        loss = (
            c.alignment_weight * alignment
            + c.stretch_weight * stretch
            + c.terminal_velocity_weight * mean_speed
            + c.smoothness_weight * smoothness
        )
        success = alignment <= c.success_alignment and max_speed <= c.success_max_speed
        return {
            "loss": float(loss),
            "alignment_error": alignment,
            "mean_stretch_error": stretch,
            "terminal_mean_speed": mean_speed,
            "terminal_max_speed": max_speed,
            "target_smoothness": smoothness,
            "success": bool(success),
        }
