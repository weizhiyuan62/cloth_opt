from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from cloth_opt.sim import ClothAction, ClothEnv, ClothEnvConfig


class DiagonalFoldPhase(str, Enum):
    INITIAL_SETTLE = "initial_settle"
    GRASP_HOLD = "grasp_hold"
    EARLY_LIFT = "early_lift"
    ROTATE = "rotate"
    PLACE = "place"
    HOLD = "hold"
    RELEASE = "release"
    FINAL_SETTLE = "final_settle"
    DONE = "done"


@dataclass(frozen=True)
class DiagonalFoldParameters:
    early_lift_angle: float = 35.0
    rotate_end_angle: float = 160.0
    final_angle: float = 178.0
    layer_gap: float = 0.01
    grasp_hold_frames: int = 20
    early_lift_frames: int = 30
    rotate_frames: int = 120
    place_frames: int = 40
    hold_frames: int = 50
    position_gain: float = 800.0
    max_force: float = 150.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DiagonalFoldParameters":
        converted = dict(values)
        for key in (
            "grasp_hold_frames",
            "early_lift_frames",
            "rotate_frames",
            "place_frames",
            "hold_frames",
        ):
            converted[key] = max(1, int(round(float(converted[key]))))
        for key in (
            "early_lift_angle",
            "rotate_end_angle",
            "final_angle",
            "layer_gap",
            "position_gain",
            "max_force",
        ):
            converted[key] = float(converted[key])
        parameters = cls(**converted)
        if not (
            0.0 < parameters.early_lift_angle
            < parameters.rotate_end_angle
            < parameters.final_angle
            <= 180.0
        ):
            raise ValueError(
                "angles must satisfy 0 < early_lift_angle < rotate_end_angle "
                "< final_angle <= 180"
            )
        return parameters


@dataclass(frozen=True)
class DiagonalFoldPolicyConfig:
    fold_diagonal: str = "main"
    controlled_corner: str = "top_right"
    pin_fold_endpoints: bool = True
    initial_settle_frames: int = 30
    final_settle_frames: int = 100
    alignment_weight: float = 10.0
    stationary_weight: float = 2.0
    crease_weight: float = 1.0
    stretch_weight: float = 1.0
    terminal_velocity_weight: float = 0.5
    smoothness_weight: float = 0.05
    success_alignment: float = 0.10
    success_max_speed: float = 0.10


@dataclass
class DiagonalFoldResult:
    parameters: DiagonalFoldParameters
    metrics: dict[str, float | bool]
    controlled_indices: np.ndarray
    anchor_indices: np.ndarray
    moving_indices: np.ndarray
    stationary_indices: np.ndarray
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
                moving_indices=self.moving_indices,
                stationary_indices=self.stationary_indices,
            )
        with (directory / "final.obj").open("w", encoding="utf-8") as file:
            for x, y, z in self.final_positions:
                file.write(f"v {x:.12g} {y:.12g} {z:.12g}\n")
            for i, j, k in self.triangles:
                file.write(f"f {i + 1} {j + 1} {k + 1}\n")


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _rotate_about_axis(
    points: np.ndarray,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    """Rotate points around a 3-D line using Rodrigues' formula."""

    relative = np.asarray(points, dtype=np.float64) - axis_origin
    axis = np.asarray(axis_direction, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    return (
        axis_origin
        + relative * cosine
        + np.cross(axis, relative) * sine
        + np.outer(relative @ axis, axis) * (1.0 - cosine)
    )


class DiagonalFoldStateMachine:
    def __init__(
        self,
        start_target: np.ndarray,
        controlled_index: int,
        axis_origin: np.ndarray,
        axis_direction: np.ndarray,
        rotation_sign: float,
        parameters: DiagonalFoldParameters,
        policy_config: DiagonalFoldPolicyConfig,
    ) -> None:
        self.start_target = np.asarray(start_target, dtype=np.float64).reshape(1, 3)
        self.controlled_index = int(controlled_index)
        self.axis_origin = np.asarray(axis_origin, dtype=np.float64)
        self.axis_direction = np.asarray(axis_direction, dtype=np.float64)
        self.rotation_sign = float(rotation_sign)
        self.parameters = parameters
        self.policy_config = policy_config
        self.phase = DiagonalFoldPhase.INITIAL_SETTLE
        self.phase_frame = 0

        self.early_target = self._angle_target(parameters.early_lift_angle)
        self.rotate_target = self._angle_target(parameters.rotate_end_angle)
        self.place_target = self._angle_target(parameters.final_angle)
        self.place_target[:, 1] += parameters.layer_gap

    @property
    def done(self) -> bool:
        return self.phase == DiagonalFoldPhase.DONE

    def _duration(self) -> int:
        p, c = self.parameters, self.policy_config
        return {
            DiagonalFoldPhase.INITIAL_SETTLE: c.initial_settle_frames,
            DiagonalFoldPhase.GRASP_HOLD: p.grasp_hold_frames,
            DiagonalFoldPhase.EARLY_LIFT: p.early_lift_frames,
            DiagonalFoldPhase.ROTATE: p.rotate_frames,
            DiagonalFoldPhase.PLACE: p.place_frames,
            DiagonalFoldPhase.HOLD: p.hold_frames,
            DiagonalFoldPhase.RELEASE: 1,
            DiagonalFoldPhase.FINAL_SETTLE: c.final_settle_frames,
        }[self.phase]

    def _advance(self) -> None:
        order = [
            DiagonalFoldPhase.INITIAL_SETTLE,
            DiagonalFoldPhase.GRASP_HOLD,
            DiagonalFoldPhase.EARLY_LIFT,
            DiagonalFoldPhase.ROTATE,
            DiagonalFoldPhase.PLACE,
            DiagonalFoldPhase.HOLD,
            DiagonalFoldPhase.RELEASE,
            DiagonalFoldPhase.FINAL_SETTLE,
            DiagonalFoldPhase.DONE,
        ]
        self.phase = order[order.index(self.phase) + 1]
        self.phase_frame = 0

    def _angle_target(self, angle_degrees: float) -> np.ndarray:
        angle = self.rotation_sign * np.deg2rad(angle_degrees)
        return _rotate_about_axis(
            self.start_target, self.axis_origin, self.axis_direction, angle
        )

    def _interpolated_angle_target(
        self, start_angle: float, end_angle: float, progress: float
    ) -> np.ndarray:
        angle = start_angle + progress * (end_angle - start_angle)
        return self._angle_target(angle)

    def _position_action(self, target: np.ndarray) -> ClothAction:
        return ClothAction(
            mode="position",
            vertex_indices=np.asarray([self.controlled_index], dtype=np.int64),
            values=target,
            params={
                "gain": self.parameters.position_gain,
                "max_force": self.parameters.max_force,
            },
        )

    def next_action(
        self,
    ) -> tuple[ClothAction | None, DiagonalFoldPhase, np.ndarray]:
        if self.done:
            raise RuntimeError("state machine already completed")

        phase = self.phase
        duration = max(1, self._duration())
        progress = _smoothstep((self.phase_frame + 1) / duration)
        p = self.parameters

        if phase == DiagonalFoldPhase.INITIAL_SETTLE:
            target = self.start_target
            action = None
        elif phase == DiagonalFoldPhase.GRASP_HOLD:
            target = self.start_target
            action = self._position_action(target)
        elif phase == DiagonalFoldPhase.EARLY_LIFT:
            target = self._interpolated_angle_target(0.0, p.early_lift_angle, progress)
            action = self._position_action(target)
        elif phase == DiagonalFoldPhase.ROTATE:
            target = self._interpolated_angle_target(
                p.early_lift_angle, p.rotate_end_angle, progress
            )
            action = self._position_action(target)
        elif phase == DiagonalFoldPhase.PLACE:
            target = self._interpolated_angle_target(
                p.rotate_end_angle, p.final_angle, progress
            )
            target[:, 1] += progress * p.layer_gap
            action = self._position_action(target)
        elif phase == DiagonalFoldPhase.HOLD:
            target = self.place_target
            action = self._position_action(target)
        elif phase == DiagonalFoldPhase.RELEASE:
            target = self.place_target
            action = ClothAction(mode="clear")
        elif phase == DiagonalFoldPhase.FINAL_SETTLE:
            target = self.place_target
            action = None
        else:
            raise AssertionError(phase)

        self.phase_frame += 1
        if self.phase_frame >= duration:
            self._advance()
        return action, phase, target.copy()


class DiagonalFoldPolicy:
    _CORNERS = {"top_left", "top_right", "bottom_left", "bottom_right"}

    def __init__(
        self,
        env_config: ClothEnvConfig,
        policy_config: DiagonalFoldPolicyConfig,
    ) -> None:
        self.env_config = env_config
        self.policy_config = policy_config

    def _validate_scene(self, env: ClothEnv) -> None:
        scene = env.config.scene
        if scene.width != scene.height:
            raise ValueError("diagonal fold currently requires a square cloth grid")
        if self.policy_config.fold_diagonal not in {"main", "anti"}:
            raise ValueError("fold_diagonal must be 'main' or 'anti'")
        if self.policy_config.controlled_corner not in self._CORNERS:
            raise ValueError(f"unsupported controlled_corner: {self.policy_config.controlled_corner}")
        compatible = {
            "main": {"top_right", "bottom_left"},
            "anti": {"top_left", "bottom_right"},
        }
        if self.policy_config.controlled_corner not in compatible[self.policy_config.fold_diagonal]:
            raise ValueError(
                f"{self.policy_config.controlled_corner} lies on or is incompatible with "
                f"the {self.policy_config.fold_diagonal} diagonal"
            )

    @staticmethod
    def _corner_coordinate(env: ClothEnv, corner: str) -> tuple[int, int]:
        last_row = env.config.scene.height - 1
        last_column = env.config.scene.width - 1
        return {
            "top_left": (0, 0),
            "top_right": (0, last_column),
            "bottom_left": (last_row, 0),
            "bottom_right": (last_row, last_column),
        }[corner]

    def _fold_geometry(
        self, env: ClothEnv, initial_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        last = env.config.scene.height - 1
        endpoints = (
            ((0, 0), (last, last))
            if self.policy_config.fold_diagonal == "main"
            else ((0, last), (last, 0))
        )
        anchor_indices = np.asarray(
            [env.engine.grid_index(*coordinate) for coordinate in endpoints], dtype=np.int64
        )
        axis_origin = initial_positions[anchor_indices[0]].copy()
        axis_direction = initial_positions[anchor_indices[1]] - axis_origin

        corner_coordinate = self._corner_coordinate(env, self.policy_config.controlled_corner)
        controlled_index = env.engine.grid_index(*corner_coordinate)
        radial = initial_positions[controlled_index] - axis_origin
        lift_direction = float(np.cross(axis_direction, radial)[1])
        if abs(lift_direction) < 1e-12:
            raise ValueError("controlled corner lies on the selected fold axis")
        rotation_sign = 1.0 if lift_direction > 0.0 else -1.0
        return (
            np.asarray([controlled_index], dtype=np.int64),
            anchor_indices,
            axis_direction,
            rotation_sign,
        )

    def _fold_correspondence(
        self, env: ClothEnv
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scene = env.config.scene
        last = scene.height - 1
        moving: list[int] = []
        stationary: list[int] = []
        crease: list[int] = []
        corner = self.policy_config.controlled_corner

        for row in range(scene.height):
            for column in range(scene.width):
                if self.policy_config.fold_diagonal == "main":
                    side = column - row
                    is_moving = side > 0 if corner == "top_right" else side < 0
                    partner = (column, row)
                else:
                    side = row + column - last
                    is_moving = side < 0 if corner == "top_left" else side > 0
                    partner = (last - column, last - row)

                index = env.engine.grid_index(row, column)
                if side == 0:
                    crease.append(index)
                elif is_moving:
                    moving.append(index)
                    stationary.append(env.engine.grid_index(*partner))

        return (
            np.asarray(moving, dtype=np.int64),
            np.asarray(stationary, dtype=np.int64),
            np.asarray(crease, dtype=np.int64),
        )

    def rollout(
        self,
        parameters: DiagonalFoldParameters,
        record: bool = False,
        frame_callback: Callable[
            [int, DiagonalFoldPhase, dict[str, Any], np.ndarray, np.ndarray], None
        ]
        | None = None,
    ) -> DiagonalFoldResult:
        env = ClothEnv(self.env_config)
        observation = env.reset()
        self._validate_scene(env)
        initial_positions = observation["positions"].copy()
        controlled, anchor, axis_direction, rotation_sign = self._fold_geometry(
            env, initial_positions
        )
        moving, stationary, crease = self._fold_correspondence(env)

        if self.policy_config.pin_fold_endpoints:
            pin_flags = env.engine.get_cloth_pin_flags()
            pin_flags[anchor] = True
            env.engine.set_cloth_pin_flags(pin_flags)

        machine = DiagonalFoldStateMachine(
            start_target=initial_positions[controlled],
            controlled_index=int(controlled[0]),
            axis_origin=initial_positions[anchor[0]],
            axis_direction=axis_direction,
            rotation_sign=rotation_sign,
            parameters=parameters,
            policy_config=self.policy_config,
        )

        positions_all = [initial_positions.copy()] if record else None
        velocities_all = [observation["velocities"].copy()] if record else None
        times = [observation["time"]] if record else None
        targets_all: list[np.ndarray] = []
        phases: list[str] = []
        actions: list[dict[str, Any] | None] | None = [] if record else None
        frame_index = 0

        try:
            while not machine.done:
                action, phase, target = machine.next_action()
                observation = env.step(action)
                targets_all.append(target)
                phases.append(phase.value)
                if record:
                    positions_all.append(observation["positions"].copy())
                    velocities_all.append(observation["velocities"].copy())
                    times.append(observation["time"])
                    actions.append(None if action is None else action.to_dict())
                if frame_callback is not None:
                    frame_callback(frame_index, phase, observation, controlled, target)
                frame_index += 1

            final_positions = observation["positions"].copy()
            final_velocities = observation["velocities"].copy()
            triangles = env.engine.mesh.triangles.copy()
            metrics = self._evaluate(
                env,
                initial_positions,
                final_positions,
                final_velocities,
                moving,
                stationary,
                crease,
                np.asarray(targets_all),
                parameters,
            )
        finally:
            env.close()

        return DiagonalFoldResult(
            parameters=parameters,
            metrics=metrics,
            controlled_indices=controlled,
            anchor_indices=anchor,
            moving_indices=moving,
            stationary_indices=stationary,
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
        initial_positions: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        moving: np.ndarray,
        stationary: np.ndarray,
        crease: np.ndarray,
        targets: np.ndarray,
        parameters: DiagonalFoldParameters,
    ) -> dict[str, float | bool]:
        desired = positions[stationary].copy()
        desired[:, 1] += parameters.layer_gap
        alignment = float(np.linalg.norm(positions[moving] - desired, axis=1).mean())
        stationary_drift = float(
            np.linalg.norm(positions[stationary] - initial_positions[stationary], axis=1).mean()
        )
        crease_error = float(
            np.linalg.norm(positions[crease] - initial_positions[crease], axis=1).mean()
        )

        scene = env.config.scene
        stretch_values: list[float] = []
        for row in range(scene.height):
            for column in range(scene.width):
                index = env.engine.grid_index(row, column)
                if column + 1 < scene.width:
                    neighbor = env.engine.grid_index(row, column + 1)
                    stretch_values.append(
                        abs(np.linalg.norm(positions[index] - positions[neighbor]) / scene.spacing - 1.0)
                    )
                if row + 1 < scene.height:
                    neighbor = env.engine.grid_index(row + 1, column)
                    stretch_values.append(
                        abs(np.linalg.norm(positions[index] - positions[neighbor]) / scene.spacing - 1.0)
                    )
        stretch = float(np.mean(stretch_values))
        speeds = np.linalg.norm(velocities, axis=1)
        mean_speed = float(speeds.mean())
        max_speed = float(speeds.max())

        if len(targets) >= 3:
            second_difference = targets[2:] - 2.0 * targets[1:-1] + targets[:-2]
            smoothness = float(np.square(second_difference).mean())
        else:
            smoothness = 0.0

        c = self.policy_config
        loss = (
            c.alignment_weight * alignment
            + c.stationary_weight * stationary_drift
            + c.crease_weight * crease_error
            + c.stretch_weight * stretch
            + c.terminal_velocity_weight * mean_speed
            + c.smoothness_weight * smoothness
        )
        success = alignment <= c.success_alignment and max_speed <= c.success_max_speed
        return {
            "loss": float(loss),
            "alignment_error": alignment,
            "stationary_drift": stationary_drift,
            "crease_error": crease_error,
            "mean_stretch_error": stretch,
            "terminal_mean_speed": mean_speed,
            "terminal_max_speed": max_speed,
            "target_smoothness": smoothness,
            "success": bool(success),
        }
