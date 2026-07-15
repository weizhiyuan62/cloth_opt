from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from cloth_opt.sim import ClothAction, ClothEnv, ClothEnvConfig
from cloth_opt.policy.fold_metrics import evaluate_fold_trajectory


class FoldPhase(str, Enum):
    INITIAL_SETTLE = "initial_settle"
    LIFT = "lift"
    TRANSFER = "transfer"
    PLACE = "place"
    HOLD = "hold"
    RELEASE = "release"
    FINAL_SETTLE = "final_settle"
    DONE = "done"


@dataclass(frozen=True)
class SymmetricFoldParameters:
    lift_angle: float = 45.0
    transfer_angle: float = 165.0
    final_angle: float = 180.0
    layer_gap: float = 0.01
    lift_frames: int = 50
    transfer_frames: int = 120
    place_frames: int = 50
    hold_frames: int = 60
    position_gain: float = 2000.0
    max_force: float = 500.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SymmetricFoldParameters":
        converted = dict(values)
        for key in ("lift_frames", "transfer_frames", "place_frames", "hold_frames"):
            converted[key] = max(1, int(round(float(converted[key]))))
        for key in (
            "lift_angle",
            "transfer_angle",
            "final_angle",
            "layer_gap",
            "position_gain",
            "max_force",
        ):
            converted[key] = float(converted[key])
        parameters = cls(**converted)
        if not 0.0 < parameters.lift_angle < parameters.transfer_angle < parameters.final_angle <= 180.0:
            raise ValueError("angles must satisfy 0 < lift < transfer < final <= 180")
        return parameters


@dataclass(frozen=True)
class SymmetricFoldPolicyConfig:
    controlled_edge: str = "bottom"
    pinned_line_offsets: tuple[int, ...] = (0, 2, 4)
    pin_final_state: bool = True
    initial_settle_frames: int = 20
    final_settle_frames: int = 40
    alignment_weight: float = 10.0
    energy_weight: float = 0.05
    stretch_weight: float = 1.0
    max_stretch_weight: float = 2.0
    penetration_weight: float = 10.0
    terminal_velocity_weight: float = 0.5
    smoothness_weight: float = 0.05
    success_alignment: float = 0.08
    success_max_speed: float = 0.10
    success_aesthetic_quality: float = 70.0
    success_max_stretch: float = 0.35

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SymmetricFoldPolicyConfig":
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


def _rotate_about_axis(
    points: np.ndarray,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    relative = np.asarray(points, dtype=np.float64) - axis_origin
    axis = np.asarray(axis_direction, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cosine, sine = np.cos(angle_radians), np.sin(angle_radians)
    return (
        axis_origin
        + relative * cosine
        + np.cross(axis, relative) * sine
        + np.outer(relative @ axis, axis) * (1.0 - cosine)
    )


class SymmetricFoldStateMachine:
    def __init__(
        self,
        start_targets: np.ndarray,
        controlled_indices: np.ndarray,
        axis_origin: np.ndarray,
        axis_direction: np.ndarray,
        rotation_sign: float,
        parameters: SymmetricFoldParameters,
        policy_config: SymmetricFoldPolicyConfig,
    ) -> None:
        self.start_targets = start_targets.copy()
        self.controlled_indices = controlled_indices.copy()
        self.axis_origin = axis_origin.copy()
        self.axis_direction = axis_direction.copy()
        self.rotation_sign = float(rotation_sign)
        self.parameters = parameters
        self.policy_config = policy_config
        self.phase = FoldPhase.INITIAL_SETTLE
        self.phase_frame = 0

        self.place_targets = self._angle_target(parameters.final_angle)
        self.place_targets[:, 1] += parameters.layer_gap

    def _angle_target(self, angle_degrees: float) -> np.ndarray:
        return _rotate_about_axis(
            self.start_targets,
            self.axis_origin,
            self.axis_direction,
            self.rotation_sign * np.deg2rad(angle_degrees),
        )

    def _interpolated_target(self, start: float, end: float, progress: float) -> np.ndarray:
        return self._angle_target(start + progress * (end - start))

    @property
    def done(self) -> bool:
        return self.phase == FoldPhase.DONE

    def _duration(self) -> int:
        p, c = self.parameters, self.policy_config
        return {
            FoldPhase.INITIAL_SETTLE: c.initial_settle_frames,
            FoldPhase.LIFT: p.lift_frames,
            FoldPhase.TRANSFER: p.transfer_frames,
            FoldPhase.PLACE: p.place_frames,
            FoldPhase.HOLD: p.hold_frames,
            FoldPhase.RELEASE: 1,
            FoldPhase.FINAL_SETTLE: c.final_settle_frames,
        }[self.phase]

    def _advance(self) -> None:
        order = [
            FoldPhase.INITIAL_SETTLE,
            FoldPhase.LIFT,
            FoldPhase.TRANSFER,
            FoldPhase.PLACE,
            FoldPhase.HOLD,
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
            targets = self._interpolated_target(0.0, p.lift_angle, progress)
            action = self._position_action(targets)
        elif phase == FoldPhase.TRANSFER:
            targets = self._interpolated_target(p.lift_angle, p.transfer_angle, progress)
            action = self._position_action(targets)
        elif phase == FoldPhase.PLACE:
            targets = self._interpolated_target(p.transfer_angle, p.final_angle, progress)
            targets[:, 1] += progress * p.layer_gap
            action = self._position_action(targets)
        elif phase == FoldPhase.HOLD:
            targets = self.place_targets
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


class SymmetricFoldPolicy:
    def __init__(
        self,
        env_config: ClothEnvConfig,
        policy_config: SymmetricFoldPolicyConfig,
    ) -> None:
        self.env_config = env_config
        self.policy_config = policy_config

    def _fold_regions(self, env: ClothEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scene = env.config.scene
        if self.policy_config.controlled_edge not in {"top", "bottom"}:
            raise NotImplementedError("symmetric fold currently supports top/bottom folds")
        if scene.height % 2 != 0:
            raise ValueError("symmetric fold currently requires an even grid height")
        half = scene.height // 2
        moving_rows = range(half, scene.height) if self.policy_config.controlled_edge == "bottom" else range(half)
        moving, stationary = [], []
        for row in moving_rows:
            partner_row = scene.height - 1 - row
            for column in range(scene.width):
                moving.append(env.engine.grid_index(row, column))
                stationary.append(env.engine.grid_index(partner_row, column))
        if self.policy_config.controlled_edge == "top":
            pinned_rows = [half + offset for offset in self.policy_config.pinned_line_offsets]
        else:
            pinned_rows = [half - 1 - offset for offset in self.policy_config.pinned_line_offsets]
        pinned_rows = [row for row in pinned_rows if 0 <= row < scene.height]
        pinned = [env.engine.grid_index(row, column) for row in pinned_rows for column in range(scene.width)]
        return np.asarray(moving), np.asarray(stationary), np.asarray(pinned)

    def rollout(
        self,
        parameters: SymmetricFoldParameters,
        record: bool = False,
        frame_callback: Callable[[int, FoldPhase, dict[str, Any], np.ndarray, np.ndarray], None] | None = None,
    ) -> SymmetricFoldResult:
        env = ClothEnv(self.env_config)
        observation = env.reset()
        initial_positions = observation["positions"].copy()
        controlled, _, pinned = self._fold_regions(env)
        middle_row = env.config.scene.height // 2
        upper_row, lower_row = middle_row - 1, middle_row
        axis_origin = 0.5 * (
            initial_positions[env.engine.grid_index(upper_row, 0)]
            + initial_positions[env.engine.grid_index(lower_row, 0)]
        )
        axis_end = 0.5 * (
            initial_positions[env.engine.grid_index(upper_row, env.config.scene.width - 1)]
            + initial_positions[env.engine.grid_index(lower_row, env.config.scene.width - 1)]
        )
        axis_direction = axis_end - axis_origin
        radial = initial_positions[controlled[0]] - axis_origin
        lift_direction = float(np.cross(axis_direction, radial)[1])
        rotation_sign = 1.0 if lift_direction > 0.0 else -1.0

        pin_flags = env.engine.get_cloth_pin_flags()
        pin_flags[pinned] = True
        env.engine.set_cloth_pin_flags(pin_flags)

        start_targets = initial_positions[controlled]
        machine = SymmetricFoldStateMachine(
            start_targets,
            controlled,
            axis_origin,
            axis_direction,
            rotation_sign,
            parameters,
            self.policy_config,
        )

        positions_all = [observation["positions"].copy()] if record else None
        evaluation_positions = [observation["positions"].copy()]
        velocities_all = [observation["velocities"].copy()] if record else None
        times = [observation["time"]] if record else None
        targets_all: list[np.ndarray] = []
        phases: list[str] = []
        actions: list[dict[str, Any] | None] | None = [] if record else None
        frame_index = 0

        try:
            while not machine.done:
                action, phase, targets = machine.next_action()
                if phase == FoldPhase.RELEASE and self.policy_config.pin_final_state:
                    final_pin_flags = env.engine.get_cloth_pin_flags()
                    final_pin_flags[controlled] = True
                    env.engine.set_cloth_pin_flags(final_pin_flags)
                observation = env.step(action)
                evaluation_positions.append(observation["positions"].copy())
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
                env,
                np.asarray(evaluation_positions),
                final_velocities,
                np.asarray(targets_all),
                parameters,
            )
        finally:
            env.close()

        return SymmetricFoldResult(
            parameters=parameters,
            metrics=metrics,
            controlled_indices=controlled,
            anchor_indices=pinned,
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
        trajectory_positions: np.ndarray,
        velocities: np.ndarray,
        targets: np.ndarray,
        parameters: SymmetricFoldParameters,
    ) -> dict[str, float | bool]:
        moving, stationary, _ = self._fold_regions(env)
        scene = env.config.scene
        metrics = evaluate_fold_trajectory(
            trajectory_positions,
            velocities,
            targets,
            moving,
            stationary,
            scene.width,
            scene.height,
            scene.spacing,
            scene.dt * env.config.n_substeps,
            parameters.layer_gap,
            env.engine.grid_index,
        )
        c = self.policy_config
        loss = (
            c.alignment_weight * float(metrics["alignment_error"])
            + c.energy_weight * float(metrics["control_effort_proxy"])
            + c.stretch_weight * float(metrics["trajectory_mean_stretch"])
            + c.max_stretch_weight * float(metrics["trajectory_max_stretch"])
            + c.penetration_weight * float(metrics["layer_penetration_proxy"])
            + c.terminal_velocity_weight * float(metrics["terminal_mean_speed"])
            + c.smoothness_weight * float(metrics["target_smoothness"])
        )
        structural_integrity = float(metrics["trajectory_max_stretch"]) <= c.success_max_stretch
        success = (
            float(metrics["alignment_error"]) <= c.success_alignment
            and float(metrics["terminal_max_speed"]) <= c.success_max_speed
            and float(metrics["aesthetic_quality"]) >= c.success_aesthetic_quality
            and structural_integrity
        )
        metrics.update(
            loss=float(loss),
            structural_integrity=bool(structural_integrity),
            success=bool(success),
        )
        return metrics
