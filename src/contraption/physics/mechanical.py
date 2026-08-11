"""Focused electromechanical and mechanical reference systems."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .backend import Array, Backend
from .electrical import _control, _parameter
from .simulator import SimulationResult, simulate


class DCMotorSystem:
    """Armature-controlled permanent-magnet DC motor."""

    state_names = ("armature_current", "angular_velocity", "shaft_angle")
    control_names = ("voltage", "load_torque")
    output_names = state_names + ("electromagnetic_torque",)

    def __init__(self, resistance: Any = 2.0, inductance: Any = 0.01, torque_constant: Any = 0.08, back_emf_constant: Any = 0.08, inertia: Any = 0.002, viscous_friction: Any = 0.001, initial_state: Sequence[Any] = (0.0, 0.0, 0.0)) -> None:
        self.default_parameters = {"resistance": resistance, "inductance": inductance, "torque_constant": torque_constant, "back_emf_constant": back_emf_constant, "inertia": inertia, "viscous_friction": viscous_friction}
        self.parameter_bounds = {name: (1e-15, None) for name in self.default_parameters}
        self.parameter_uncertainty = {name: {"std": abs(value) * 0.02} for name, value in self.default_parameters.items()}
        self.initial_state = list(initial_state)

    def derivative(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Array:
        current, speed = state[:, 0], state[:, 1]
        current_rate = (_control(controls, "voltage", "armature_voltage") - _parameter(parameters, "resistance") * current - _parameter(parameters, "back_emf_constant") * speed) / _parameter(parameters, "inductance")
        speed_rate = (_parameter(parameters, "torque_constant") * current - _parameter(parameters, "viscous_friction") * speed - _control(controls, "load_torque")) / _parameter(parameters, "inertia")
        return backend.stack((current_rate, speed_rate, speed), axis=-1)

    def observe(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Mapping[str, Array]:
        return {"armature_current": state[:, 0], "angular_velocity": state[:, 1], "shaft_angle": state[:, 2], "electromagnetic_torque": _parameter(parameters, "torque_constant") * state[:, 0]}

    def simulate(self, **options: Any) -> SimulationResult:
        return simulate(self, **options)


class PlanarRigidBodySystem:
    """Planar rigid body with world-frame forces and optional roughness."""

    state_names = ("x", "y", "yaw", "velocity_x", "velocity_y", "angular_velocity")
    control_names = ("force_x", "force_y", "torque")

    def __init__(self, mass: Any = 1.0, moment_of_inertia: Any = 0.1, linear_drag: Any = 0.0, angular_drag: Any = 0.0, roughness_std: Any = 0.0, initial_state: Sequence[Any] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) -> None:
        self.default_parameters = {"mass": mass, "moment_of_inertia": moment_of_inertia, "linear_drag": linear_drag, "angular_drag": angular_drag, "roughness_std": roughness_std}
        self.parameter_bounds = {"mass": (1e-12, None), "moment_of_inertia": (1e-12, None), "linear_drag": (0.0, None), "angular_drag": (0.0, None), "roughness_std": (0.0, None)}
        self.initial_state = list(initial_state)

    def derivative(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Array:
        vx, vy, omega = state[:, 3], state[:, 4], state[:, 5]
        ax = (_control(controls, "force_x") - _parameter(parameters, "linear_drag") * vx) / _parameter(parameters, "mass")
        ay = (_control(controls, "force_y") - _parameter(parameters, "linear_drag") * vy) / _parameter(parameters, "mass")
        angular_acceleration = (_control(controls, "torque") - _parameter(parameters, "angular_drag") * omega) / _parameter(parameters, "moment_of_inertia")
        return backend.stack((vx, vy, omega, ax, ay, angular_acceleration), axis=-1)

    def process_noise(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], dt: Any, rng: Any, backend: Backend) -> Array:
        scale = _parameter(parameters, "roughness_std") * backend.sqrt(dt)
        return backend.concatenate((backend.normal((int(state.shape[0]), 3), rng) * scale[:, None], backend.zeros((int(state.shape[0]), 3))), axis=-1)

    def simulate(self, **options: Any) -> SimulationResult:
        return simulate(self, **options)


__all__ = ["DCMotorSystem", "PlanarRigidBodySystem"]
