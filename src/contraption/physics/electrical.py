"""Focused lumped electrical reference systems."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .backend import Array, Backend
from .simulator import SimulationResult, simulate


def _parameter(parameters: Mapping[str, Array], name: str) -> Array:
    try:
        return parameters[name]
    except KeyError as exc:
        raise KeyError(f"Required physical parameter {name!r} was not supplied") from exc


def _control(controls: Mapping[str, Array], *names: str, default: Any = 0.0) -> Any:
    return next((controls[name] for name in names if name in controls), default)


class RCCircuit:
    """Series resistor-capacitor circuit driven by a voltage source."""

    state_names = ("capacitor_voltage",)
    control_names = ("voltage",)
    output_names = ("capacitor_voltage", "resistor_current")

    def __init__(self, resistance: Any = 1_000.0, capacitance: Any = 1e-3, initial_voltage: Any = 0.0) -> None:
        self.default_parameters = {"resistance": resistance, "capacitance": capacitance}
        self.parameter_bounds = {"resistance": (1e-12, None), "capacitance": (1e-15, None)}
        self.parameter_uncertainty = {"resistance": {"std": 0.01 * resistance}, "capacitance": {"std": 0.02 * capacitance}}
        self.initial_state = [initial_voltage]

    def derivative(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Array:
        voltage = _control(controls, "voltage", "source_voltage", "input_voltage")
        return ((voltage - state[:, 0]) / (_parameter(parameters, "resistance") * _parameter(parameters, "capacitance")))[:, None]

    def observe(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Mapping[str, Array]:
        voltage = _control(controls, "voltage", "source_voltage", "input_voltage")
        return {"capacitor_voltage": state[:, 0], "resistor_current": (voltage - state[:, 0]) / _parameter(parameters, "resistance")}

    def simulate(self, **options: Any) -> SimulationResult:
        return simulate(self, **options)


class RLCircuit:
    """Series resistor-inductor circuit driven by a voltage source."""

    state_names = ("inductor_current",)
    control_names = ("voltage",)
    output_names = ("inductor_current", "resistor_voltage", "inductor_voltage")

    def __init__(self, resistance: Any = 10.0, inductance: Any = 0.1, initial_current: Any = 0.0) -> None:
        self.default_parameters = {"resistance": resistance, "inductance": inductance}
        self.parameter_bounds = {"resistance": (1e-12, None), "inductance": (1e-15, None)}
        self.parameter_uncertainty = {"resistance": {"std": 0.01 * resistance}, "inductance": {"std": 0.02 * inductance}}
        self.initial_state = [initial_current]

    def derivative(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Array:
        voltage = _control(controls, "voltage", "source_voltage", "input_voltage")
        return ((voltage - _parameter(parameters, "resistance") * state[:, 0]) / _parameter(parameters, "inductance"))[:, None]

    def observe(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Mapping[str, Array]:
        source = _control(controls, "voltage", "source_voltage", "input_voltage")
        resistor = _parameter(parameters, "resistance") * state[:, 0]
        return {"inductor_current": state[:, 0], "resistor_voltage": resistor, "inductor_voltage": source - resistor}

    def simulate(self, **options: Any) -> SimulationResult:
        return simulate(self, **options)


__all__ = ["RCCircuit", "RLCircuit"]
