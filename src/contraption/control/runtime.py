"""Backend-native interpreter for validated ``control-1`` programs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping
import warnings

import numpy as np

from ..physics.backend import Backend, get_backend
from ..physics.dsl import Binary, Call, Comparison, Conditional, Expression, Literal, Symbol, Unary, parse_expression
from .observer import (
    AffineObserverModel,
    ObservabilityDiagnostic,
    observability_diagnostics,
)
from .specs import ControlSpec, control_digest


class ControlRuntimeError(RuntimeError):
    """A valid controller cannot execute the supplied frame."""


class UnobservableImplicitInputWarning(RuntimeWarning):
    """A declared implicit input has no measurement sensitivity."""


@dataclass(frozen=True, slots=True)
class ImplicitInputState:
    mean: Any
    variance: Any


@dataclass(frozen=True, slots=True)
class ControlFrame:
    time: float
    active_mode: str
    next_mode: str
    outputs: Mapping[str, Any]
    registers: Mapping[str, Any]
    implicit_inputs: Mapping[str, ImplicitInputState]
    derived: Mapping[str, Any]
    emergency: bool


def _lookup(name: str, values: Mapping[str, Any]) -> Any:
    if name == "pi":
        return math.pi
    if name == "e":
        return math.e
    try:
        return values[name]
    except KeyError as exc:
        raise ControlRuntimeError(f"no value supplied for control symbol {name!r}") from exc


def evaluate_control_expression(
    expression: str | Expression,
    values: Mapping[str, Any],
    backend: Backend,
) -> Any:
    """Evaluate the shared PMDL expression IR without crossing backend boundaries."""

    node = parse_expression(expression) if isinstance(expression, str) else expression

    def evaluate(item: Expression) -> Any:
        if isinstance(item, Literal):
            if isinstance(item.value, bool):
                boolean_dtype = (
                    backend.torch.bool
                    if bool(getattr(backend, "is_torch", False))
                    else np.bool_
                )
                return backend.asarray(item.value, dtype=boolean_dtype)
            return backend.asarray(item.value)
        if isinstance(item, Symbol):
            if item.name == "pi":
                return backend.asarray(math.pi)
            if item.name == "e":
                return backend.asarray(math.e)
            return _lookup(item.name, values)
        if isinstance(item, Unary):
            value = evaluate(item.operand)
            if item.operator == "+":
                return +value
            if item.operator == "-":
                return -value
            if item.operator == "not":
                return backend.logical_not(value)
            raise ControlRuntimeError(f"unsupported unary operator {item.operator!r}")
        if isinstance(item, Binary):
            left = evaluate(item.left)
            if item.operator == "and":
                return (
                    evaluate(item.right)
                    if _truth(left, "left operand of and")
                    else left
                )
            if item.operator == "or":
                return (
                    left
                    if _truth(left, "left operand of or")
                    else evaluate(item.right)
                )
            right = evaluate(item.right)
            if item.operator == "+":
                return left + right
            if item.operator == "-":
                return left - right
            if item.operator == "*":
                return left * right
            if item.operator == "/":
                return left / right
            if item.operator == "**":
                return left**right
            raise ControlRuntimeError(f"unsupported binary operator {item.operator!r}")
        if isinstance(item, Comparison):
            left, right = evaluate(item.left), evaluate(item.right)
            operations = {
                "<": lambda: left < right,
                "<=": lambda: left <= right,
                ">": lambda: left > right,
                ">=": lambda: left >= right,
                "==": lambda: left == right,
                "!=": lambda: left != right,
            }
            try:
                return operations[item.operator]()
            except KeyError as exc:
                raise ControlRuntimeError(
                    f"unsupported comparison {item.operator!r}"
                ) from exc
        if isinstance(item, Conditional):
            condition = evaluate(item.condition)
            return evaluate(
                item.when_true
                if _truth(condition, "conditional expression")
                else item.when_false
            )
        if isinstance(item, Call):
            if item.function == "der":
                raise ControlRuntimeError("der() is PMDL-only and forbidden in control expressions")
            if item.function == "where":
                condition = evaluate(item.arguments[0])
                return evaluate(
                    item.arguments[1]
                    if _truth(condition, "where() condition")
                    else item.arguments[2]
                )
            args = tuple(evaluate(argument) for argument in item.arguments)
            functions = {
                "abs": backend.abs,
                "sqrt": backend.sqrt,
                "sin": backend.sin,
                "cos": backend.cos,
                "tan": backend.tan,
                "tanh": backend.tanh,
                "asin": backend.asin,
                "acos": backend.acos,
                "atan": backend.atan,
                "atan2": backend.atan2,
                "exp": backend.exp,
                "log": backend.log,
                "log10": backend.log10,
                "min": backend.minimum,
                "max": backend.maximum,
                "clip": backend.clip,
                "sign": backend.sign,
            }
            if item.function == "smooth_abs":
                epsilon = args[1] if len(args) == 2 else 1e-12
                return backend.sqrt(args[0] * args[0] + epsilon * epsilon)
            try:
                return functions[item.function](*args)
            except KeyError as exc:
                raise ControlRuntimeError(
                    f"unsupported control function {item.function!r}"
                ) from exc
        raise ControlRuntimeError(f"unsupported expression node {type(item).__name__}")

    return evaluate(node)


def _truth(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise ControlRuntimeError(f"{context} must be scalar")
        return bool(value.detach().item())
    array = np.asarray(value)
    if array.size != 1:
        raise ControlRuntimeError(f"{context} must be scalar")
    return bool(array.reshape(-1)[0])


def _diagnostic_number(value: Any, context: str) -> float:
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise ControlRuntimeError(f"{context} must be scalar")
        result = float(value.detach().item())
    else:
        array = np.asarray(value)
        if array.size != 1:
            raise ControlRuntimeError(f"{context} must be scalar")
        result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ControlRuntimeError(f"{context} must be finite")
    return result


def _require_finite_array(value: Any, context: str) -> None:
    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ControlRuntimeError(f"{context} contains a non-finite value")


class ControlRuntime:
    """Execute the same synchronous controller represented by generated targets."""

    def __init__(
        self,
        spec: ControlSpec,
        *,
        observer: AffineObserverModel | None = None,
        backend: str | Backend = "numpy",
        device: str | None = None,
        emit_observability_warnings: bool = True,
    ) -> None:
        if not isinstance(spec, ControlSpec):
            raise TypeError("ControlRuntime requires a parsed ControlSpec")
        self.spec = spec
        self.backend = get_backend(backend, device=device)
        self._explicit_inputs = {item.name: item for item in spec.explicit_inputs}
        self._outputs = {item.name: item for item in spec.outputs}
        self._registers = {item.name: item for item in spec.registers}
        self._modes = {item.name: item for item in spec.modes}
        self._implicit_inputs = {item.name: item for item in spec.implicit_inputs}
        if bool(spec.implicit_inputs) != (observer is not None):
            raise ControlRuntimeError(
                "a resolved affine observer is required exactly when implicit inputs exist"
            )
        if observer is not None:
            if not isinstance(observer, AffineObserverModel):
                raise TypeError("observer must be an AffineObserverModel")
            expected_measurements = tuple(
                item.name
                for item in spec.explicit_inputs
                if item.source == "sensor" and item.dtype == "real"
            )
            if (
                observer.controller_id != spec.id
                or observer.controller_digest != control_digest(spec)
                or observer.latent_names != tuple(self._implicit_inputs)
                or observer.measurement_names != expected_measurements
                or len(observer.input_names) != len(set(observer.input_names))
                or any(name not in self._outputs for name in observer.input_names)
                or any(
                    self._outputs[name].dtype != "real"
                    for name in observer.input_names
                )
                or observer.input_names
                != tuple(name for name in self._outputs if name in observer.input_names)
                or abs(observer.period_s - spec.period_s) > 1e-15
            ):
                raise ControlRuntimeError(
                    "observer identity/manifests do not match the controller"
                )
        self.observer = observer
        self.observability = (
            () if observer is None else observability_diagnostics(observer)
        )
        if emit_observability_warnings:
            for diagnostic in self.observability:
                if not diagnostic.observable:
                    warnings.warn(
                        f"controller {spec.id!r} implicit input {diagnostic.implicit_input!r} "
                        f"bound to {diagnostic.variable!r} is unobservable from "
                        f"{list(diagnostic.measurement_names)} at the admitted operating point",
                        UnobservableImplicitInputWarning,
                        stacklevel=2,
                    )
        self._observer_arrays = (
            {}
            if observer is None
            else {
                name: self.backend.asarray(
                    np.array(getattr(observer, name), copy=True)
                )
                for name in (
                    "A",
                    "B",
                    "dynamics_bias",
                    "C",
                    "D",
                    "measurement_bias",
                    "L",
                    "M",
                    "latent_bias",
                    "process_covariance",
                    "transition",
                    "discrete_input",
                    "discrete_bias",
                    "discrete_process_covariance",
                    "measurement_variance",
                    "initial_state",
                    "initial_covariance",
                )
            }
        )
        self.reset()

    def _value(self, value: Any, dtype: str) -> Any:
        if dtype == "bool":
            boolean_dtype = (
                self.backend.torch.bool
                if bool(getattr(self.backend, "is_torch", False))
                else np.bool_
            )
            return self.backend.asarray(value, dtype=boolean_dtype)
        return self.backend.asarray(value)

    def reset(self) -> None:
        self.mode = self.spec.initial_mode
        self.time = 0.0
        self.time_in_mode = 0.0
        self.registers = {
            item.name: self._value(item.initial, item.dtype)
            for item in self.spec.registers
        }
        self.outputs = {
            item.name: self._value(item.default, item.dtype)
            for item in self.spec.outputs
        }
        if self.observer is None:
            self.observer_state = None
            self.observer_covariance = None
            self.implicit_inputs = {}
        else:
            self.observer_state = self.backend.clone(
                self._observer_arrays["initial_state"]
            )
            self.observer_covariance = self.backend.clone(
                self._observer_arrays["initial_covariance"]
            )
            self.implicit_inputs = self._project_implicit_inputs()

    def _normalize_explicit_inputs(
        self, supplied: Mapping[str, Any]
    ) -> dict[str, Any]:
        unknown = sorted(set(supplied) - set(self._explicit_inputs))
        if unknown:
            raise ControlRuntimeError(f"unknown controller input(s): {unknown}")
        result: dict[str, Any] = {}
        for name, spec in self._explicit_inputs.items():
            value = self._value(supplied.get(name, spec.default), spec.dtype)
            if spec.dtype == "bool":
                _truth(value, f"input {name}")
            else:
                number = _diagnostic_number(value, f"input {name}")
                if not spec.bounds.contains(number):
                    raise ControlRuntimeError(
                        f"input {name!r}={number} is outside bounds "
                        f"[{spec.bounds.lower}, {spec.bounds.upper}]"
                    )
            result[name] = value
        return result

    def _observer_control_vector(self) -> Any:
        assert self.observer is not None
        return self.backend.stack(
            [self.outputs[name] for name in self.observer.input_names]
        )

    def _project_implicit_inputs(self) -> dict[str, ImplicitInputState]:
        assert self.observer is not None
        arrays = self._observer_arrays
        control = self._observer_control_vector()
        means = (
            arrays["L"] @ self.observer_state
            + arrays["M"] @ control
            + arrays["latent_bias"]
        )
        result: dict[str, ImplicitInputState] = {}
        for index, name in enumerate(self.observer.latent_names):
            row = arrays["L"][index]
            variance = row @ self.observer_covariance @ row
            variance = self.backend.maximum(variance, self.backend.asarray(0.0))
            latent_spec = self._implicit_inputs[name]
            _diagnostic_number(means[index], f"implicit input {name}.raw_mean")
            mean = self.backend.clip(
                means[index], latent_spec.bounds.lower, latent_spec.bounds.upper
            )
            _diagnostic_number(mean, f"implicit input {name}.mean")
            _diagnostic_number(variance, f"implicit input {name}.variance")
            result[name] = ImplicitInputState(mean, variance)
        return result

    def _update_observer(
        self, explicit_inputs: Mapping[str, Any], elapsed: float
    ) -> None:
        if self.observer is None:
            return
        arrays = self._observer_arrays
        nx = len(self.observer.state_names)
        identity = self.backend.eye(nx)
        control = self._observer_control_vector()
        state = (
            arrays["transition"] @ self.observer_state
            + arrays["discrete_input"] @ control
            + arrays["discrete_bias"]
        )
        covariance = (
            arrays["transition"]
            @ self.observer_covariance
            @ arrays["transition"].T
            + arrays["discrete_process_covariance"]
        )
        for index, name in enumerate(self.observer.measurement_names):
            row = arrays["C"][index]
            measurement = explicit_inputs[name]
            predicted = (
                row @ state
                + arrays["D"][index] @ control
                + arrays["measurement_bias"][index]
            )
            innovation = measurement - predicted
            innovation_variance = (
                row @ covariance @ row + arrays["measurement_variance"][index]
            )
            if _diagnostic_number(
                innovation_variance, f"observer innovation variance for {name}"
            ) <= 0.0:
                raise ControlRuntimeError(
                    f"observer innovation variance for {name!r} is not positive"
                )
            gain = (covariance @ row) / innovation_variance
            state = state + gain * innovation
            residual = identity - gain[:, None] * row[None, :]
            covariance = (
                residual @ covariance @ residual.T
                + (gain[:, None] * arrays["measurement_variance"][index])
                @ gain[None, :]
            )
            covariance = 0.5 * (covariance + covariance.T)
        _require_finite_array(state, "observer state")
        _require_finite_array(covariance, "observer covariance")
        self.observer_state = state
        self.observer_covariance = covariance
        self.implicit_inputs = self._project_implicit_inputs()

    def _environment(
        self, explicit_inputs: Mapping[str, Any], dt: float
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "time": self.backend.asarray(self.time),
            "time_in_mode": self.backend.asarray(self.time_in_mode),
            "dt": self.backend.asarray(dt),
        }
        values.update(
            {f"input.{name}": value for name, value in explicit_inputs.items()}
        )
        values.update(
            {
                f"parameter.{item.name}": self._value(item.default, item.dtype)
                for item in self.spec.parameters
            }
        )
        values.update(
            {f"register.{name}": value for name, value in self.registers.items()}
        )
        values.update({f"output.{name}": value for name, value in self.outputs.items()})
        for name, state in self.implicit_inputs.items():
            values[f"implicit.{name}.mean"] = state.mean
            values[f"implicit.{name}.variance"] = state.variance
            values[f"implicit.{name}.std"] = self.backend.sqrt(state.variance)
        return values

    def step(
        self,
        explicit_inputs: Mapping[str, Any] | None = None,
    ) -> ControlFrame:
        previous_state = (
            None
            if self.observer_state is None
            else self.backend.clone(self.observer_state)
        )
        previous_covariance = (
            None
            if self.observer_covariance is None
            else self.backend.clone(self.observer_covariance)
        )
        previous_implicit = dict(self.implicit_inputs)
        previous_outputs = dict(self.outputs)
        previous_registers = dict(self.registers)
        previous_mode = self.mode
        previous_time = self.time
        previous_time_in_mode = self.time_in_mode
        try:
            return self._step_impl(explicit_inputs)
        except Exception:
            self.observer_state = previous_state
            self.observer_covariance = previous_covariance
            self.implicit_inputs = previous_implicit
            self.outputs = previous_outputs
            self.registers = previous_registers
            self.mode = previous_mode
            self.time = previous_time
            self.time_in_mode = previous_time_in_mode
            raise

    def _step_impl(
        self,
        explicit_inputs: Mapping[str, Any] | None = None,
    ) -> ControlFrame:
        elapsed = self.spec.period_s
        normalized = self._normalize_explicit_inputs(
            {} if explicit_inputs is None else explicit_inputs
        )
        self._update_observer(normalized, elapsed)
        values = self._environment(normalized, elapsed)
        derived: dict[str, Any] = {}
        for item in self.spec.derived:
            result = evaluate_control_expression(item.expression, values, self.backend)
            if item.dtype == "real":
                _diagnostic_number(result, f"derived value {item.name}")
            derived[item.name] = result
            values[f"derived.{item.name}"] = result

        emergency = (
            False
            if self.spec.emergency_when is None
            else _truth(
                evaluate_control_expression(
                    self.spec.emergency_when, values, self.backend
                ),
                "emergency condition",
            )
        )
        active_name = self.mode
        active = self._modes[active_name]
        next_outputs: dict[str, Any] = {}
        for name, output_spec in self._outputs.items():
            raw = evaluate_control_expression(
                active.outputs[name], values, self.backend
            )
            if emergency and output_spec.emergency_value is not None:
                raw = self._value(output_spec.emergency_value, output_spec.dtype)
            if output_spec.dtype == "real":
                raw = self.backend.clip(
                    raw, output_spec.bounds.lower, output_spec.bounds.upper
                )
                if output_spec.slew_rate is not None and not emergency:
                    previous = self.outputs[name]
                    delta = output_spec.slew_rate * elapsed
                    raw = self.backend.minimum(
                        self.backend.maximum(raw, previous - delta),
                        previous + delta,
                    )
            next_outputs[name] = raw

        next_registers = dict(self.registers)
        for name, expression in active.updates.items():
            register_spec = self._registers[name]
            value = evaluate_control_expression(expression, values, self.backend)
            if register_spec.dtype == "real":
                _diagnostic_number(value, f"register {name}")
                value = self.backend.clip(
                    value,
                    register_spec.bounds.lower,
                    register_spec.bounds.upper,
                )
            next_registers[name] = value

        next_mode = active_name
        for transition in sorted(
            active.transitions, key=lambda item: item.priority, reverse=True
        ):
            if _truth(
                evaluate_control_expression(
                    transition.guard, values, self.backend
                ),
                f"transition {active_name}->{transition.target}",
            ):
                next_mode = transition.target
                break

        for name, output_spec in self._outputs.items():
            if output_spec.dtype == "real":
                _diagnostic_number(next_outputs[name], f"final output {name}")
        for name, register_spec in self._registers.items():
            if register_spec.dtype == "real":
                _diagnostic_number(next_registers[name], f"final register {name}")
        next_time = self.time + elapsed
        next_time_in_mode = (
            0.0 if next_mode != active_name else self.time_in_mode + elapsed
        )
        _diagnostic_number(next_time, "controller time")
        _diagnostic_number(next_time_in_mode, "controller time_in_mode")

        self.outputs = next_outputs
        self.registers = next_registers
        self.mode = next_mode
        self.time = next_time
        self.time_in_mode = next_time_in_mode
        return ControlFrame(
            self.time,
            active_name,
            next_mode,
            dict(self.outputs),
            dict(self.registers),
            dict(self.implicit_inputs),
            derived,
            emergency,
        )


__all__ = [
    "ControlFrame",
    "ControlRuntime",
    "ControlRuntimeError",
    "ImplicitInputState",
    "ObservabilityDiagnostic",
    "UnobservableImplicitInputWarning",
    "evaluate_control_expression",
    "observability_diagnostics",
]
