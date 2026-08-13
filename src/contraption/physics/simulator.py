"""Differentiable, vectorized offline simulation for Phase 1.

The engine accepts either an explicit ODE object exposing ``derivative`` or an
acausal descriptor object exposing the universal residual
``F(t, z, zdot, theta, u) = 0``.  Explicit systems use RK4 by default;
descriptor systems use an implicit backward-Euler/Newton path.  Both paths
operate on a leading Monte Carlo batch dimension and share NumPy/PyTorch
backend operations.

Models are structural protocols rather than subclasses.  Consequently the DSL
layer can remain a data-only module, and future physical domains can join the
simulator without importing or modifying this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import inspect
import math
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from ..control import ControlFrame, ControlRuntime
from ..verification import evaluate_verification
from .backend import Array, Backend, as_jsonable, get_backend, infer_backend
from .dsl import Binary, Call, Comparison, Conditional, Expression, Literal, Symbol, Unary, parse_expression
from .uq import (
    DistributionSummary,
    sample_gaussian,
    sample_parameters,
    split_seed,
    summarize_samples,
)


class ExplicitDynamics(Protocol):
    state_names: Sequence[str]

    def derivative(
        self,
        t: Any,
        state: Array,
        parameters: Mapping[str, Array],
        controls: Mapping[str, Array],
        backend: Backend,
    ) -> Array: ...


class DescriptorDynamics(Protocol):
    state_names: Sequence[str]

    def residual(
        self,
        t: Any,
        state: Array,
        state_derivative: Array,
        parameters: Mapping[str, Array],
        controls: Mapping[str, Array],
        backend: Backend,
    ) -> Array: ...


class UnsupportedPMDLSemanticsError(NotImplementedError):
    """A declared PMDL behavior cannot be honored by this simulator.

    PMDL declarations are part of the executable contract.  The simulator
    raises this error at admission instead of accepting a model and silently
    dropping semantics that could change its result.
    """


@dataclass(frozen=True)
class SimulationConfig:
    """Reusable defaults for :class:`OfflineSimulator`."""

    dt: float = 0.01
    num_samples: int = 256
    seed: int = 0
    backend: str | Backend = "numpy"
    device: str | None = None
    dtype: Any | None = None
    integrator: str = "auto"
    quantiles: tuple[float, ...] = (0.025, 0.5, 0.975)
    confidence_level: float = 0.95
    use_model_uncertainty: bool = True
    process_noise: bool = True
    newton_tolerance: float = 1e-9
    newton_max_iterations: int = 12

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.integrator not in {"auto", "euler", "rk4", "implicit_euler"}:
            raise ValueError(f"Unsupported integrator {self.integrator!r}")


@dataclass(frozen=True)
class ControllerTrace:
    """Hardware-equivalent controller outputs and posterior state over time.

    Numeric arrays use ``[sample,time,channel]`` layout and stay on the active
    backend so Torch traces remain differentiable. Mode names are diagnostic
    discrete data indexed as ``[time][sample]``.
    """

    controller_id: str
    output_names: tuple[str, ...]
    output_samples: Array
    implicit_input_names: tuple[str, ...]
    implicit_means: Array
    implicit_variances: Array
    emergency_samples: Array
    active_modes: tuple[tuple[str, ...], ...]
    tick_mask: tuple[bool, ...]
    frame_times_s: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "output_names": list(self.output_names),
            "output_samples": as_jsonable(self.output_samples),
            "implicit_input_names": list(self.implicit_input_names),
            "implicit_means": as_jsonable(self.implicit_means),
            "implicit_variances": as_jsonable(self.implicit_variances),
            "emergency_samples": as_jsonable(self.emergency_samples),
            "active_modes": [list(frame) for frame in self.active_modes],
            "tick_mask": list(self.tick_mask),
            "frame_times_s": list(self.frame_times_s),
        }


@dataclass(frozen=True)
class SimulationResult:
    """Trajectory ensemble plus pointwise state and output distributions.

    ``samples`` is shaped ``[sample, time, state]``.  ``output_samples`` uses
    the analogous ``[sample, time, output]`` layout.  Torch tensors remain live
    autograd values; only :meth:`to_dict` intentionally crosses the graph
    boundary to make a JSON-compatible representation.
    """

    time: Array
    state_names: tuple[str, ...]
    samples: Array
    output_names: tuple[str, ...]
    output_samples: Array
    summary: DistributionSummary
    output_summary: DistributionSummary
    controller_traces: Mapping[str, ControllerTrace] = field(default_factory=dict)
    verification_reports: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mean(self) -> Array:
        return self.summary.mean

    @property
    def covariance(self) -> Array:
        return self.summary.covariance

    @property
    def quantiles(self) -> Mapping[float, Array]:
        return self.summary.quantiles

    @property
    def confidence_interval(self) -> tuple[Array, Array]:
        return self.summary.interval

    @property
    def named_samples(self) -> dict[str, Array]:
        return {name: self.samples[..., index] for index, name in enumerate(self.state_names)}

    @property
    def named_outputs(self) -> dict[str, Array]:
        return {
            name: self.output_samples[..., index]
            for index, name in enumerate(self.output_names)
        }

    def series(self, name: str, *, outputs_first: bool = False) -> Array:
        """Return a named trajectory ensemble shaped ``[sample, time]``."""

        if outputs_first and name in self.output_names:
            return self.output_samples[..., self.output_names.index(name)]
        if name in self.state_names:
            return self.samples[..., self.state_names.index(name)]
        if name in self.output_names:
            return self.output_samples[..., self.output_names.index(name)]
        raise KeyError(f"Unknown state/output {name!r}")

    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "time": as_jsonable(self.time),
            "state_names": list(self.state_names),
            "output_names": list(self.output_names),
            "state_distribution": self.summary.to_dict(),
            "output_distribution": self.output_summary.to_dict(),
            "controllers": {
                name: trace.to_dict() for name, trace in self.controller_traces.items()
            },
            "verifications": {
                name: report.to_dict() for name, report in self.verification_reports.items()
            },
            "metadata": as_jsonable(dict(self.metadata)),
        }
        if include_samples:
            payload["samples"] = as_jsonable(self.samples)
            payload["output_samples"] = as_jsonable(self.output_samples)
        return payload


@dataclass(frozen=True)
class ResidualSystem:
    """Typed convenience wrapper for a residual descriptor callable."""

    state_names: tuple[str, ...]
    residual_function: Callable[..., Any]
    initial_state: Any
    default_parameters: Mapping[str, Any] = field(default_factory=dict)
    control_names: tuple[str, ...] = ()
    parameter_bounds: Mapping[str, tuple[Any | None, Any | None]] = field(default_factory=dict)
    jacobian_function: Callable[..., Any] | None = None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.default_parameters)

    def residual(
        self,
        t: Any,
        state: Array,
        state_derivative: Array,
        parameters: Mapping[str, Array],
        controls: Mapping[str, Array],
        backend: Backend,
    ) -> Any:
        return _invoke(self.residual_function, t, state, state_derivative, parameters, controls, backend=backend)

    def jacobian(
        self,
        t: Any,
        state: Array,
        state_derivative: Array,
        parameters: Mapping[str, Array],
        controls: Mapping[str, Array],
        backend: Backend,
    ) -> Any:
        if self.jacobian_function is None:
            raise AttributeError("No analytic descriptor Jacobian was supplied")
        return _invoke(self.jacobian_function, t, state, state_derivative, parameters, controls, backend=backend)


@dataclass(frozen=True)
class Linearization:
    """Continuous-time local linearization ``zdot ~= A dz + B du``."""

    state_matrix: Array
    control_matrix: Array
    state_names: tuple[str, ...]
    control_names: tuple[str, ...]
    operating_state: Array
    operating_controls: Mapping[str, Array]

    @property
    def A(self) -> Array:
        return self.state_matrix

    @property
    def B(self) -> Array:
        return self.control_matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "A": as_jsonable(self.A),
            "B": as_jsonable(self.B),
            "state_names": list(self.state_names),
            "control_names": list(self.control_names),
            "operating_state": as_jsonable(self.operating_state),
            "operating_controls": as_jsonable(dict(self.operating_controls)),
        }


def _invoke(function: Callable[..., Any], *args: Any, backend: Backend) -> Any:
    """Call a protocol method with an optional backend keyword."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):  # pragma: no cover - rare extension callable
        return function(*args, backend)
    parameters = signature.parameters
    if "backend" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return function(*args, backend=backend)
    return function(*args)


def _as_vector(value: Any, backend: Backend, batch_size: int) -> Array:
    if isinstance(value, Mapping):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        pieces = [backend.asarray(item) for item in value]
        pieces = [
            backend.broadcast_to(piece, (batch_size,)) if len(piece.shape) == 0 else piece
            for piece in pieces
        ]
        result = backend.stack(pieces, axis=-1)
    else:
        result = backend.asarray(value)
    if len(result.shape) == 1:
        result = backend.broadcast_to(result, (batch_size, int(result.shape[0])))
    if len(result.shape) != 2 or int(result.shape[0]) != batch_size:
        raise ValueError(
            f"Dynamics must return [sample, variable], received shape {tuple(result.shape)}"
        )
    return result


def _state_names(system: Any) -> tuple[str, ...]:
    names = getattr(system, "state_names", None)
    if names is None and hasattr(system, "states"):
        names = [getattr(item, "name", str(index)) for index, item in enumerate(system.states)]
    if names is None:
        raise TypeError("A simulation system must expose state_names")
    names = tuple(str(name) for name in names)
    if not names or len(set(names)) != len(names):
        raise ValueError("state_names must be non-empty and unique")
    return names


def _parameter_defaults(system: Any) -> dict[str, Any]:
    defaults = getattr(system, "default_parameters", None)
    if defaults is not None:
        return dict(defaults)
    parameters = getattr(system, "parameters", None)
    if isinstance(parameters, Mapping):
        return dict(parameters)
    if parameters is not None:
        result = {}
        for item in parameters:
            name = getattr(item, "name")
            if hasattr(item, "default"):
                result[name] = item.default
            elif hasattr(item, "value"):
                result[name] = item.value
            else:
                raise TypeError(f"Parameter {name!r} has no default/value")
        return result
    return {}


def _parameter_bounds(system: Any) -> dict[str, tuple[Any | None, Any | None]]:
    explicit = getattr(system, "parameter_bounds", None)
    if explicit is not None:
        return dict(explicit)
    result: dict[str, tuple[Any | None, Any | None]] = {}
    for item in getattr(system, "parameters", ()):
        bounds = getattr(item, "bounds", None)
        if bounds is not None:
            if hasattr(bounds, "lower") and hasattr(bounds, "upper"):
                result[str(item.name)] = (bounds.lower, bounds.upper)
            else:
                result[str(item.name)] = tuple(bounds)
    return result


def _parameter_uncertainty(system: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Translate declarative ParameterSpec uncertainty into sampler inputs.

    The sampler itself owns distribution semantics.  Keeping this bridge
    structural means simulation can consume a ModelSpec without importing its
    concrete spec classes or mutating the otherwise data-only model.
    """

    declared: dict[str, Any] = {}
    correlated: list[str] = []
    for item in getattr(system, "parameters", ()):
        uncertainty = getattr(item, "uncertainty", None)
        if uncertainty is None:
            continue
        if isinstance(uncertainty, Mapping):
            distribution = str(uncertainty.get("distribution", "fixed"))
            values = uncertainty.get("parameters", {})
            correlation_group = uncertainty.get("correlation_group")
        else:
            distribution = str(getattr(uncertainty, "distribution", "fixed"))
            values = getattr(uncertainty, "parameters", {})
            correlation_group = getattr(uncertainty, "correlation_group", None)
        if distribution == "fixed":
            continue
        if correlation_group is not None:
            correlated.append(f"{item.name}:{correlation_group}")
        declared[str(item.name)] = {
            "distribution": distribution,
            "parameters": dict(values),
        }
    return declared, tuple(correlated)


def _lookup_expression_symbol(name: str, environment: Mapping[str, Any]) -> Any:
    if name == "pi":
        return math.pi
    if name == "e":
        return math.e
    if name in environment:
        return environment[name]
    if "." in name:
        current: Any = environment
        try:
            for part in name.split("."):
                current = current[part] if isinstance(current, Mapping) else getattr(current, part)
            return current
        except (KeyError, AttributeError, TypeError) as exc:
            raise KeyError(f"No value supplied for PMDL symbol {name!r}") from exc
    raise KeyError(f"No value supplied for PMDL symbol {name!r}")


def _evaluate_backend_expression(
    expression: Expression,
    environment: Mapping[str, Any],
    backend: Backend,
) -> Any:
    """Evaluate allow-listed PMDL IR using only backend-native primitives.

    This mirrors the safe DSL interpreter without dispatching numerical calls
    through NumPy.  In particular, torch tensors stay on their requested
    device and retain their autograd graph.
    """

    if isinstance(expression, Literal):
        return expression.value
    if isinstance(expression, Symbol):
        return _lookup_expression_symbol(expression.name, environment)
    if isinstance(expression, Unary):
        operand = _evaluate_backend_expression(expression.operand, environment, backend)
        if expression.operator == "+":
            return +operand
        if expression.operator == "-":
            return -operand
        if expression.operator == "not":
            return backend.logical_not(operand)
        raise ValueError(f"Unsupported PMDL unary operator {expression.operator!r}")
    if isinstance(expression, Binary):
        left = _evaluate_backend_expression(expression.left, environment, backend)
        right = _evaluate_backend_expression(expression.right, environment, backend)
        if expression.operator == "+":
            return left + right
        if expression.operator == "-":
            return left - right
        if expression.operator == "*":
            return left * right
        if expression.operator == "/":
            return left / right
        if expression.operator == "**":
            return left**right
        if expression.operator == "and":
            return backend.logical_and(left, right)
        if expression.operator == "or":
            return backend.logical_or(left, right)
        raise ValueError(f"Unsupported PMDL binary operator {expression.operator!r}")
    if isinstance(expression, Comparison):
        left = _evaluate_backend_expression(expression.left, environment, backend)
        right = _evaluate_backend_expression(expression.right, environment, backend)
        operations = {
            "<": lambda: left < right,
            "<=": lambda: left <= right,
            ">": lambda: left > right,
            ">=": lambda: left >= right,
            "==": lambda: left == right,
            "!=": lambda: left != right,
        }
        try:
            return operations[expression.operator]()
        except KeyError as exc:
            raise ValueError(f"Unsupported PMDL comparison {expression.operator!r}") from exc
    if isinstance(expression, Call):
        if expression.function == "der":
            argument = expression.arguments[0]
            if not isinstance(argument, Symbol):
                raise ValueError("PMDL der() requires a state symbol")
            return _lookup_expression_symbol(f"{argument.name}_dot", environment)
        arguments = tuple(
            _evaluate_backend_expression(argument, environment, backend)
            for argument in expression.arguments
        )
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
            "where": backend.where,
        }
        if expression.function == "smooth_abs":
            epsilon = arguments[1] if len(arguments) == 2 else 1e-12
            return backend.sqrt(arguments[0] * arguments[0] + epsilon * epsilon)
        try:
            return functions[expression.function](*arguments)
        except KeyError as exc:
            raise ValueError(f"Unsupported PMDL function {expression.function!r}") from exc
    if isinstance(expression, Conditional):
        condition = _evaluate_backend_expression(expression.condition, environment, backend)
        when_true = _evaluate_backend_expression(expression.when_true, environment, backend)
        when_false = _evaluate_backend_expression(expression.when_false, environment, backend)
        return backend.where(condition, when_true, when_false)
    raise TypeError(f"Unsupported PMDL expression node {type(expression).__name__}")


def _require_known_controls(system: Any, controls: Mapping[str, Any]) -> None:
    """Reject misspelled or undeclared control channels before evaluation.

    An omitted declared channel retains the documented zero/default behavior,
    but an extra channel is almost always a wiring or spelling error.  Allowing
    it to disappear into a model would produce a plausible-looking, incorrect
    trajectory, so this check intentionally runs on every dynamically supplied
    control frame.
    """

    declared = tuple(str(name) for name in getattr(system, "control_names", ()))
    unknown = sorted(set(controls) - set(declared))
    if unknown:
        model_name = type(getattr(system, "model", system)).__name__
        raise KeyError(
            f"{model_name} received unknown control channel(s) {unknown}; "
            f"declared channels are {list(declared)}"
        )


def _validity_contract(
    system: Any,
) -> tuple[dict[str, Any], float | None]:
    validity = getattr(system, "validity", None)
    if validity is None:
        return {}, None
    raw_ranges = (
        validity.get("ranges", {})
        if isinstance(validity, Mapping)
        else getattr(validity, "ranges", {})
    )
    if not isinstance(raw_ranges, Mapping):
        raise TypeError("PMDL validity.ranges must be a symbol-to-bounds mapping")
    raw_maximum = (
        validity.get("max_timestep")
        if isinstance(validity, Mapping)
        else getattr(validity, "max_timestep", None)
    )
    maximum = None if raw_maximum is None else float(raw_maximum)
    if maximum is not None and (not math.isfinite(maximum) or maximum <= 0.0):
        raise ValueError("PMDL validity.max_timestep must be finite and positive")
    return dict(raw_ranges), maximum


def _validity_bounds(value: Any, symbol: str) -> tuple[float | None, float | None]:
    if isinstance(value, Mapping):
        lower, upper = value.get("lower"), value.get("upper")
    else:
        lower, upper = getattr(value, "lower", None), getattr(value, "upper", None)
    lower = None if lower is None else float(lower)
    upper = None if upper is None else float(upper)
    if lower is not None and not math.isfinite(lower):
        raise ValueError(f"PMDL validity lower bound for {symbol!r} must be finite")
    if upper is not None and not math.isfinite(upper):
        raise ValueError(f"PMDL validity upper bound for {symbol!r} must be finite")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(
            f"PMDL validity lower bound exceeds upper bound for {symbol!r}"
        )
    return lower, upper


def _validity_diagnostic(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _require_runtime_validity(
    system: Any,
    *,
    t: float,
    state: Array,
    parameters: Mapping[str, Array],
    controls: Mapping[str, Array],
    phase: str,
) -> None:
    ranges, _ = _validity_contract(system)
    if not ranges:
        return
    values: dict[str, Any] = {**parameters, **controls, "t": t}
    names = tuple(str(name) for name in getattr(system, "state_names", ()))
    if len(getattr(state, "shape", ())) != 2 or int(state.shape[-1]) != len(names):
        raise ValueError(
            "cannot evaluate PMDL runtime validity because state shape does not "
            "match declared state_names"
        )
    values.update({name: state[:, index] for index, name in enumerate(names)})
    missing = sorted(set(ranges) - set(values))
    if missing:
        raise UnsupportedPMDLSemanticsError(
            "PMDL runtime validity cannot be evaluated for unavailable symbol(s) "
            f"{missing}; refusing to ignore their declared ranges."
        )
    for symbol, bounds in sorted(ranges.items()):
        lower, upper = _validity_bounds(bounds, symbol)
        diagnostic = _validity_diagnostic(values[symbol])
        invalid = ~np.isfinite(diagnostic)
        if lower is not None:
            invalid = invalid | (diagnostic < lower)
        if upper is not None:
            invalid = invalid | (diagnostic > upper)
        if not np.any(invalid):
            continue
        flattened = np.asarray(invalid).reshape(-1)
        flat_index = int(np.flatnonzero(flattened)[0])
        location = (
            tuple(int(item) for item in np.unravel_index(flat_index, diagnostic.shape))
            if diagnostic.shape
            else ()
        )
        sample = location[0] if location else 0
        actual = float(np.asarray(diagnostic).reshape(-1)[flat_index])
        interval = (
            f"[{('-inf' if lower is None else format(lower, '.17g'))}, "
            f"{('inf' if upper is None else format(upper, '.17g'))}]"
        )
        raise ValueError(
            "PMDL validity range violation during "
            f"{phase} at time={t:.17g}: symbol={symbol!r}, sample={sample}, "
            f"value={actual:.17g}, allowed={interval}"
        )


def _require_valid_timestep(system: Any, grid: np.ndarray) -> None:
    _, maximum = _validity_contract(system)
    if maximum is None:
        return
    steps = np.diff(grid)
    tolerance = max(1e-15, abs(maximum) * 1e-12)
    violations = np.flatnonzero(steps > maximum + tolerance)
    if violations.size:
        index = int(violations[0])
        raise ValueError(
            "PMDL validity.max_timestep exceeded at "
            f"timestep={index + 1}: requested={steps[index]:.17g}, "
            f"declared_max={maximum:.17g}"
        )


class _ModelSpecAdapter:
    """Adapter for the data-only DSL ModelSpec structural contract."""

    def __init__(self, model: Any) -> None:
        self.model = model
        physical_states = tuple(getattr(model, "state_names", _state_names(model)))
        algebraics = tuple(getattr(model, "algebraic_names", ()))
        self._differential_state_count = len(physical_states)
        self.state_names = physical_states + algebraics
        self.differential_state_indices = tuple(range(len(physical_states)))
        self.algebraic_indices = tuple(
            range(len(physical_states), len(self.state_names))
        )
        self.default_parameters = _parameter_defaults(model)
        self.parameter_bounds = _parameter_bounds(model)
        self.parameter_uncertainty, self._correlated_uncertainty = _parameter_uncertainty(model)
        self.validity = getattr(model, "validity", None)
        initial = []
        for state in getattr(model, "states", ()):
            initial.append(getattr(state, "initial", 0.0))
        for algebraic in getattr(model, "algebraics", ()):
            initial.append(getattr(algebraic, "initial", 0.0))
        self.initial_state = initial or [0.0] * len(self.state_names)
        self.control_names = tuple(getattr(model, "input_names", ()))
        noise_spec = getattr(model, "process_noise", None)
        self.has_process_noise = bool(getattr(noise_spec, "enabled", False))
        self.process_noise_seed_policy = getattr(
            noise_spec, "seed_policy", "simulation_seed"
        )
        self.process_noise_reproducibility = getattr(
            noise_spec, "reproducibility", "same_backend_device"
        )
        self.process_noise_channel_names = tuple(
            sorted(channel.name for channel in getattr(noise_spec, "channels", ()))
        )
        physical_state_indices = {
            name: index for index, name in enumerate(physical_states)
        }
        compiled_noise_increments = []
        noise_runtime_symbols = {
            "t",
            "dt",
            *self.state_names,
            *self.default_parameters,
            *self.control_names,
            *self.process_noise_channel_names,
        }
        for increment in getattr(noise_spec, "increments", ()):
            if increment.target not in physical_state_indices:
                raise UnsupportedPMDLSemanticsError(
                    "PMDL process-noise increment target "
                    f"{increment.target!r} is not a differential state"
                )
            expression = parse_expression(increment.expression)
            unavailable = sorted(expression.variables() - noise_runtime_symbols)
            if unavailable:
                raise UnsupportedPMDLSemanticsError(
                    "PMDL process-noise increment for standalone simulation "
                    f"references unavailable symbol(s) {unavailable}; assemble the "
                    "physical network so port variables have runtime values"
                )
            compiled_noise_increments.append(
                (
                    physical_state_indices[increment.target],
                    increment.target,
                    expression,
                )
            )
        self._process_noise_increments = tuple(compiled_noise_increments)
        relations = getattr(model, "relations", None)
        relation_names = tuple(
            str(getattr(relation, "name", index))
            for index, relation in enumerate(relations or ())
        )
        self._reject_unsupported_semantics(model, relation_names)
        self._relations = None if relations is None else tuple(
            relation.expression
            if isinstance(relation.expression, Expression)
            else parse_expression(str(relation.expression))
            for relation in relations
        )

        ranges, _ = _validity_contract(self)
        runtime_symbols = {
            "t",
            *self.state_names,
            *self.default_parameters,
            *self.control_names,
        }
        unavailable = sorted(set(ranges) - runtime_symbols)
        if unavailable:
            raise UnsupportedPMDLSemanticsError(
                "PMDL validity ranges reference symbol(s) unavailable to the "
                "standalone simulator: "
                f"{unavailable}. Assemble the physical network so those port/output "
                "symbols have runtime values; the ranges will not be ignored."
            )

    @staticmethod
    def _reject_unsupported_semantics(
        model: Any, relation_names: tuple[str, ...]
    ) -> None:
        modes = tuple(getattr(model, "modes", ()) or ())
        if modes:
            names = [str(getattr(mode, "name", index)) for index, mode in enumerate(modes)]
            raise UnsupportedPMDLSemanticsError(
                "PMDL discrete modes are declared but mode transitions/reset maps "
                f"are not implemented by the simulator (modes={names}); refusing "
                "to integrate an always-active approximation."
            )

        initialization = getattr(model, "initialization", None)
        constraints = tuple(getattr(initialization, "constraints", ()) or ())
        if constraints:
            raise UnsupportedPMDLSemanticsError(
                "PMDL initialization constraints are declared but a consistent-"
                "initialization solver is not implemented; refusing to ignore "
                f"{len(constraints)} constraint(s)."
            )
        strategy = getattr(initialization, "strategy", "consistent")
        if initialization is not None and strategy != "consistent":
            raise UnsupportedPMDLSemanticsError(
                "PMDL initialization strategy "
                f"{strategy!r} is unsupported; only explicit defaults under the "
                "'consistent' contract are currently admitted."
            )

        fidelity_levels = tuple(getattr(model, "fidelity_levels", ()) or ())
        if len(fidelity_levels) > 1:
            names = [
                str(getattr(level, "name", index))
                for index, level in enumerate(fidelity_levels)
            ]
            raise UnsupportedPMDLSemanticsError(
                "PMDL declares multiple fidelity levels but simulate() has no "
                f"fidelity selector (levels={names}); refusing to choose silently."
            )
        if fidelity_levels:
            level = fidelity_levels[0]
            active = tuple(getattr(level, "active_relations", ()) or ())
            overrides = dict(getattr(level, "parameter_overrides", {}) or {})
            if set(active) != set(relation_names) or overrides:
                raise UnsupportedPMDLSemanticsError(
                    "PMDL fidelity level "
                    f"{getattr(level, 'name', 'unnamed')!r} changes active relations "
                    "or parameter values, but fidelity execution is not implemented; "
                    "refusing to run the unmodified base model."
                )

        properties = tuple(getattr(model, "properties", ()) or ())
        if properties:
            names = [
                str(getattr(prop, "name", index))
                for index, prop in enumerate(properties)
            ]
            raise UnsupportedPMDLSemanticsError(
                "PMDL machine-checkable property tests are declared but the "
                "simulator has no property-test executor "
                f"(properties={names}); refusing to treat type-checking as a pass."
            )

    def residual(
        self,
        t: Any,
        state: Array,
        state_derivative: Array,
        parameters: Mapping[str, Array],
        controls: Mapping[str, Array],
        backend: Backend,
    ) -> Any:
        if self._relations is not None:
            environment: dict[str, Any] = {"t": t}
            for index, name in enumerate(self.state_names):
                environment[name] = state[:, index]
            for index, state_spec in enumerate(getattr(self.model, "states", ())):
                derivative_name = getattr(state_spec, "derivative", None) or f"{state_spec.name}_dot"
                derivative_value = state_derivative[:, index]
                environment[derivative_name] = derivative_value
                # der(x) has a canonical x_dot lookup even when a ModelSpec
                # explicitly names the derivative symbol.
                environment[f"{state_spec.name}_dot"] = derivative_value
            environment.update(parameters)
            environment.update(controls)
            return tuple(
                _evaluate_backend_expression(expression, environment, backend)
                for expression in self._relations
            )
        evaluator = getattr(self.model, "residual", None)
        if evaluator is None:
            evaluator = self.model.evaluate_residual
        # ModelSpec's safe evaluator is variable-major: each sequence element is
        # one named variable.  The engine is sample-major for vectorization, so
        # expose each column without copying or detaching torch tensors.
        z = tuple(state[:, index] for index in range(int(state.shape[-1])))
        zdot = tuple(
            state_derivative[:, index] for index in range(self._differential_state_count)
        )
        return _invoke(evaluator, t, z, zdot, parameters, controls, backend=backend)

    def process_noise(
        self,
        t: Any,
        state: Array,
        parameters: Mapping[str, Array],
        controls: Mapping[str, Array],
        dt: Any,
        rng: Any,
        backend: Backend,
    ) -> Array:
        """Evaluate PMDL noise draws and increment expressions without detaching."""

        batch_size = int(state.shape[0])
        if not self.has_process_noise:
            return backend.zeros(tuple(state.shape))
        draws = backend.normal(
            (batch_size, len(self.process_noise_channel_names)), rng
        )
        environment: dict[str, Any] = {"t": t, "dt": dt}
        for index, name in enumerate(self.state_names):
            environment[name] = state[:, index]
        environment.update(parameters)
        environment.update(controls)
        for index, name in enumerate(self.process_noise_channel_names):
            environment[name] = draws[:, index]
        increments: dict[int, Array] = {}
        for target_index, target_name, expression in self._process_noise_increments:
            raw = backend.asarray(
                _evaluate_backend_expression(expression, environment, backend)
            )
            if len(raw.shape) == 0:
                raw = backend.broadcast_to(raw, (batch_size,))
            if tuple(raw.shape) != (batch_size,):
                raise ValueError(
                    f"PMDL process-noise increment {target_name!r} must produce "
                    f"[sample], got {tuple(raw.shape)}"
                )
            increments[target_index] = raw
        zero = backend.zeros((batch_size,))
        return backend.stack(
            [increments.get(index, zero) for index in range(len(self.state_names))],
            axis=-1,
        )


def _resolve_system(candidate: Any) -> tuple[Any, Mapping[str, Any]]:
    controllers = getattr(candidate, "controllers", {})
    if controllers is None:
        controllers = {}
    if not isinstance(controllers, Mapping):
        raise TypeError("contraption.controllers must be a mapping of resolved controllers")
    system = candidate
    if not hasattr(system, "derivative") and not hasattr(system, "residual"):
        for attribute in ("dynamics", "system", "simulation_model", "model"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                system = nested
                break
    # ModelSpec.process_noise is a declarative data record, not an executable
    # Python hook; wrap all ModelSpec-shaped objects before generic dispatch.
    if not hasattr(system, "derivative") and hasattr(system, "evaluate_residual"):
        system = _ModelSpecAdapter(system)
    elif not hasattr(system, "derivative") and not hasattr(system, "residual"):
        raise TypeError("System must expose derivative(...) or residual(...)")
    return system, controllers


def _sample_scalar(value: Any, index: int, count: int, backend: Backend, context: str) -> Any:
    array = backend.asarray(value)
    if len(array.shape) == 0:
        return array
    if tuple(array.shape) == (count,):
        return array[index]
    raise ValueError(f"{context} must be scalar or [sample], got shape {tuple(array.shape)}")


class _ResolvedControllerExecutor:
    """Execute one resolved controller once per posterior sample.

    A hardware controller is scalar and has discrete mode state. Keeping one
    runtime per posterior sample preserves that exact behavior while numeric
    expressions remain backend-native and differentiable within each selected
    branch.
    """

    def __init__(self, resolved: Any, count: int, backend: Backend) -> None:
        self.resolved = resolved
        self.count = count
        self.backend = backend
        self.runtimes = tuple(
            ControlRuntime(
                resolved.spec,
                observer=resolved.observer,
                backend=backend,
                emit_observability_warnings=sample_index == 0,
            )
            for sample_index in range(count)
        )
        self.frames: list[tuple[ControlFrame, ...]] = []
        self.tick_mask: list[bool] = []
        self._frame_times_s: list[float] = []
        self._current_frames: tuple[ControlFrame, ...] | None = None
        self._held_outputs: dict[str, Array] | None = None

    def initialize(self, time_s: float) -> dict[str, Array]:
        """Publish the untouched hardware reset state at simulation time zero."""

        if self._current_frames is not None:
            raise RuntimeError(
                f"controller {self.resolved.id!r} is already initialized"
            )
        sample_frames = tuple(
            ControlFrame(
                time=runtime.time,
                active_mode=runtime.mode,
                next_mode=runtime.mode,
                outputs=dict(runtime.outputs),
                registers=dict(runtime.registers),
                implicit_inputs=dict(runtime.implicit_inputs),
                derived={},
                emergency=False,
            )
            for runtime in self.runtimes
        )
        output_columns: dict[str, list[Any]] = {
            name: []
            for name, binding in self.resolved.output_bindings.items()
            if binding.kind == "signal"
        }
        for frame in sample_frames:
            for output_name in output_columns:
                output_columns[output_name].append(frame.outputs[output_name])
        result: dict[str, Array] = {}
        for output_name, values in output_columns.items():
            source = self.resolved.output_bindings[output_name].source
            if source in result:
                raise RuntimeError(
                    f"controller {self.resolved.id!r} drives source {source!r} more than once"
                )
            result[source] = self.backend.stack(values, axis=0)
        self.frames.append(sample_frames)
        self.tick_mask.append(False)
        self._frame_times_s.append(float(time_s))
        self._current_frames = sample_frames
        self._held_outputs = result
        return dict(result)

    @property
    def external_names(self) -> frozenset[str]:
        return frozenset(
            binding.source
            for binding in self.resolved.explicit_input_bindings.values()
            if binding.kind == "external"
        )

    def step(
        self,
        state: Array,
        external_inputs: Mapping[str, Any],
        time_s: float,
    ) -> dict[str, Array]:
        sample_frames: list[ControlFrame] = []
        output_columns: dict[str, list[Any]] = {
            name: []
            for name, binding in self.resolved.output_bindings.items()
            if binding.kind == "signal"
        }
        for sample_index, runtime in enumerate(self.runtimes):
            inputs: dict[str, Any] = {}
            for name, binding in self.resolved.explicit_input_bindings.items():
                if binding.kind == "sensor":
                    if binding.state_index is None:
                        raise RuntimeError(
                            f"controller {self.resolved.id!r} sensor {name!r} lost its state index"
                        )
                    inputs[name] = state[sample_index, binding.state_index]
                elif binding.kind == "external":
                    if binding.source in external_inputs:
                        inputs[name] = _sample_scalar(
                            external_inputs[binding.source],
                            sample_index,
                            self.count,
                            self.backend,
                            f"controller input {binding.source!r}",
                        )
                else:
                    raise RuntimeError(
                        f"controller {self.resolved.id!r} has invalid input binding kind {binding.kind!r}"
                    )
            frame = runtime.step(inputs)
            sample_frames.append(frame)
            for output_name in output_columns:
                output_columns[output_name].append(frame.outputs[output_name])
        current_frames = tuple(sample_frames)
        self.frames.append(current_frames)
        self.tick_mask.append(True)
        self._frame_times_s.append(float(time_s))
        result: dict[str, Array] = {}
        for output_name, values in output_columns.items():
            source = self.resolved.output_bindings[output_name].source
            if source in result:
                raise RuntimeError(
                    f"controller {self.resolved.id!r} drives source {source!r} more than once"
                )
            result[source] = self.backend.stack(values, axis=0)
        self._current_frames = current_frames
        self._held_outputs = result
        return result

    def hold(self, time_s: float) -> dict[str, Array]:
        """Publish the previous hardware frame without advancing controller state."""

        if self._current_frames is None or self._held_outputs is None:
            raise RuntimeError(
                f"controller {self.resolved.id!r} cannot hold before its initial tick"
            )
        self.frames.append(self._current_frames)
        self.tick_mask.append(False)
        self._frame_times_s.append(float(time_s))
        return dict(self._held_outputs)

    def trace(self) -> ControllerTrace:
        output_names = tuple(item.name for item in self.resolved.spec.outputs)
        implicit_names = tuple(item.name for item in self.resolved.spec.implicit_inputs)
        output_by_time = [
            self.backend.stack(
                [
                    self.backend.stack(
                        [frame.outputs[name] for name in output_names], axis=-1
                    )
                    for frame in sample_frames
                ],
                axis=0,
            )
            for sample_frames in self.frames
        ]
        output_samples = self.backend.stack(output_by_time, axis=1)
        if implicit_names:
            means_by_time = [
                self.backend.stack(
                    [
                        self.backend.stack(
                            [frame.implicit_inputs[name].mean for name in implicit_names],
                            axis=-1,
                        )
                        for frame in sample_frames
                    ],
                    axis=0,
                )
                for sample_frames in self.frames
            ]
            variances_by_time = [
                self.backend.stack(
                    [
                        self.backend.stack(
                            [frame.implicit_inputs[name].variance for name in implicit_names],
                            axis=-1,
                        )
                        for frame in sample_frames
                    ],
                    axis=0,
                )
                for sample_frames in self.frames
            ]
            implicit_means = self.backend.stack(means_by_time, axis=1)
            implicit_variances = self.backend.stack(variances_by_time, axis=1)
        else:
            implicit_means = self.backend.zeros((self.count, len(self.frames), 0))
            implicit_variances = self.backend.zeros((self.count, len(self.frames), 0))
        emergency_samples = self.backend.stack(
            [
                self.backend.stack(
                    [self.backend.asarray(frame.emergency) for frame in sample_frames],
                    axis=0,
                )
                for sample_frames in self.frames
            ],
            axis=1,
        )
        return ControllerTrace(
            controller_id=self.resolved.id,
            output_names=output_names,
            output_samples=output_samples,
            implicit_input_names=implicit_names,
            implicit_means=implicit_means,
            implicit_variances=implicit_variances,
            emergency_samples=emergency_samples,
            active_modes=tuple(
                tuple(frame.active_mode for frame in sample_frames)
                for sample_frames in self.frames
            ),
            tick_mask=tuple(self.tick_mask),
            frame_times_s=tuple(self._frame_times_s),
        )


def _validate_accepted_step(
    contraption: Any,
    *,
    step_index: int,
    time_s: float,
    state: Array,
    state_names: tuple[str, ...],
    backend: Backend,
) -> None:
    """Run an optional fail-closed validator on each accepted state frame."""

    validator = getattr(contraption, "validate_simulation_step", None)
    if callable(validator):
        validator(
            step_index=step_index,
            time_s=time_s,
            state=state,
            state_names=state_names,
            backend=backend,
        )


def _time_grid(
    t_span: tuple[float, float] | None,
    duration: float | None,
    dt: float,
    times: Sequence[float] | None,
) -> np.ndarray:
    if times is not None:
        grid = np.asarray(times, dtype=np.float64)
        if grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0):
            raise ValueError("times must be a strictly increasing 1-D sequence")
        return grid
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive")
    if t_span is None:
        t_span = (0.0, 1.0 if duration is None else float(duration))
    elif duration is not None:
        raise ValueError("Specify either t_span or duration, not both")
    start, stop = map(float, t_span)
    if not stop > start:
        raise ValueError("t_span stop must be greater than start")
    count = int(math.ceil((stop - start) / dt))
    grid = start + np.arange(count + 1, dtype=np.float64) * dt
    grid[-1] = stop
    return grid


_CONTROLLER_SCHEDULE_ABSOLUTE_TOLERANCE = 1e-12
_CONTROLLER_SCHEDULE_RELATIVE_TOLERANCE = 1e-9


def _controller_tick_stride(period: float, physics_dt: float) -> int:
    period = float(period)
    physics_dt = float(physics_dt)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("controller periods must be finite and positive")
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError("physics dt must be finite and positive")
    ratio = period / physics_dt
    stride = int(round(ratio))
    tolerance = max(
        _CONTROLLER_SCHEDULE_ABSOLUTE_TOLERANCE,
        _CONTROLLER_SCHEDULE_RELATIVE_TOLERANCE * max(period, physics_dt),
    )
    if stride < 1 or abs(period - stride * physics_dt) > tolerance:
        raise ValueError(
            f"controller period {period:.17g}s is not commensurate with physics dt "
            f"{physics_dt:.17g}s; every controller period must be a positive integer "
            f"multiple of dt within tolerance {tolerance:.3g}s"
        )
    return stride


def controller_time_step(
    periods: Iterable[float],
    requested: float | None = None,
    *,
    default: float = 0.01,
) -> float:
    """Return or validate a common physics subdivision for controller periods.

    Authored periods are decimal JSON numbers. Converting their string forms to
    exact rational values gives a deterministic greatest common divisor for the
    default grid while runtime admission still uses an explicit floating-point
    tolerance.
    """

    values = tuple(float(period) for period in periods)
    if any(not math.isfinite(period) or period <= 0.0 for period in values):
        raise ValueError("controller periods must be finite and positive")
    if requested is not None:
        result = float(requested)
    elif not values:
        result = float(default)
    else:
        fractions = tuple(Fraction(str(period)) for period in values)
        denominator = 1
        for value in fractions:
            denominator = math.lcm(denominator, value.denominator)
        numerator = 0
        for value in fractions:
            scaled = value.numerator * (denominator // value.denominator)
            numerator = math.gcd(numerator, abs(scaled))
        result = float(Fraction(numerator, denominator))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("physics dt must be finite and positive")
    for period in values:
        _controller_tick_stride(period, result)
    return result


def _call_external_input(function: Callable[..., Any], t: float, backend: Backend) -> Any:
    """Call an open-loop provider without ever exposing plant state."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(t)
    parameters = list(signature.parameters.values())
    accepts_kwargs = any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters
    )
    names = {item.name for item in parameters}
    kwargs = {"backend": backend} if "backend" in names or accepts_kwargs else {}
    positional = [
        item
        for item in parameters
        if item.name != "backend"
        and item.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) > 1:
        raise TypeError(
            "open-loop input/control providers may accept only time and optional "
            "backend; plant state is available only to resolved control DSL sensors"
        )
    return function(t, **kwargs) if positional else function(**kwargs)


def _external_input_value(
    value: Any,
    t: float,
    backend: Backend,
    times: np.ndarray,
) -> Array:
    if callable(value):
        value = _call_external_input(value, t, backend)
    elif hasattr(value, "evaluate"):
        value = _call_external_input(value.evaluate, t, backend)
    if isinstance(value, (list, tuple, np.ndarray)) or hasattr(value, "shape"):
        array = backend.asarray(value)
        if len(array.shape) > 0 and int(array.shape[0]) == len(times):
            upper = int(np.searchsorted(times, t, side="right"))
            if upper <= 0:
                value = array[0]
            elif upper >= len(times):
                value = array[-1]
            else:
                lower = upper - 1
                fraction = (t - times[lower]) / (times[upper] - times[lower])
                value = array[lower] * (1.0 - fraction) + array[upper] * fraction
            return backend.asarray(value)
    return backend.asarray(value)


def _evaluate_controller_inputs(
    source: Any,
    t: float,
    backend: Backend,
    times: np.ndarray,
    allowed: frozenset[str],
    sampled: frozenset[str] | None = None,
) -> dict[str, Array]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        values = source
    else:
        evaluator = source.evaluate if hasattr(source, "evaluate") else source
        values = _call_external_input(evaluator, t, backend)
        if not isinstance(values, Mapping):
            raise TypeError("A controller-input provider must return a mapping")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise KeyError(
            f"Unknown external controller input(s) {unknown}; declared inputs are {sorted(allowed)}"
        )
    selected = allowed if sampled is None else sampled
    return {
        str(name): _external_input_value(values[name], t, backend, times)
        for name in selected
        if name in values
    }


def _control_value(
    value: Any,
    t: float,
    backend: Backend,
    times: np.ndarray,
) -> Array:
    if callable(value):
        value = _call_external_input(value, t, backend)
    elif hasattr(value, "evaluate"):
        value = _call_external_input(value.evaluate, t, backend)
    if isinstance(value, (list, tuple, np.ndarray)) or hasattr(value, "shape"):
        array = backend.asarray(value)
        if len(array.shape) > 0 and int(array.shape[0]) == len(times):
            upper = int(np.searchsorted(times, t, side="right"))
            if upper <= 0:
                value = array[0]
            elif upper >= len(times):
                value = array[-1]
            else:
                lower = upper - 1
                fraction = (t - times[lower]) / (times[upper] - times[lower])
                value = array[lower] * (1.0 - fraction) + array[upper] * fraction
            return backend.asarray(value)
    return backend.asarray(value)


def _evaluate_controls(
    source: Any,
    t: float,
    backend: Backend,
    times: np.ndarray,
) -> dict[str, Array]:
    result: dict[str, Array] = {}
    if source is not None:
        if isinstance(source, Mapping):
            result.update(
                {
                    str(name): _control_value(value, t, backend, times)
                    for name, value in source.items()
                }
            )
        else:
            evaluator = source.evaluate if hasattr(source, "evaluate") else source
            values = _call_external_input(evaluator, t, backend)
            if not isinstance(values, Mapping):
                raise TypeError("A control provider must return a mapping")
            result.update({str(name): backend.asarray(value) for name, value in values.items()})
    return result


def _merged_controls(
    external: Mapping[str, Array], internal: Mapping[str, Array]
) -> dict[str, Array]:
    """Merge independent open-loop and controller-driven actuator sources."""

    overlap = sorted(set(external) & set(internal))
    if overlap:
        raise ValueError(
            "physical actuator sources cannot be driven externally and by a controller: "
            + ", ".join(overlap)
        )
    result = dict(external)
    result.update(internal)
    return result


def _initial_state(
    system: Any,
    state_names: tuple[str, ...],
    supplied: Any,
    covariance: Any | None,
    count: int,
    backend: Backend,
    rng: Any,
) -> Array:
    value = supplied
    if value is None:
        value = getattr(system, "initial_state", None)
        if callable(value):
            try:
                value = _invoke(value, count, rng, backend=backend)
            except TypeError:
                value = value()
        if value is None:
            value = [0.0] * len(state_names)
    if isinstance(value, Mapping):
        missing = set(state_names) - set(value)
        if missing:
            raise KeyError(f"Initial state is missing {sorted(missing)}")
        value = backend.stack([backend.asarray(value[name]) for name in state_names], axis=-1)
    else:
        value = backend.asarray(value)
    if covariance is not None:
        if len(value.shape) != 1:
            raise ValueError("initial_covariance requires a single mean initial_state [state]")
        result = sample_gaussian(value, covariance, count, backend=backend, rng=rng)
    elif len(value.shape) == 1:
        result = backend.broadcast_to(value, (count, int(value.shape[0])))
    elif len(value.shape) == 2 and int(value.shape[0]) == count:
        result = value
    else:
        raise ValueError("initial_state must have shape [state] or [sample,state]")
    if int(result.shape[-1]) != len(state_names):
        raise ValueError(f"Expected {len(state_names)} initial states, got {result.shape[-1]}")
    return result


def _explicit_derivative(
    system: Any,
    t: float,
    state: Array,
    parameters: Mapping[str, Array],
    controls: Mapping[str, Array],
    backend: Backend,
) -> Array:
    _require_known_controls(system, controls)
    _require_runtime_validity(
        system,
        t=float(_validity_diagnostic(t).item()),
        state=state,
        parameters=parameters,
        controls=controls,
        phase="explicit dynamics evaluation",
    )
    value = _invoke(system.derivative, backend.asarray(t), state, parameters, controls, backend=backend)
    return _as_vector(value, backend, int(state.shape[0]))


def _explicit_step(
    method: str,
    system: Any,
    t: float,
    dt: float,
    state: Array,
    parameters: Mapping[str, Array],
    control_source: Any,
    held_internal_controls: Mapping[str, Array],
    backend: Backend,
    times: np.ndarray,
) -> tuple[Array, Mapping[str, Array]]:
    c1 = _merged_controls(
        _evaluate_controls(control_source, t, backend, times),
        held_internal_controls,
    )
    k1 = _explicit_derivative(system, t, state, parameters, c1, backend)
    if method == "euler":
        return state + dt * k1, c1
    midpoint = state + 0.5 * dt * k1
    c2 = _merged_controls(
        _evaluate_controls(control_source, t + 0.5 * dt, backend, times),
        held_internal_controls,
    )
    k2 = _explicit_derivative(system, t + 0.5 * dt, midpoint, parameters, c2, backend)
    midpoint = state + 0.5 * dt * k2
    c3 = _merged_controls(
        _evaluate_controls(control_source, t + 0.5 * dt, backend, times),
        held_internal_controls,
    )
    k3 = _explicit_derivative(system, t + 0.5 * dt, midpoint, parameters, c3, backend)
    endpoint = state + dt * k3
    c4 = _merged_controls(
        _evaluate_controls(control_source, t + dt, backend, times),
        held_internal_controls,
    )
    k4 = _explicit_derivative(system, t + dt, endpoint, parameters, c4, backend)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), c1


def _descriptor_residual(
    system: Any,
    t: float,
    state: Array,
    state_derivative: Array,
    parameters: Mapping[str, Array],
    controls: Mapping[str, Array],
    backend: Backend,
) -> Array:
    _require_known_controls(system, controls)
    _require_runtime_validity(
        system,
        t=float(_validity_diagnostic(t).item()),
        state=state,
        parameters=parameters,
        controls=controls,
        phase="implicit residual evaluation",
    )
    value = _invoke(
        system.residual,
        backend.asarray(t),
        state,
        state_derivative,
        parameters,
        controls,
        backend=backend,
    )
    return _as_vector(value, backend, int(state.shape[0]))


def _analytic_descriptor_jacobian(
    system: Any,
    t: float,
    next_state: Array,
    previous_state: Array,
    dt: float,
    parameters: Mapping[str, Array],
    controls: Mapping[str, Array],
    backend: Backend,
) -> Array | None:
    direct = getattr(system, "backward_euler_jacobian", None)
    derivative = (next_state - previous_state) / dt
    if direct is not None:
        value = _invoke(
            direct,
            backend.asarray(t),
            next_state,
            derivative,
            parameters,
            controls,
            backend.asarray(dt),
            backend=backend,
        )
    else:
        jacobian = getattr(system, "jacobian", None)
        if jacobian is None or (
            isinstance(system, ResidualSystem) and system.jacobian_function is None
        ):
            return None
        value = _invoke(
            jacobian,
            backend.asarray(t),
            next_state,
            derivative,
            parameters,
            controls,
            backend=backend,
        )
        if isinstance(value, Mapping):
            state_jacobian = value.get("state", value.get("z"))
            derivative_jacobian = value.get("state_derivative", value.get("zdot"))
            if state_jacobian is None or derivative_jacobian is None:
                raise ValueError("Jacobian mapping requires state/z and state_derivative/zdot")
            value = backend.asarray(state_jacobian) + backend.asarray(derivative_jacobian) / dt
        elif isinstance(value, tuple) and len(value) == 2:
            value = backend.asarray(value[0]) + backend.asarray(value[1]) / dt
    if hasattr(value, "toarray"):
        value = value.toarray()
    value = backend.asarray(value)
    if len(value.shape) == 2:
        value = backend.broadcast_to(value, (int(next_state.shape[0]),) + tuple(value.shape))
    return value


def _newton_diagnostic_array(value: Array, backend: Backend) -> np.ndarray:
    """Detach only the copy used to decide whether a Newton solve is valid."""

    if backend.is_torch:
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _require_newton_finite(
    value: Array,
    quantity: str,
    backend: Backend,
    *,
    step_index: int,
    time: float,
) -> np.ndarray:
    diagnostic = _newton_diagnostic_array(value, backend)
    locations = np.argwhere(~np.isfinite(diagnostic))
    if locations.size:
        location = tuple(int(index) for index in locations[0])
        sample = location[0] if location else 0
        component = location[1:] if len(location) > 1 else ()
        component_text = ",".join(str(index) for index in component) or "scalar"
        invalid = diagnostic[location] if location else diagnostic.item()
        raise FloatingPointError(
            f"Implicit Newton {quantity} is non-finite at timestep={step_index}, "
            f"time={time:.17g}, sample={sample}, component={component_text}, value={invalid!r}"
        )
    return diagnostic


def _implicit_step(
    system: Any,
    t: float,
    dt: float,
    state: Array,
    parameters: Mapping[str, Array],
    controls: Mapping[str, Array],
    backend: Backend,
    *,
    tolerance: float,
    max_iterations: int,
    step_index: int,
) -> Array:
    batch_size, dimension = map(int, state.shape)
    if max_iterations < 1:
        raise ValueError("newton_max_iterations must be positive")
    next_time = t + dt
    _require_newton_finite(
        state, "state", backend, step_index=step_index, time=next_time
    )
    guess = backend.clone(state)

    def residual(candidate: Array) -> Array:
        return _descriptor_residual(
            system,
            next_time,
            candidate,
            (candidate - state) / dt,
            parameters,
            controls,
            backend,
        )

    iterations = 0
    for iteration in range(max_iterations):
        iterations = iteration + 1
        _require_newton_finite(
            guess, "state", backend, step_index=step_index, time=next_time
        )
        value = residual(guess)
        value_diagnostic = _require_newton_finite(
            value, "residual", backend, step_index=step_index, time=next_time
        )
        if tuple(value.shape) != (batch_size, dimension):
            raise ValueError(
                "Implicit descriptor residual must match state shape "
                f"{(batch_size, dimension)}, received {tuple(value.shape)} at "
                f"timestep={step_index}, time={next_time:.17g}"
            )
        jacobian = _analytic_descriptor_jacobian(
            system, next_time, guess, state, dt, parameters, controls, backend
        )
        if jacobian is None:
            columns = []
            for index in range(dimension):
                basis = backend.stack(
                    [backend.asarray(1.0 if j == index else 0.0) for j in range(dimension)]
                )
                step = 1e-6 * backend.maximum(backend.abs(guess[:, index]), backend.asarray(1.0))
                perturbation = step[:, None] * basis[None, :]
                positive = residual(guess + perturbation)
                negative = residual(guess - perturbation)
                _require_newton_finite(
                    positive,
                    "residual during positive Jacobian perturbation",
                    backend,
                    step_index=step_index,
                    time=next_time,
                )
                _require_newton_finite(
                    negative,
                    "residual during negative Jacobian perturbation",
                    backend,
                    step_index=step_index,
                    time=next_time,
                )
                columns.append(
                    (positive - negative) / (2.0 * step[:, None])
                )
            jacobian = backend.stack(columns, axis=-1)
        if tuple(jacobian.shape) != (batch_size, dimension, dimension):
            raise ValueError(
                "Implicit descriptor Jacobian must have shape "
                f"{(batch_size, dimension, dimension)}, received {tuple(jacobian.shape)} "
                f"at timestep={step_index}, time={next_time:.17g}"
            )
        jacobian_diagnostic = _require_newton_finite(
            jacobian, "Jacobian", backend, step_index=step_index, time=next_time
        )
        ranks = np.asarray(np.linalg.matrix_rank(jacobian_diagnostic))
        singular_samples = np.flatnonzero(ranks < dimension)
        if singular_samples.size:
            sample_index = int(singular_samples[0])
            rank = int(ranks[sample_index])
            residual_max = float(np.max(np.abs(value_diagnostic[sample_index])))
            raise RuntimeError(
                "Implicit Newton Jacobian is singular at "
                f"timestep={step_index}, time={next_time:.17g}, sample={sample_index}; "
                f"rank={rank}/{dimension}, residual_max={residual_max:.17g}"
            )
        try:
            update = backend.solve(jacobian, -value[..., None])[..., 0]
        except (np.linalg.LinAlgError, RuntimeError) as exc:
            raise RuntimeError(
                "Implicit Newton linear solve failed at "
                f"timestep={step_index}, time={next_time:.17g}: {exc}"
            ) from exc
        update_diagnostic = _require_newton_finite(
            update, "update", backend, step_index=step_index, time=next_time
        )
        guess = guess + update
        if float(np.max(np.abs(update_diagnostic))) <= tolerance:
            break
    _require_newton_finite(
        guess, "state", backend, step_index=step_index, time=next_time
    )
    final_residual = residual(guess)
    final_diagnostic = _require_newton_finite(
        final_residual, "residual", backend, step_index=step_index, time=next_time
    )
    residual_by_sample = np.max(np.abs(final_diagnostic), axis=1)
    threshold = max(float(tolerance) * 10.0, 1e-8)
    worst_sample = int(np.argmax(residual_by_sample))
    worst_residual = float(residual_by_sample[worst_sample])
    if worst_residual > threshold:
        residual_vector = np.asarray(final_diagnostic[worst_sample]).tolist()
        raise RuntimeError(
            "Implicit Newton solve did not converge at "
            f"timestep={step_index}, time={next_time:.17g}, sample={worst_sample} "
            f"after {iterations} iteration(s); residual_max={worst_residual:.17g}, "
            f"residual={residual_vector}"
        )
    return guess


def _observe(
    system: Any,
    t: float,
    state: Array,
    parameters: Mapping[str, Array],
    controls: Mapping[str, Array],
    backend: Backend,
    state_names: tuple[str, ...],
) -> tuple[tuple[str, ...], Array]:
    observer = getattr(system, "observe", None)
    if observer is None:
        return state_names, state
    value = _invoke(observer, backend.asarray(t), state, parameters, controls, backend=backend)
    if isinstance(value, Mapping):
        names = tuple(str(name) for name in value)
        output = _as_vector(value, backend, int(state.shape[0]))
    else:
        names = tuple(getattr(system, "output_names", ()))
        output = _as_vector(value, backend, int(state.shape[0]))
        if not names:
            names = tuple(f"output_{index}" for index in range(int(output.shape[-1])))
    if len(names) != int(output.shape[-1]):
        raise ValueError("output_names length does not match observe() output")
    return names, output


def simulate(
    contraption: Any,
    t_span: tuple[float, float] | None = None,
    dt: float = 0.01,
    *,
    duration: float | None = None,
    times: Sequence[float] | None = None,
    controls: Any = None,
    controller_inputs: Any = None,
    parameters: Mapping[str, Any] | None = None,
    parameter_distribution: Mapping[str, Any] | Any | None = None,
    parameter_distributions: Mapping[str, Any] | Any | None = None,
    initial_state: Any = None,
    initial_covariance: Any | None = None,
    num_samples: int = 256,
    seed: int = 0,
    backend: str | Backend = "numpy",
    device: str | None = None,
    dtype: Any | None = None,
    integrator: str = "auto",
    quantiles: Sequence[float] = (0.025, 0.5, 0.975),
    confidence_level: float = 0.95,
    use_model_uncertainty: bool = True,
    process_noise: bool = True,
    newton_tolerance: float = 1e-9,
    newton_max_iterations: int = 12,
) -> SimulationResult:
    """Simulate an explicit or descriptor system as a Monte Carlo ensemble.

    Parameter uncertainty accepts independent concise forms—``std``,
    ``(mean, std)``, :class:`~contraption.physics.uq.Normal`, or a mapping with
    ``mean/std/lower/upper``—as well as a joint distribution object whose
    ``sample`` method returns a name-to-batch mapping.

    A contraption wrapper may expose ``validate_simulation_step`` to reject an
    invalid consistently initialized or accepted state before the engine
    advances or publishes it.  ``validate_simulation_result`` remains the
    final whole-trajectory admission hook.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if parameter_distribution is not None and parameter_distributions is not None:
        raise ValueError("Use only one of parameter_distribution/parameter_distributions")
    distributions = parameter_distribution if parameter_distribution is not None else parameter_distributions
    numerical = get_backend(backend, device=device, dtype=dtype)
    system, resolved_controllers = _resolve_system(contraption)
    names = _state_names(system)
    grid = _time_grid(t_span, duration, dt, times)
    _require_valid_timestep(system, grid)
    controller_tick_strides: dict[str, int] = {}
    if resolved_controllers:
        if times is None:
            physics_dt = float(dt)
        else:
            intervals = np.diff(grid)
            physics_dt = float(intervals[0])
            if not np.allclose(
                intervals,
                physics_dt,
                rtol=_CONTROLLER_SCHEDULE_RELATIVE_TOLERANCE,
                atol=_CONTROLLER_SCHEDULE_ABSOLUTE_TOLERANCE,
            ):
                raise ValueError(
                    "controllers require a uniform explicit time grid whose step is a "
                    "common subdivision of every controller period"
                )
        for controller_id, resolved_controller in resolved_controllers.items():
            try:
                controller_tick_strides[controller_id] = _controller_tick_stride(
                    float(resolved_controller.spec.period_s), physics_dt
                )
            except ValueError as exc:
                raise ValueError(f"controller {controller_id!r}: {exc}") from exc
    else:
        physics_dt = float(np.diff(grid)[0])
    defaults = _parameter_defaults(system)
    if parameters:
        unknown_parameters = sorted(set(parameters) - set(defaults))
        if unknown_parameters:
            raise KeyError(
                f"Unknown physical parameter override(s) {unknown_parameters}; "
                f"declared parameters are {sorted(defaults)}"
            )
        defaults.update(parameters)
    if distributions is None and use_model_uncertainty and num_samples > 1:
        correlated = tuple(getattr(system, "_correlated_uncertainty", ()))
        if correlated:
            raise NotImplementedError(
                "Default ModelSpec UQ cannot infer covariance for correlation_group "
                f"declarations {list(correlated)}; supply parameter_distribution explicitly"
            )
        distributions = getattr(system, "parameter_uncertainty", None)
    bounds = _parameter_bounds(system)
    parameter_rng = numerical.make_rng(split_seed(seed, "parameters"))
    sampled_parameters = sample_parameters(
        defaults,
        distributions,
        num_samples,
        backend=numerical,
        rng=parameter_rng,
        bounds=bounds,
    )
    state_rng = numerical.make_rng(split_seed(seed, "initial-state"))
    state = _initial_state(
        system,
        names,
        initial_state,
        initial_covariance,
        num_samples,
        numerical,
        state_rng,
    )
    controller_executors = tuple(
        _ResolvedControllerExecutor(controller, num_samples, numerical)
        for controller in resolved_controllers.values()
    )
    external_controller_names = frozenset(
        name for executor in controller_executors for name in executor.external_names
    )
    explicit = hasattr(system, "derivative")
    if integrator == "auto":
        integrator = "rk4" if explicit else "implicit_euler"
    if integrator in {"euler", "rk4"} and not explicit:
        raise ValueError(f"{integrator} requires an explicit derivative")
    if integrator == "implicit_euler" and not hasattr(system, "residual"):
        raise ValueError("implicit_euler requires a descriptor residual")
    if integrator not in {"euler", "rk4", "implicit_euler"}:
        raise ValueError(f"Unsupported integrator {integrator!r}")

    external_controls = _evaluate_controls(
        controls, float(grid[0]), numerical, grid
    )
    held_internal_controls: dict[str, Array] = {}
    for executor in controller_executors:
        held_internal_controls = _merged_controls(
            held_internal_controls,
            executor.initialize(float(grid[0])),
        )
    first_controls = _merged_controls(external_controls, held_internal_controls)
    initializer = getattr(system, "consistent_initial_state", None)
    if callable(initializer):
        state = _invoke(
            initializer,
            numerical.asarray(float(grid[0])),
            state,
            sampled_parameters,
            first_controls,
            backend=numerical,
        )
    # Time zero is the hardware reset snapshot: declared/default actuator
    # outputs are held, descriptor algebraics are consistent under those
    # outputs, and no controller period or sensor-fusion update has elapsed.
    _require_runtime_validity(
        system,
        t=float(grid[0]),
        state=state,
        parameters=sampled_parameters,
        controls=first_controls,
        phase="initialization",
    )
    # The first published frame must be the accepted, consistently initialized
    # assembly state.  Retaining the caller's zero-filled algebraic guess here
    # would make visualization and downstream metrics observe a configuration
    # that the descriptor equations explicitly rejected.
    states = [state]
    invariant_check = getattr(system, "require_network_invariants", None)
    if callable(invariant_check):
        invariant_check(
            state,
            first_controls,
            numerical,
            tolerance=max(float(newton_tolerance) * 10.0, 1e-8),
            time=float(grid[0]),
        )
    _validate_accepted_step(
        contraption,
        step_index=0,
        time_s=float(grid[0]),
        state=state,
        state_names=names,
        backend=numerical,
    )
    output_names, first_output = _observe(
        system, float(grid[0]), state, sampled_parameters, first_controls, numerical, names
    )
    outputs = [first_output]
    noise_rng = numerical.make_rng(split_seed(seed, "process-noise"))
    for index in range(len(grid) - 1):
        t = float(grid[index])
        step_size = float(grid[index + 1] - grid[index])
        if explicit:
            next_state, step_controls = _explicit_step(
                integrator,
                system,
                t,
                step_size,
                state,
                sampled_parameters,
                controls,
                held_internal_controls,
                numerical,
                grid,
            )
        else:
            step_controls = _merged_controls(
                _evaluate_controls(controls, t + step_size, numerical, grid),
                held_internal_controls,
            )
            next_state = _implicit_step(
                system,
                t,
                step_size,
                state,
                sampled_parameters,
                step_controls,
                numerical,
                tolerance=newton_tolerance,
                max_iterations=newton_max_iterations,
                step_index=index + 1,
            )
        noise_function = getattr(system, "process_noise", None)
        noise_declared = bool(
            getattr(system, "has_process_noise", noise_function is not None)
        )
        if process_noise and noise_declared and noise_function is not None:
            increment = _invoke(
                noise_function,
                numerical.asarray(t + step_size),
                next_state,
                sampled_parameters,
                step_controls,
                numerical.asarray(step_size),
                noise_rng,
                backend=numerical,
            )
            next_state = next_state + _as_vector(increment, numerical, num_samples)
            algebraic_indices = tuple(getattr(system, "algebraic_indices", ()))
            if algebraic_indices:
                reconciler = getattr(system, "consistent_initial_state", None)
                if not callable(reconciler):
                    raise UnsupportedPMDLSemanticsError(
                        "process-noise increments changed differential state in a "
                        "descriptor model with algebraic unknowns, but the system has "
                        "no consistent-state reconciliation solver"
                    )
                next_state = _invoke(
                    reconciler,
                    numerical.asarray(t + step_size),
                    next_state,
                    sampled_parameters,
                    step_controls,
                    backend=numerical,
                )
        state = next_state
        states.append(state)
        if explicit:
            next_external_controls = _evaluate_controls(
                controls, float(grid[index + 1]), numerical, grid
            )
            output_controls = _merged_controls(
                next_external_controls, held_internal_controls
            )
        else:
            # The accepted descriptor state satisfies its algebraic/control
            # equations for the command held during this interval.  Comparing
            # it against newly evaluated feedback would check a command that
            # was never applied and can report a false constraint violation.
            output_controls = dict(step_controls)
        if callable(invariant_check):
            invariant_check(
                state,
                output_controls,
                numerical,
                tolerance=max(float(newton_tolerance) * 10.0, 1e-8),
                time=float(grid[index + 1]),
            )
        _require_runtime_validity(
            system,
            t=float(grid[index + 1]),
            state=state,
            parameters=sampled_parameters,
            controls=output_controls,
            phase=f"accepted timestep {index + 1}",
        )
        _validate_accepted_step(
            contraption,
            step_index=index + 1,
            time_s=float(grid[index + 1]),
            state=state,
            state_names=names,
            backend=numerical,
        )
        current_names, output = _observe(
            system,
            float(grid[index + 1]),
            state,
            sampled_parameters,
            output_controls,
            numerical,
            names,
        )
        if current_names != output_names:
            raise ValueError("observe() returned inconsistent output names across time")
        outputs.append(output)
        frame_time = float(grid[index + 1])
        frame_index = index + 1
        regular_grid_frame = (
            times is not None
            or frame_index < len(grid) - 1
            or math.isclose(
                step_size,
                physics_dt,
                rel_tol=_CONTROLLER_SCHEDULE_RELATIVE_TOLERANCE,
                abs_tol=_CONTROLLER_SCHEDULE_ABSOLUTE_TOLERANCE,
            )
        )
        ticking = tuple(
            regular_grid_frame
            and frame_index % controller_tick_strides[executor.resolved.id] == 0
            for executor in controller_executors
        )
        sampled_external_names = frozenset(
            name
            for executor, should_tick in zip(controller_executors, ticking)
            if should_tick
            for name in executor.external_names
        )
        next_controller_inputs = (
            _evaluate_controller_inputs(
                controller_inputs,
                frame_time,
                numerical,
                grid,
                external_controller_names,
                sampled_external_names,
            )
            if any(ticking)
            else {}
        )
        held_internal_controls = {}
        for executor, should_tick in zip(controller_executors, ticking):
            controller_controls = (
                executor.step(
                    state,
                    next_controller_inputs,
                    frame_time,
                )
                if should_tick
                else executor.hold(frame_time)
            )
            held_internal_controls = _merged_controls(
                held_internal_controls,
                controller_controls,
            )

    trajectory = numerical.stack(states, axis=1)
    output_trajectory = numerical.stack(outputs, axis=1)
    summary = summarize_samples(
        trajectory,
        backend=numerical,
        quantiles=quantiles,
        confidence_level=confidence_level,
    )
    output_summary = summarize_samples(
        output_trajectory,
        backend=numerical,
        quantiles=quantiles,
        confidence_level=confidence_level,
    )
    controller_traces = {
        executor.resolved.id: executor.trace() for executor in controller_executors
    }
    verification_reports: dict[str, Any] = {}
    for verification_id, verification in getattr(
        contraption, "verifications", {}
    ).items():
        verification_inputs = {
            name: trajectory[..., binding.state_index]
            for name, binding in verification.input_bindings.items()
        }
        verification_reports[verification_id] = evaluate_verification(
            verification.spec,
            verification_inputs,
            time=numerical.asarray(grid),
        )
    result = SimulationResult(
        time=numerical.asarray(grid),
        state_names=names,
        samples=trajectory,
        output_names=output_names,
        output_samples=output_trajectory,
        summary=summary,
        output_summary=output_summary,
        controller_traces=controller_traces,
        verification_reports=verification_reports,
        metadata={
            "backend": numerical.name,
            "device": str(numerical.device),
            "integrator": integrator,
            "seed": int(seed),
            "sample_count": int(num_samples),
            "interval_kind": summary.interval_kind,
            "process_noise": bool(process_noise),
            "process_noise_declared": bool(
                getattr(
                    system,
                    "has_process_noise",
                    getattr(system, "process_noise", None) is not None,
                )
            ),
            "process_noise_seed_policy": getattr(
                system, "process_noise_seed_policy", None
            ),
            "process_noise_reproducibility": getattr(
                system, "process_noise_reproducibility", None
            ),
            "physics_dt_s": physics_dt,
            "controller_periods_s": {
                controller_id: float(controller.spec.period_s)
                for controller_id, controller in resolved_controllers.items()
            },
            "controllers": list(controller_traces),
            "verifications": list(verification_reports),
            **(
                {"assembly_sha256": str(system.assembly_sha256)}
                if getattr(system, "assembly_sha256", None) is not None
                else {}
            ),
            **(
                {"pmdl_sha256": str(system.pmdl_sha256)}
                if getattr(system, "pmdl_sha256", None) is not None
                else {}
            ),
        },
    )
    result_validator = getattr(contraption, "validate_simulation_result", None)
    if callable(result_validator):
        # Geometry is not a viewer concern: a resolved assembly reconstructs
        # every sample/time configuration from these exact PMDL states and
        # rejects any mechanical boundary drift before the result can escape.
        result_validator(result)
    return result


class OfflineSimulator:
    """Configured facade around :func:`simulate`."""

    def __init__(self, config: SimulationConfig | None = None, **overrides: Any) -> None:
        if config is not None and overrides:
            raise ValueError("Pass a SimulationConfig or keyword defaults, not both")
        self.config = config or SimulationConfig(**overrides)

    def simulate(self, contraption: Any, **overrides: Any) -> SimulationResult:
        values = {
            "dt": self.config.dt,
            "num_samples": self.config.num_samples,
            "seed": self.config.seed,
            "backend": self.config.backend,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "integrator": self.config.integrator,
            "quantiles": self.config.quantiles,
            "confidence_level": self.config.confidence_level,
            "use_model_uncertainty": self.config.use_model_uncertainty,
            "process_noise": self.config.process_noise,
            "newton_tolerance": self.config.newton_tolerance,
            "newton_max_iterations": self.config.newton_max_iterations,
        }
        values.update(overrides)
        return simulate(contraption, **values)

def linearize_dynamics(
    system: Any,
    state: Any,
    parameters: Mapping[str, Any] | None = None,
    controls: Mapping[str, Any] | None = None,
    *,
    t: float = 0.0,
    control_names: Sequence[str] | None = None,
    backend: str | Backend | None = None,
    device: str | None = None,
    relative_step: float = 1e-6,
) -> Linearization:
    """Compute a sparse-compiler-friendly local ODE linearization.

    The returned dense reference matrices are deliberately plain arrays.  A
    compiler may inspect their zero pattern and emit a sparse representation.
    Torch inputs remain differentiable through this finite-difference operation.
    """

    if backend is None:
        backend = infer_backend(state)
    numerical = get_backend(backend, device=device)
    resolved, _ = _resolve_system(system)
    if not hasattr(resolved, "derivative"):
        raise TypeError("linearize_dynamics currently requires an explicit ODE")
    names = _state_names(resolved)
    point = numerical.asarray(state)
    if tuple(point.shape) != (len(names),):
        raise ValueError(f"state must have shape {(len(names),)}")
    defaults = _parameter_defaults(resolved)
    if parameters:
        defaults.update(parameters)
    batched_parameters = sample_parameters(defaults, None, 1, backend=numerical)
    controls = {} if controls is None else dict(controls)
    if control_names is None:
        control_names = tuple(getattr(resolved, "control_names", tuple(controls)))
    control_names = tuple(control_names)
    operating_controls = {
        name: numerical.asarray(controls.get(name, 0.0)) for name in control_names
    }

    def derivative_at(value: Array, control_values: Mapping[str, Array]) -> Array:
        batched = value[None, :]
        return _explicit_derivative(
            resolved, t, batched, batched_parameters, control_values, numerical
        )[0]

    state_columns = []
    for index in range(len(names)):
        basis = numerical.stack(
            [numerical.asarray(1.0 if j == index else 0.0) for j in range(len(names))]
        )
        step = relative_step * numerical.maximum(numerical.abs(point[index]), numerical.asarray(1.0))
        state_columns.append(
            (
                derivative_at(point + step * basis, operating_controls)
                - derivative_at(point - step * basis, operating_controls)
            )
            / (2.0 * step)
        )
    state_matrix = numerical.stack(state_columns, axis=-1)
    control_columns = []
    for name in control_names:
        nominal = operating_controls[name]
        step = relative_step * numerical.maximum(numerical.abs(nominal), numerical.asarray(1.0))
        above = dict(operating_controls)
        below = dict(operating_controls)
        above[name] = nominal + step
        below[name] = nominal - step
        control_columns.append(
            (derivative_at(point, above) - derivative_at(point, below)) / (2.0 * step)
        )
    control_matrix = (
        numerical.stack(control_columns, axis=-1)
        if control_columns
        else numerical.zeros((len(names), 0))
    )
    return Linearization(
        state_matrix,
        control_matrix,
        names,
        control_names,
        point,
        operating_controls,
    )


__all__ = [
    "DescriptorDynamics",
    "ExplicitDynamics",
    "Linearization",
    "OfflineSimulator",
    "ResidualSystem",
    "SimulationConfig",
    "SimulationResult",
    "controller_time_step",
    "linearize_dynamics",
    "simulate",
]
