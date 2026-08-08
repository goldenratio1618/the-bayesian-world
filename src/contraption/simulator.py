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
import inspect
import math
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

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


def _resolve_system(candidate: Any) -> tuple[Any, Any | None]:
    controller = getattr(candidate, "controller", None)
    system = candidate
    if not hasattr(system, "derivative") and not hasattr(system, "residual"):
        for attribute in ("dynamics", "system", "simulation_model", "model"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                system = nested
                break
    if not hasattr(system, "derivative") and hasattr(system, "evaluate_residual"):
        system = _ModelSpecAdapter(system)
    elif not hasattr(system, "derivative") and not hasattr(system, "residual"):
        raise TypeError("System must expose derivative(...) or residual(...)")
    return system, controller


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


def _call_control(function: Callable[..., Any], t: float, state: Array, backend: Backend) -> Any:
    try:
        signature = inspect.signature(function)
        parameters = list(signature.parameters.values())
        accepts_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters)
        names = {item.name for item in parameters}
        kwargs = {"backend": backend} if "backend" in names or accepts_kwargs else {}
        positional = [
            item
            for item in parameters
            if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2:
            return function(t, state, **kwargs)
        return function(t, **kwargs)
    except (TypeError, ValueError):
        return function(t)


def _control_value(
    value: Any,
    t: float,
    state: Array,
    backend: Backend,
    times: np.ndarray,
) -> Array:
    if callable(value):
        value = _call_control(value, t, state, backend)
    elif hasattr(value, "evaluate"):
        value = _call_control(value.evaluate, t, state, backend)
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
    controller: Any | None,
    t: float,
    state: Array,
    backend: Backend,
    times: np.ndarray,
) -> dict[str, Array]:
    result: dict[str, Array] = {}
    if source is not None:
        if isinstance(source, Mapping):
            result.update(
                {
                    str(name): _control_value(value, t, state, backend, times)
                    for name, value in source.items()
                }
            )
        else:
            evaluator = source.evaluate if hasattr(source, "evaluate") else source
            values = _call_control(evaluator, t, state, backend)
            if not isinstance(values, Mapping):
                raise TypeError("A control provider must return a mapping")
            result.update({str(name): backend.asarray(value) for name, value in values.items()})
    if controller is not None:
        evaluator = controller.evaluate if hasattr(controller, "evaluate") else controller
        internal = _call_control(evaluator, t, state, backend)
        if not isinstance(internal, Mapping):
            raise TypeError("An internal controller must return a mapping")
        # Internal commands deliberately take precedence over equally-named
        # external settings; external values remain available under other names.
        result.update({str(name): backend.asarray(value) for name, value in internal.items()})
    return result


def _merged_controls(
    external: Mapping[str, Array], internal: Mapping[str, Array]
) -> dict[str, Array]:
    """Merge held internal commands over externally supplied settings."""

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
        _evaluate_controls(control_source, None, t, state, backend, times),
        held_internal_controls,
    )
    k1 = _explicit_derivative(system, t, state, parameters, c1, backend)
    if method == "euler":
        return state + dt * k1, c1
    midpoint = state + 0.5 * dt * k1
    c2 = _merged_controls(
        _evaluate_controls(control_source, None, t + 0.5 * dt, midpoint, backend, times),
        held_internal_controls,
    )
    k2 = _explicit_derivative(system, t + 0.5 * dt, midpoint, parameters, c2, backend)
    midpoint = state + 0.5 * dt * k2
    c3 = _merged_controls(
        _evaluate_controls(control_source, None, t + 0.5 * dt, midpoint, backend, times),
        held_internal_controls,
    )
    k3 = _explicit_derivative(system, t + 0.5 * dt, midpoint, parameters, c3, backend)
    endpoint = state + dt * k3
    c4 = _merged_controls(
        _evaluate_controls(control_source, None, t + dt, endpoint, backend, times),
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
    ``(mean, std)``, :class:`~contraption.uq.Normal`, or a mapping with
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
    system, controller = _resolve_system(contraption)
    names = _state_names(system)
    grid = _time_grid(t_span, duration, dt, times)
    _require_valid_timestep(system, grid)
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
    if controller is not None and hasattr(controller, "reset"):
        controller.reset()
    explicit = hasattr(system, "derivative")
    if integrator == "auto":
        integrator = "rk4" if explicit else "implicit_euler"
    if integrator in {"euler", "rk4"} and not explicit:
        raise ValueError(f"{integrator} requires an explicit derivative")
    if integrator == "implicit_euler" and not hasattr(system, "residual"):
        raise ValueError("implicit_euler requires a descriptor residual")
    if integrator not in {"euler", "rk4", "implicit_euler"}:
        raise ValueError(f"Unsupported integrator {integrator!r}")

    external_controls = _evaluate_controls(controls, None, float(grid[0]), state, numerical, grid)
    held_internal_controls = _evaluate_controls(
        None, controller, float(grid[0]), state, numerical, grid
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
        # Initial commands are sample-and-hold inputs to the consistency solve.
        # Re-evaluating feedback against solved algebraics here would change the
        # equations after solving them and publish an inconsistent first frame.
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
                _evaluate_controls(controls, None, t + step_size, state, numerical, grid),
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
        if process_noise and noise_function is not None:
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
        state = next_state
        states.append(state)
        if explicit:
            next_external_controls = _evaluate_controls(
                controls, None, float(grid[index + 1]), state, numerical, grid
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
        held_internal_controls = _evaluate_controls(
            None, controller, float(grid[index + 1]), state, numerical, grid
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
    result = SimulationResult(
        time=numerical.asarray(grid),
        state_names=names,
        samples=trajectory,
        output_names=output_names,
        output_samples=output_trajectory,
        summary=summary,
        output_summary=output_summary,
        metadata={
            "backend": numerical.name,
            "device": str(numerical.device),
            "integrator": integrator,
            "seed": int(seed),
            "sample_count": int(num_samples),
            "interval_kind": summary.interval_kind,
            "process_noise": bool(process_noise),
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


def _parameter(parameters: Mapping[str, Array], name: str) -> Array:
    try:
        return parameters[name]
    except KeyError as exc:
        raise KeyError(f"Required physical parameter {name!r} was not supplied") from exc


def _control(controls: Mapping[str, Array], *names: str, default: Any = 0.0) -> Any:
    for name in names:
        if name in controls:
            return controls[name]
    return default


class RCCircuit:
    """Series resistor-capacitor circuit driven by a voltage source."""

    state_names = ("capacitor_voltage",)
    control_names = ("voltage",)
    output_names = ("capacitor_voltage", "resistor_current")

    def __init__(
        self,
        resistance: Any = 1_000.0,
        capacitance: Any = 1e-3,
        initial_voltage: Any = 0.0,
    ) -> None:
        self.default_parameters = {
            "resistance": resistance,
            "capacitance": capacitance,
        }
        self.parameter_bounds = {
            "resistance": (1e-12, None),
            "capacitance": (1e-15, None),
        }
        self.parameter_uncertainty = {
            "resistance": {"std": 0.01 * resistance},
            "capacitance": {"std": 0.02 * capacitance},
        }
        self.initial_state = [initial_voltage]

    def derivative(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Array:
        voltage = _control(controls, "voltage", "source_voltage", "input_voltage")
        rate = (voltage - state[:, 0]) / (
            _parameter(parameters, "resistance") * _parameter(parameters, "capacitance")
        )
        return rate[:, None]

    def observe(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Mapping[str, Array]:
        voltage = _control(controls, "voltage", "source_voltage", "input_voltage")
        return {
            "capacitor_voltage": state[:, 0],
            "resistor_current": (voltage - state[:, 0]) / _parameter(parameters, "resistance"),
        }

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
        self.parameter_uncertainty = {
            "resistance": {"std": 0.01 * resistance},
            "inductance": {"std": 0.02 * inductance},
        }
        self.initial_state = [initial_current]

    def derivative(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Array:
        voltage = _control(controls, "voltage", "source_voltage", "input_voltage")
        rate = (voltage - _parameter(parameters, "resistance") * state[:, 0]) / _parameter(parameters, "inductance")
        return rate[:, None]

    def observe(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Mapping[str, Array]:
        source = _control(controls, "voltage", "source_voltage", "input_voltage")
        resistor = _parameter(parameters, "resistance") * state[:, 0]
        return {
            "inductor_current": state[:, 0],
            "resistor_voltage": resistor,
            "inductor_voltage": source - resistor,
        }

    def simulate(self, **options: Any) -> SimulationResult:
        return simulate(self, **options)


class DCMotorSystem:
    """Armature-controlled permanent-magnet DC motor."""

    state_names = ("armature_current", "angular_velocity", "shaft_angle")
    control_names = ("voltage", "load_torque")
    output_names = state_names + ("electromagnetic_torque",)

    def __init__(
        self,
        resistance: Any = 2.0,
        inductance: Any = 0.01,
        torque_constant: Any = 0.08,
        back_emf_constant: Any = 0.08,
        inertia: Any = 0.002,
        viscous_friction: Any = 0.001,
        initial_state: Sequence[Any] = (0.0, 0.0, 0.0),
    ) -> None:
        self.default_parameters = {
            "resistance": resistance,
            "inductance": inductance,
            "torque_constant": torque_constant,
            "back_emf_constant": back_emf_constant,
            "inertia": inertia,
            "viscous_friction": viscous_friction,
        }
        self.parameter_bounds = {name: (1e-15, None) for name in self.default_parameters}
        self.parameter_uncertainty = {
            name: {"std": abs(value) * 0.02}
            for name, value in self.default_parameters.items()
        }
        self.initial_state = list(initial_state)

    def derivative(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Array:
        current, speed = state[:, 0], state[:, 1]
        voltage = _control(controls, "voltage", "armature_voltage")
        load = _control(controls, "load_torque", default=0.0)
        current_rate = (
            voltage
            - _parameter(parameters, "resistance") * current
            - _parameter(parameters, "back_emf_constant") * speed
        ) / _parameter(parameters, "inductance")
        speed_rate = (
            _parameter(parameters, "torque_constant") * current
            - _parameter(parameters, "viscous_friction") * speed
            - load
        ) / _parameter(parameters, "inertia")
        return backend.stack((current_rate, speed_rate, speed), axis=-1)

    def observe(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Mapping[str, Array]:
        return {
            "armature_current": state[:, 0],
            "angular_velocity": state[:, 1],
            "shaft_angle": state[:, 2],
            "electromagnetic_torque": _parameter(parameters, "torque_constant") * state[:, 0],
        }

    def simulate(self, **options: Any) -> SimulationResult:
        return simulate(self, **options)


class PlanarRigidBodySystem:
    """Planar rigid body with world-frame forces and optional drag/roughness."""

    state_names = ("x", "y", "yaw", "velocity_x", "velocity_y", "angular_velocity")
    control_names = ("force_x", "force_y", "torque")

    def __init__(
        self,
        mass: Any = 1.0,
        moment_of_inertia: Any = 0.1,
        linear_drag: Any = 0.0,
        angular_drag: Any = 0.0,
        roughness_std: Any = 0.0,
        initial_state: Sequence[Any] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.default_parameters = {
            "mass": mass,
            "moment_of_inertia": moment_of_inertia,
            "linear_drag": linear_drag,
            "angular_drag": angular_drag,
            "roughness_std": roughness_std,
        }
        self.parameter_bounds = {
            "mass": (1e-12, None),
            "moment_of_inertia": (1e-12, None),
            "linear_drag": (0.0, None),
            "angular_drag": (0.0, None),
            "roughness_std": (0.0, None),
        }
        self.initial_state = list(initial_state)

    def derivative(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], backend: Backend) -> Array:
        vx, vy, omega = state[:, 3], state[:, 4], state[:, 5]
        ax = (_control(controls, "force_x") - _parameter(parameters, "linear_drag") * vx) / _parameter(parameters, "mass")
        ay = (_control(controls, "force_y") - _parameter(parameters, "linear_drag") * vy) / _parameter(parameters, "mass")
        angular_acceleration = (
            _control(controls, "torque") - _parameter(parameters, "angular_drag") * omega
        ) / _parameter(parameters, "moment_of_inertia")
        return backend.stack((vx, vy, omega, ax, ay, angular_acceleration), axis=-1)

    def process_noise(self, t: Any, state: Array, parameters: Mapping[str, Array], controls: Mapping[str, Array], dt: Any, rng: Any, backend: Backend) -> Array:
        scale = _parameter(parameters, "roughness_std") * backend.sqrt(dt)
        noise = backend.normal((int(state.shape[0]), 3), rng) * scale[:, None]
        zeros = backend.zeros((int(state.shape[0]), 3))
        return backend.concatenate((noise, zeros), axis=-1)

    def simulate(self, **options: Any) -> SimulationResult:
        return simulate(self, **options)


# Friendly compatibility aliases used in examples and downstream notebooks.
RCSystem = RCCircuit
RLSystem = RLCircuit
DCMotor = DCMotorSystem
PlanarRigidBody = PlanarRigidBodySystem


__all__ = [
    "DCMotor",
    "DCMotorSystem",
    "DescriptorDynamics",
    "ExplicitDynamics",
    "Linearization",
    "OfflineSimulator",
    "PlanarRigidBody",
    "PlanarRigidBodySystem",
    "RCCircuit",
    "RCSystem",
    "RLCircuit",
    "RLSystem",
    "ResidualSystem",
    "SimulationConfig",
    "SimulationResult",
    "linearize_dynamics",
    "simulate",
]
