"""Backend-preserving execution of parsed verification trajectory programs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any

import numpy as np

from ..physics.dsl import (
    Binary,
    Call,
    Comparison,
    Conditional,
    DSLParseError,
    Expression,
    Literal,
    Symbol,
    Unary,
)
from .specs import VerificationProgram, VerificationRuntimeError


def _is_torch(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch"


class _ArrayOps:
    def __init__(self, reference: Any | None) -> None:
        self.reference = reference
        self.is_torch = reference is not None
        if self.is_torch:
            import torch

            self.module = torch
        else:
            self.module = np

    def array(self, value: Any) -> Any:
        if self.is_torch:
            if _is_torch(value):
                return value
            raw = np.asarray(value)
            if raw.dtype.kind == "b":
                return self.module.as_tensor(
                    value, dtype=self.module.bool, device=self.reference.device
                )
            return self.reference.new_tensor(value)
        return np.asarray(value)

    def isfinite(self, value: Any) -> bool:
        if self.is_torch:
            return bool(self.module.isfinite(value).all().detach().cpu().item())
        return bool(np.isfinite(np.asarray(value)).all())

    def broadcast(self, value: Any, shape: tuple[int, ...]) -> Any:
        if self.is_torch:
            return value.expand(shape)
        return np.broadcast_to(value, shape)

    def reduce(self, reducer: str, value: Any, time: Any) -> Any:
        if reducer == "initial":
            return value[:, 0]
        if reducer == "final":
            return value[:, -1]
        if reducer == "mean":
            return self._time_average(value, time)
        if reducer == "min":
            return value.min(dim=1).values if self.is_torch else np.min(value, axis=1)
        if reducer == "max":
            return value.max(dim=1).values if self.is_torch else np.max(value, axis=1)
        if reducer == "rmse":
            return self.module.sqrt(self._time_average(value * value, time))
        raise VerificationRuntimeError(f"unsupported trajectory reducer {reducer!r}")

    def _time_average(self, value: Any, time: Any) -> Any:
        """Trapezoidal time average, preserving the active numeric backend."""

        interval_widths = time[1:] - time[:-1]
        segments = 0.5 * (value[:, :-1] + value[:, 1:]) * interval_widths
        integral = (
            segments.sum(dim=1) if self.is_torch else np.sum(segments, axis=1)
        )
        return integral / (time[-1] - time[0])

    def bool_array(self, value: Any) -> bool:
        if self.is_torch:
            return value.dtype == self.module.bool
        return np.asarray(value).dtype.kind == "b"

    def count_true(self, value: Any) -> int:
        if self.is_torch:
            return int(value.sum().detach().cpu().item())
        return int(np.count_nonzero(value))

    def scalar_bool(self, value: Any) -> bool:
        if self.is_torch:
            return bool(value.detach().cpu().item())
        return bool(np.asarray(value).item())

    def logical_not(self, value: Any) -> Any:
        return self.module.logical_not(value)

    def size(self, value: Any) -> int:
        return int(value.numel()) if self.is_torch else int(np.asarray(value).size)

    def all_positive(self, value: Any) -> bool:
        if self.is_torch:
            return bool((value > 0).all().detach().cpu().item())
        return bool(np.all(np.asarray(value) > 0))

    def plain(self, value: Any) -> Any:
        if self.is_torch and _is_torch(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        return value


def _lookup(name: str, values: Mapping[str, Any]) -> Any:
    if name in {"pi", "e"}:
        return math.pi if name == "pi" else math.e
    if name in values:
        return values[name]
    if "." in name:
        current: Any = values
        try:
            for part in name.split("."):
                current = current[part] if isinstance(current, Mapping) else getattr(current, part)
            return current
        except (KeyError, AttributeError, TypeError) as exc:
            raise VerificationRuntimeError(f"no value supplied for symbol {name!r}") from exc
    raise VerificationRuntimeError(f"no value supplied for symbol {name!r}")


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(size) for size in value.shape)


def _masked_value(value: Any, mask: Any, ops: _ArrayOps) -> Any:
    """Select only active elements from an evaluation environment value."""

    if isinstance(value, Mapping):
        return {key: _masked_value(item, mask, ops) for key, item in value.items()}
    array = ops.array(value)
    if _shape(array) == ():
        return array
    try:
        broadcast = ops.broadcast(array, _shape(mask))
    except (RuntimeError, ValueError) as exc:
        raise VerificationRuntimeError(
            f"cannot apply Boolean mask {_shape(mask)} to value {_shape(array)}"
        ) from exc
    return broadcast[mask]


def _masked_environment(
    values: Mapping[str, Any], mask: Any, ops: _ArrayOps
) -> dict[str, Any]:
    return {key: _masked_value(value, mask, ops) for key, value in values.items()}


def _selected_branch(value: Any, count: int, ops: _ArrayOps) -> Any:
    result = ops.array(value)
    if _shape(result) == ():
        return ops.broadcast(result, (count,))
    if _shape(result) != (count,):
        raise VerificationRuntimeError(
            "masked verification branch must produce one value per selected element; "
            f"expected {(count,)}, got {_shape(result)}"
        )
    return result


def _merge_masked(
    condition: Any, when_true: Any, when_false: Any, ops: _ArrayOps
) -> Any:
    true_count = ops.count_true(condition)
    false_count = ops.size(condition) - true_count
    true_values = _selected_branch(when_true, true_count, ops)
    false_values = _selected_branch(when_false, false_count, ops)
    if ops.is_torch:
        dtype = ops.module.promote_types(true_values.dtype, false_values.dtype)
        result = ops.module.empty(
            _shape(condition), dtype=dtype, device=condition.device
        )
        result[condition] = true_values.to(dtype=dtype)
        result[ops.logical_not(condition)] = false_values.to(dtype=dtype)
        return result
    dtype = np.result_type(true_values.dtype, false_values.dtype)
    result = np.empty(_shape(condition), dtype=dtype)
    result[condition] = true_values
    result[np.logical_not(condition)] = false_values
    return result


def _lazy_select(
    condition: Any,
    values: Mapping[str, Any],
    ops: _ArrayOps,
    when_true: Callable[[Mapping[str, Any]], Any],
    when_false: Callable[[Mapping[str, Any]], Any],
) -> Any:
    """Evaluate only the selected branch, elementwise for array conditions.

    Numeric branch values remain backend-native and differentiable. The Boolean
    mask is intentionally discrete: it controls which subgraph is constructed,
    and no gradient is defined through that branch decision.
    """

    boolean = ops.array(condition)
    if not ops.bool_array(boolean):
        raise VerificationRuntimeError("verification conditional must be boolean")
    if _shape(boolean) == ():
        return when_true(values) if ops.scalar_bool(boolean) else when_false(values)

    true_count = ops.count_true(boolean)
    total_count = ops.size(boolean)
    if true_count == total_count:
        return when_true(values)
    if true_count == 0:
        return when_false(values)

    true_values = _masked_environment(values, boolean, ops)
    false_mask = ops.logical_not(boolean)
    false_values = _masked_environment(values, false_mask, ops)
    return _merge_masked(
        boolean,
        when_true(true_values),
        when_false(false_values),
        ops,
    )


def _evaluate_backend(
    expression: Expression, values: Mapping[str, Any], ops: _ArrayOps
) -> Any:
    """Evaluate the allow-listed AST with backend-native, masked semantics."""

    module = ops.module
    if isinstance(expression, Literal):
        return expression.value
    if isinstance(expression, Symbol):
        return _lookup(expression.name, values)
    if isinstance(expression, Unary):
        value = _evaluate_backend(expression.operand, values, ops)
        if expression.operator == "+":
            return +value
        if expression.operator == "-":
            return -value
        if expression.operator == "not":
            return ops.logical_not(ops.array(value))
    if isinstance(expression, Binary):
        left = _evaluate_backend(expression.left, values, ops)
        if expression.operator == "and":
            return _lazy_select(
                left,
                values,
                ops,
                lambda selected: _evaluate_backend(expression.right, selected, ops),
                lambda selected: False,
            )
        if expression.operator == "or":
            return _lazy_select(
                left,
                values,
                ops,
                lambda selected: True,
                lambda selected: _evaluate_backend(expression.right, selected, ops),
            )
        right = _evaluate_backend(expression.right, values, ops)
        functions = {
            "+": lambda: left + right,
            "-": lambda: left - right,
            "*": lambda: left * right,
            "/": lambda: left / right,
            "**": lambda: left**right,
        }
        try:
            return functions[expression.operator]()
        except KeyError as exc:
            raise VerificationRuntimeError(
                f"unsupported verification binary operator {expression.operator!r}"
            ) from exc
    if isinstance(expression, Comparison):
        left = _evaluate_backend(expression.left, values, ops)
        right = _evaluate_backend(expression.right, values, ops)
        functions = {
            "<": lambda: left < right,
            "<=": lambda: left <= right,
            ">": lambda: left > right,
            ">=": lambda: left >= right,
            "==": lambda: left == right,
            "!=": lambda: left != right,
        }
        return functions[expression.operator]()
    if isinstance(expression, Conditional):
        condition = _evaluate_backend(expression.condition, values, ops)
        return _lazy_select(
            condition,
            values,
            ops,
            lambda selected: _evaluate_backend(expression.when_true, selected, ops),
            lambda selected: _evaluate_backend(expression.when_false, selected, ops),
        )
    if isinstance(expression, Call):
        if expression.function == "where":
            condition = _evaluate_backend(expression.arguments[0], values, ops)
            return _lazy_select(
                condition,
                values,
                ops,
                lambda selected: _evaluate_backend(expression.arguments[1], selected, ops),
                lambda selected: _evaluate_backend(expression.arguments[2], selected, ops),
            )
        args = tuple(
            ops.array(_evaluate_backend(item, values, ops))
            for item in expression.arguments
        )
        inverse_sine = module.asin if ops.is_torch else module.arcsin
        inverse_cosine = module.acos if ops.is_torch else module.arccos
        inverse_tangent = module.atan if ops.is_torch else module.arctan
        inverse_tangent2 = module.atan2 if ops.is_torch else module.arctan2
        functions = {
            "abs": module.abs,
            "sqrt": module.sqrt,
            "sin": module.sin,
            "cos": module.cos,
            "tan": module.tan,
            "tanh": module.tanh,
            "asin": inverse_sine,
            "acos": inverse_cosine,
            "atan": inverse_tangent,
            "atan2": inverse_tangent2,
            "exp": module.exp,
            "log": module.log,
            "log10": module.log10,
            "min": module.minimum,
            "max": module.maximum,
            "clip": module.clamp if ops.is_torch else module.clip,
            "sign": module.sign,
            "smooth_abs": lambda x, epsilon=1e-12: module.sqrt(
                x * x + epsilon * epsilon
            ),
        }
        try:
            return functions[expression.function](*args)
        except KeyError as exc:
            raise VerificationRuntimeError(
                f"unsupported verification function {expression.function!r}"
            ) from exc
    raise VerificationRuntimeError(
        f"unsupported verification expression node {type(expression).__name__}"
    )


def _evaluate(expression: Expression, values: Mapping[str, Any], ops: _ArrayOps) -> Any:
    try:
        return _evaluate_backend(expression, values, ops)
    except VerificationRuntimeError:
        raise
    except (DSLParseError, ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
        raise VerificationRuntimeError(f"verification expression evaluation failed: {exc}") from exc


def _as_trajectory(value: Any, name: str, ops: _ArrayOps) -> Any:
    result = ops.array(value)
    if _shape(result) == ():
        raise VerificationRuntimeError(
            f"verification input {name!r} must have shape [sample,time], not a scalar"
        )
    if len(_shape(result)) != 2:
        raise VerificationRuntimeError(
            f"verification input {name!r} must have shape [sample,time], got {_shape(result)}"
        )
    if not ops.isfinite(result):
        raise VerificationRuntimeError(f"verification input {name!r} contains non-finite values")
    return result


def _as_time(value: Any, expected_count: int, ops: _ArrayOps) -> Any:
    result = ops.array(value)
    if _shape(result) != (expected_count,):
        raise VerificationRuntimeError(
            "verification time must have shape [time] matching the trajectory axis; "
            f"expected {(expected_count,)}, got {_shape(result)}"
        )
    if not ops.isfinite(result):
        raise VerificationRuntimeError("verification time contains non-finite values")
    if not ops.all_positive(result[1:] - result[:-1]):
        raise VerificationRuntimeError("verification time must be strictly increasing")
    return result


def _as_metric_trajectory(
    value: Any, metric_name: str, expected_shape: tuple[int, int], ops: _ArrayOps
) -> Any:
    result = ops.array(value)
    if _shape(result) == ():
        result = ops.broadcast(result, expected_shape)
    if _shape(result) != expected_shape:
        raise VerificationRuntimeError(
            f"verification metric {metric_name!r} expression must produce "
            f"[sample,time] {expected_shape}, got {_shape(result)}"
        )
    if not ops.isfinite(result):
        raise VerificationRuntimeError(
            f"verification metric {metric_name!r} expression produced non-finite values"
        )
    return result


def _as_criterion_samples(
    value: Any, criterion_name: str, sample_count: int, ops: _ArrayOps
) -> Any:
    result = ops.array(value)
    if _shape(result) == ():
        result = ops.broadcast(result, (sample_count,))
    if _shape(result) != (sample_count,):
        raise VerificationRuntimeError(
            f"verification criterion {criterion_name!r} must produce [sample] "
            f"{(sample_count,)}, got {_shape(result)}"
        )
    if not ops.bool_array(result):
        raise VerificationRuntimeError(
            f"verification criterion {criterion_name!r} did not produce booleans"
        )
    return result


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Continuous per-sample values; Torch values retain their autograd graph."""

    name: str
    unit: str
    reducer: str
    values: Any

    def to_dict(self) -> dict[str, Any]:
        ops = _ArrayOps(self.values if _is_torch(self.values) else None)
        return {
            "name": self.name,
            "unit": self.unit,
            "reducer": self.reducer,
            "values": ops.plain(self.values),
        }


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """Discrete finite-sample admission derived from Boolean criterion outcomes.

    ``passes``, counts, confidence bounds, and ``accepted`` are reporting values,
    not differentiable surrogates. Differentiable numeric results live in the
    corresponding :class:`MetricResult` objects.
    """

    name: str
    expression: str
    passes: Any
    pass_count: int
    sample_count: int
    effective_sample_count: float
    probability_estimate: float
    confidence_level: float
    probability_lower_bound: float
    probability_upper_bound: float
    minimum_probability: float
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        ops = _ArrayOps(self.passes if _is_torch(self.passes) else None)
        return {
            "name": self.name,
            "expression": self.expression,
            "passes": ops.plain(self.passes),
            "pass_count": self.pass_count,
            "sample_count": self.sample_count,
            "effective_sample_count": self.effective_sample_count,
            "probability_estimate": self.probability_estimate,
            "confidence_level": self.confidence_level,
            "probability_lower_bound": self.probability_lower_bound,
            "probability_upper_bound": self.probability_upper_bound,
            "minimum_probability": self.minimum_probability,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class VerificationReport:
    program_id: str
    program_version: str
    program_sha256: str
    sample_count: int
    time_count: int
    metrics: Mapping[str, MetricResult]
    criteria: Mapping[str, CriterionResult]
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "contraption.verification-report/v2",
            "program": {
                "id": self.program_id,
                "version": self.program_version,
                "sha256": self.program_sha256,
            },
            "sample_count": self.sample_count,
            "time_count": self.time_count,
            "metrics": {name: result.to_dict() for name, result in self.metrics.items()},
            "criteria": {name: result.to_dict() for name, result in self.criteria.items()},
            "accepted": self.accepted,
        }


def _wilson_score_interval(
    pass_count: int, sample_count: int, confidence_level: float
) -> tuple[float, float, float]:
    """One-sided Wilson lower/upper bounds for equal-weight Bernoulli samples.

    The normal quantile is deterministic and uses only the Python standard
    library. Admission consumes the conservative lower bound; the point
    estimate and upper bound are reported for uncertainty context.
    """

    estimate = pass_count / sample_count
    z = NormalDist().inv_cdf(confidence_level)
    z_squared = z * z
    denominator = 1.0 + z_squared / sample_count
    center = (estimate + z_squared / (2.0 * sample_count)) / denominator
    half_width = (
        z
        / denominator
        * math.sqrt(
            estimate * (1.0 - estimate) / sample_count
            + z_squared / (4.0 * sample_count * sample_count)
        )
    )
    return estimate, max(0.0, center - half_width), min(1.0, center + half_width)


def evaluate_verification(
    program: VerificationProgram,
    trajectories: Mapping[str, Any],
    *,
    time: Any,
) -> VerificationReport:
    """Evaluate equal-weight posterior trajectories on their exact time grid.

    Metric expressions and reducers stay backend-native. Criteria cross an
    explicit continuous-to-discrete boundary: Boolean outcomes are counted and
    admitted only when a conservative finite-sample confidence lower bound
    reaches the authored minimum probability.
    """

    if not isinstance(program, VerificationProgram):
        raise TypeError("program must be a VerificationProgram")
    if not isinstance(trajectories, Mapping):
        raise TypeError("trajectories must be a name-to-array mapping")
    expected = {item.name for item in program.inputs}
    supplied = set(trajectories)
    if expected != supplied:
        raise VerificationRuntimeError(
            f"verification input coverage mismatch; missing={sorted(expected - supplied)}, "
            f"unknown={sorted(supplied - expected)}"
        )
    torch_values = [value for value in trajectories.values() if _is_torch(value)]
    reference = torch_values[0] if torch_values else None
    ops = _ArrayOps(reference)
    environment: dict[str, Any] = {
        name: _as_trajectory(trajectories[name], name, ops) for name in sorted(expected)
    }
    shapes = {_shape(value) for value in environment.values()}
    if len(shapes) != 1:
        raise VerificationRuntimeError(
            f"verification input trajectories must share one [sample,time] shape, got {sorted(shapes)}"
        )
    sample_count, time_count = next(iter(shapes))
    if sample_count < 1:
        raise VerificationRuntimeError("verification trajectories require at least one sample")
    if time_count < 2:
        raise VerificationRuntimeError(
            "verification trajectories require at least two time points"
        )
    time_values = _as_time(time, time_count, ops)
    environment.update({item.name: item.value for item in program.parameters})

    metric_results: dict[str, MetricResult] = {}
    metric_values: dict[str, Any] = {}
    for metric in program.metrics:
        trajectory = _as_metric_trajectory(
            _evaluate(metric.parsed_expression, environment, ops),
            metric.name,
            (sample_count, time_count),
            ops,
        )
        values = ops.reduce(metric.reducer, trajectory, time_values)
        if not ops.isfinite(values):
            raise VerificationRuntimeError(
                f"verification metric {metric.name!r} reducer produced non-finite values"
            )
        metric_values[metric.name] = values
        metric_results[metric.name] = MetricResult(
            metric.name, metric.unit, metric.reducer, values
        )

    criterion_environment = {
        **{item.name: item.value for item in program.parameters},
        **metric_values,
    }
    criterion_results: dict[str, CriterionResult] = {}
    for criterion in program.criteria:
        passes = _as_criterion_samples(
            _evaluate(criterion.parsed_expression, criterion_environment, ops),
            criterion.name,
            sample_count,
            ops,
        )
        pass_count = ops.count_true(passes)
        probability, lower_bound, upper_bound = _wilson_score_interval(
            pass_count, sample_count, criterion.confidence_level
        )
        accepted = lower_bound >= criterion.minimum_probability
        criterion_results[criterion.name] = CriterionResult(
            criterion.name,
            criterion.expression,
            passes,
            pass_count,
            sample_count,
            float(sample_count),
            probability,
            criterion.confidence_level,
            lower_bound,
            upper_bound,
            criterion.minimum_probability,
            accepted,
        )

    return VerificationReport(
        program.id,
        program.version,
        program.sha256,
        sample_count,
        time_count,
        metric_results,
        criterion_results,
        all(result.accepted for result in criterion_results.values()),
    )


class VerificationRuntime:
    """Small configured facade for repeated evaluation of one immutable program."""

    def __init__(self, program: VerificationProgram) -> None:
        if not isinstance(program, VerificationProgram):
            raise TypeError("program must be a VerificationProgram")
        self.program = program

    def evaluate(self, trajectories: Mapping[str, Any], *, time: Any) -> VerificationReport:
        return evaluate_verification(self.program, trajectories, time=time)


__all__ = [
    "CriterionResult",
    "MetricResult",
    "VerificationReport",
    "VerificationRuntime",
    "evaluate_verification",
]
