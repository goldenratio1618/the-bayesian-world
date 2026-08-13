"""Strict, hashable records for the ``verification-1`` trajectory DSL.

Verification artifacts are data-only. Their expressions use the same safe,
allow-listed AST as PMDL, while their records define how scalar trajectory
expressions are reduced to one metric value per posterior sample.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from ..physics.dsl import (
    BOOLEAN,
    Binary,
    Call,
    Comparison,
    Conditional,
    DSLParseError,
    Expression,
    ExpressionType,
    ExpressionTypeError,
    Literal,
    Symbol,
    Unary,
    parse_expression,
)
from ..physics.units import Unit, UnitError, parse_unit


class VerificationError(ValueError):
    """Base class for verification artifact and execution failures."""


class VerificationSpecError(VerificationError):
    """A ``verification-1`` artifact is malformed or ill-typed."""


class VerificationRuntimeError(VerificationError):
    """Trajectory data cannot be evaluated against a valid program."""


_ARTIFACT_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_REDUCERS = frozenset({"initial", "final", "mean", "min", "max", "rmse"})


class DifferentiabilityClass(str, Enum):
    """Typed boundary between numeric paths and verification decisions."""

    SMOOTH = "smooth_on_valid_domain"
    PIECEWISE_SMOOTH = "piecewise_smooth_on_valid_domain"
    DISCRETE = "discrete"


_BACKEND_FUNCTIONS = frozenset(
    {
        "abs", "sqrt", "sin", "cos", "tan", "tanh", "asin", "acos", "atan",
        "atan2", "exp", "log", "log10", "min", "max", "clip", "sign", "where",
        "smooth_abs",
    }
)
_PIECEWISE_FUNCTIONS = frozenset({"abs", "min", "max", "clip", "sign", "where"})


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise VerificationSpecError(f"{context} must be an object with string keys")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise VerificationSpecError(f"{context} must be an array")
    return value


def _keys(
    value: Mapping[str, Any],
    allowed: Sequence[str] | set[str],
    context: str,
    required: Sequence[str] | set[str] = (),
) -> None:
    unknown = sorted(set(value) - set(allowed))
    missing = sorted(set(required) - set(value))
    if unknown:
        raise VerificationSpecError(f"unknown {context} field(s): {', '.join(unknown)}")
    if missing:
        raise VerificationSpecError(f"missing {context} field(s): {', '.join(missing)}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerificationSpecError(f"{context} must be a non-empty string")
    return value


def _artifact_id(value: Any, context: str) -> str:
    result = _text(value, context)
    if _ARTIFACT_ID.fullmatch(result) is None:
        raise VerificationSpecError(f"{context} is not a valid artifact identifier")
    return result


def _symbol(value: Any, context: str) -> str:
    result = _text(value, context)
    if _SYMBOL.fullmatch(result) is None or result in {"pi", "e"}:
        raise VerificationSpecError(f"{context} is not a valid verification symbol")
    return result


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationSpecError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise VerificationSpecError(f"{context} must be finite")
    return result


def _unit(value: Any, context: str) -> Unit:
    text = _text(value, context)
    try:
        return parse_unit(text)
    except UnitError as exc:
        raise VerificationSpecError(f"{context} is invalid: {exc}") from exc


def _expression(value: Any, context: str) -> tuple[str, Expression]:
    source = _text(value, context)
    try:
        parsed = parse_expression(source)
    except DSLParseError as exc:
        raise VerificationSpecError(f"{context} is invalid: {exc}") from exc
    _reject_derivative(parsed, context)
    _validate_backend_expression(parsed, context)
    return source, parsed


def _reject_derivative(expression: Expression, context: str) -> None:
    """Reject PMDL's state-only ``der()`` primitive in trajectory expressions."""

    if isinstance(expression, Call) and expression.function == "der":
        raise VerificationSpecError(
            f"{context} may not use der(); declare a derivative trajectory input explicitly"
        )
    for record_field in dataclass_fields(expression):
        value = getattr(expression, record_field.name)
        if isinstance(value, Expression):
            _reject_derivative(value, context)
        elif isinstance(value, tuple):
            for item in value:
                if isinstance(item, Expression):
                    _reject_derivative(item, context)


def _validate_backend_expression(expression: Expression, context: str) -> None:
    """Prove that the AST has a backend-native NumPy/Torch implementation."""

    if isinstance(expression, (Literal, Symbol)):
        return
    if isinstance(expression, Unary):
        if expression.operator not in {"+", "-", "not"}:
            raise VerificationSpecError(
                f"{context} uses unsupported backend operator {expression.operator!r}"
            )
        _validate_backend_expression(expression.operand, context)
        return
    if isinstance(expression, Binary):
        if expression.operator not in {"+", "-", "*", "/", "**", "and", "or"}:
            raise VerificationSpecError(
                f"{context} uses unsupported backend operator {expression.operator!r}"
            )
        _validate_backend_expression(expression.left, context)
        _validate_backend_expression(expression.right, context)
        return
    if isinstance(expression, Comparison):
        if expression.operator not in {"<", "<=", ">", ">=", "==", "!="}:
            raise VerificationSpecError(
                f"{context} uses unsupported backend comparison {expression.operator!r}"
            )
        _validate_backend_expression(expression.left, context)
        _validate_backend_expression(expression.right, context)
        return
    if isinstance(expression, Conditional):
        _validate_backend_expression(expression.condition, context)
        _validate_backend_expression(expression.when_true, context)
        _validate_backend_expression(expression.when_false, context)
        return
    if isinstance(expression, Call):
        if expression.function not in _BACKEND_FUNCTIONS:
            raise VerificationSpecError(
                f"{context} uses numeric function {expression.function!r} without a "
                "backend-native differentiability contract"
            )
        for argument in expression.arguments:
            _validate_backend_expression(argument, context)
        return
    raise VerificationSpecError(
        f"{context} uses unsupported expression node {type(expression).__name__!r}"
    )


def _numeric_differentiability(
    expression: Expression, context: str
) -> DifferentiabilityClass:
    """Classify a real AST after backend support and type validation."""

    if isinstance(expression, (Literal, Symbol)):
        return DifferentiabilityClass.SMOOTH
    if isinstance(expression, Unary):
        if expression.operator == "not":
            raise VerificationSpecError(
                f"{context} contains a discrete operator in a real numeric path"
            )
        return _numeric_differentiability(expression.operand, context)
    if isinstance(expression, Binary):
        if expression.operator in {"and", "or"}:
            raise VerificationSpecError(
                f"{context} contains a discrete operator in a real numeric path"
            )
        children = (
            _numeric_differentiability(expression.left, context),
            _numeric_differentiability(expression.right, context),
        )
        return (
            DifferentiabilityClass.PIECEWISE_SMOOTH
            if DifferentiabilityClass.PIECEWISE_SMOOTH in children
            else DifferentiabilityClass.SMOOTH
        )
    if isinstance(expression, Conditional):
        _numeric_differentiability(expression.when_true, context)
        _numeric_differentiability(expression.when_false, context)
        return DifferentiabilityClass.PIECEWISE_SMOOTH
    if isinstance(expression, Call):
        arguments = (
            expression.arguments[1:]
            if expression.function == "where"
            else expression.arguments
        )
        child_classes = tuple(
            _numeric_differentiability(argument, context) for argument in arguments
        )
        if (
            expression.function in _PIECEWISE_FUNCTIONS
            or DifferentiabilityClass.PIECEWISE_SMOOTH in child_classes
        ):
            return DifferentiabilityClass.PIECEWISE_SMOOTH
        return DifferentiabilityClass.SMOOTH
    raise VerificationSpecError(
        f"{context} cannot be classified as a differentiable real numeric path"
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationSpecError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise VerificationSpecError(f"non-finite JSON number {value!r} is forbidden")


@dataclass(frozen=True, slots=True)
class VerificationInputSpec:
    """One scalar real trajectory supplied by the resolved contraption runtime."""

    name: str
    unit: str
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _symbol(self.name, "verification input name"))
        _unit(self.unit, f"verification input {self.name!r} unit")
        if not isinstance(self.description, str):
            raise VerificationSpecError("verification input description must be a string")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationInputSpec":
        data = _mapping(value, "verification input")
        _keys(data, {"name", "unit", "description"}, "verification input", {"name", "unit"})
        return cls(data["name"], data["unit"], data.get("description", ""))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "unit": self.unit, "description": self.description}


@dataclass(frozen=True, slots=True)
class VerificationParameterSpec:
    """A hash-bound scalar constant available to metric and criterion expressions."""

    name: str
    unit: str
    value: float
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _symbol(self.name, "verification parameter name"))
        _unit(self.unit, f"verification parameter {self.name!r} unit")
        object.__setattr__(
            self, "value", _finite(self.value, f"verification parameter {self.name!r} value")
        )
        if not isinstance(self.description, str):
            raise VerificationSpecError("verification parameter description must be a string")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationParameterSpec":
        data = _mapping(value, "verification parameter")
        _keys(
            data,
            {"name", "unit", "value", "description"},
            "verification parameter",
            {"name", "unit", "value"},
        )
        return cls(data["name"], data["unit"], data["value"], data.get("description", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": self.value,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryMetricSpec:
    """A backend-native real expression reduced independently per sample."""

    name: str
    expression: str
    reducer: str
    unit: str
    description: str = ""
    parsed_expression: Expression = field(init=False, repr=False, compare=False)
    differentiability: DifferentiabilityClass = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _symbol(self.name, "verification metric name"))
        source, parsed = _expression(
            self.expression, f"verification metric {self.name!r} expression"
        )
        object.__setattr__(self, "expression", source)
        object.__setattr__(self, "parsed_expression", parsed)
        object.__setattr__(
            self,
            "differentiability",
            _numeric_differentiability(
                parsed, f"verification metric {self.name!r} expression"
            ),
        )
        if self.reducer not in _REDUCERS:
            raise VerificationSpecError(
                f"verification metric {self.name!r} reducer must be one of {sorted(_REDUCERS)}"
            )
        _unit(self.unit, f"verification metric {self.name!r} unit")
        if not isinstance(self.description, str):
            raise VerificationSpecError("verification metric description must be a string")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrajectoryMetricSpec":
        data = _mapping(value, "verification metric")
        names = {"name", "expression", "reducer", "unit", "description"}
        _keys(data, names, "verification metric", names - {"description"})
        return cls(
            data["name"],
            data["expression"],
            data["reducer"],
            data["unit"],
            data.get("description", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "reducer": self.reducer,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class VerificationCriterionSpec:
    """A per-sample Boolean gate with conservative finite-sample admission.

    ``minimum_probability`` is compared with a one-sided lower confidence
    bound, never with the raw observed pass fraction. ``confidence_level``
    authors the coverage used for that explicitly discrete decision.
    """

    name: str
    expression: str
    minimum_probability: float
    confidence_level: float
    description: str = ""
    parsed_expression: Expression = field(init=False, repr=False, compare=False)
    differentiability: DifferentiabilityClass = field(
        init=False,
        default=DifferentiabilityClass.DISCRETE,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _symbol(self.name, "verification criterion name"))
        source, parsed = _expression(
            self.expression, f"verification criterion {self.name!r} expression"
        )
        object.__setattr__(self, "expression", source)
        object.__setattr__(self, "parsed_expression", parsed)
        minimum = _finite(
            self.minimum_probability,
            f"verification criterion {self.name!r} minimum_probability",
        )
        if not 0.0 <= minimum <= 1.0:
            raise VerificationSpecError(
                f"verification criterion {self.name!r} minimum_probability must be within [0, 1]"
            )
        object.__setattr__(self, "minimum_probability", minimum)
        confidence = _finite(
            self.confidence_level,
            f"verification criterion {self.name!r} confidence_level",
        )
        if not 0.5 < confidence < 1.0:
            raise VerificationSpecError(
                f"verification criterion {self.name!r} confidence_level must be within (0.5, 1)"
            )
        object.__setattr__(self, "confidence_level", confidence)
        if not isinstance(self.description, str):
            raise VerificationSpecError("verification criterion description must be a string")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationCriterionSpec":
        data = _mapping(value, "verification criterion")
        names = {
            "name", "expression", "minimum_probability", "confidence_level", "description"
        }
        _keys(data, names, "verification criterion", names - {"description"})
        return cls(
            data["name"],
            data["expression"],
            data["minimum_probability"],
            data["confidence_level"],
            data.get("description", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "minimum_probability": self.minimum_probability,
            "confidence_level": self.confidence_level,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class VerificationProgram:
    """A fully parsed and dimension-checked ``verification-1`` artifact."""

    format: str
    id: str
    name: str
    version: str
    inputs: tuple[VerificationInputSpec, ...]
    parameters: tuple[VerificationParameterSpec, ...]
    metrics: tuple[TrajectoryMetricSpec, ...]
    criteria: tuple[VerificationCriterionSpec, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if self.format != "verification-1":
            raise VerificationSpecError(
                f"verification format must be 'verification-1', got {self.format!r}"
            )
        object.__setattr__(self, "id", _artifact_id(self.id, "verification id"))
        _text(self.name, "verification name")
        _text(self.version, "verification version")
        if not isinstance(self.description, str):
            raise VerificationSpecError("verification description must be a string")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "criteria", tuple(self.criteria))
        if not self.inputs:
            raise VerificationSpecError("verification program requires at least one input")
        if not self.metrics:
            raise VerificationSpecError("verification program requires at least one metric")
        if not self.criteria:
            raise VerificationSpecError("verification program requires at least one criterion")

        input_names = self._unique((item.name for item in self.inputs), "input")
        parameter_names = self._unique((item.name for item in self.parameters), "parameter")
        metric_names = self._unique((item.name for item in self.metrics), "metric")
        self._unique((item.name for item in self.criteria), "criterion")
        collisions = sorted(
            (input_names & parameter_names)
            | (input_names & metric_names)
            | (parameter_names & metric_names)
        )
        if collisions:
            raise VerificationSpecError(
                "verification inputs, parameters, and metrics must use distinct symbols: "
                + ", ".join(collisions)
            )

        trajectory_symbols: dict[str, ExpressionType] = {
            item.name: ExpressionType("real", _unit(item.unit, item.name).dimension)
            for item in (*self.inputs, *self.parameters)
        }
        metric_symbols: dict[str, ExpressionType] = {
            item.name: ExpressionType("real", _unit(item.unit, item.name).dimension)
            for item in self.parameters
        }
        for metric in self.metrics:
            try:
                result_type = metric.parsed_expression.infer_type(trajectory_symbols)
            except ExpressionTypeError as exc:
                raise VerificationSpecError(
                    f"verification metric {metric.name!r} is ill-typed: {exc}"
                ) from exc
            expected = ExpressionType("real", _unit(metric.unit, metric.name).dimension)
            if result_type != expected:
                raise VerificationSpecError(
                    f"verification metric {metric.name!r} has type "
                    f"{result_type.kind}/{result_type.dimension.describe()}, expected "
                    f"real/{expected.dimension.describe()}"
                )
            metric_symbols[metric.name] = expected

        for criterion in self.criteria:
            try:
                result_type = criterion.parsed_expression.infer_type(metric_symbols)
            except ExpressionTypeError as exc:
                raise VerificationSpecError(
                    f"verification criterion {criterion.name!r} is ill-typed: {exc}"
                ) from exc
            if result_type != BOOLEAN:
                raise VerificationSpecError(
                    f"verification criterion {criterion.name!r} must be boolean"
                )

    @staticmethod
    def _unique(values: Any, context: str) -> set[str]:
        sequence = list(values)
        if len(sequence) != len(set(sequence)):
            raise VerificationSpecError(f"duplicate verification {context} name")
        return set(sequence)

    @property
    def differentiability(self) -> dict[str, Any]:
        """Machine-readable continuous numeric versus discrete decision boundary."""

        return {
            "metrics": {
                item.name: item.differentiability.value for item in self.metrics
            },
            "criteria": {
                item.name: item.differentiability.value for item in self.criteria
            },
            "admission": DifferentiabilityClass.DISCRETE.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerificationProgram":
        data = _mapping(value, "verification program")
        names = {
            "format", "id", "name", "version", "description", "inputs",
            "parameters", "metrics", "criteria",
        }
        required = {"format", "id", "name", "version", "inputs", "metrics", "criteria"}
        _keys(data, names, "verification program", required)
        return cls(
            data["format"],
            data["id"],
            data["name"],
            data["version"],
            tuple(
                VerificationInputSpec.from_dict(_mapping(item, f"inputs[{index}]"))
                for index, item in enumerate(_sequence(data["inputs"], "verification inputs"))
            ),
            tuple(
                VerificationParameterSpec.from_dict(_mapping(item, f"parameters[{index}]"))
                for index, item in enumerate(
                    _sequence(data.get("parameters", []), "verification parameters")
                )
            ),
            tuple(
                TrajectoryMetricSpec.from_dict(_mapping(item, f"metrics[{index}]"))
                for index, item in enumerate(_sequence(data["metrics"], "verification metrics"))
            ),
            tuple(
                VerificationCriterionSpec.from_dict(_mapping(item, f"criteria[{index}]"))
                for index, item in enumerate(_sequence(data["criteria"], "verification criteria"))
            ),
            data.get("description", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "inputs": [item.to_dict() for item in self.inputs],
            "parameters": [item.to_dict() for item in self.parameters],
            "metrics": [item.to_dict() for item in self.metrics],
            "criteria": [item.to_dict() for item in self.criteria],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def parse_verification(
    source: str | bytes | Mapping[str, Any], *, source_name: str = "<memory>"
) -> VerificationProgram:
    """Parse and eagerly validate a ``verification-1`` JSON artifact."""

    if isinstance(source, Mapping):
        data = source
    else:
        try:
            data = json.loads(
                source.decode("utf-8") if isinstance(source, bytes) else source,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except VerificationSpecError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VerificationSpecError(f"invalid verification JSON in {source_name}: {exc}") from exc
    try:
        return VerificationProgram.from_dict(_mapping(data, "verification program"))
    except VerificationSpecError as exc:
        if source_name == "<memory>":
            raise
        raise VerificationSpecError(f"{source_name}: {exc}") from exc


def load_verification(path: str | Path) -> VerificationProgram:
    """Load one verification artifact without executing authored code."""

    source = Path(path).expanduser().resolve()
    return parse_verification(source.read_bytes(), source_name=str(source))


__all__ = [
    "DifferentiabilityClass",
    "TrajectoryMetricSpec",
    "VerificationCriterionSpec",
    "VerificationError",
    "VerificationInputSpec",
    "VerificationParameterSpec",
    "VerificationProgram",
    "VerificationRuntimeError",
    "VerificationSpecError",
    "load_verification",
    "parse_verification",
]
