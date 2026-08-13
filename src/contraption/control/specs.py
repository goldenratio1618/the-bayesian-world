"""Strict, data-only ``control-1`` language records.

The control language deliberately reuses PMDL's safe expression parser and
dimensional type system.  Expressions are strings in the source document, but
are parsed and type-checked eagerly; no controller document is executable
Python.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
    Expression,
    ExpressionType,
    Literal,
    Symbol,
    Unary,
    parse_expression,
)
from ..physics.specs import BoundsSpec, FrozenDict, StrictRecord
from ..physics.units import DIMENSIONLESS, TIME, UnitError, parse_unit


class ControlSpecError(ValueError):
    """A controller document violates the ``control-1`` contract."""


class ControlValidationError(ControlSpecError):
    """A structurally valid record has invalid control semantics."""


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_DTYPES = frozenset({"real", "bool"})


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlSpecError(f"{context} must be an object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ControlSpecError(f"{context} must be an array")
    return value


def _keys(
    value: Mapping[str, Any],
    allowed: Iterable[str],
    context: str,
    required: Iterable[str] = (),
) -> None:
    allowed_set = set(allowed)
    extras = sorted(set(value) - allowed_set)
    missing = sorted(set(required) - set(value))
    if extras:
        raise ControlSpecError(f"{context} has unknown field(s): {', '.join(extras)}")
    if missing:
        raise ControlSpecError(f"{context} is missing field(s): {', '.join(missing)}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ControlSpecError(f"{context} must be a non-empty string")
    return value


def _name(value: Any, context: str) -> str:
    text = _text(value, context)
    if _NAME.fullmatch(text) is None:
        raise ControlSpecError(f"{context} must match {_NAME.pattern!r}")
    return text


def _identifier(value: Any, context: str) -> str:
    text = _text(value, context)
    if _ID.fullmatch(text) is None:
        raise ControlSpecError(f"{context} must match {_ID.pattern!r}")
    return text


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlSpecError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ControlSpecError(f"{context} must be finite")
    return result


def _dtype(value: Any, context: str) -> str:
    result = _text(value, context)
    if result not in _DTYPES:
        raise ControlSpecError(f"{context} must be one of {sorted(_DTYPES)}")
    return result


def _typed_scalar(value: Any, dtype: str, context: str) -> bool | float:
    if dtype == "bool":
        if not isinstance(value, bool):
            raise ControlSpecError(f"{context} must be boolean")
        return value
    return _finite(value, context)


def _bounds(value: Any, context: str) -> BoundsSpec:
    try:
        return BoundsSpec.from_dict({} if value is None else value)
    except Exception as exc:
        raise ControlSpecError(f"{context}: {exc}") from exc


def _dimension(unit: str, context: str):
    try:
        return parse_unit(unit).dimension
    except UnitError as exc:
        raise ControlValidationError(f"{context}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ExplicitInputSpec(StrictRecord):
    name: str
    source: str
    dtype: str = "real"
    unit: str = "1"
    default: bool | float = 0.0
    bounds: BoundsSpec = BoundsSpec()
    measurement_variance: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        _name(self.name, "input.name")
        if self.source not in {"external", "sensor"}:
            raise ControlSpecError("input.source must be 'external' or 'sensor'")
        if self.dtype not in _DTYPES:
            raise ControlSpecError(f"unsupported input dtype {self.dtype!r}")
        _typed_scalar(self.default, self.dtype, f"input {self.name}.default")
        if self.source == "sensor":
            if self.dtype == "real" and self.measurement_variance is not None and (
                not math.isfinite(self.measurement_variance)
                or self.measurement_variance <= 0.0
            ):
                raise ControlSpecError(
                    "sensor measurement_variance must be positive and finite"
                )
            if self.dtype == "bool" and self.measurement_variance is not None:
                raise ControlSpecError(
                    "boolean sensor inputs may not declare measurement_variance"
                )
        elif self.measurement_variance is not None:
            raise ControlSpecError(
                "external inputs may not declare measurement_variance"
            )
        if self.dtype == "bool":
            if self.bounds.lower is not None or self.bounds.upper is not None:
                raise ControlSpecError("boolean inputs cannot have numeric bounds")
        elif not self.bounds.contains(float(self.default)):
            raise ControlSpecError(f"input {self.name!r} default is outside bounds")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplicitInputSpec":
        data = _object(value, "input")
        names = {
            "name",
            "source",
            "dtype",
            "unit",
            "default",
            "bounds",
            "measurement_variance",
            "description",
        }
        _keys(data, names, "input", {"name", "source"})
        dtype = _dtype(data.get("dtype", "real"), "input.dtype")
        return cls(
            _name(data["name"], "input.name"),
            _text(data["source"], "input.source"),
            dtype,
            _text(data.get("unit", "1"), "input.unit"),
            _typed_scalar(data.get("default", False if dtype == "bool" else 0.0), dtype, "input.default"),
            _bounds(data.get("bounds"), "input.bounds"),
            None
            if data.get("measurement_variance") is None
            else _finite(data["measurement_variance"], "input.measurement_variance"),
            _text(data.get("description", ""), "input.description") if data.get("description", "") else "",
        )


@dataclass(frozen=True, slots=True)
class OutputSpec(StrictRecord):
    name: str
    dtype: str = "real"
    unit: str = "1"
    default: bool | float = 0.0
    bounds: BoundsSpec = BoundsSpec()
    slew_rate: float | None = None
    emergency_value: bool | float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        _name(self.name, "output.name")
        if self.dtype not in _DTYPES:
            raise ControlSpecError(f"unsupported output dtype {self.dtype!r}")
        _typed_scalar(self.default, self.dtype, f"output {self.name}.default")
        if self.emergency_value is not None:
            _typed_scalar(
                self.emergency_value,
                self.dtype,
                f"output {self.name}.emergency_value",
            )
        if self.dtype == "bool":
            if self.bounds.lower is not None or self.bounds.upper is not None:
                raise ControlSpecError("boolean outputs cannot have numeric bounds")
            if self.slew_rate is not None:
                raise ControlSpecError("boolean outputs cannot have slew_rate")
        else:
            if not self.bounds.contains(float(self.default)):
                raise ControlSpecError(f"output {self.name!r} default is outside bounds")
            if self.emergency_value is not None and not self.bounds.contains(
                float(self.emergency_value)
            ):
                raise ControlSpecError(
                    f"output {self.name!r} emergency_value is outside bounds"
                )
            if self.slew_rate is not None and self.slew_rate <= 0.0:
                raise ControlSpecError("output.slew_rate must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OutputSpec":
        data = _object(value, "output")
        names = {
            "name",
            "dtype",
            "unit",
            "default",
            "bounds",
            "slew_rate",
            "emergency_value",
            "description",
        }
        _keys(data, names, "output", {"name"})
        dtype = _dtype(data.get("dtype", "real"), "output.dtype")
        slew = data.get("slew_rate")
        emergency = data.get("emergency_value")
        return cls(
            _name(data["name"], "output.name"),
            dtype,
            _text(data.get("unit", "1"), "output.unit"),
            _typed_scalar(data.get("default", False if dtype == "bool" else 0.0), dtype, "output.default"),
            _bounds(data.get("bounds"), "output.bounds"),
            None if slew is None else _finite(slew, "output.slew_rate"),
            None if emergency is None else _typed_scalar(emergency, dtype, "output.emergency_value"),
            _text(data.get("description", ""), "output.description") if data.get("description", "") else "",
        )


@dataclass(frozen=True, slots=True)
class ParameterSpec(StrictRecord):
    name: str
    dtype: str = "real"
    unit: str = "1"
    default: bool | float = 0.0
    bounds: BoundsSpec = BoundsSpec()

    def __post_init__(self) -> None:
        _name(self.name, "parameter.name")
        if self.dtype not in _DTYPES:
            raise ControlSpecError(f"unsupported parameter dtype {self.dtype!r}")
        _typed_scalar(self.default, self.dtype, f"parameter {self.name}.default")
        if self.dtype == "bool":
            if self.bounds.lower is not None or self.bounds.upper is not None:
                raise ControlSpecError("boolean parameters cannot have bounds")
        elif not self.bounds.contains(float(self.default)):
            raise ControlSpecError(f"parameter {self.name!r} default is outside bounds")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterSpec":
        data = _object(value, "parameter")
        _keys(data, {"name", "dtype", "unit", "default", "bounds"}, "parameter", {"name"})
        dtype = _dtype(data.get("dtype", "real"), "parameter.dtype")
        return cls(
            _name(data["name"], "parameter.name"),
            dtype,
            _text(data.get("unit", "1"), "parameter.unit"),
            _typed_scalar(data.get("default", False if dtype == "bool" else 0.0), dtype, "parameter.default"),
            _bounds(data.get("bounds"), "parameter.bounds"),
        )


@dataclass(frozen=True, slots=True)
class RegisterSpec(StrictRecord):
    name: str
    dtype: str = "real"
    unit: str = "1"
    initial: bool | float = 0.0
    bounds: BoundsSpec = BoundsSpec()

    def __post_init__(self) -> None:
        _name(self.name, "register.name")
        if self.dtype not in _DTYPES:
            raise ControlSpecError(f"unsupported register dtype {self.dtype!r}")
        _typed_scalar(self.initial, self.dtype, f"register {self.name}.initial")
        if self.dtype == "bool":
            if self.bounds.lower is not None or self.bounds.upper is not None:
                raise ControlSpecError("boolean registers cannot have bounds")
        elif not self.bounds.contains(float(self.initial)):
            raise ControlSpecError(f"register {self.name!r} initial is outside bounds")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegisterSpec":
        data = _object(value, "register")
        _keys(data, {"name", "dtype", "unit", "initial", "bounds"}, "register", {"name"})
        dtype = _dtype(data.get("dtype", "real"), "register.dtype")
        return cls(
            _name(data["name"], "register.name"),
            dtype,
            _text(data.get("unit", "1"), "register.unit"),
            _typed_scalar(data.get("initial", False if dtype == "bool" else 0.0), dtype, "register.initial"),
            _bounds(data.get("bounds"), "register.bounds"),
        )


@dataclass(frozen=True, slots=True)
class DerivedSpec(StrictRecord):
    name: str
    expression: str
    dtype: str = "real"
    unit: str = "1"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DerivedSpec":
        data = _object(value, "derived")
        _keys(data, {"name", "expression", "dtype", "unit"}, "derived", {"name", "expression"})
        return cls(
            _name(data["name"], "derived.name"),
            _text(data["expression"], "derived.expression"),
            _dtype(data.get("dtype", "real"), "derived.dtype"),
            _text(data.get("unit", "1"), "derived.unit"),
        )


@dataclass(frozen=True, slots=True)
class TransitionSpec(StrictRecord):
    target: str
    guard: str
    priority: int = 0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransitionSpec":
        data = _object(value, "transition")
        _keys(data, {"target", "guard", "priority"}, "transition", {"target", "guard"})
        priority = data.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ControlSpecError("transition.priority must be an integer")
        return cls(
            _name(data["target"], "transition.target"),
            _text(data["guard"], "transition.guard"),
            priority,
        )


@dataclass(frozen=True, slots=True)
class ModeSpec(StrictRecord):
    name: str
    outputs: FrozenDict[str]
    updates: FrozenDict[str] = FrozenDict()
    transitions: tuple[TransitionSpec, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModeSpec":
        data = _object(value, "mode")
        _keys(data, {"name", "outputs", "updates", "transitions"}, "mode", {"name", "outputs"})
        outputs = _object(data["outputs"], "mode.outputs")
        updates = _object(data.get("updates", {}), "mode.updates")
        return cls(
            _name(data["name"], "mode.name"),
            FrozenDict(
                (_name(key, "mode output name"), _text(item, f"mode.outputs.{key}"))
                for key, item in outputs.items()
            ),
            FrozenDict(
                (_name(key, "mode register name"), _text(item, f"mode.updates.{key}"))
                for key, item in updates.items()
            ),
            tuple(
                TransitionSpec.from_dict(item)
                for item in _sequence(data.get("transitions", []), "mode.transitions")
            ),
        )


@dataclass(frozen=True, slots=True)
class ImplicitInputSpec(StrictRecord):
    """One latent scalar projected from a resolved plant-derived observer."""

    name: str
    unit: str
    initial_variance: float = 1.0
    process_variance_per_s: float = 0.0
    bounds: BoundsSpec = BoundsSpec()

    def __post_init__(self) -> None:
        _name(self.name, "implicit input.name")
        for field_name in (
            "initial_variance",
            "process_variance_per_s",
        ):
            _finite(getattr(self, field_name), f"implicit input {self.name}.{field_name}")
        if self.initial_variance < 0.0 or self.process_variance_per_s < 0.0:
            raise ControlSpecError(
                "implicit input initial/process variance must be non-negative"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImplicitInputSpec":
        data = _object(value, "implicit input")
        names = {
            "name",
            "unit",
            "initial_variance",
            "process_variance_per_s",
            "bounds",
        }
        _keys(data, names, "implicit input", {"name", "unit"})
        return cls(
            _name(data["name"], "implicit input.name"),
            _text(data["unit"], "implicit input.unit"),
            _finite(data.get("initial_variance", 1.0), "implicit input.initial_variance"),
            _finite(
                data.get("process_variance_per_s", 0.0),
                "implicit input.process_variance_per_s",
            ),
            _bounds(data.get("bounds"), "implicit input.bounds"),
        )


@dataclass(frozen=True, slots=True)
class ObserverSpec(StrictRecord):
    """Explicit admission and numerical settings for PMDL local linearization."""

    kind: str
    nonlinear_approximation: str
    acknowledged_open_gates: tuple[str, ...]
    sample_radius_relative: float
    maximum_sampled_remainder: float
    relative_step: float = 1e-6
    newton_tolerance: float = 1e-10
    newton_max_iterations: int = 20
    maximum_condition_number: float = 1e12

    def __post_init__(self) -> None:
        if self.kind != "local_affine":
            raise ControlSpecError("observer.kind must be 'local_affine'")
        if self.nonlinear_approximation != "approved":
            raise ControlSpecError(
                "observer.nonlinear_approximation must explicitly be 'approved'"
            )
        if len(self.acknowledged_open_gates) != len(
            set(self.acknowledged_open_gates)
        ):
            raise ControlSpecError("observer acknowledged_open_gates must be unique")
        for gate in self.acknowledged_open_gates:
            _identifier(gate, "observer acknowledged open gate")
        for field_name in (
            "sample_radius_relative",
            "maximum_sampled_remainder",
            "relative_step",
            "newton_tolerance",
            "maximum_condition_number",
        ):
            value = _finite(getattr(self, field_name), f"observer.{field_name}")
            if value <= 0.0:
                raise ControlSpecError(f"observer.{field_name} must be positive")
        if self.maximum_condition_number <= 1.0:
            raise ControlSpecError("observer.maximum_condition_number must exceed one")
        if (
            isinstance(self.newton_max_iterations, bool)
            or not isinstance(self.newton_max_iterations, int)
            or self.newton_max_iterations <= 0
        ):
            raise ControlSpecError("observer.newton_max_iterations must be a positive integer")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObserverSpec":
        data = _object(value, "observer")
        names = {
            "kind",
            "nonlinear_approximation",
            "acknowledged_open_gates",
            "sample_radius_relative",
            "maximum_sampled_remainder",
            "relative_step",
            "newton_tolerance",
            "newton_max_iterations",
            "maximum_condition_number",
        }
        _keys(
            data,
            names,
            "observer",
            {
                "kind",
                "nonlinear_approximation",
                "acknowledged_open_gates",
                "sample_radius_relative",
                "maximum_sampled_remainder",
            },
        )
        iterations = data.get("newton_max_iterations", 20)
        return cls(
            _text(data["kind"], "observer.kind"),
            _text(
                data["nonlinear_approximation"],
                "observer.nonlinear_approximation",
            ),
            tuple(
                _identifier(item, "observer acknowledged open gate")
                for item in _sequence(
                    data["acknowledged_open_gates"],
                    "observer.acknowledged_open_gates",
                )
            ),
            _finite(
                data["sample_radius_relative"],
                "observer.sample_radius_relative",
            ),
            _finite(
                data["maximum_sampled_remainder"],
                "observer.maximum_sampled_remainder",
            ),
            _finite(data.get("relative_step", 1e-6), "observer.relative_step"),
            _finite(
                data.get("newton_tolerance", 1e-10),
                "observer.newton_tolerance",
            ),
            iterations,
            _finite(
                data.get("maximum_condition_number", 1e12),
                "observer.maximum_condition_number",
            ),
        )


@dataclass(frozen=True, slots=True)
class ControlSpec(StrictRecord):
    format: str
    id: str
    name: str
    version: str
    period_s: float
    explicit_inputs: tuple[ExplicitInputSpec, ...]
    outputs: tuple[OutputSpec, ...]
    modes: tuple[ModeSpec, ...]
    initial_mode: str
    parameters: tuple[ParameterSpec, ...] = ()
    registers: tuple[RegisterSpec, ...] = ()
    implicit_inputs: tuple[ImplicitInputSpec, ...] = ()
    observer: ObserverSpec | None = None
    derived: tuple[DerivedSpec, ...] = ()
    emergency_when: str | None = None
    metadata: FrozenDict[Any] = FrozenDict()

    def __post_init__(self) -> None:
        if self.format != "control-1":
            raise ControlSpecError(f"unsupported control format {self.format!r}")
        _identifier(self.id, "control.id")
        if not self.name or not self.version:
            raise ControlSpecError("control name/version may not be empty")
        if not math.isfinite(self.period_s) or self.period_s <= 0.0:
            raise ControlSpecError("control.period_s must be positive and finite")
        object.__setattr__(self, "metadata", FrozenDict(self.metadata))
        validate_control(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ControlSpec":
        data = _object(value, "control")
        names = {
            "format",
            "id",
            "name",
            "version",
            "period_s",
            "explicit_inputs",
            "outputs",
            "parameters",
            "registers",
            "implicit_inputs",
            "observer",
            "derived",
            "modes",
            "initial_mode",
            "emergency_when",
            "metadata",
        }
        required = {"format", "id", "name", "version", "period_s", "explicit_inputs", "outputs", "modes", "initial_mode"}
        _keys(data, names, "control", required)
        return cls(
            _text(data["format"], "control.format"),
            _identifier(data["id"], "control.id"),
            _text(data["name"], "control.name"),
            _text(data["version"], "control.version"),
            _finite(data["period_s"], "control.period_s"),
            tuple(ExplicitInputSpec.from_dict(item) for item in _sequence(data["explicit_inputs"], "control.explicit_inputs")),
            tuple(OutputSpec.from_dict(item) for item in _sequence(data["outputs"], "control.outputs")),
            tuple(ModeSpec.from_dict(item) for item in _sequence(data["modes"], "control.modes")),
            _name(data["initial_mode"], "control.initial_mode"),
            tuple(ParameterSpec.from_dict(item) for item in _sequence(data.get("parameters", []), "control.parameters")),
            tuple(RegisterSpec.from_dict(item) for item in _sequence(data.get("registers", []), "control.registers")),
            tuple(ImplicitInputSpec.from_dict(item) for item in _sequence(data.get("implicit_inputs", []), "control.implicit_inputs")),
            None
            if data.get("observer") is None
            else ObserverSpec.from_dict(_object(data["observer"], "control.observer")),
            tuple(DerivedSpec.from_dict(item) for item in _sequence(data.get("derived", []), "control.derived")),
            None if data.get("emergency_when") is None else _text(data["emergency_when"], "control.emergency_when"),
            FrozenDict(_object(data.get("metadata", {}), "control.metadata")),
        )


def _unique(items: Iterable[str], context: str) -> set[str]:
    values = list(items)
    if len(values) != len(set(values)):
        raise ControlValidationError(f"duplicate {context}")
    return set(values)


def _expression_type(dtype: str, unit: str, context: str) -> ExpressionType:
    dimension = _dimension(unit, context)
    if dtype == "bool":
        if dimension != DIMENSIONLESS:
            raise ControlValidationError(f"{context}: boolean values must be dimensionless")
        return BOOLEAN
    return ExpressionType("real", dimension)


def _require_expression(
    source: str,
    symbols: Mapping[str, ExpressionType],
    expected: ExpressionType,
    context: str,
) -> None:
    try:
        expression = parse_expression(source)
        actual = expression.infer_type(symbols)
    except Exception as exc:
        raise ControlValidationError(f"{context}: {exc}") from exc
    if actual != expected:
        raise ControlValidationError(
            f"{context}: expression type is {actual.kind}/{actual.dimension.describe()}, "
            f"expected {expected.kind}/{expected.dimension.describe()}"
        )
    _admit_expression(expression, symbols, context)


_BACKEND_NUMERIC_CALLS = frozenset(
    {
        "abs",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "tanh",
        "asin",
        "acos",
        "atan",
        "atan2",
        "exp",
        "log",
        "log10",
        "min",
        "max",
        "clip",
        "where",
        "smooth_abs",
    }
)
_PIECEWISE_CALLS = frozenset({"abs", "min", "max", "clip", "where"})


def _admit_expression(
    expression: Expression,
    symbols: Mapping[str, ExpressionType],
    context: str,
) -> tuple[str, frozenset[str]]:
    """Admit a typed expression to every runtime/compiler numeric graph.

    The returned regularity is one of ``smooth``, ``piecewise_smooth``, or
    ``discrete``.  Boolean expressions are the deliberately discrete graph;
    real-valued expressions must remain backend-native and at least
    piecewise-smooth on their admitted domain.
    """

    def visit(node: Expression) -> tuple[str, set[str]]:
        node_type = node.infer_type(symbols)
        if isinstance(node, Literal):
            return ("discrete" if isinstance(node.value, bool) else "smooth", set())
        if isinstance(node, Symbol):
            return ("discrete" if node_type.kind == "boolean" else "smooth", set())
        if isinstance(node, Unary):
            regularity, boundaries = visit(node.operand)
            if node.operator == "not":
                return "discrete", boundaries
            return regularity, boundaries
        if isinstance(node, Binary):
            left, left_boundaries = visit(node.left)
            right, right_boundaries = visit(node.right)
            boundaries = left_boundaries | right_boundaries
            if node.operator in {"and", "or"}:
                return "discrete", boundaries
            if node.operator == "**":
                if (
                    not isinstance(node.right, Literal)
                    or isinstance(node.right.value, bool)
                    or int(node.right.value) != node.right.value
                    or not 0 <= int(node.right.value) <= 8
                ):
                    raise ControlValidationError(
                        f"{context}: exponentiation requires a non-negative integer "
                        "literal from 0 through 8 for common target lowering"
                    )
            regularity = (
                "piecewise_smooth"
                if "piecewise_smooth" in {left, right}
                else "smooth"
            )
            return regularity, boundaries
        if isinstance(node, Comparison):
            _, left_boundaries = visit(node.left)
            _, right_boundaries = visit(node.right)
            return "discrete", left_boundaries | right_boundaries | {"comparison"}
        if isinstance(node, Conditional):
            _, condition_boundaries = visit(node.condition)
            left, left_boundaries = visit(node.when_true)
            right, right_boundaries = visit(node.when_false)
            boundaries = (
                condition_boundaries
                | left_boundaries
                | right_boundaries
                | {"conditional"}
            )
            if node_type.kind == "boolean":
                return "discrete", boundaries
            return "piecewise_smooth", boundaries
        if isinstance(node, Call):
            child_results = [visit(argument) for argument in node.arguments]
            boundaries: set[str] = set()
            for _, child_boundaries in child_results:
                boundaries.update(child_boundaries)
            if node.function == "der":
                raise ControlValidationError(
                    f"{context}: der() is PMDL-only and cannot appear in a controller"
                )
            if node.function == "sign":
                raise ControlValidationError(
                    f"{context}: sign() is a discontinuous numeric graph operation; "
                    "use a typed boolean guard or a smooth approximation"
                )
            if node.function not in _BACKEND_NUMERIC_CALLS:
                raise ControlValidationError(
                    f"{context}: function {node.function!r} has no backend-native "
                    "differentiable controller lowering"
                )
            if node.function in _PIECEWISE_CALLS:
                boundaries.add(node.function)
                return (
                    "discrete" if node_type.kind == "boolean" else "piecewise_smooth",
                    boundaries,
                )
            if any(result == "piecewise_smooth" for result, _ in child_results):
                return "piecewise_smooth", boundaries
            return "smooth", boundaries
        raise ControlValidationError(
            f"{context}: unsupported controller expression node {type(node).__name__}"
        )

    return_regular, return_boundaries = visit(expression)
    if expression.infer_type(symbols).kind == "real" and return_regular == "discrete":
        raise ControlValidationError(
            f"{context}: a discrete boolean graph cannot be used as a real-valued path"
        )
    return return_regular, frozenset(return_boundaries)


@dataclass(frozen=True, slots=True)
class _DomainInterval:
    lower: float
    upper: float


def _domain_product(left: _DomainInterval, right: _DomainInterval) -> _DomainInterval:
    products = (
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    )
    if any(math.isnan(value) for value in products):
        return _DomainInterval(-math.inf, math.inf)
    return _DomainInterval(min(products), max(products))


def _domain_power(value: _DomainInterval, exponent: int) -> _DomainInterval:
    if exponent == 0:
        return _DomainInterval(1.0, 1.0)
    if exponent % 2:
        return _DomainInterval(value.lower**exponent, value.upper**exponent)
    lower = (
        0.0
        if value.lower <= 0.0 <= value.upper
        else min(value.lower**exponent, value.upper**exponent)
    )
    return _DomainInterval(
        lower, max(value.lower**exponent, value.upper**exponent)
    )


def _refined_domains(
    condition: Expression,
    environment: Mapping[str, _DomainInterval | None],
) -> tuple[dict[str, _DomainInterval | None], dict[str, _DomainInterval | None]]:
    when_true = dict(environment)
    when_false = dict(environment)
    if not isinstance(condition, Comparison):
        return when_true, when_false

    symbol: Symbol | None = None
    literal: Literal | None = None
    operator = condition.operator
    if isinstance(condition.left, Symbol) and isinstance(condition.right, Literal):
        symbol, literal = condition.left, condition.right
    elif isinstance(condition.right, Symbol) and isinstance(condition.left, Literal):
        symbol, literal = condition.right, condition.left
        operator = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(
            operator, operator
        )
    if (
        symbol is None
        or literal is None
        or isinstance(literal.value, bool)
        or environment.get(symbol.name) is None
    ):
        return when_true, when_false

    current = environment[symbol.name]
    assert current is not None
    boundary = float(literal.value)
    if operator in {">", ">="}:
        when_true[symbol.name] = _DomainInterval(
            max(current.lower, boundary), current.upper
        )
        when_false[symbol.name] = _DomainInterval(
            current.lower, min(current.upper, boundary)
        )
    elif operator in {"<", "<="}:
        when_true[symbol.name] = _DomainInterval(
            current.lower, min(current.upper, boundary)
        )
        when_false[symbol.name] = _DomainInterval(
            max(current.lower, boundary), current.upper
        )
    return when_true, when_false


def _validate_expression_domains(spec: ControlSpec) -> None:
    """Prove domain-sensitive operations safe for every admitted input value."""

    environment: dict[str, _DomainInterval | None] = {
        "time": _DomainInterval(0.0, math.inf),
        "time_in_mode": _DomainInterval(0.0, math.inf),
        "dt": _DomainInterval(spec.period_s, spec.period_s),
    }
    for item in spec.explicit_inputs:
        environment[f"input.{item.name}"] = (
            None
            if item.dtype == "bool"
            else _DomainInterval(
                -math.inf if item.bounds.lower is None else item.bounds.lower,
                math.inf if item.bounds.upper is None else item.bounds.upper,
            )
        )
    for item in spec.outputs:
        environment[f"output.{item.name}"] = (
            None
            if item.dtype == "bool"
            else _DomainInterval(
                -math.inf if item.bounds.lower is None else item.bounds.lower,
                math.inf if item.bounds.upper is None else item.bounds.upper,
            )
        )
    for item in spec.parameters:
        environment[f"parameter.{item.name}"] = (
            None
            if item.dtype == "bool"
            else _DomainInterval(float(item.default), float(item.default))
        )
    for item in spec.registers:
        environment[f"register.{item.name}"] = (
            None
            if item.dtype == "bool"
            else _DomainInterval(
                -math.inf if item.bounds.lower is None else item.bounds.lower,
                math.inf if item.bounds.upper is None else item.bounds.upper,
            )
        )
    for item in spec.implicit_inputs:
        environment[f"implicit.{item.name}.mean"] = _DomainInterval(
            -math.inf if item.bounds.lower is None else item.bounds.lower,
            math.inf if item.bounds.upper is None else item.bounds.upper,
        )
        environment[f"implicit.{item.name}.variance"] = _DomainInterval(0.0, math.inf)
        environment[f"implicit.{item.name}.std"] = _DomainInterval(0.0, math.inf)

    def numeric(value: _DomainInterval | None, context: str) -> _DomainInterval:
        if value is None:
            raise ControlValidationError(
                f"{context}: boolean value entered a numeric domain proof"
            )
        return value

    def evaluate(
        node: Expression,
        values: Mapping[str, _DomainInterval | None],
        context: str,
    ) -> _DomainInterval | None:
        if isinstance(node, Literal):
            return (
                None
                if isinstance(node.value, bool)
                else _DomainInterval(float(node.value), float(node.value))
            )
        if isinstance(node, Symbol):
            if node.name == "pi":
                return _DomainInterval(math.pi, math.pi)
            if node.name == "e":
                return _DomainInterval(math.e, math.e)
            return values[node.name]
        if isinstance(node, Unary):
            operand = evaluate(node.operand, values, context)
            if node.operator == "not":
                return None
            interval = numeric(operand, context)
            return (
                _DomainInterval(-interval.upper, -interval.lower)
                if node.operator == "-"
                else interval
            )
        if isinstance(node, Binary):
            left = evaluate(node.left, values, context)
            right = evaluate(node.right, values, context)
            if node.operator in {"and", "or"}:
                return None
            lhs, rhs = numeric(left, context), numeric(right, context)
            if node.operator == "+":
                return _DomainInterval(lhs.lower + rhs.lower, lhs.upper + rhs.upper)
            if node.operator == "-":
                return _DomainInterval(lhs.lower - rhs.upper, lhs.upper - rhs.lower)
            if node.operator == "*":
                return _domain_product(lhs, rhs)
            if node.operator == "/":
                if rhs.lower <= 0.0 <= rhs.upper:
                    raise ControlValidationError(
                        f"{context}: divisor interval [{rhs.lower:.17g}, "
                        f"{rhs.upper:.17g}] does not exclude zero"
                    )
                reciprocal = _DomainInterval(1.0 / rhs.upper, 1.0 / rhs.lower)
                if reciprocal.lower > reciprocal.upper:
                    reciprocal = _DomainInterval(reciprocal.upper, reciprocal.lower)
                return _domain_product(lhs, reciprocal)
            if node.operator == "**":
                assert isinstance(node.right, Literal) and not isinstance(node.right.value, bool)
                return _domain_power(lhs, int(node.right.value))
        if isinstance(node, Comparison):
            evaluate(node.left, values, context)
            evaluate(node.right, values, context)
            return None
        if isinstance(node, Conditional):
            evaluate(node.condition, values, context)
            true_values, false_values = _refined_domains(node.condition, values)
            left = evaluate(node.when_true, true_values, f"{context}.true")
            right = evaluate(node.when_false, false_values, f"{context}.false")
            if left is None or right is None:
                return None
            return _DomainInterval(
                min(left.lower, right.lower), max(left.upper, right.upper)
            )
        if isinstance(node, Call):
            if node.function == "where":
                condition = node.arguments[0]
                evaluate(condition, values, context)
                true_values, false_values = _refined_domains(condition, values)
                left = evaluate(node.arguments[1], true_values, f"{context}.where_true")
                right = evaluate(node.arguments[2], false_values, f"{context}.where_false")
                if left is None or right is None:
                    return None
                return _DomainInterval(
                    min(left.lower, right.lower), max(left.upper, right.upper)
                )
            arguments = [
                numeric(evaluate(argument, values, context), context)
                for argument in node.arguments
            ]
            if node.function == "sqrt":
                if arguments[0].lower < 0.0:
                    raise ControlValidationError(
                        f"{context}: sqrt argument interval [{arguments[0].lower:.17g}, "
                        f"{arguments[0].upper:.17g}] includes negative values"
                    )
                return _DomainInterval(
                    math.sqrt(arguments[0].lower), math.sqrt(arguments[0].upper)
                )
            if node.function in {"sin", "cos", "tanh"}:
                return _DomainInterval(-1.0, 1.0)
            if node.function == "tan":
                interval = arguments[0]
                if not math.isfinite(interval.lower) or not math.isfinite(interval.upper):
                    raise ControlValidationError(
                        f"{context}: tan requires finite argument bounds excluding its poles"
                    )
                first_pole = math.ceil((interval.lower - math.pi / 2.0) / math.pi)
                last_pole = math.floor((interval.upper - math.pi / 2.0) / math.pi)
                if first_pole <= last_pole:
                    raise ControlValidationError(
                        f"{context}: tan argument interval crosses a pole"
                    )
                endpoints = (math.tan(interval.lower), math.tan(interval.upper))
                return _DomainInterval(min(endpoints), max(endpoints))
            if node.function in {"asin", "acos"}:
                interval = arguments[0]
                if interval.lower < -1.0 or interval.upper > 1.0:
                    raise ControlValidationError(
                        f"{context}: {node.function} argument interval must stay within [-1, 1]"
                    )
                endpoints = (
                    getattr(math, node.function)(interval.lower),
                    getattr(math, node.function)(interval.upper),
                )
                return _DomainInterval(min(endpoints), max(endpoints))
            if node.function == "atan":
                return _DomainInterval(
                    math.atan(arguments[0].lower), math.atan(arguments[0].upper)
                )
            if node.function == "atan2":
                if (
                    arguments[0].lower <= 0.0 <= arguments[0].upper
                    and arguments[1].lower <= 0.0 <= arguments[1].upper
                ):
                    raise ControlValidationError(
                        f"{context}: atan2 arguments admit the undefined point (0, 0)"
                    )
                return _DomainInterval(-math.pi, math.pi)
            if node.function == "exp":
                try:
                    return _DomainInterval(
                        math.exp(arguments[0].lower), math.exp(arguments[0].upper)
                    )
                except OverflowError as exc:
                    raise ControlValidationError(
                        f"{context}: exp overflows over the admitted argument interval"
                    ) from exc
            if node.function in {"log", "log10"}:
                if arguments[0].lower <= 0.0:
                    raise ControlValidationError(
                        f"{context}: {node.function} argument interval must be strictly positive"
                    )
                function = math.log if node.function == "log" else math.log10
                return _DomainInterval(
                    function(arguments[0].lower), function(arguments[0].upper)
                )
            if node.function == "abs":
                maximum = max(abs(arguments[0].lower), abs(arguments[0].upper))
                minimum = (
                    0.0
                    if arguments[0].lower <= 0.0 <= arguments[0].upper
                    else min(abs(arguments[0].lower), abs(arguments[0].upper))
                )
                return _DomainInterval(minimum, maximum)
            if node.function == "min":
                return _DomainInterval(
                    min(arguments[0].lower, arguments[1].lower),
                    min(arguments[0].upper, arguments[1].upper),
                )
            if node.function == "max":
                return _DomainInterval(
                    max(arguments[0].lower, arguments[1].lower),
                    max(arguments[0].upper, arguments[1].upper),
                )
            if node.function == "clip":
                if arguments[1].upper > arguments[2].lower:
                    raise ControlValidationError(
                        f"{context}: clip lower/upper intervals are not provably ordered"
                    )
                return _DomainInterval(arguments[1].lower, arguments[2].upper)
            if node.function == "smooth_abs":
                square = _domain_power(arguments[0], 2)
                epsilon = (
                    arguments[1]
                    if len(arguments) == 2
                    else _DomainInterval(1e-12, 1e-12)
                )
                epsilon_square = _domain_power(epsilon, 2)
                return _DomainInterval(
                    math.sqrt(max(0.0, square.lower + epsilon_square.lower)),
                    math.sqrt(square.upper + epsilon_square.upper),
                )
        raise ControlValidationError(
            f"{context}: could not prove expression domains for {type(node).__name__}"
        )

    for item in spec.derived:
        environment[f"derived.{item.name}"] = evaluate(
            parse_expression(item.expression), environment, f"derived {item.name}"
        )
    if spec.emergency_when is not None:
        evaluate(parse_expression(spec.emergency_when), environment, "emergency_when")
    for mode in spec.modes:
        for name, source in mode.outputs.items():
            evaluate(parse_expression(source), environment, f"mode {mode.name}.outputs.{name}")
        for name, source in mode.updates.items():
            evaluate(parse_expression(source), environment, f"mode {mode.name}.updates.{name}")
        for index, transition in enumerate(mode.transitions):
            evaluate(
                parse_expression(transition.guard),
                environment,
                f"mode {mode.name}.transitions[{index}].guard",
            )


def validate_control(spec: ControlSpec) -> None:
    """Eagerly validate names, bindings, units, and every expression."""

    _unique((item.name for item in spec.explicit_inputs), "explicit input name")
    output_names = _unique((item.name for item in spec.outputs), "output name")
    parameter_names = _unique((item.name for item in spec.parameters), "parameter name")
    register_names = _unique((item.name for item in spec.registers), "register name")
    implicit_input_names = _unique(
        (item.name for item in spec.implicit_inputs), "implicit input name"
    )
    derived_names = _unique((item.name for item in spec.derived), "derived name")
    mode_names = _unique((item.name for item in spec.modes), "mode name")
    if bool(spec.implicit_inputs) != (spec.observer is not None):
        raise ControlValidationError(
            "observer must be declared exactly when implicit_inputs are present"
        )
    if spec.observer is not None:
        missing_variances = [
            item.name
            for item in spec.explicit_inputs
            if item.source == "sensor"
            and item.dtype == "real"
            and item.measurement_variance is None
        ]
        if missing_variances:
            raise ControlValidationError(
                "real sensor inputs used by the affine observer require "
                f"measurement_variance: {missing_variances}"
            )
    if not spec.outputs or not spec.modes:
        raise ControlValidationError("a controller needs outputs and modes")
    if spec.initial_mode not in mode_names:
        raise ControlValidationError("initial_mode does not name a declared mode")
    for group_a, group_b, label in (
        (parameter_names, register_names, "parameters/registers"),
        (implicit_input_names, derived_names, "implicit inputs/derived values"),
    ):
        overlap = sorted(group_a & group_b)
        if overlap:
            raise ControlValidationError(f"{label} overlap: {overlap}")

    symbols: dict[str, ExpressionType] = {
        "time": ExpressionType("real", TIME),
        "time_in_mode": ExpressionType("real", TIME),
        "dt": ExpressionType("real", TIME),
    }
    for item in spec.explicit_inputs:
        symbols[f"input.{item.name}"] = _expression_type(item.dtype, item.unit, f"input {item.name}.unit")
    for item in spec.outputs:
        symbols[f"output.{item.name}"] = _expression_type(item.dtype, item.unit, f"output {item.name}.unit")
    for item in spec.parameters:
        symbols[f"parameter.{item.name}"] = _expression_type(item.dtype, item.unit, f"parameter {item.name}.unit")
    for item in spec.registers:
        symbols[f"register.{item.name}"] = _expression_type(item.dtype, item.unit, f"register {item.name}.unit")
    for item in spec.implicit_inputs:
        state_dimension = _dimension(item.unit, f"implicit input {item.name}.unit")
        symbols[f"implicit.{item.name}.mean"] = ExpressionType("real", state_dimension)
        symbols[f"implicit.{item.name}.variance"] = ExpressionType(
            "real", state_dimension * state_dimension
        )
        symbols[f"implicit.{item.name}.std"] = ExpressionType("real", state_dimension)

    for item in spec.derived:
        expected = _expression_type(item.dtype, item.unit, f"derived {item.name}.unit")
        _require_expression(item.expression, symbols, expected, f"derived {item.name}.expression")
        symbols[f"derived.{item.name}"] = expected

    if spec.emergency_when is not None:
        _require_expression(spec.emergency_when, symbols, BOOLEAN, "control.emergency_when")
        missing_emergency_values = [
            item.name for item in spec.outputs if item.emergency_value is None
        ]
        if missing_emergency_values:
            raise ControlValidationError(
                "emergency_when requires emergency_value on every output; "
                f"missing={missing_emergency_values}"
            )

    output_types = {
        item.name: _expression_type(item.dtype, item.unit, f"output {item.name}.unit")
        for item in spec.outputs
    }
    register_types = {
        item.name: _expression_type(item.dtype, item.unit, f"register {item.name}.unit")
        for item in spec.registers
    }
    for mode in spec.modes:
        if set(mode.outputs) != output_names:
            raise ControlValidationError(
                f"mode {mode.name!r} output coverage mismatch; "
                f"missing={sorted(output_names - set(mode.outputs))}, "
                f"unknown={sorted(set(mode.outputs) - output_names)}"
            )
        unknown_updates = sorted(set(mode.updates) - register_names)
        if unknown_updates:
            raise ControlValidationError(
                f"mode {mode.name!r} updates unknown registers {unknown_updates}"
            )
        for name, expression in mode.outputs.items():
            _require_expression(expression, symbols, output_types[name], f"mode {mode.name}.outputs.{name}")
        for name, expression in mode.updates.items():
            _require_expression(expression, symbols, register_types[name], f"mode {mode.name}.updates.{name}")
        priorities: set[int] = set()
        for index, transition in enumerate(mode.transitions):
            if transition.target not in mode_names:
                raise ControlValidationError(
                    f"mode {mode.name!r} transitions to missing mode {transition.target!r}"
                )
            if transition.priority in priorities:
                raise ControlValidationError(
                    f"mode {mode.name!r} has ambiguous priority {transition.priority}"
                )
            priorities.add(transition.priority)
            _require_expression(
                transition.guard,
                symbols,
                BOOLEAN,
                f"mode {mode.name}.transitions[{index}].guard",
            )
    _validate_expression_domains(spec)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ControlSpecError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_control(
    source: str | bytes | Mapping[str, Any], *, source_name: str = "<memory>"
) -> ControlSpec:
    if isinstance(source, Mapping):
        data = source
    else:
        try:
            data = json.loads(source, object_pairs_hook=_reject_duplicate_pairs)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ControlSpecError(f"invalid control JSON in {source_name}: {exc}") from exc
    try:
        return ControlSpec.from_dict(_object(data, f"control root in {source_name}"))
    except ControlSpecError:
        raise
    except Exception as exc:
        raise ControlSpecError(f"invalid control document in {source_name}: {exc}") from exc


def load_control(path: str | Path) -> ControlSpec:
    source_path = Path(path)
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ControlSpecError(f"could not read {source_path}: {exc}") from exc
    return parse_control(source, source_name=str(source_path))


def dump_control(spec: ControlSpec, path: str | Path | None = None) -> str:
    source = spec.to_json(indent=2) + "\n"
    if path is not None:
        Path(path).write_text(source, encoding="utf-8")
    return source


def control_digest(spec: ControlSpec) -> str:
    payload = spec.to_json().encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "ControlSpec",
    "ControlSpecError",
    "ControlValidationError",
    "DerivedSpec",
    "ExplicitInputSpec",
    "ImplicitInputSpec",
    "ModeSpec",
    "ObserverSpec",
    "OutputSpec",
    "ParameterSpec",
    "RegisterSpec",
    "TransitionSpec",
    "control_digest",
    "dump_control",
    "load_control",
    "parse_control",
    "validate_control",
]
