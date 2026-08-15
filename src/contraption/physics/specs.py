"""Immutable, deterministic data contracts for models and contraptions.

All ``from_dict`` methods reject unknown keys.  The records contain data only;
model equations remain strings until :mod:`contraption.physics.dsl` parses them into
an allow-listed expression tree.  This separation is the primary trust
boundary for generated or imported component artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import MISSING, dataclass, fields, is_dataclass
import json
import math
from pathlib import Path
import re
import types
from typing import Any, ClassVar, Generic, TypeVar, Union, get_args, get_origin, get_type_hints


class SpecError(ValueError):
    """Raised when serialized specification data violates its schema."""


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_T = TypeVar("_T")


def _identifier(value: Any, context: str, *, symbol: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(f"{context} must be a non-empty string")
    pattern = _SYMBOL if symbol else _IDENTIFIER
    if pattern.fullmatch(value) is None:
        raise SpecError(f"{context} is not a valid identifier: {value!r}")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise SpecError(f"{context} must be a string")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise SpecError(f"{context} must be a boolean")
    return value


def _number(value: Any, context: str, *, finite: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{context} must be numeric")
    result = float(value)
    if finite and not math.isfinite(result):
        raise SpecError(f"{context} must be finite")
    return result


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SpecError(f"{context} must be an object with string keys")
    return value


def _sequence(value: Any, context: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise SpecError(f"{context} must be an array")
    return value


def _keys(data: Mapping[str, Any], allowed: Iterable[str], context: str, required: Iterable[str] = ()) -> None:
    allowed_set, required_set = set(allowed), set(required)
    unknown = sorted(set(data) - allowed_set)
    missing = sorted(required_set - set(data))
    if unknown:
        raise SpecError(f"unknown {context} key(s): {', '.join(unknown)}")
    if missing:
        raise SpecError(f"missing {context} key(s): {', '.join(missing)}")


def _strings(value: Any, context: str) -> tuple[str, ...]:
    return tuple(_string(item, f"{context}[{index}]") for index, item in enumerate(_sequence(value, context)))


def _freeze(value: Any, context: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SpecError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise SpecError(f"{context} has a non-string object key")
        return FrozenDict((key, _freeze(item, f"{context}.{key}")) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{context}[{index}]") for index, item in enumerate(value))
    if isinstance(value, StrictRecord):
        return value
    raise SpecError(f"{context} contains unsupported type {type(value).__name__}")


class FrozenDict(Mapping[str, _T], Generic[_T]):
    """A small insertion-independent immutable mapping."""

    __slots__ = ("_items", "_dict")

    def __init__(self, values: Mapping[str, _T] | Iterable[tuple[str, _T]] = ()) -> None:
        source = dict(values)
        if any(not isinstance(key, str) for key in source):
            raise SpecError("FrozenDict keys must be strings")
        self._items = tuple(sorted(source.items()))
        self._dict = dict(self._items)

    def __getitem__(self, key: str) -> _T:
        return self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._items)!r})"

    def __hash__(self) -> int:
        return hash(tuple((key, _hashable(value)) for key, value in self._items))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self._items) == dict(other)


def _hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value


def _to_data(value: Any) -> Any:
    if isinstance(value, StrictRecord):
        return {field.name: _to_data(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _to_data(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_to_data(item) for item in value]
    return value


class StrictRecord:
    """Mixin for canonical serialization; subclasses implement ``from_dict``."""

    SCHEMA: ClassVar[str] = "contraption.phase1"

    def to_dict(self) -> dict[str, Any]:
        return _to_data(self)

    def to_json(self, *, indent: int | None = None) -> str:
        separators = None if indent is not None else (",", ":")
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=separators, indent=indent,
            ensure_ascii=False, allow_nan=False,
        )

    @classmethod
    def from_json(cls, source: str) -> Any:
        try:
            data = json.loads(source)
        except json.JSONDecodeError as exc:
            raise SpecError(f"invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}") from exc
        if not isinstance(data, dict):
            raise SpecError(f"{cls.__name__} JSON must contain an object")
        return cls.from_dict(data)  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class BoundsSpec(StrictRecord):
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.lower is not None and not math.isfinite(self.lower):
            raise SpecError("bounds.lower must be finite or null")
        if self.upper is not None and not math.isfinite(self.upper):
            raise SpecError("bounds.upper must be finite or null")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise SpecError("bounds.lower may not exceed bounds.upper")

    def contains(self, value: float) -> bool:
        return (self.lower is None or value >= self.lower) and (self.upper is None or value <= self.upper)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | list[Any] | tuple[Any, ...]) -> "BoundsSpec":
        if isinstance(data, (list, tuple)):
            if len(data) != 2:
                raise SpecError("bounds array must have [lower, upper]")
            lower, upper = data
        else:
            data = _object(data, "bounds")
            _keys(data, ("lower", "upper"), "bounds")
            lower, upper = data.get("lower"), data.get("upper")
        return cls(
            None if lower is None else _number(lower, "bounds.lower"),
            None if upper is None else _number(upper, "bounds.upper"),
        )


@dataclass(frozen=True, slots=True)
class UncertaintySpec(StrictRecord):
    distribution: str = "fixed"
    parameters: FrozenDict[Any] = FrozenDict()
    correlation_group: str | None = None

    def __post_init__(self) -> None:
        allowed = {"fixed", "normal", "lognormal", "uniform", "triangular", "empirical"}
        if self.distribution not in allowed:
            raise SpecError(f"unsupported uncertainty distribution {self.distribution!r}")
        object.__setattr__(self, "parameters", _freeze(self.parameters, "uncertainty.parameters"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UncertaintySpec":
        data = _object(data, "uncertainty")
        _keys(data, ("distribution", "parameters", "correlation_group"), "uncertainty")
        return cls(
            _string(data.get("distribution", "fixed"), "uncertainty.distribution"),
            _freeze(_object(data.get("parameters", {}), "uncertainty.parameters")),
            None if data.get("correlation_group") is None else _identifier(data["correlation_group"], "uncertainty.correlation_group"),
        )


@dataclass(frozen=True, slots=True)
class StateSpec(StrictRecord):
    name: str
    unit: str = "1"
    initial: float = 0.0
    derivative: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        _identifier(self.name, "state.name", symbol=True)
        if self.derivative is not None:
            _identifier(self.derivative, "state.derivative", symbol=True)
        _number(self.initial, f"state {self.name}.initial")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateSpec":
        data = _object(data, "state")
        _keys(data, ("name", "unit", "initial", "derivative", "description"), "state", ("name",))
        return cls(
            _identifier(data["name"], "state.name", symbol=True),
            _string(data.get("unit", "1"), "state.unit"),
            _number(data.get("initial", 0.0), "state.initial"),
            None if data.get("derivative") is None else _identifier(data["derivative"], "state.derivative", symbol=True),
            _string(data.get("description", ""), "state.description"),
        )


@dataclass(frozen=True, slots=True)
class AlgebraicSpec(StrictRecord):
    name: str
    unit: str = "1"
    initial: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        _identifier(self.name, "algebraic.name", symbol=True)
        _number(self.initial, f"algebraic {self.name}.initial")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AlgebraicSpec":
        data = _object(data, "algebraic")
        _keys(data, ("name", "unit", "initial", "description"), "algebraic", ("name",))
        return cls(
            _identifier(data["name"], "algebraic.name", symbol=True),
            _string(data.get("unit", "1"), "algebraic.unit"),
            _number(data.get("initial", 0.0), "algebraic.initial"),
            _string(data.get("description", ""), "algebraic.description"),
        )


@dataclass(frozen=True, slots=True)
class ParameterSpec(StrictRecord):
    name: str
    unit: str = "1"
    default: float = 0.0
    bounds: BoundsSpec = BoundsSpec()
    uncertainty: UncertaintySpec = UncertaintySpec()
    learnable: bool = True
    description: str = ""

    @property
    def value(self) -> float:
        return self.default

    def __post_init__(self) -> None:
        _identifier(self.name, "parameter.name", symbol=True)
        _number(self.default, f"parameter {self.name}.default")
        if not self.bounds.contains(self.default):
            raise SpecError(f"parameter {self.name!r} default is outside its bounds")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterSpec":
        data = _object(data, "parameter")
        _keys(data, ("name", "unit", "default", "bounds", "uncertainty", "learnable", "description"), "parameter", ("name",))
        return cls(
            _identifier(data["name"], "parameter.name", symbol=True),
            _string(data.get("unit", "1"), "parameter.unit"),
            _number(data.get("default", 0.0), "parameter.default"),
            BoundsSpec.from_dict(data.get("bounds", {})),
            UncertaintySpec.from_dict(data.get("uncertainty", {})),
            _boolean(data.get("learnable", True), "parameter.learnable"),
            _string(data.get("description", ""), "parameter.description"),
        )


@dataclass(frozen=True, slots=True)
class PowerPortSpec(StrictRecord):
    name: str
    domain: str
    effort: str
    flow: str
    effort_unit: str
    flow_unit: str
    orientation: str = "into_component"
    frame: str = "body"
    reference: str = "declared"
    description: str = ""

    def __post_init__(self) -> None:
        _identifier(self.name, "power_port.name", symbol=True)
        _identifier(self.domain, "power_port.domain")
        _identifier(self.effort, "power_port.effort", symbol=True)
        _identifier(self.flow, "power_port.flow", symbol=True)
        if self.orientation not in {"into_component", "out_of_component", "bidirectional"}:
            raise SpecError(f"invalid orientation {self.orientation!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PowerPortSpec":
        data = _object(data, "power_port")
        names = ("name", "domain", "effort", "flow", "effort_unit", "flow_unit", "orientation", "frame", "reference", "description")
        _keys(data, names, "power_port", names[:6])
        return cls(*(
            _string(data[name], f"power_port.{name}") for name in names[:6]
        ), _string(data.get("orientation", "into_component"), "power_port.orientation"),
            _string(data.get("frame", "body"), "power_port.frame"),
            _string(data.get("reference", "declared"), "power_port.reference"),
            _string(data.get("description", ""), "power_port.description"))


@dataclass(frozen=True, slots=True)
class SignalPortSpec(StrictRecord):
    name: str
    direction: str
    unit: str = "1"
    dtype: str = "float64"
    shape: tuple[int, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        _identifier(self.name, "signal_port.name", symbol=True)
        if self.direction not in {"input", "output"}:
            raise SpecError("signal_port.direction must be 'input' or 'output'")
        if self.dtype not in {"float32", "float64", "int32", "bool"}:
            raise SpecError(f"unsupported signal dtype {self.dtype!r}")
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in self.shape):
            raise SpecError("signal_port.shape entries must be positive integers")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalPortSpec":
        data = _object(data, "signal_port")
        _keys(data, ("name", "direction", "unit", "dtype", "shape", "description"), "signal_port", ("name", "direction"))
        shape = tuple(_sequence(data.get("shape", []), "signal_port.shape"))
        return cls(
            _identifier(data["name"], "signal_port.name", symbol=True),
            _string(data["direction"], "signal_port.direction"),
            _string(data.get("unit", "1"), "signal_port.unit"),
            _string(data.get("dtype", "float64"), "signal_port.dtype"),
            shape,
            _string(data.get("description", ""), "signal_port.description"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactPortSpec(StrictRecord):
    """A typed non-scalar artifact stream kept outside the equation symbol table.

    Artifact ports carry hash-bound records such as images, depth maps, point
    clouds, and reconstruction deltas.  ``controller_stream`` is a bounded wire
    contract for future controller/FPGA bridges; this release does not implement
    such a bridge.
    """

    name: str
    direction: str
    artifact_type: str
    timing: str = "event"
    transport: str = "content_addressed"
    sample_period_s: float | None = None
    max_payload_bytes: int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        _identifier(self.name, "artifact_port.name", symbol=True)
        if self.direction not in {"input", "output"}:
            raise SpecError("artifact_port.direction must be 'input' or 'output'")
        if re.fullmatch(r"[a-z][a-z0-9.-]*/[a-z][a-z0-9.-]*@[1-9][0-9]*", self.artifact_type) is None:
            raise SpecError("artifact_port.artifact_type must be a versioned type such as 'contraption/optical-observation@1'")
        if self.timing not in {"sampled", "event"}:
            raise SpecError("artifact_port.timing must be 'sampled' or 'event'")
        if self.transport not in {"in_process", "content_addressed", "shared_memory", "network", "controller_stream"}:
            raise SpecError(f"unsupported artifact transport {self.transport!r}")
        if self.sample_period_s is not None and _number(self.sample_period_s, "artifact_port.sample_period_s") <= 0.0:
            raise SpecError("artifact_port.sample_period_s must be positive")
        if self.timing == "sampled" and self.sample_period_s is None:
            raise SpecError("sampled artifact ports require sample_period_s")
        if self.max_payload_bytes is not None and (isinstance(self.max_payload_bytes, bool) or not isinstance(self.max_payload_bytes, int) or self.max_payload_bytes <= 0):
            raise SpecError("artifact_port.max_payload_bytes must be a positive integer")
        if self.transport == "controller_stream" and self.max_payload_bytes is None:
            raise SpecError("controller_stream artifact ports require max_payload_bytes")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactPortSpec":
        data = _object(data, "artifact_port")
        names = ("name", "direction", "artifact_type", "timing", "transport", "sample_period_s", "max_payload_bytes", "description")
        _keys(data, names, "artifact_port", names[:3])
        return cls(
            _identifier(data["name"], "artifact_port.name", symbol=True), _string(data["direction"], "artifact_port.direction"),
            _string(data["artifact_type"], "artifact_port.artifact_type"), _string(data.get("timing", "event"), "artifact_port.timing"),
            _string(data.get("transport", "content_addressed"), "artifact_port.transport"),
            None if data.get("sample_period_s") is None else _number(data["sample_period_s"], "artifact_port.sample_period_s"),
            data.get("max_payload_bytes"), _string(data.get("description", ""), "artifact_port.description"),
        )


@dataclass(frozen=True, slots=True)
class RelationSpec(StrictRecord):
    name: str
    expression: str
    description: str = ""

    def __post_init__(self) -> None:
        _identifier(self.name, "relation.name", symbol=True)
        if not self.expression.strip():
            raise SpecError(f"relation {self.name!r} expression may not be empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationSpec":
        data = _object(data, "relation")
        _keys(data, ("name", "expression", "description"), "relation", ("name", "expression"))
        return cls(_identifier(data["name"], "relation.name", symbol=True),
                   _string(data["expression"], "relation.expression"),
                   _string(data.get("description", ""), "relation.description"))


@dataclass(frozen=True, slots=True)
class NamedExpressionSpec(StrictRecord):
    name: str
    expression: str
    unit: str
    nonnegative: bool = False
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], context: str = "expression") -> "NamedExpressionSpec":
        data = _object(data, context)
        _keys(data, ("name", "expression", "unit", "nonnegative", "description"), context, ("name", "expression", "unit"))
        return cls(_identifier(data["name"], f"{context}.name", symbol=True),
                   _string(data["expression"], f"{context}.expression"),
                   _string(data["unit"], f"{context}.unit"),
                   _boolean(data.get("nonnegative", False), f"{context}.nonnegative"),
                   _string(data.get("description", ""), f"{context}.description"))


EnergySpec = NamedExpressionSpec
DissipationSpec = NamedExpressionSpec
SourceSpec = NamedExpressionSpec


@dataclass(frozen=True, slots=True)
class ProcessNoiseChannelSpec(StrictRecord):
    """One dimensionless stochastic driver for accepted-step state increments."""

    name: str
    distribution: str
    description: str = ""

    def __post_init__(self) -> None:
        _identifier(self.name, "process_noise.channel.name", symbol=True)
        if self.distribution != "standard_normal":
            raise SpecError(
                "process_noise.channel.distribution must be 'standard_normal'"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcessNoiseChannelSpec":
        data = _object(data, "process_noise.channel")
        _keys(
            data,
            ("name", "distribution", "description"),
            "process_noise.channel",
            ("name", "distribution"),
        )
        return cls(
            _identifier(data["name"], "process_noise.channel.name", symbol=True),
            _string(data["distribution"], "process_noise.channel.distribution"),
            _string(data.get("description", ""), "process_noise.channel.description"),
        )


@dataclass(frozen=True, slots=True)
class ProcessNoiseIncrementSpec(StrictRecord):
    """A backend-native expression added to one differential state per step."""

    target: str
    expression: str
    description: str = ""

    def __post_init__(self) -> None:
        _identifier(self.target, "process_noise.increment.target", symbol=True)
        if not self.expression.strip():
            raise SpecError("process_noise.increment.expression may not be empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcessNoiseIncrementSpec":
        data = _object(data, "process_noise.increment")
        _keys(
            data,
            ("target", "expression", "description"),
            "process_noise.increment",
            ("target", "expression"),
        )
        return cls(
            _identifier(data["target"], "process_noise.increment.target", symbol=True),
            _string(data["expression"], "process_noise.increment.expression"),
            _string(data.get("description", ""), "process_noise.increment.description"),
        )


@dataclass(frozen=True, slots=True)
class ProcessNoiseSpec(StrictRecord):
    """Declarative stochastic increments with an explicit replay contract.

    The only admitted policy derives one RNG stream from ``simulate(seed=...)``.
    Draws are reproducible for an unchanged PMDL closure, time grid, sample
    count, numerical backend, device, and dtype.  Increment expressions receive
    the actual accepted-step ``dt`` and are applied after the deterministic
    integration step.
    """

    seed_policy: str = "simulation_seed"
    reproducibility: str = "same_backend_device"
    application: str = "accepted_step_increment"
    channels: tuple[ProcessNoiseChannelSpec, ...] = ()
    increments: tuple[ProcessNoiseIncrementSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.seed_policy != "simulation_seed":
            raise SpecError("process_noise.seed_policy must be 'simulation_seed'")
        if self.reproducibility != "same_backend_device":
            raise SpecError(
                "process_noise.reproducibility must be 'same_backend_device'"
            )
        if self.application != "accepted_step_increment":
            raise SpecError(
                "process_noise.application must be 'accepted_step_increment'"
            )
        if bool(self.channels) != bool(self.increments):
            raise SpecError(
                "process_noise.channels and process_noise.increments must either "
                "both be empty or both be non-empty"
            )

    @property
    def enabled(self) -> bool:
        return bool(self.channels)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcessNoiseSpec":
        data = _object(data, "process_noise")
        names = (
            "seed_policy",
            "reproducibility",
            "application",
            "channels",
            "increments",
        )
        # An explicit empty object is canonical no-noise, identical to omission.
        # A non-empty block must spell out the complete replay contract.
        _keys(data, names, "process_noise", () if not data else names)
        if not data:
            return cls()
        return cls(
            _string(data["seed_policy"], "process_noise.seed_policy"),
            _string(data["reproducibility"], "process_noise.reproducibility"),
            _string(data["application"], "process_noise.application"),
            tuple(
                ProcessNoiseChannelSpec.from_dict(item)
                for item in _sequence(data["channels"], "process_noise.channels")
            ),
            tuple(
                ProcessNoiseIncrementSpec.from_dict(item)
                for item in _sequence(data["increments"], "process_noise.increments")
            ),
        )


@dataclass(frozen=True, slots=True)
class TransitionSpec(StrictRecord):
    target: str
    guard: str
    resets: FrozenDict[str] = FrozenDict()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransitionSpec":
        data = _object(data, "transition")
        _keys(data, ("target", "guard", "resets"), "transition", ("target", "guard"))
        resets = _object(data.get("resets", {}), "transition.resets")
        return cls(_identifier(data["target"], "transition.target", symbol=True),
                   _string(data["guard"], "transition.guard"),
                   FrozenDict((name, _string(expr, f"transition.resets.{name}")) for name, expr in resets.items()))


@dataclass(frozen=True, slots=True)
class ModeSpec(StrictRecord):
    name: str
    active_relations: tuple[str, ...] = ()
    transitions: tuple[TransitionSpec, ...] = ()
    initial: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModeSpec":
        data = _object(data, "mode")
        _keys(data, ("name", "active_relations", "transitions", "initial"), "mode", ("name",))
        return cls(_identifier(data["name"], "mode.name", symbol=True),
                   _strings(data.get("active_relations", []), "mode.active_relations"),
                   tuple(TransitionSpec.from_dict(item) for item in _sequence(data.get("transitions", []), "mode.transitions")),
                   _boolean(data.get("initial", False), "mode.initial"))


@dataclass(frozen=True, slots=True)
class InitializationSpec(StrictRecord):
    strategy: str = "consistent"
    constraints: tuple[RelationSpec, ...] = ()
    required: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InitializationSpec":
        data = _object(data, "initialization")
        _keys(data, ("strategy", "constraints", "required"), "initialization")
        return cls(_string(data.get("strategy", "consistent"), "initialization.strategy"),
                   tuple(RelationSpec.from_dict(item) for item in _sequence(data.get("constraints", []), "initialization.constraints")),
                   _strings(data.get("required", []), "initialization.required"))


@dataclass(frozen=True, slots=True)
class ValiditySpec(StrictRecord):
    ranges: FrozenDict[BoundsSpec] = FrozenDict()
    assumptions: tuple[str, ...] = ()
    max_timestep: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValiditySpec":
        data = _object(data, "validity")
        _keys(data, ("ranges", "assumptions", "max_timestep"), "validity")
        ranges = _object(data.get("ranges", {}), "validity.ranges")
        max_step = data.get("max_timestep")
        if max_step is not None and _number(max_step, "validity.max_timestep") <= 0:
            raise SpecError("validity.max_timestep must be positive")
        return cls(FrozenDict((name, BoundsSpec.from_dict(bounds)) for name, bounds in ranges.items()),
                   _strings(data.get("assumptions", []), "validity.assumptions"),
                   None if max_step is None else float(max_step))


@dataclass(frozen=True, slots=True)
class FidelitySpec(StrictRecord):
    name: str
    description: str = ""
    active_relations: tuple[str, ...] = ()
    parameter_overrides: FrozenDict[Any] = FrozenDict()
    approximation_error: str = "unspecified"
    relative_cost: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FidelitySpec":
        data = _object(data, "fidelity")
        names = ("name", "description", "active_relations", "parameter_overrides", "approximation_error", "relative_cost")
        _keys(data, names, "fidelity", ("name",))
        cost = _number(data.get("relative_cost", 1.0), "fidelity.relative_cost")
        if cost <= 0:
            raise SpecError("fidelity.relative_cost must be positive")
        return cls(_identifier(data["name"], "fidelity.name", symbol=True),
                   _string(data.get("description", ""), "fidelity.description"),
                   _strings(data.get("active_relations", []), "fidelity.active_relations"),
                   _freeze(_object(data.get("parameter_overrides", {}), "fidelity.parameter_overrides")),
                   _string(data.get("approximation_error", "unspecified"), "fidelity.approximation_error"), cost)


@dataclass(frozen=True, slots=True)
class PropertyTestSpec(StrictRecord):
    name: str
    kind: str
    expression: str
    expected: bool = True
    samples: int = 32
    tolerance: float = 1e-9
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PropertyTestSpec":
        data = _object(data, "property")
        names = ("name", "kind", "expression", "expected", "samples", "tolerance", "description")
        _keys(data, names, "property", ("name", "kind", "expression"))
        samples = data.get("samples", 32)
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise SpecError("property.samples must be a positive integer")
        tolerance = _number(data.get("tolerance", 1e-9), "property.tolerance")
        if tolerance < 0:
            raise SpecError("property.tolerance must be nonnegative")
        return cls(_identifier(data["name"], "property.name", symbol=True),
                   _string(data["kind"], "property.kind"), _string(data["expression"], "property.expression"),
                   _boolean(data.get("expected", True), "property.expected"), samples, tolerance,
                   _string(data.get("description", ""), "property.description"))


@dataclass(frozen=True, slots=True)
class EvidenceSpec(StrictRecord):
    kind: str
    reference: str
    summary: str = ""
    date: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceSpec":
        data = _object(data, "evidence")
        _keys(data, ("kind", "reference", "summary", "date"), "evidence", ("kind", "reference"))
        return cls(_string(data["kind"], "evidence.kind"), _string(data["reference"], "evidence.reference"),
                   _string(data.get("summary", ""), "evidence.summary"),
                   None if data.get("date") is None else _string(data["date"], "evidence.date"))


@dataclass(frozen=True, slots=True)
class TrustSpec(StrictRecord):
    structural: str = "unverified"
    physical: str = "unverified"
    numerical: str = "unverified"
    empirical: str = "unverified"
    evidence: tuple[EvidenceSpec, ...] = ()

    LEVELS: ClassVar[set[str]] = {"unverified", "reviewed", "tested", "validated", "certified"}

    def __post_init__(self) -> None:
        for field_name in ("structural", "physical", "numerical", "empirical"):
            if getattr(self, field_name) not in self.LEVELS:
                raise SpecError(f"trust.{field_name} has an unsupported level")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrustSpec":
        data = _object(data, "trust")
        names = ("structural", "physical", "numerical", "empirical", "evidence")
        _keys(data, names, "trust")
        return cls(*(_string(data.get(name, "unverified"), f"trust.{name}") for name in names[:4]),
                   tuple(EvidenceSpec.from_dict(item) for item in _sequence(data.get("evidence", []), "trust.evidence")))


@dataclass(frozen=True, slots=True)
class ModelSpec(StrictRecord):
    """Acausal descriptor model in universal ``F(t,z,zdot,theta,u)=0`` form."""

    format: str
    id: str
    name: str
    version: str
    domains: tuple[str, ...]
    implements: str
    description: str = ""
    power_ports: tuple[PowerPortSpec, ...] = ()
    signal_ports: tuple[SignalPortSpec, ...] = ()
    artifact_ports: tuple[ArtifactPortSpec, ...] = ()
    states: tuple[StateSpec, ...] = ()
    algebraics: tuple[AlgebraicSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    relations: tuple[RelationSpec, ...] = ()
    stored_energy: tuple[NamedExpressionSpec, ...] = ()
    dissipation: tuple[NamedExpressionSpec, ...] = ()
    sources: tuple[NamedExpressionSpec, ...] = ()
    process_noise: ProcessNoiseSpec = ProcessNoiseSpec()
    modes: tuple[ModeSpec, ...] = ()
    initialization: InitializationSpec = InitializationSpec()
    validity: ValiditySpec = ValiditySpec()
    fidelity_levels: tuple[FidelitySpec, ...] = ()
    properties: tuple[PropertyTestSpec, ...] = ()
    trust: TrustSpec = TrustSpec()
    metadata: FrozenDict[Any] = FrozenDict()

    def __post_init__(self) -> None:
        if self.format != "pmdl-1":
            raise SpecError(f"unsupported model format {self.format!r}")
        _identifier(self.id, "model.id")
        if not self.name:
            raise SpecError("model.name may not be empty")
        if not self.version:
            raise SpecError("model.version may not be empty")
        if not self.domains:
            raise SpecError("model.domains may not be empty")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "model.metadata"))

    @property
    def residuals(self) -> tuple[RelationSpec, ...]:
        return self.relations

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(state.name for state in self.states)

    @property
    def algebraic_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.algebraics)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(port.name for port in self.signal_ports if port.direction == "input")

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(port.name for port in self.signal_ports if port.direction == "output")

    def evaluate_residual(self, t: Any, z: Any, zdot: Any, theta: Any = None, u: Any = None) -> Any:
        from .dsl import evaluate_model_residual
        return evaluate_model_residual(self, t, z, zdot, theta, u)

    residual = evaluate_residual

    def to_dict(self) -> dict[str, Any]:
        result = StrictRecord.to_dict(self)
        if not self.process_noise.enabled:
            result.pop("process_noise", None)
        if not self.artifact_ports:
            result.pop("artifact_ports", None)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        data = _object(data, "model")
        names = (
            "format", "id", "name", "version", "domains", "implements", "description", "power_ports", "signal_ports", "artifact_ports",
            "states", "algebraics", "parameters", "relations", "stored_energy", "dissipation", "sources", "process_noise", "modes",
            "initialization", "validity", "fidelity_levels", "properties", "trust", "metadata",
        )
        _keys(data, names, "model", names[:6])
        seq = lambda key: _sequence(data.get(key, []), f"model.{key}")
        return cls(
            _string(data["format"], "model.format"), _identifier(data["id"], "model.id"),
            _string(data["name"], "model.name"), _string(data["version"], "model.version"),
            tuple(_identifier(item, "model.domains[]") for item in seq("domains")),
            _identifier(data["implements"], "model.implements"), _string(data.get("description", ""), "model.description"),
            tuple(PowerPortSpec.from_dict(item) for item in seq("power_ports")),
            tuple(SignalPortSpec.from_dict(item) for item in seq("signal_ports")),
            tuple(ArtifactPortSpec.from_dict(item) for item in seq("artifact_ports")),
            tuple(StateSpec.from_dict(item) for item in seq("states")),
            tuple(AlgebraicSpec.from_dict(item) for item in seq("algebraics")),
            tuple(ParameterSpec.from_dict(item) for item in seq("parameters")),
            tuple(RelationSpec.from_dict(item) for item in seq("relations")),
            tuple(NamedExpressionSpec.from_dict(item, "stored_energy") for item in seq("stored_energy")),
            tuple(NamedExpressionSpec.from_dict(item, "dissipation") for item in seq("dissipation")),
            tuple(NamedExpressionSpec.from_dict(item, "source") for item in seq("sources")),
            ProcessNoiseSpec.from_dict(data.get("process_noise", {})),
            tuple(ModeSpec.from_dict(item) for item in seq("modes")),
            InitializationSpec.from_dict(data.get("initialization", {})), ValiditySpec.from_dict(data.get("validity", {})),
            tuple(FidelitySpec.from_dict(item) for item in seq("fidelity_levels")),
            tuple(PropertyTestSpec.from_dict(item) for item in seq("properties")), TrustSpec.from_dict(data.get("trust", {})),
            _freeze(_object(data.get("metadata", {}), "model.metadata")),
        )


@dataclass(frozen=True, slots=True)
class ComponentReferenceSpec(StrictRecord):
    """A contraption-local name bound to one catalog model instance."""

    id: str
    instantiation: str

    def __post_init__(self) -> None:
        _identifier(self.id, "component.id")
        _identifier(self.instantiation, "component.instantiation")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ComponentReferenceSpec":
        data = _object(data, "component")
        _keys(data, ("id", "instantiation"), "component", ("id", "instantiation"))
        return cls(
            id=_identifier(data["id"], "component.id"),
            instantiation=_identifier(data["instantiation"], "component.instantiation"),
        )


@dataclass(frozen=True, slots=True)
class PortRef(StrictRecord):
    component: str
    port: str

    def __post_init__(self) -> None:
        _identifier(self.component, "port_ref.component")
        _identifier(self.port, "port_ref.port", symbol=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | str) -> "PortRef":
        if isinstance(data, str):
            try:
                component, port = data.rsplit(".", 1)
            except ValueError as exc:
                raise SpecError("port reference string must be 'component.port'") from exc
            return cls(component, port)
        data = _object(data, "port_ref")
        _keys(data, ("component", "port"), "port_ref", ("component", "port"))
        return cls(_identifier(data["component"], "port_ref.component"), _identifier(data["port"], "port_ref.port", symbol=True))


@dataclass(frozen=True, slots=True)
class JointCoordinateBindingSpec(StrictRecord):
    """Map a PMDL angle state onto one physical revolute coordinate."""

    state: str
    joint_angle_at_state_zero_rad: float

    def __post_init__(self) -> None:
        PortRef.from_dict(self.state)
        value = self.joint_angle_at_state_zero_rad
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SpecError("joint coordinate zero angle must be finite")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JointCoordinateBindingSpec":
        value = _object(data, "joint coordinate binding")
        names = ("state", "joint_angle_at_state_zero_rad")
        _keys(value, names, "joint coordinate binding", names)
        angle = value["joint_angle_at_state_zero_rad"]
        if isinstance(angle, bool) or not isinstance(angle, (int, float)):
            raise SpecError("joint coordinate zero angle must be numeric")
        return cls(
            _string(value["state"], "joint coordinate binding.state"),
            float(angle),
        )


@dataclass(frozen=True, slots=True)
class JointSpec(StrictRecord):
    """Typed physical semantics for an attachment connection."""

    kind: str
    behavior_binding: str
    coordinate_bindings: tuple[JointCoordinateBindingSpec, ...]
    coordinate: str | None = None
    zero_angle_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"fixed", "revolute"}:
            raise SpecError("joint.kind must be 'fixed' or 'revolute'")
        if self.behavior_binding not in {"kinematic_only", "pmdl"}:
            raise SpecError("joint.behavior_binding must be 'kinematic_only' or 'pmdl'")
        if not math.isfinite(float(self.zero_angle_rad)):
            raise SpecError("joint.zero_angle_rad must be finite")
        if len({binding.state for binding in self.coordinate_bindings}) != len(self.coordinate_bindings):
            raise SpecError("joint coordinate bindings must name unique PMDL states")
        if self.kind == "fixed":
            if self.coordinate is not None or self.coordinate_bindings or self.zero_angle_rad != 0.0:
                raise SpecError("fixed joints cannot declare a coordinate, bindings, or nonzero zero angle")
        else:
            if self.coordinate is None:
                raise SpecError("revolute joints require a coordinate")
            PortRef.from_dict(self.coordinate)
            if not self.coordinate_bindings:
                raise SpecError("revolute joints require coordinate_bindings")
            if self.coordinate_bindings[0].state != self.coordinate:
                raise SpecError("the first revolute coordinate binding must name joint.coordinate")
            if not math.isclose(
                self.coordinate_bindings[0].joint_angle_at_state_zero_rad,
                self.zero_angle_rad,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise SpecError("joint.zero_angle_rad must match the primary coordinate binding")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JointSpec":
        value = _object(data, "joint")
        names = ("kind", "behavior_binding", "coordinate_bindings", "coordinate", "zero_angle_rad")
        _keys(value, names, "joint", names[:3])
        zero = value.get("zero_angle_rad", 0.0)
        if isinstance(zero, bool) or not isinstance(zero, (int, float)):
            raise SpecError("joint.zero_angle_rad must be numeric")
        return cls(
            _string(value["kind"], "joint.kind"),
            _string(value["behavior_binding"], "joint.behavior_binding"),
            tuple(
                JointCoordinateBindingSpec.from_dict(item)
                for item in _sequence(value["coordinate_bindings"], "joint.coordinate_bindings")
            ),
            None if value.get("coordinate") is None else _string(value["coordinate"], "joint.coordinate"),
            float(zero),
        )


@dataclass(frozen=True, slots=True)
class ConnectionSpec(StrictRecord):
    id: str
    kind: str
    endpoints: tuple[PortRef, ...]
    domain: str | None = None
    joint: JointSpec | None = None
    metadata: FrozenDict[Any] = FrozenDict()

    def __post_init__(self) -> None:
        _identifier(self.id, "connection.id")
        if self.kind not in {"power", "signal", "attachment", "constraint"}:
            raise SpecError(f"unsupported connection kind {self.kind!r}")
        if len(self.endpoints) < 2:
            raise SpecError("connection requires at least two endpoints")
        if self.kind == "attachment":
            if len(self.endpoints) != 2 or self.joint is None:
                raise SpecError("attachment connections require two endpoints and a joint")
            endpoint_components = {endpoint.component for endpoint in self.endpoints}
            invalid = sorted(
                binding.state.rsplit(".", 1)[0]
                for binding in self.joint.coordinate_bindings
                if binding.state.rsplit(".", 1)[0] not in endpoint_components
            )
            if invalid:
                raise SpecError(f"joint coordinate bindings must reference endpoint components: {invalid}")
        elif self.joint is not None:
            raise SpecError("only attachment connections may declare a joint")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "connection.metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConnectionSpec":
        data = _object(data, "connection")
        _keys(data, ("id", "kind", "endpoints", "domain", "joint", "metadata"), "connection", ("id", "kind", "endpoints"))
        return cls(_identifier(data["id"], "connection.id"), _string(data["kind"], "connection.kind"),
                   tuple(PortRef.from_dict(item) for item in _sequence(data["endpoints"], "connection.endpoints")),
                   None if data.get("domain") is None else _identifier(data["domain"], "connection.domain"),
                   None if data.get("joint") is None else JointSpec.from_dict(_object(data["joint"], "connection.joint")),
                   _freeze(_object(data.get("metadata", {}), "connection.metadata")))


@dataclass(frozen=True, slots=True)
class ActuatorBindingSpec(StrictRecord):
    """One resolved controller/external output driving a PMDL signal input."""

    id: str
    source: str
    target: PortRef
    settings: FrozenDict[Any] = FrozenDict()
    external: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActuatorBindingSpec":
        data = _object(data, "actuator binding")
        names = ("id", "source", "target", "settings", "external")
        _keys(data, names, "actuator binding", names[:3])
        return cls(
            _identifier(data["id"], "actuator_binding.id"),
            _string(data["source"], "actuator_binding.source"),
            PortRef.from_dict(data["target"]),
            _freeze(_object(data.get("settings", {}), "actuator_binding.settings")),
            _boolean(data.get("external", False), "actuator_binding.external"),
        )


@dataclass(frozen=True, slots=True)
class CatalogLinkSpec(StrictRecord):
    """Relative catalog root included in a contraption closure."""

    path: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | str) -> "CatalogLinkSpec":
        if isinstance(data, str):
            return cls(_string(data, "catalog path"))
        value = _object(data, "catalog link")
        _keys(value, ("path",), "catalog link", ("path",))
        return cls(_string(value["path"], "catalog path"))


@dataclass(frozen=True, slots=True)
class ArtifactLinkSpec(StrictRecord):
    """Hash-bound artifact path relative to the contraption document."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise SpecError("artifact path must be a non-empty, contained relative path")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.sha256) is None:
            raise SpecError("artifact sha256 must be 'sha256:' plus 64 lowercase hex digits")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactLinkSpec":
        value = _object(data, "artifact link")
        _keys(value, ("path", "sha256"), "artifact link", ("path", "sha256"))
        return cls(
            _string(value["path"], "artifact path"),
            _string(value["sha256"], "artifact sha256"),
        )


@dataclass(frozen=True, slots=True)
class ExplicitInputBindingSpec(StrictRecord):
    """Physical/external wiring for one controller explicit input pin."""

    signal: str | None = None
    external: str | None = None

    def __post_init__(self) -> None:
        if (self.signal is None) == (self.external is None):
            raise SpecError("explicit input binding requires exactly one of signal/external")
        if self.signal is not None:
            PortRef.from_dict(self.signal)
        if self.external is not None:
            _identifier(self.external, "explicit input external name", symbol=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExplicitInputBindingSpec":
        value = _object(data, "explicit input binding")
        _keys(value, ("signal", "external"), "explicit input binding")
        return cls(
            None if value.get("signal") is None else _string(value["signal"], "input signal"),
            None if value.get("external") is None else _string(value["external"], "external input"),
        )


@dataclass(frozen=True, slots=True)
class ControllerOutputBindingSpec(StrictRecord):
    """Physical signal wiring or one named external hardware/telemetry pin."""

    signal: str | None = None
    external: str | None = None

    def __post_init__(self) -> None:
        if (self.signal is None) == (self.external is None):
            raise SpecError(
                "controller output binding requires exactly one of signal/external"
            )
        if self.signal is not None:
            PortRef.from_dict(self.signal)
        if self.external is not None:
            _identifier(self.external, "controller output external name", symbol=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControllerOutputBindingSpec":
        value = _object(data, "controller output binding")
        _keys(value, ("signal", "external"), "controller output binding")
        return cls(
            None
            if value.get("signal") is None
            else _string(value["signal"], "controller output signal"),
            None
            if value.get("external") is None
            else _string(value["external"], "controller output external name"),
        )


@dataclass(frozen=True, slots=True)
class ControllerLinkSpec(StrictRecord):
    """One deployable controller plus its physical pin wiring."""

    id: str
    program: ArtifactLinkSpec
    explicit_inputs: FrozenDict[Any]
    implicit_inputs: FrozenDict[Any]
    outputs: FrozenDict[Any]

    def __post_init__(self) -> None:
        _identifier(self.id, "controller link id")
        if (
            not isinstance(self.explicit_inputs, FrozenDict)
            or not isinstance(self.implicit_inputs, FrozenDict)
            or not isinstance(self.outputs, FrozenDict)
        ):
            raise SpecError("controller link bindings must be frozen mappings")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControllerLinkSpec":
        value = _object(data, "controller link")
        names = ("id", "program", "explicit_inputs", "implicit_inputs", "outputs")
        _keys(value, names, "controller link", names)
        raw_inputs = _object(value["explicit_inputs"], "controller explicit_inputs")
        raw_implicit = _object(value["implicit_inputs"], "controller implicit_inputs")
        raw_outputs = _object(value["outputs"], "controller outputs")
        inputs = {
            _identifier(name, "controller input name", symbol=True):
            ExplicitInputBindingSpec.from_dict(_object(binding, f"controller input {name}"))
            for name, binding in raw_inputs.items()
        }
        outputs = {
            _identifier(name, "controller output name", symbol=True):
            ControllerOutputBindingSpec.from_dict(
                _object(binding, f"controller output {name}")
            )
            for name, binding in raw_outputs.items()
        }
        implicit_inputs = {
            _identifier(name, "controller implicit input name", symbol=True):
            _string(target, f"controller implicit input {name}")
            for name, target in raw_implicit.items()
        }
        return cls(
            _identifier(value["id"], "controller link id"),
            ArtifactLinkSpec.from_dict(_object(value["program"], "controller program")),
            FrozenDict(inputs),
            FrozenDict(implicit_inputs),
            FrozenDict(outputs),
        )


@dataclass(frozen=True, slots=True)
class VerificationLinkSpec(StrictRecord):
    """One posterior verification program and its observable bindings."""

    id: str
    program: ArtifactLinkSpec
    inputs: FrozenDict[Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationLinkSpec":
        value = _object(data, "verification link")
        names = ("id", "program", "inputs")
        _keys(value, names, "verification link", names)
        bindings = {
            _identifier(name, "verification input name", symbol=True):
            _string(source, f"verification input {name}")
            for name, source in _object(value["inputs"], "verification inputs").items()
        }
        return cls(
            _identifier(value["id"], "verification link id"),
            ArtifactLinkSpec.from_dict(_object(value["program"], "verification program")),
            FrozenDict(bindings),
        )


@dataclass(frozen=True, slots=True)
class ContraptionSpec(StrictRecord):
    format: str
    id: str
    name: str
    version: str
    catalogs: tuple[CatalogLinkSpec, ...]
    physical_root: FrozenDict[Any]
    components: tuple[ComponentReferenceSpec, ...]
    connections: tuple[ConnectionSpec, ...] = ()
    actuators: tuple[ActuatorBindingSpec, ...] = ()
    controllers: tuple[ControllerLinkSpec, ...] = ()
    verifications: tuple[VerificationLinkSpec, ...] = ()
    environment: FrozenDict[Any] = FrozenDict()
    metadata: FrozenDict[Any] = FrozenDict()

    def __post_init__(self) -> None:
        if self.format != "contraption-4":
            raise SpecError(f"unsupported contraption format {self.format!r}")
        _identifier(self.id, "contraption.id")
        if not self.name or not self.version:
            raise SpecError("contraption name and version may not be empty")
        object.__setattr__(self, "environment", _freeze(self.environment, "contraption.environment"))
        object.__setattr__(self, "metadata", _freeze(self.metadata, "contraption.metadata"))
        if not self.catalogs:
            raise SpecError("contraption requires at least one linked catalog")
        if len({item.id for item in self.controllers}) != len(self.controllers):
            raise SpecError("contraption controller ids must be unique")
        if len({item.id for item in self.verifications}) != len(self.verifications):
            raise SpecError("contraption verification ids must be unique")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContraptionSpec":
        data = _object(data, "contraption")
        names = (
            "format", "id", "name", "version", "catalogs", "physical_root",
            "components", "connections", "actuators", "controllers", "verifications",
            "environment", "metadata",
        )
        _keys(data, names, "contraption", (*names[:7], "metadata"))
        return cls(_string(data["format"], "contraption.format"), _identifier(data["id"], "contraption.id"),
                   _string(data["name"], "contraption.name"), _string(data["version"], "contraption.version"),
                   tuple(CatalogLinkSpec.from_dict(item) for item in _sequence(data["catalogs"], "contraption.catalogs")),
                   _freeze(_object(data["physical_root"], "contraption.physical_root")),
                   tuple(ComponentReferenceSpec.from_dict(item) for item in _sequence(data["components"], "contraption.components")),
                   tuple(ConnectionSpec.from_dict(item) for item in _sequence(data.get("connections", []), "contraption.connections")),
                   tuple(ActuatorBindingSpec.from_dict(item) for item in _sequence(data.get("actuators", []), "contraption.actuators")),
                   tuple(ControllerLinkSpec.from_dict(item) for item in _sequence(data.get("controllers", []), "contraption.controllers")),
                   tuple(VerificationLinkSpec.from_dict(item) for item in _sequence(data.get("verifications", []), "contraption.verifications")),
                   _freeze(_object(data.get("environment", {}), "contraption.environment")),
                   _freeze(_object(data.get("metadata", {}), "contraption.metadata")))


def json_schema_for(record_type: type[StrictRecord]) -> dict[str, Any]:
    """Generate a strict Draft 2020-12 JSON Schema for a record graph.

    The runtime dataclasses remain authoritative, but this export gives agents,
    editors, and staging pipelines a language-neutral contract.  Every record
    object sets ``additionalProperties`` to false, matching ``from_dict``.
    """

    if not isinstance(record_type, type) or not issubclass(record_type, StrictRecord) or not is_dataclass(record_type):
        raise TypeError("record_type must be a StrictRecord dataclass")
    definitions: dict[str, Any] = {}
    in_progress: set[type[Any]] = set()

    def type_schema(annotation: Any) -> dict[str, Any]:
        if annotation is Any:
            return {}
        origin, arguments = get_origin(annotation), get_args(annotation)
        if origin in (types.UnionType, Union):
            return {"anyOf": [type_schema(argument) if argument is not type(None) else {"type": "null"} for argument in arguments]}
        if origin in (tuple, list, Sequence):
            item_types = arguments[:-1] if origin is tuple and arguments and arguments[-1] is not Ellipsis else arguments[:1]
            item_schema = type_schema(item_types[0]) if item_types and all(item == item_types[0] for item in item_types) else {}
            result: dict[str, Any] = {"type": "array", "items": item_schema}
            if origin is tuple and arguments and arguments[-1] is not Ellipsis:
                result["minItems"] = len(arguments)
                result["maxItems"] = len(arguments)
            return result
        if origin in (dict, Mapping, FrozenDict) or annotation is FrozenDict:
            value_type = arguments[-1] if arguments else Any
            return {"type": "object", "additionalProperties": type_schema(value_type)}
        if isinstance(annotation, type) and issubclass(annotation, StrictRecord) and is_dataclass(annotation):
            visit(annotation)
            return {"$ref": f"#/$defs/{annotation.__name__}"}
        primitives = {str: "string", bool: "boolean", int: "integer", float: "number"}
        if annotation in primitives:
            return {"type": primitives[annotation]}
        return {}

    def visit(cls: type[StrictRecord]) -> None:
        if cls.__name__ in definitions or cls in in_progress:
            return
        in_progress.add(cls)
        hints = get_type_hints(cls)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for field in fields(cls):
            properties[field.name] = type_schema(hints.get(field.name, Any))
            if field.default is MISSING and field.default_factory is MISSING:
                required.append(field.name)
        constants = {
            ("ModelSpec", "format"): "pmdl-1",
            ("ContraptionSpec", "format"): "contraption-4",
        }
        for field_name in properties:
            constant = constants.get((cls.__name__, field_name))
            if constant is not None:
                properties[field_name] = {"const": constant}
        definitions[cls.__name__] = {
            "type": "object", "additionalProperties": False,
            "properties": properties, "required": required,
        }
        in_progress.remove(cls)

    visit(record_type)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://contraption.local/schema/{record_type.__name__}.schema.json",
        "$ref": f"#/$defs/{record_type.__name__}",
        "$defs": {name: definitions[name] for name in sorted(definitions)},
    }


def write_json_schema(record_type: type[StrictRecord], path: str | Path) -> Path:
    target = Path(path)
    target.write_text(json.dumps(json_schema_for(record_type), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return target
