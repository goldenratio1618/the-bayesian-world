"""Restricted declarative controllers for simulated and embedded contraptions.

The control format is deliberately data-only.  Expressions are allow-listed
trees, state changes are explicit transitions, and mutable values are declared
bounded registers.  In particular, this module never calls ``eval``/``exec``
and does not import user-authored Python modules.

A controller tick has synchronous (Moore-machine) semantics: outputs, register
updates, and transition guards all observe the same pre-tick snapshot.  Register
updates are committed together and a selected transition becomes active for the
next tick.  This makes an offline run deterministic and straightforward to
mirror in an embedded runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence, Union


JSONScalar = Union[bool, int, float]
Value = Union[bool, float]


class ControlError(ValueError):
    """Base class for control-program errors."""


class ControlValidationError(ControlError):
    """Raised when a declarative program violates the control schema."""


class ControlRuntimeError(ControlError):
    """Raised when otherwise-valid control math is undefined at runtime."""


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_VALUE_TYPES = frozenset({"number", "boolean"})
_SOURCES = frozenset({"external", "sensor", "internal", "output"})
_MAX_EXPRESSION_DEPTH = 32
_MAX_EXPRESSION_NODES = 512


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ControlValidationError(
            f"{label} must match {_IDENTIFIER.pattern!r}; received {value!r}"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlValidationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ControlValidationError(f"{label} must be finite")
    return result


def _scalar(value: Any, label: str) -> Value:
    if isinstance(value, bool):
        return value
    return _finite_number(value, label)


def _unknown(data: Mapping[str, Any], allowed: Iterable[str], label: str) -> None:
    extras = sorted(set(data) - set(allowed))
    if extras:
        raise ControlValidationError(f"unknown {label} keys: {', '.join(extras)}")


@dataclass(frozen=True)
class Expression:
    """An immutable node in the approved control expression language."""

    op: str
    args: tuple["Expression", ...] = ()
    value: Value | str | None = None

    @classmethod
    def literal(cls, value: JSONScalar) -> "Expression":
        return cls("literal", value=_scalar(value, "expression literal"))

    @classmethod
    def reference(cls, path: str) -> "Expression":
        if not isinstance(path, str):
            raise ControlValidationError("expression reference must be a string")
        return cls("ref", value=path)

    @classmethod
    def operation(cls, op: str, *args: Any) -> "Expression":
        return cls(str(op), tuple(cls.from_data(arg) for arg in args))

    @classmethod
    def from_data(cls, data: Any) -> "Expression":
        """Parse an expression from JSON-compatible data.

        Numbers and booleans are literals.  All other expressions must be one
        of ``{"ref": "..."}``, ``{"signal": "..."}``, or
        ``{"op": "add", "args": [...]}``.  Strings are intentionally not
        treated as code or implicit references.
        """

        if isinstance(data, Expression):
            return data
        if isinstance(data, (bool, int, float)):
            return cls.literal(data)
        if not isinstance(data, Mapping):
            raise ControlValidationError(
                "expression must be a scalar or a declarative expression object"
            )
        if "literal" in data or "const" in data:
            _unknown(data, {"literal", "const"}, "literal expression")
            if "literal" in data and "const" in data:
                raise ControlValidationError("use either literal or const, not both")
            return cls.literal(data.get("literal", data.get("const")))
        if "ref" in data or "signal" in data:
            _unknown(data, {"ref", "signal"}, "reference expression")
            if "ref" in data and "signal" in data:
                raise ControlValidationError("use either ref or signal, not both")
            raw = data.get("ref")
            if raw is None:
                raw = data.get("signal")
            return cls.reference(raw)
        _unknown(data, {"op", "args"}, "operation expression")
        op = data.get("op")
        args = data.get("args")
        if not isinstance(op, str) or not isinstance(args, Sequence) or isinstance(
            args, (str, bytes)
        ):
            raise ControlValidationError("operation requires string op and array args")
        return cls(op, tuple(cls.from_data(arg) for arg in args))

    def to_dict(self) -> dict[str, Any]:
        if self.op == "literal":
            return {"literal": self.value}
        if self.op == "ref":
            return {"ref": self.value}
        return {"op": self.op, "args": [arg.to_dict() for arg in self.args]}


@dataclass(frozen=True)
class SignalSpec:
    """A typed, optionally bounded control input or output."""

    name: str
    value_type: str = "number"
    source: str = "external"
    default: Value = 0.0
    minimum: float | None = None
    maximum: float | None = None
    unit: str = "1"
    description: str = ""

    def __post_init__(self) -> None:
        _name(self.name, "signal name")
        if self.value_type not in _VALUE_TYPES:
            raise ControlValidationError(
                f"signal {self.name!r} has unsupported type {self.value_type!r}"
            )
        if self.source not in _SOURCES:
            raise ControlValidationError(
                f"signal {self.name!r} has unsupported source {self.source!r}"
            )
        if not isinstance(self.unit, str) or not self.unit:
            raise ControlValidationError(f"signal {self.name!r} needs a unit string")
        if self.value_type == "boolean":
            if not isinstance(self.default, bool):
                raise ControlValidationError(
                    f"boolean signal {self.name!r} needs a boolean default"
                )
            if self.minimum is not None or self.maximum is not None:
                raise ControlValidationError("boolean signals cannot have numeric bounds")
        else:
            default = _finite_number(self.default, f"default for {self.name}")
            low = (
                _finite_number(self.minimum, f"minimum for {self.name}")
                if self.minimum is not None
                else None
            )
            high = (
                _finite_number(self.maximum, f"maximum for {self.name}")
                if self.maximum is not None
                else None
            )
            if low is not None and high is not None and low > high:
                raise ControlValidationError(f"invalid bounds for {self.name!r}")
            if low is not None and default < low or high is not None and default > high:
                raise ControlValidationError(f"default for {self.name!r} is out of bounds")

    @classmethod
    def from_data(
        cls, data: str | Mapping[str, Any], *, default_source: str = "external"
    ) -> "SignalSpec":
        if isinstance(data, str):
            return cls(data, source=default_source)
        if not isinstance(data, Mapping):
            raise ControlValidationError("signal declaration must be a name or object")
        _unknown(
            data,
            {
                "name",
                "type",
                "value_type",
                "source",
                "default",
                "minimum",
                "maximum",
                "min",
                "max",
                "unit",
                "description",
            },
            "signal",
        )
        if "name" not in data:
            raise ControlValidationError("signal declaration is missing name")
        return cls(
            name=data["name"],
            value_type=data.get("value_type", data.get("type", "number")),
            source=data.get("source", default_source),
            default=data.get("default", False if data.get("type") == "boolean" else 0.0),
            minimum=data.get("minimum", data.get("min")),
            maximum=data.get("maximum", data.get("max")),
            unit=data.get("unit", "1"),
            description=data.get("description", ""),
        )

    def normalize(self, value: Any, label: str | None = None) -> Value:
        label = label or self.name
        if self.value_type == "boolean":
            if not isinstance(value, bool):
                raise ControlRuntimeError(f"{label} must be boolean")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ControlRuntimeError(f"{label} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ControlRuntimeError(f"{label} must be finite")
        if self.minimum is not None and result < self.minimum:
            raise ControlRuntimeError(f"{label} is below {self.minimum}")
        if self.maximum is not None and result > self.maximum:
            raise ControlRuntimeError(f"{label} is above {self.maximum}")
        return result


@dataclass(frozen=True)
class RegisterSpec:
    """A synchronously updated controller memory cell."""

    name: str
    initial: Value = 0.0
    value_type: str = "number"
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        SignalSpec(
            self.name,
            self.value_type,
            "internal",
            self.initial,
            self.minimum,
            self.maximum,
        )

    @classmethod
    def from_data(cls, name: str, data: Any) -> "RegisterSpec":
        if isinstance(data, (bool, int, float)):
            return cls(name, _scalar(data, f"register {name}"), "boolean" if isinstance(data, bool) else "number")
        if not isinstance(data, Mapping):
            raise ControlValidationError(f"register {name!r} must be a scalar or object")
        _unknown(
            data,
            {"initial", "type", "value_type", "minimum", "maximum", "min", "max"},
            f"register {name}",
        )
        value_type = data.get("value_type", data.get("type", "number"))
        initial = data.get("initial", False if value_type == "boolean" else 0.0)
        return cls(
            name,
            _scalar(initial, f"initial value for {name}"),
            value_type,
            data.get("minimum", data.get("min")),
            data.get("maximum", data.get("max")),
        )


@dataclass(frozen=True)
class Transition:
    """A prioritized, boolean-guarded state transition."""

    target: str
    condition: Expression
    priority: int = 0
    name: str = ""

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "Transition":
        if not isinstance(data, Mapping):
            raise ControlValidationError("transition must be an object")
        _unknown(data, {"target", "when", "condition", "priority", "name"}, "transition")
        if "target" not in data:
            raise ControlValidationError("transition is missing target")
        raw_condition = data.get("condition", data.get("when"))
        if raw_condition is None:
            raise ControlValidationError("transition is missing condition")
        priority = data.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ControlValidationError("transition priority must be an integer")
        return cls(
            target=_name(data["target"], "transition target"),
            condition=Expression.from_data(raw_condition),
            priority=priority,
            name=data.get("name", ""),
        )


@dataclass(frozen=True)
class ControlState:
    """Outputs, memory updates, and exits active in a named state."""

    name: str
    outputs: Mapping[str, Expression] = field(default_factory=dict)
    updates: Mapping[str, Expression] = field(default_factory=dict)
    transitions: tuple[Transition, ...] = ()

    @classmethod
    def from_data(cls, data: Mapping[str, Any], *, name: str | None = None) -> "ControlState":
        if not isinstance(data, Mapping):
            raise ControlValidationError("state must be an object")
        _unknown(data, {"name", "outputs", "updates", "transitions"}, "state")
        state_name = name if name is not None else data.get("name")
        if state_name is None:
            raise ControlValidationError("state is missing name")
        raw_outputs = data.get("outputs", {})
        raw_updates = data.get("updates", {})
        raw_transitions = data.get("transitions", [])
        if not isinstance(raw_outputs, Mapping) or not isinstance(raw_updates, Mapping):
            raise ControlValidationError("state outputs and updates must be objects")
        if not isinstance(raw_transitions, Sequence) or isinstance(raw_transitions, (str, bytes)):
            raise ControlValidationError("state transitions must be an array")
        return cls(
            _name(state_name, "state name"),
            {_name(key, "output name"): Expression.from_data(value) for key, value in raw_outputs.items()},
            {_name(key, "register name"): Expression.from_data(value) for key, value in raw_updates.items()},
            tuple(Transition.from_data(item) for item in raw_transitions),
        )


_ARITIES: dict[str, tuple[int, int]] = {
    "add": (2, 2),
    "sub": (2, 2),
    "mul": (2, 2),
    "div": (2, 2),
    "pow": (2, 2),
    "neg": (1, 1),
    "abs": (1, 1),
    "min": (2, 16),
    "max": (2, 16),
    "clamp": (3, 3),
    "sqrt": (1, 1),
    "exp": (1, 1),
    "log": (1, 1),
    "sin": (1, 1),
    "cos": (1, 1),
    "tanh": (1, 1),
    "lt": (2, 2),
    "le": (2, 2),
    "gt": (2, 2),
    "ge": (2, 2),
    "eq": (2, 2),
    "ne": (2, 2),
    "and": (2, 16),
    "or": (2, 16),
    "not": (1, 1),
    "select": (3, 3),
}


def _validate_expression(
    expression: Expression,
    symbols: Mapping[str, str],
    *,
    depth: int = 0,
    budget: list[int] | None = None,
) -> str:
    budget = budget if budget is not None else [_MAX_EXPRESSION_NODES]
    budget[0] -= 1
    if budget[0] < 0:
        raise ControlValidationError("expression exceeds node limit")
    if depth > _MAX_EXPRESSION_DEPTH:
        raise ControlValidationError("expression exceeds depth limit")
    if expression.op == "literal":
        return "boolean" if isinstance(expression.value, bool) else "number"
    if expression.op == "ref":
        if not isinstance(expression.value, str) or expression.value not in symbols:
            raise ControlValidationError(f"unknown control reference {expression.value!r}")
        return symbols[expression.value]
    if expression.op not in _ARITIES:
        raise ControlValidationError(f"unsupported control operation {expression.op!r}")
    low, high = _ARITIES[expression.op]
    if not low <= len(expression.args) <= high:
        raise ControlValidationError(
            f"operation {expression.op!r} expects {low}..{high} arguments"
        )
    child_types = [
        _validate_expression(arg, symbols, depth=depth + 1, budget=budget)
        for arg in expression.args
    ]
    if expression.op in {"and", "or", "not"}:
        if any(kind != "boolean" for kind in child_types):
            raise ControlValidationError(f"operation {expression.op!r} requires booleans")
        return "boolean"
    if expression.op == "select":
        if child_types[0] != "boolean" or child_types[1] != child_types[2]:
            raise ControlValidationError("select requires boolean condition and matching branches")
        return child_types[1]
    if expression.op in {"eq", "ne"}:
        if child_types[0] != child_types[1]:
            raise ControlValidationError(f"operation {expression.op!r} compares unlike types")
        return "boolean"
    if expression.op in {"lt", "le", "gt", "ge"}:
        if any(kind != "number" for kind in child_types):
            raise ControlValidationError(f"operation {expression.op!r} requires numbers")
        return "boolean"
    if any(kind != "number" for kind in child_types):
        raise ControlValidationError(f"operation {expression.op!r} requires numbers")
    return "number"


@dataclass(frozen=True)
class ControlProgram:
    """A validated, serializable internal-control state machine."""

    inputs: tuple[SignalSpec, ...]
    outputs: tuple[SignalSpec, ...]
    states: tuple[ControlState, ...]
    initial_state: str
    parameters: Mapping[str, Value] = field(default_factory=dict)
    registers: tuple[RegisterSpec, ...] = ()
    name: str = "controller"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        _name(self.name, "controller name")
        if not isinstance(self.version, str) or not self.version:
            raise ControlValidationError("controller version must be a non-empty string")
        self.validate()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControlProgram":
        if not isinstance(data, Mapping):
            raise ControlValidationError("control program must be an object")
        _unknown(
            data,
            {
                "name",
                "version",
                "inputs",
                "outputs",
                "parameters",
                "registers",
                "states",
                "initial_state",
            },
            "control program",
        )
        raw_inputs = data.get("inputs", [])
        raw_outputs = data.get("outputs", [])
        raw_states = data.get("states", [])
        raw_parameters = data.get("parameters", {})
        raw_registers = data.get("registers", {})
        if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, (str, bytes)):
            raise ControlValidationError("inputs must be an array")
        if not isinstance(raw_outputs, Sequence) or isinstance(raw_outputs, (str, bytes)):
            raise ControlValidationError("outputs must be an array")
        if not isinstance(raw_parameters, Mapping) or not isinstance(raw_registers, Mapping):
            raise ControlValidationError("parameters and registers must be objects")
        parameters = {
            _name(key, "parameter name"): _scalar(value, f"parameter {key}")
            for key, value in raw_parameters.items()
        }
        if isinstance(raw_states, Mapping):
            states = tuple(
                ControlState.from_data(value, name=key) for key, value in raw_states.items()
            )
        elif isinstance(raw_states, Sequence) and not isinstance(raw_states, (str, bytes)):
            states = tuple(ControlState.from_data(value) for value in raw_states)
        else:
            raise ControlValidationError("states must be an array or object")
        return cls(
            inputs=tuple(SignalSpec.from_data(value) for value in raw_inputs),
            outputs=tuple(
                SignalSpec.from_data(value, default_source="output") for value in raw_outputs
            ),
            states=states,
            initial_state=data.get("initial_state", states[0].name if states else ""),
            parameters=parameters,
            registers=tuple(
                RegisterSpec.from_data(_name(key, "register name"), value)
                for key, value in raw_registers.items()
            ),
            name=data.get("name", "controller"),
            version=data.get("version", "1.0.0"),
        )

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "ControlProgram":
        return cls.from_dict(data)

    def validate(self) -> None:
        def unique(items: Iterable[str], label: str) -> set[str]:
            values = list(items)
            if len(values) != len(set(values)):
                raise ControlValidationError(f"duplicate {label}")
            return set(values)

        input_names = unique((item.name for item in self.inputs), "input name")
        output_names = unique((item.name for item in self.outputs), "output name")
        register_names = unique((item.name for item in self.registers), "register name")
        state_names = unique((item.name for item in self.states), "state name")
        parameter_names = unique(self.parameters, "parameter name")
        if not self.states:
            raise ControlValidationError("control program needs at least one state")
        if self.initial_state not in state_names:
            raise ControlValidationError("initial_state does not name a declared state")
        if input_names & output_names:
            raise ControlValidationError("input and output names must be distinct")
        if (input_names | output_names | register_names) & parameter_names:
            raise ControlValidationError("parameter names must be distinct from signals/registers")

        symbols: dict[str, str] = {
            "time": "number",
            "time_in_state": "number",
            "dt": "number",
        }
        for signal in self.inputs:
            symbols[signal.name] = signal.value_type
            symbols[f"input.{signal.name}"] = signal.value_type
            symbols[f"{signal.source}.{signal.name}"] = signal.value_type
        for key, value in self.parameters.items():
            symbols[f"parameter.{key}"] = "boolean" if isinstance(value, bool) else "number"
        for register in self.registers:
            symbols[f"register.{register.name}"] = register.value_type

        output_types = {item.name: item.value_type for item in self.outputs}
        register_types = {item.name: item.value_type for item in self.registers}
        for state in self.states:
            missing = output_names - set(state.outputs)
            extras = set(state.outputs) - output_names
            if missing or extras:
                raise ControlValidationError(
                    f"state {state.name!r} output mismatch; missing={sorted(missing)}, "
                    f"unknown={sorted(extras)}"
                )
            extras = set(state.updates) - register_names
            if extras:
                raise ControlValidationError(
                    f"state {state.name!r} updates unknown registers {sorted(extras)}"
                )
            for key, expr in state.outputs.items():
                if _validate_expression(expr, symbols) != output_types[key]:
                    raise ControlValidationError(f"output {key!r} has the wrong type")
            for key, expr in state.updates.items():
                if _validate_expression(expr, symbols) != register_types[key]:
                    raise ControlValidationError(f"register update {key!r} has the wrong type")
            priorities: set[int] = set()
            for transition in state.transitions:
                if transition.target not in state_names:
                    raise ControlValidationError(
                        f"state {state.name!r} transitions to unknown state {transition.target!r}"
                    )
                if transition.priority in priorities:
                    raise ControlValidationError(
                        f"state {state.name!r} has ambiguous transition priority "
                        f"{transition.priority}"
                    )
                priorities.add(transition.priority)
                if _validate_expression(transition.condition, symbols) != "boolean":
                    raise ControlValidationError("transition conditions must be boolean")

    def to_dict(self) -> dict[str, Any]:
        def signal(item: SignalSpec) -> dict[str, Any]:
            return {
                "name": item.name,
                "type": item.value_type,
                "source": item.source,
                "default": item.default,
                "minimum": item.minimum,
                "maximum": item.maximum,
                "unit": item.unit,
                "description": item.description,
            }

        return {
            "name": self.name,
            "version": self.version,
            "inputs": [signal(item) for item in self.inputs],
            "outputs": [signal(item) for item in self.outputs],
            "parameters": dict(self.parameters),
            "registers": {
                item.name: {
                    "type": item.value_type,
                    "initial": item.initial,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                }
                for item in self.registers
            },
            "states": [
                {
                    "name": state.name,
                    "outputs": {key: value.to_dict() for key, value in state.outputs.items()},
                    "updates": {key: value.to_dict() for key, value in state.updates.items()},
                    "transitions": [
                        {
                            "name": transition.name,
                            "target": transition.target,
                            "priority": transition.priority,
                            "condition": transition.condition.to_dict(),
                        }
                        for transition in state.transitions
                    ],
                }
                for state in self.states
            ],
            "initial_state": self.initial_state,
        }


def _evaluate(expression: Expression, values: Mapping[str, Value]) -> Value:
    if expression.op == "literal":
        assert isinstance(expression.value, (bool, float))
        return expression.value
    if expression.op == "ref":
        assert isinstance(expression.value, str)
        return values[expression.value]
    args = [_evaluate(arg, values) for arg in expression.args]
    op = expression.op
    try:
        if op == "add":
            result = float(args[0]) + float(args[1])
        elif op == "sub":
            result = float(args[0]) - float(args[1])
        elif op == "mul":
            result = float(args[0]) * float(args[1])
        elif op == "div":
            if float(args[1]) == 0.0:
                raise ControlRuntimeError("division by zero in control expression")
            result = float(args[0]) / float(args[1])
        elif op == "pow":
            result = math.pow(float(args[0]), float(args[1]))
        elif op == "neg":
            result = -float(args[0])
        elif op == "abs":
            result = abs(float(args[0]))
        elif op == "min":
            result = min(float(value) for value in args)
        elif op == "max":
            result = max(float(value) for value in args)
        elif op == "clamp":
            low, high = float(args[1]), float(args[2])
            if low > high:
                raise ControlRuntimeError("clamp lower bound exceeds upper bound")
            result = min(max(float(args[0]), low), high)
        elif op == "sqrt":
            result = math.sqrt(float(args[0]))
        elif op == "exp":
            result = math.exp(float(args[0]))
        elif op == "log":
            result = math.log(float(args[0]))
        elif op == "sin":
            result = math.sin(float(args[0]))
        elif op == "cos":
            result = math.cos(float(args[0]))
        elif op == "tanh":
            result = math.tanh(float(args[0]))
        elif op == "lt":
            return float(args[0]) < float(args[1])
        elif op == "le":
            return float(args[0]) <= float(args[1])
        elif op == "gt":
            return float(args[0]) > float(args[1])
        elif op == "ge":
            return float(args[0]) >= float(args[1])
        elif op == "eq":
            return args[0] == args[1]
        elif op == "ne":
            return args[0] != args[1]
        elif op == "and":
            return all(bool(value) for value in args)
        elif op == "or":
            return any(bool(value) for value in args)
        elif op == "not":
            return not bool(args[0])
        elif op == "select":
            return args[1] if bool(args[0]) else args[2]
        else:  # pragma: no cover - validation makes this unreachable
            raise ControlRuntimeError(f"unsupported operation {op!r}")
    except (OverflowError, ValueError) as exc:
        raise ControlRuntimeError(f"invalid math in {op!r}: {exc}") from exc
    if not math.isfinite(result):
        raise ControlRuntimeError(f"operation {op!r} produced a non-finite value")
    return result


def _numeric_backend(backend: Any, context: Mapping[str, Any]) -> Any:
    """Resolve NumPy/Torch without making either mandatory for scalar control."""

    if backend is not None and not isinstance(backend, str):
        return backend
    name = backend
    if name is None:
        module_roots = {
            type(value).__module__.split(".")[0]
            for value in context.values()
            if not isinstance(value, (bool, int, float))
        }
        name = "torch" if "torch" in module_roots else "numpy" if "numpy" in module_roots else None
    if name is None or name == "scalar":
        return None
    if name not in {"numpy", "torch"}:
        raise ControlRuntimeError("vectorized backend must be 'numpy', 'torch', or a module")
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ControlRuntimeError(f"requested control backend {name!r} is unavailable") from exc


def evaluate_expression(
    expression: Expression | Mapping[str, Any] | JSONScalar,
    context: Mapping[str, Any],
    backend: Any = None,
) -> Any:
    """Evaluate one approved tree with scalar, NumPy, or Torch values.

    This pure evaluator preserves array/tensor operations and autograd graphs;
    it never converts a tensor to a Python scalar.  Callers should validate the
    expression as part of a :class:`ControlProgram` before using this lower-level
    helper directly.
    """

    expr = Expression.from_data(expression)
    module = _numeric_backend(backend, context)

    def tensor_pair(left: Any, right: Any) -> tuple[Any, Any]:
        if module is None or getattr(module, "__name__", "") != "torch":
            return left, right
        if module.is_tensor(left) and not module.is_tensor(right):
            right = module.as_tensor(right, dtype=left.dtype, device=left.device)
        elif module.is_tensor(right) and not module.is_tensor(left):
            left = module.as_tensor(left, dtype=right.dtype, device=right.device)
        return left, right

    def apply(node: Expression) -> Any:
        if node.op == "literal":
            return node.value
        if node.op == "ref":
            if node.value not in context:
                raise ControlRuntimeError(f"missing vectorized control value {node.value!r}")
            return context[node.value]
        args = [apply(arg) for arg in node.args]
        op = node.op
        if op == "add": return args[0] + args[1]
        if op == "sub": return args[0] - args[1]
        if op == "mul": return args[0] * args[1]
        if op == "div": return args[0] / args[1]
        if op == "pow": return args[0] ** args[1]
        if op == "neg": return -args[0]
        if op == "abs": return abs(args[0])
        if op in {"min", "max"}:
            result = args[0]
            function_name = "minimum" if op == "min" else "maximum"
            for item in args[1:]:
                if module is None:
                    result = min(result, item) if op == "min" else max(result, item)
                else:
                    left, right = tensor_pair(result, item)
                    result = getattr(module, function_name)(left, right)
            return result
        if op == "clamp":
            if module is None:
                return min(max(args[0], args[1]), args[2])
            value, low = tensor_pair(args[0], args[1])
            value = module.maximum(value, low)
            value, high = tensor_pair(value, args[2])
            return module.minimum(value, high)
        if op in {"sqrt", "exp", "log", "sin", "cos", "tanh"}:
            return getattr(math if module is None else module, op)(args[0])
        if op == "lt": return args[0] < args[1]
        if op == "le": return args[0] <= args[1]
        if op == "gt": return args[0] > args[1]
        if op == "ge": return args[0] >= args[1]
        if op == "eq": return args[0] == args[1]
        if op == "ne": return args[0] != args[1]
        if op in {"and", "or"}:
            result = args[0]
            function_name = "logical_and" if op == "and" else "logical_or"
            for item in args[1:]:
                if module is None:
                    result = (result and item) if op == "and" else (result or item)
                else:
                    result = getattr(module, function_name)(result, item)
            return result
        if op == "not":
            return (not args[0]) if module is None else module.logical_not(args[0])
        if op == "select":
            if module is None:
                return args[1] if args[0] else args[2]
            left, right = tensor_pair(args[1], args[2])
            return module.where(args[0], left, right)
        raise ControlRuntimeError(f"unsupported vectorized control operation {op!r}")

    return apply(expr)


def evaluate_state_outputs(
    program: ControlProgram | Mapping[str, Any],
    state_name: str,
    context: Mapping[str, Any],
    backend: Any = None,
) -> dict[str, Any]:
    """Pure batch-capable output evaluation for a fixed controller state.

    The simulator can call this once per Monte Carlo batch with sensor tensors.
    NumPy arrays remain vectorized, and Torch tensors retain their gradient
    graph.  State-machine transitions stay in :class:`ControllerRuntime`, where
    individual batch members may legitimately diverge between states.
    """

    parsed = program if isinstance(program, ControlProgram) else ControlProgram.from_dict(program)
    state = next((item for item in parsed.states if item.name == state_name), None)
    if state is None:
        raise ControlRuntimeError(f"unknown controller state {state_name!r}")
    values: dict[str, Any] = {
        "time": context.get("time", 0.0),
        "time_in_state": context.get("time_in_state", 0.0),
        "dt": context.get("dt", 0.0),
    }
    for signal in parsed.inputs:
        if signal.name in context:
            value = context[signal.name]
        elif f"input.{signal.name}" in context:
            value = context[f"input.{signal.name}"]
        elif f"{signal.source}.{signal.name}" in context:
            value = context[f"{signal.source}.{signal.name}"]
        else:
            value = signal.default
        values[signal.name] = value
        values[f"input.{signal.name}"] = value
        values[f"{signal.source}.{signal.name}"] = value
    for name, value in parsed.parameters.items():
        values[f"parameter.{name}"] = context.get(f"parameter.{name}", value)
    for register in parsed.registers:
        values[f"register.{register.name}"] = context.get(
            f"register.{register.name}", context.get(register.name, register.initial)
        )
    return {
        name: evaluate_expression(expression, values, backend)
        for name, expression in state.outputs.items()
    }


@dataclass(frozen=True)
class ControlFrame:
    """Observable result of one synchronous controller tick."""

    time: float
    state: str
    outputs: Mapping[str, Value]
    registers: Mapping[str, Value]
    transitioned_from: str | None = None

    def __getitem__(self, output_name: str) -> Value:
        return self.outputs[output_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "state": self.state,
            "outputs": dict(self.outputs),
            "registers": dict(self.registers),
            "transitioned_from": self.transitioned_from,
        }


class ControllerRuntime:
    """Stateful interpreter for a :class:`ControlProgram`."""

    def __init__(self, program: ControlProgram | Mapping[str, Any]):
        self.program = (
            program if isinstance(program, ControlProgram) else ControlProgram.from_dict(program)
        )
        self._states = {state.name: state for state in self.program.states}
        self._inputs = {item.name: item for item in self.program.inputs}
        self._outputs = {item.name: item for item in self.program.outputs}
        self._register_specs = {item.name: item for item in self.program.registers}
        self.reset()

    def reset(self) -> None:
        self.state = self.program.initial_state
        self.time = 0.0
        self.time_in_state = 0.0
        self.registers: dict[str, Value] = {
            item.name: item.initial for item in self.program.registers
        }

    def _context(self, supplied: Mapping[str, Any], dt: float) -> dict[str, Value]:
        unknown = sorted(set(supplied) - set(self._inputs))
        if unknown:
            raise ControlRuntimeError(f"unknown controller inputs: {', '.join(unknown)}")
        values: dict[str, Value] = {
            "time": self.time,
            "time_in_state": self.time_in_state,
            "dt": dt,
        }
        for name, spec in self._inputs.items():
            value = spec.normalize(supplied.get(name, spec.default), f"input {name}")
            values[name] = value
            values[f"input.{name}"] = value
            values[f"{spec.source}.{name}"] = value
        for name, value in self.program.parameters.items():
            values[f"parameter.{name}"] = value
        for name, value in self.registers.items():
            values[f"register.{name}"] = value
        return values

    def step(
        self,
        inputs: Mapping[str, Any] | None = None,
        dt: float = 0.01,
        *,
        external: Mapping[str, Any] | None = None,
        sensors: Mapping[str, Any] | None = None,
    ) -> ControlFrame:
        """Advance one tick and return outputs from the pre-transition state.

        ``inputs`` is the compact API.  ``external`` and ``sensors`` may be used
        instead; their keys are checked against each signal's declared source.
        """

        dt_value = _finite_number(dt, "control dt")
        if dt_value <= 0.0:
            raise ControlRuntimeError("control dt must be positive")
        supplied = dict(inputs or {})
        for source_name, source_values in (("external", external), ("sensor", sensors)):
            if source_values is None:
                continue
            for key, value in source_values.items():
                if key in supplied:
                    raise ControlRuntimeError(f"controller input {key!r} supplied twice")
                spec = self._inputs.get(key)
                if spec is None or spec.source != source_name:
                    raise ControlRuntimeError(
                        f"{key!r} is not declared as a {source_name} input"
                    )
                supplied[key] = value

        active_name = self.state
        active = self._states[active_name]
        values = self._context(supplied, dt_value)
        outputs = {
            name: self._outputs[name].normalize(_evaluate(expr, values), f"output {name}")
            for name, expr in active.outputs.items()
        }
        updates = dict(self.registers)
        for name, expr in active.updates.items():
            raw = _evaluate(expr, values)
            spec = self._register_specs[name]
            # Registers saturate at declared numeric bounds.  This is useful for
            # safe integrators while input/output bounds remain strict checks.
            if spec.value_type == "number":
                value = float(raw)
                if spec.minimum is not None:
                    value = max(value, spec.minimum)
                if spec.maximum is not None:
                    value = min(value, spec.maximum)
                updates[name] = value
            else:
                updates[name] = bool(raw)

        selected: Transition | None = None
        for transition in sorted(active.transitions, key=lambda item: -item.priority):
            if bool(_evaluate(transition.condition, values)):
                selected = transition
                break
        self.registers = updates
        self.time += dt_value
        if selected is not None:
            self.state = selected.target
            self.time_in_state = 0.0
        else:
            self.time_in_state += dt_value
        return ControlFrame(
            time=self.time,
            state=self.state,
            outputs=outputs,
            registers=dict(self.registers),
            transitioned_from=active_name if selected is not None else None,
        )

    def run(
        self,
        inputs: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        dt: float,
        steps: int | None = None,
    ) -> tuple[ControlFrame, ...]:
        """Run a fixed input or one input mapping per tick."""

        if isinstance(inputs, Mapping):
            if steps is None or isinstance(steps, bool) or steps < 0:
                raise ControlRuntimeError("steps must be a non-negative integer")
            sequence = [inputs] * steps
        else:
            sequence = list(inputs)
            if steps is not None and steps != len(sequence):
                raise ControlRuntimeError("steps does not match input sequence length")
        return tuple(self.step(item, dt) for item in sequence)


# Concise aliases used by callers and older design documents.
InternalController = ControllerRuntime
ControlMachine = ControllerRuntime


def load_control_program(source: ControlProgram | Mapping[str, Any] | str | Path) -> ControlProgram:
    """Load a program from an object, JSON text, or JSON file (never Python)."""

    if isinstance(source, ControlProgram):
        return source
    if isinstance(source, Mapping):
        return ControlProgram.from_dict(source)
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, str):
        stripped = source.lstrip()
        text = source if stripped.startswith("{") else Path(source).read_text(encoding="utf-8")
    else:
        raise ControlValidationError("control source must be a mapping, JSON string, or path")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ControlValidationError(f"invalid control JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ControlValidationError("control JSON root must be an object")
    return ControlProgram.from_dict(data)


def execute_control_program(
    program: ControlProgram | Mapping[str, Any] | str | Path,
    inputs: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    dt: float,
    *,
    steps: int | None = None,
) -> tuple[ControlFrame, ...]:
    """Convenience entry point for a deterministic offline controller run."""

    return ControllerRuntime(load_control_program(program)).run(inputs, dt, steps)


__all__ = [
    "ControlError",
    "ControlFrame",
    "ControlMachine",
    "ControlProgram",
    "ControlRuntimeError",
    "ControlState",
    "ControlValidationError",
    "ControllerRuntime",
    "Expression",
    "InternalController",
    "RegisterSpec",
    "SignalSpec",
    "Transition",
    "evaluate_expression",
    "evaluate_state_outputs",
    "execute_control_program",
    "load_control_program",
]
