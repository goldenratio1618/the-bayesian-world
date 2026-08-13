"""Parser and interpreter for the restricted Physical Model DSL (PMDL).

PMDL files are strict JSON envelopes described by :class:`ModelSpec`.  Their
mathematics is a custom scalar expression language whose concrete syntax is a
safe subset of familiar mathematical Python notation.  ``ast.parse`` is used
only as a tokenizer/parser; every node is translated to immutable PMDL nodes,
and neither ``eval`` nor ``exec`` is used anywhere.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .specs import ModelSpec, SpecError
from .units import DIMENSIONLESS, TIME, Dimension


class DSLParseError(SpecError):
    """Raised for syntax or allow-list violations in PMDL."""


class ExpressionTypeError(DSLParseError):
    """Raised for type or dimension errors in a PMDL expression."""


@dataclass(frozen=True, slots=True)
class ExpressionType:
    kind: str
    dimension: Dimension = DIMENSIONLESS

    def __post_init__(self) -> None:
        if self.kind not in {"real", "boolean"}:
            raise ExpressionTypeError(f"unsupported expression type {self.kind!r}")
        if self.kind == "boolean" and self.dimension != DIMENSIONLESS:
            raise ExpressionTypeError("boolean expressions cannot carry dimensions")


REAL = ExpressionType("real")
BOOLEAN = ExpressionType("boolean")


class Expression:
    """Base class for immutable, executable-only-by-interpreter PMDL nodes."""

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def infer_type(self, symbols: Mapping[str, ExpressionType | Dimension]) -> ExpressionType:
        raise NotImplementedError

    def variables(self) -> frozenset[str]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Literal(Expression):
    value: float | bool

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        return self.value

    def infer_type(self, symbols: Mapping[str, ExpressionType | Dimension]) -> ExpressionType:
        return BOOLEAN if isinstance(self.value, bool) else REAL

    def variables(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True, slots=True)
class Symbol(Expression):
    name: str

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        if self.name in values:
            return values[self.name]
        if "." in self.name:
            parts = self.name.split(".")
            current: Any = values
            try:
                for part in parts:
                    current = current[part] if isinstance(current, Mapping) else getattr(current, part)
                return current
            except (KeyError, AttributeError, TypeError) as exc:
                raise DSLParseError(f"no value supplied for symbol {self.name!r}") from exc
        raise DSLParseError(f"no value supplied for symbol {self.name!r}")

    def infer_type(self, symbols: Mapping[str, ExpressionType | Dimension]) -> ExpressionType:
        if self.name in {"pi", "e"}:
            return REAL
        try:
            value = symbols[self.name]
        except KeyError as exc:
            raise ExpressionTypeError(f"unknown symbol {self.name!r}") from exc
        return ExpressionType("real", value) if isinstance(value, Dimension) else value

    def variables(self) -> frozenset[str]:
        return frozenset((self.name,)) if self.name not in {"pi", "e"} else frozenset()


@dataclass(frozen=True, slots=True)
class Unary(Expression):
    operator: str
    operand: Expression

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        value = self.operand.evaluate(values)
        if self.operator == "+":
            return +value
        if self.operator == "-":
            return -value
        if self.operator == "not":
            return np.logical_not(value)
        raise DSLParseError(f"unsupported unary operator {self.operator!r}")

    def infer_type(self, symbols: Mapping[str, ExpressionType | Dimension]) -> ExpressionType:
        operand = self.operand.infer_type(symbols)
        if self.operator == "not":
            _kind(operand, "boolean", "not")
            return BOOLEAN
        _kind(operand, "real", self.operator)
        return operand

    def variables(self) -> frozenset[str]:
        return self.operand.variables()


@dataclass(frozen=True, slots=True)
class Binary(Expression):
    operator: str
    left: Expression
    right: Expression

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        left, right = self.left.evaluate(values), self.right.evaluate(values)
        operations: dict[str, Callable[[Any, Any], Any]] = {
            "+": lambda a, b: a + b, "-": lambda a, b: a - b,
            "*": lambda a, b: a * b, "/": lambda a, b: a / b,
            "**": lambda a, b: a**b,
            "and": np.logical_and, "or": np.logical_or,
        }
        try:
            return operations[self.operator](left, right)
        except KeyError as exc:
            raise DSLParseError(f"unsupported binary operator {self.operator!r}") from exc

    def infer_type(self, symbols: Mapping[str, ExpressionType | Dimension]) -> ExpressionType:
        left, right = self.left.infer_type(symbols), self.right.infer_type(symbols)
        if self.operator in {"and", "or"}:
            _kind(left, "boolean", self.operator)
            _kind(right, "boolean", self.operator)
            return BOOLEAN
        _kind(left, "real", self.operator)
        _kind(right, "real", self.operator)
        if self.operator in {"+", "-"}:
            _same_dimension(left, right, self.operator)
            return left
        if self.operator == "*":
            return ExpressionType("real", left.dimension * right.dimension)
        if self.operator == "/":
            return ExpressionType("real", left.dimension / right.dimension)
        if self.operator == "**":
            if not isinstance(self.right, Literal) or isinstance(self.right.value, bool):
                raise ExpressionTypeError("dimensioned exponentiation requires a literal exponent")
            if not right.dimension.is_dimensionless:
                raise ExpressionTypeError("an exponent must be dimensionless")
            return ExpressionType("real", left.dimension ** float(self.right.value))
        raise ExpressionTypeError(f"unsupported binary operator {self.operator!r}")

    def variables(self) -> frozenset[str]:
        return self.left.variables() | self.right.variables()


@dataclass(frozen=True, slots=True)
class Comparison(Expression):
    operator: str
    left: Expression
    right: Expression

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        left, right = self.left.evaluate(values), self.right.evaluate(values)
        operations: dict[str, Callable[[Any, Any], Any]] = {
            "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
        }
        return operations[self.operator](left, right)

    def infer_type(self, symbols: Mapping[str, ExpressionType | Dimension]) -> ExpressionType:
        left, right = self.left.infer_type(symbols), self.right.infer_type(symbols)
        if left.kind != right.kind:
            raise ExpressionTypeError(f"comparison mixes {left.kind} and {right.kind}")
        if left.kind == "real":
            _same_dimension(left, right, self.operator)
        return BOOLEAN

    def variables(self) -> frozenset[str]:
        return self.left.variables() | self.right.variables()


@dataclass(frozen=True, slots=True)
class Call(Expression):
    function: str
    arguments: tuple[Expression, ...]

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        if self.function == "der":
            argument = self.arguments[0]
            if not isinstance(argument, Symbol):
                raise DSLParseError("der() requires a state symbol")
            return Symbol(f"{argument.name}_dot").evaluate(values)
        args = tuple(argument.evaluate(values) for argument in self.arguments)
        functions: dict[str, Callable[..., Any]] = {
            "abs": np.abs, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "tanh": np.tanh,
            "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
            "exp": np.exp, "log": np.log, "log10": np.log10, "min": np.minimum,
            "max": np.maximum, "clip": np.clip, "sign": np.sign, "where": np.where,
            "smooth_abs": lambda x, epsilon=1e-12: np.sqrt(x * x + epsilon * epsilon),
        }
        try:
            return functions[self.function](*args)
        except KeyError as exc:
            raise DSLParseError(f"unsupported function {self.function!r}") from exc

    def infer_type(self, symbols: Mapping[str, ExpressionType | Dimension]) -> ExpressionType:
        args = tuple(argument.infer_type(symbols) for argument in self.arguments)
        _arity(self.function, len(args))
        if self.function == "der":
            if not isinstance(self.arguments[0], Symbol):
                raise ExpressionTypeError("der() requires a state symbol")
            _kind(args[0], "real", "der")
            return ExpressionType("real", args[0].dimension / TIME)
        if self.function in {"sin", "cos", "tan", "tanh", "exp", "log", "log10"}:
            _dimensionless(args[0], self.function)
            return REAL
        if self.function in {"asin", "acos", "atan"}:
            _dimensionless(args[0], self.function)
            return REAL
        if self.function == "atan2":
            _same_dimension(args[0], args[1], "atan2")
            return REAL
        if self.function == "sqrt":
            _kind(args[0], "real", "sqrt")
            return ExpressionType("real", args[0].dimension ** 0.5)
        if self.function in {"abs", "smooth_abs"}:
            _kind(args[0], "real", self.function)
            if len(args) == 2:
                _same_dimension(args[0], args[1], self.function)
            return args[0]
        if self.function in {"min", "max"}:
            _same_dimension(args[0], args[1], self.function)
            return args[0]
        if self.function == "clip":
            _same_dimension(args[0], args[1], "clip")
            _same_dimension(args[0], args[2], "clip")
            return args[0]
        if self.function == "sign":
            _kind(args[0], "real", "sign")
            return REAL
        if self.function == "where":
            _kind(args[0], "boolean", "where")
            if args[1].kind != args[2].kind:
                raise ExpressionTypeError("where branches must have the same type")
            if args[1].kind == "real":
                _same_dimension(args[1], args[2], "where")
            return args[1]
        raise ExpressionTypeError(f"unsupported function {self.function!r}")

    def variables(self) -> frozenset[str]:
        result = frozenset()
        for argument in self.arguments:
            result |= argument.variables()
        return result


@dataclass(frozen=True, slots=True)
class Conditional(Expression):
    condition: Expression
    when_true: Expression
    when_false: Expression

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        return np.where(self.condition.evaluate(values), self.when_true.evaluate(values), self.when_false.evaluate(values))

    def infer_type(self, symbols: Mapping[str, ExpressionType | Dimension]) -> ExpressionType:
        condition = self.condition.infer_type(symbols)
        _kind(condition, "boolean", "conditional")
        left, right = self.when_true.infer_type(symbols), self.when_false.infer_type(symbols)
        if left.kind != right.kind:
            raise ExpressionTypeError("conditional branches must have the same type")
        if left.kind == "real":
            _same_dimension(left, right, "conditional")
        return left

    def variables(self) -> frozenset[str]:
        return self.condition.variables() | self.when_true.variables() | self.when_false.variables()


_ARITIES: dict[str, tuple[int, ...]] = {
    "abs": (1,), "sqrt": (1,), "sin": (1,), "cos": (1,), "tan": (1,),
    "tanh": (1,),
    "asin": (1,), "acos": (1,), "atan": (1,), "atan2": (2,), "exp": (1,),
    "log": (1,), "log10": (1,), "min": (2,), "max": (2,), "clip": (3,),
    "sign": (1,), "where": (3,), "smooth_abs": (1, 2), "der": (1,),
}


def _arity(function: str, count: int) -> None:
    allowed = _ARITIES.get(function)
    if allowed is None or count not in allowed:
        expected = " or ".join(str(value) for value in (allowed or ()))
        raise ExpressionTypeError(f"{function}() expects {expected or 'no allowed'} argument(s), got {count}")


def _kind(value: ExpressionType, expected: str, operation: str) -> None:
    if value.kind != expected:
        raise ExpressionTypeError(f"{operation} expects {expected}, got {value.kind}")


def _same_dimension(left: ExpressionType, right: ExpressionType, operation: str) -> None:
    _kind(left, "real", operation)
    _kind(right, "real", operation)
    if left.dimension != right.dimension:
        raise ExpressionTypeError(
            f"dimension mismatch in {operation}: {left.dimension.describe()} vs {right.dimension.describe()}"
        )


def _dimensionless(value: ExpressionType, operation: str) -> None:
    _kind(value, "real", operation)
    if not value.dimension.is_dimensionless:
        raise ExpressionTypeError(f"{operation} requires a dimensionless argument, got {value.dimension.describe()}")


class _ExpressionBuilder:
    def __init__(self, source: str, *, max_nodes: int = 256, max_depth: int = 32) -> None:
        self.source, self.max_nodes, self.max_depth = source, max_nodes, max_depth
        self.nodes = 0

    def build(self, node: ast.AST, depth: int = 0) -> Expression:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise DSLParseError(f"expression exceeds {self.max_nodes} nodes")
        if depth > self.max_depth:
            raise DSLParseError(f"expression exceeds depth {self.max_depth}")
        child = lambda item: self.build(item, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise DSLParseError("non-finite literals are forbidden")
            return Literal(node.value if isinstance(node.value, bool) else float(node.value))
        if isinstance(node, ast.Name):
            _safe_name(node.id)
            return Symbol(node.id)
        if isinstance(node, ast.Attribute):
            return Symbol(_attribute_name(node))
        if isinstance(node, ast.UnaryOp):
            operators = {ast.UAdd: "+", ast.USub: "-", ast.Not: "not"}
            operator = operators.get(type(node.op))
            if operator is None:
                raise DSLParseError(f"operator {type(node.op).__name__} is forbidden")
            return Unary(operator, child(node.operand))
        if isinstance(node, ast.BinOp):
            operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
            operator = operators.get(type(node.op))
            if operator is None:
                raise DSLParseError(f"operator {type(node.op).__name__} is forbidden")
            return Binary(operator, child(node.left), child(node.right))
        if isinstance(node, ast.BoolOp):
            operator = "and" if isinstance(node.op, ast.And) else "or" if isinstance(node.op, ast.Or) else None
            if operator is None:
                raise DSLParseError(f"operator {type(node.op).__name__} is forbidden")
            values = [child(value) for value in node.values]
            result = values[0]
            for value in values[1:]:
                result = Binary(operator, result, value)
            return result
        if isinstance(node, ast.Compare):
            operators = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
            expressions: list[Expression] = []
            left = child(node.left)
            for operator_node, comparator in zip(node.ops, node.comparators):
                right = child(comparator)
                operator = operators.get(type(operator_node))
                if operator is None:
                    raise DSLParseError(f"comparison {type(operator_node).__name__} is forbidden")
                expressions.append(Comparison(operator, left, right))
                left = right
            result = expressions[0]
            for expression in expressions[1:]:
                result = Binary("and", result, expression)
            return result
        if isinstance(node, ast.Call):
            if node.keywords or not isinstance(node.func, ast.Name):
                raise DSLParseError("only direct allow-listed calls without keyword arguments are permitted")
            if node.func.id not in _ARITIES:
                raise DSLParseError(f"function {node.func.id!r} is not allow-listed")
            _arity(node.func.id, len(node.args))
            return Call(node.func.id, tuple(child(argument) for argument in node.args))
        if isinstance(node, ast.IfExp):
            return Conditional(child(node.test), child(node.body), child(node.orelse))
        raise DSLParseError(f"syntax node {type(node).__name__} is forbidden")


def _safe_name(name: str) -> str:
    if not name or name.startswith("_") or not name.replace("_", "a").isalnum() or not (name[0].isalpha()):
        raise DSLParseError(f"unsafe symbol name {name!r}")
    return name


def _attribute_name(node: ast.Attribute) -> str:
    parts = [node.attr]
    current: ast.AST = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        raise DSLParseError("attribute access is limited to dotted PMDL symbols")
    parts.append(current.id)
    for part in parts:
        _safe_name(part)
    return ".".join(reversed(parts))


@lru_cache(maxsize=4096)
def parse_expression(source: str) -> Expression:
    """Parse an expression into an immutable allow-listed tree."""

    if not isinstance(source, str) or not source.strip():
        raise DSLParseError("expression must be a non-empty string")
    if len(source) > 16_384:
        raise DSLParseError("expression exceeds 16384 characters")
    try:
        parsed = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError) as exc:
        location = f" at column {exc.offset}" if isinstance(exc, SyntaxError) and exc.offset else ""
        raise DSLParseError(f"invalid expression syntax{location}: {exc.msg if isinstance(exc, SyntaxError) else exc}") from exc
    return _ExpressionBuilder(source).build(parsed.body)


def evaluate_expression(source: str | Expression, values: Mapping[str, Any]) -> Any:
    expression = parse_expression(source) if isinstance(source, str) else source
    environment = {"pi": math.pi, "e": math.e, **values}
    return expression.evaluate(environment)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DSLParseError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_model(source: str | bytes | Mapping[str, Any], *, source_name: str = "<memory>") -> ModelSpec:
    """Parse a PMDL JSON document and eagerly parse every expression."""

    if isinstance(source, Mapping):
        data = source
    else:
        try:
            data = json.loads(source, object_pairs_hook=_reject_duplicate_pairs)
        except DSLParseError:
            raise
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            message = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise DSLParseError(f"invalid PMDL JSON in {source_name}: {message}") from exc
    if not isinstance(data, Mapping):
        raise DSLParseError(f"PMDL root in {source_name} must be an object")
    try:
        model = ModelSpec.from_dict(data)
        for relation in model.relations:
            parse_expression(relation.expression)
        for collection in (model.stored_energy, model.dissipation, model.sources):
            for item in collection:
                parse_expression(item.expression)
        for increment in model.process_noise.increments:
            parse_expression(increment.expression)
        for constraint in model.initialization.constraints:
            parse_expression(constraint.expression)
        for mode in model.modes:
            for transition in mode.transitions:
                parse_expression(transition.guard)
                for reset in transition.resets.values():
                    parse_expression(reset)
        for property_spec in model.properties:
            parse_expression(property_spec.expression)
    except SpecError as exc:
        raise DSLParseError(f"{source_name}: {exc}") from exc
    return model


def load_model(path: str | Path) -> ModelSpec:
    model_path = Path(path)
    try:
        source = model_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DSLParseError(f"could not read {model_path}: {exc}") from exc
    return parse_model(source, source_name=str(model_path))


def dump_model(model: ModelSpec, path: str | Path | None = None, *, indent: int = 2) -> str:
    source = model.to_json(indent=indent) + "\n"
    if path is not None:
        Path(path).write_text(source, encoding="utf-8")
    return source


def _vector_mapping(value: Any, names: Sequence[str], defaults: Sequence[Any] | None, context: str) -> dict[str, Any]:
    if value is None:
        if defaults is None:
            return {}
        return dict(zip(names, defaults))
    if isinstance(value, Mapping):
        return dict(value)
    try:
        length = len(value)
    except TypeError as exc:
        raise DSLParseError(f"{context} must be a mapping or sequence") from exc
    if length != len(names):
        raise DSLParseError(f"{context} has length {length}; expected {len(names)}")
    return dict(zip(names, value))


def evaluate_model_residual(model: ModelSpec, t: Any, z: Any, zdot: Any, theta: Any = None, u: Any = None) -> Any:
    """Evaluate ``F(t,z,zdot,theta,u)`` using the safe expression interpreter.

    Port/algebraic values may be supplied by name in ``u``; this is convenient
    for isolated component tests.  A network assembler normally supplies them
    as entries in ``z`` or in the assembled environment.
    """

    z_names = model.state_names + model.algebraic_names
    z_defaults = tuple(state.initial for state in model.states) + tuple(variable.initial for variable in model.algebraics)
    environment: dict[str, Any] = {"t": t}
    environment.update(_vector_mapping(z, z_names, z_defaults, "z"))
    derivative_names = tuple((state.derivative or f"{state.name}_dot") for state in model.states)
    environment.update(_vector_mapping(zdot, derivative_names, (0.0,) * len(derivative_names), "zdot"))
    parameter_defaults = tuple(parameter.default for parameter in model.parameters)
    environment.update(_vector_mapping(theta, model.parameter_names, parameter_defaults, "theta"))
    environment.update(_vector_mapping(u, model.input_names, (0.0,) * len(model.input_names), "u"))
    values = [evaluate_expression(relation.expression, environment) for relation in model.relations]
    if not values:
        return np.empty((0,), dtype=float)
    first = values[0]
    module = type(first).__module__.split(".", 1)[0]
    if module == "torch":
        import torch  # type: ignore[import-not-found]
        return torch.stack(tuple(values), dim=-1)
    try:
        return np.stack(values, axis=-1)
    except (TypeError, ValueError):
        return np.asarray(values)


class ModelRegistry(Mapping[str, ModelSpec]):
    """Explicit model registry with duplicate protection and directory loading."""

    def __init__(self, models: Iterable[ModelSpec] = ()) -> None:
        self._models: dict[str, ModelSpec] = {}
        for model in models:
            self.register(model)

    def register(
        self,
        model: ModelSpec,
        *,
        validate: bool = True,
        interfaces: Any = None,
    ) -> ModelSpec:
        if model.id in self._models:
            raise SpecError(f"model id {model.id!r} is already registered")
        if validate:
            from .validation import validate_model
            validate_model(model, interfaces).require_valid()
        self._models[model.id] = model
        return model

    def load(
        self, path: str | Path, *, validate: bool = True, interfaces: Any = None
    ) -> ModelSpec:
        return self.register(load_model(path), validate=validate, interfaces=interfaces)

    def load_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
        validate: bool = True,
        interfaces: Any = None,
    ) -> tuple[ModelSpec, ...]:
        root = Path(directory)
        if not recursive:
            raise SpecError("model catalogs are always loaded recursively")
        from ..catalog.interfaces import concrete_model_paths, load_interface_catalog
        from .validation import validate_model

        catalog = interfaces or load_interface_catalog(root)
        loaded: list[ModelSpec] = []
        for path in concrete_model_paths(root):
            model = self.load(path, validate=False)
            if validate:
                catalog.validate_model_path(model, path, root)
                validate_model(model, catalog).require_valid()
            loaded.append(model)
        return tuple(loaded)

    def __getitem__(self, key: str) -> ModelSpec:
        return self._models[key]

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._models))

    def __len__(self) -> int:
        return len(self._models)

    def to_dict(self) -> dict[str, Any]:
        return {key: self._models[key].to_dict() for key in sorted(self._models)}
