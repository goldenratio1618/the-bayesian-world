"""Synthesizable signed fixed-point Verilog-2001 controller generator."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np

from ...physics.dsl import Binary, Call, Comparison, Conditional, Expression, Literal, Symbol, Unary, parse_expression
from ..specs import ControlSpec
from .ir import (
    ControlCompilerError,
    ControlIR,
    FixedPointFormat,
    GeneratedArtifact,
    admit_target_symbol_table,
    as_ir,
    plain_compiler_data,
)


@dataclass(frozen=True, slots=True)
class _Interval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise ValueError("invalid fixed-point interval")


class _FixedPointAdmission:
    """Conservative domain and intermediate-range proof for controller graphs."""

    def __init__(self, ir: ControlIR, fixed: FixedPointFormat) -> None:
        self.ir = ir
        self.spec = ir.spec
        self.fixed = fixed
        # Reserve the two rails as sticky runtime arithmetic-fault sentinels.
        self.safe_lower = (fixed.minimum + 1) / fixed.scale
        self.safe_upper = (fixed.maximum - 1) / fixed.scale
        self.environment: dict[str, _Interval | None] = {
            "time": _Interval(self.safe_lower, self.safe_upper),
            "time_in_mode": _Interval(self.safe_lower, self.safe_upper),
            "dt": self._constant(self.spec.period_s, "controller period"),
        }
        for item in self.spec.explicit_inputs:
            self.environment[f"input.{item.name}"] = (
                None
                if item.dtype == "bool"
                else self._declared(item.bounds.lower, item.bounds.upper, f"input {item.name}")
            )
        for item in self.spec.outputs:
            self.environment[f"output.{item.name}"] = (
                None
                if item.dtype == "bool"
                else self._declared(item.bounds.lower, item.bounds.upper, f"output {item.name}")
            )
        for item in self.spec.registers:
            self.environment[f"register.{item.name}"] = (
                None
                if item.dtype == "bool"
                else self._declared(item.bounds.lower, item.bounds.upper, f"register {item.name}")
            )
        for item in self.spec.parameters:
            self.environment[f"parameter.{item.name}"] = (
                None
                if item.dtype == "bool"
                else self._parameter_constant(item)
            )
        for item in self.spec.implicit_inputs:
            mean = self._declared(
                item.bounds.lower,
                item.bounds.upper,
                f"implicit input {item.name}.mean",
            )
            variance = _Interval(0.0, self.safe_upper)
            self.environment[f"implicit.{item.name}.mean"] = mean
            self.environment[f"implicit.{item.name}.variance"] = variance
            self.environment[f"implicit.{item.name}.std"] = _Interval(
                0.0, math.sqrt(variance.upper)
            )

    def _constant(self, value: float, context: str) -> _Interval:
        integer = self.fixed.quantize(value)
        if value != 0.0 and integer == 0:
            raise ControlCompilerError(
                f"{context}: nonzero constant {value:.17g} quantizes to zero; "
                "increase fixed-point precision"
            )
        if integer in {self.fixed.minimum, self.fixed.maximum}:
            raise ControlCompilerError(
                f"{context}: fixed-point rail values are reserved for arithmetic faults"
            )
        result = integer / self.fixed.scale
        return _Interval(result, result)

    def _lower_bound_integer(self, value: float, context: str) -> int:
        integer = math.ceil(float(value) * self.fixed.scale)
        if integer <= self.fixed.minimum or integer >= self.fixed.maximum:
            raise ControlCompilerError(
                f"{context}: directed lower bound reaches a reserved fixed-point rail"
            )
        return integer

    def _upper_bound_integer(self, value: float, context: str) -> int:
        integer = math.floor(float(value) * self.fixed.scale)
        if integer <= self.fixed.minimum or integer >= self.fixed.maximum:
            raise ControlCompilerError(
                f"{context}: directed upper bound reaches a reserved fixed-point rail"
            )
        return integer

    def _declared(
        self, lower: float | None, upper: float | None, context: str
    ) -> _Interval:
        lower_value = self.safe_lower if lower is None else float(lower)
        upper_value = self.safe_upper if upper is None else float(upper)
        if lower is not None:
            lower_value = self._lower_bound_integer(
                lower_value, f"{context} lower bound"
            ) / self.fixed.scale
        if upper is not None:
            upper_value = self._upper_bound_integer(
                upper_value, f"{context} upper bound"
            ) / self.fixed.scale
        if lower_value > upper_value:
            raise ControlCompilerError(
                f"{context}: quantized bounds are reversed"
            )
        result = _Interval(lower_value, upper_value)
        return self._check(result, context)

    def _bounded_constant(
        self,
        value: float,
        lower: float | None,
        upper: float | None,
        context: str,
    ) -> int:
        """Admit one emitted constant and its emitted quantized bounds."""

        self._constant(value, context)
        quantized = self.fixed.quantize(value)
        lower_q = None
        upper_q = None
        if lower is not None:
            lower_q = self._lower_bound_integer(lower, f"{context} lower bound")
        if upper is not None:
            upper_q = self._upper_bound_integer(upper, f"{context} upper bound")
        if (lower_q is not None and quantized < lower_q) or (
            upper_q is not None and quantized > upper_q
        ):
            raise ControlCompilerError(
                f"{context}: quantized value {quantized} is outside its quantized "
                f"authored bounds [{lower_q}, {upper_q}]"
            )
        return quantized

    def _parameter_constant(self, item: object) -> _Interval:
        value = float(item.default)
        self._bounded_constant(
            value,
            item.bounds.lower,
            item.bounds.upper,
            f"parameter {item.name} default",
        )
        return self._constant(value, f"parameter {item.name} default")

    def _positive_limit_integer(self, value: float, context: str) -> int:
        if not math.isfinite(value) or value <= 0.0:
            raise ControlCompilerError(f"{context}: limit must be finite and positive")
        integer = math.floor(value * self.fixed.scale)
        if integer == 0:
            raise ControlCompilerError(
                f"{context}: positive limit {value:.17g} quantizes to zero; "
                "increase fixed-point precision"
            )
        if integer >= self.fixed.maximum:
            raise ControlCompilerError(
                f"{context}: directed positive limit reaches a reserved fixed-point rail"
            )
        return integer

    def _admit_generated_constants(self) -> None:
        for item in self.spec.explicit_inputs:
            if item.dtype == "real":
                if item.bounds.lower is not None:
                    self._lower_bound_integer(item.bounds.lower, f"input {item.name} lower bound")
                if item.bounds.upper is not None:
                    self._upper_bound_integer(item.bounds.upper, f"input {item.name} upper bound")
        for item in self.spec.outputs:
            if item.dtype != "real":
                continue
            default_q = self._bounded_constant(
                float(item.default),
                item.bounds.lower,
                item.bounds.upper,
                f"output {item.name} default",
            )
            if item.emergency_value is not None:
                self._bounded_constant(
                    float(item.emergency_value),
                    item.bounds.lower,
                    item.bounds.upper,
                    f"output {item.name} emergency_value",
                )
            if item.slew_rate is not None:
                delta = float(item.slew_rate) * self.spec.period_s
                delta_q = self._positive_limit_integer(
                    delta, f"output {item.name} slew delta"
                )
                lower_q = (
                    self.fixed.minimum + 1
                    if item.bounds.lower is None
                    else self._lower_bound_integer(item.bounds.lower, f"output {item.name} lower bound")
                )
                upper_q = (
                    self.fixed.maximum - 1
                    if item.bounds.upper is None
                    else self._upper_bound_integer(item.bounds.upper, f"output {item.name} upper bound")
                )
                if (
                    lower_q - delta_q <= self.fixed.minimum
                    or upper_q + delta_q >= self.fixed.maximum
                ):
                    raise ControlCompilerError(
                        f"output {item.name} slew intermediate: q_sub/q_add over "
                        "the quantized output bounds can reach a reserved arithmetic-fault rail"
                    )
                if not lower_q <= default_q <= upper_q:  # defensive, with a focused diagnostic
                    raise ControlCompilerError(
                        f"output {item.name} default is outside quantized output bounds"
                    )
        for item in self.spec.registers:
            if item.dtype == "real":
                self._bounded_constant(
                    float(item.initial),
                    item.bounds.lower,
                    item.bounds.upper,
                    f"register {item.name} initial",
                )
        for item in self.spec.parameters:
            if item.dtype == "real":
                self._bounded_constant(
                    float(item.default),
                    item.bounds.lower,
                    item.bounds.upper,
                    f"parameter {item.name} default",
                )
        for item in self.spec.implicit_inputs:
            if item.bounds.lower is not None:
                self._lower_bound_integer(item.bounds.lower, f"implicit input {item.name} lower bound")
            if item.bounds.upper is not None:
                self._upper_bound_integer(item.bounds.upper, f"implicit input {item.name} upper bound")
        if self.ir.observer is not None:
            observer = self.ir.observer
            output_defaults = {
                output.name: float(output.default) for output in self.spec.outputs
            }
            initial_input = np.asarray(
                [output_defaults[name] for name in observer.input_names], dtype=float
            )
            initial_means = (
                observer.L @ observer.initial_state
                + observer.M @ initial_input
                + observer.latent_bias
            )
            initial_variances = np.einsum(
                "ij,jk,ik->i",
                observer.L,
                observer.initial_covariance,
                observer.L,
            )
            for index, name in enumerate(observer.latent_names):
                self._bounded_constant(
                    float(initial_means[index]),
                    observer.latent_lower_bounds[index],
                    observer.latent_upper_bounds[index],
                    f"implicit input {name} initial mean",
                )
                self._constant(
                    float(initial_variances[index]),
                    f"implicit input {name} initial variance",
                )

    def _check(self, value: _Interval, context: str) -> _Interval:
        if (
            not math.isfinite(value.lower)
            or not math.isfinite(value.upper)
            or value.lower < self.safe_lower
            or value.upper > self.safe_upper
        ):
            raise ControlCompilerError(
                f"{context}: interval [{value.lower:.17g}, {value.upper:.17g}] "
                f"cannot be proven inside the non-fault Q{self.fixed.total_bits - self.fixed.fractional_bits - 1}."
                f"{self.fixed.fractional_bits} range [{self.safe_lower:.17g}, {self.safe_upper:.17g}]"
            )
        return value

    def _numeric(self, value: _Interval | None, context: str) -> _Interval:
        if value is None:
            raise ControlCompilerError(
                f"{context}: a discrete boolean cannot enter a fixed-point numeric path"
            )
        return value

    def _refined_environments(
        self, condition: Expression
    ) -> tuple[dict[str, _Interval | None], dict[str, _Interval | None]]:
        when_true = dict(self.environment)
        when_false = dict(self.environment)
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
            or self.environment.get(symbol.name) is None
        ):
            return when_true, when_false
        current = self.environment[symbol.name]
        assert current is not None
        boundary = float(literal.value)
        if operator in {">", ">="}:
            when_true[symbol.name] = _Interval(
                max(current.lower, boundary), current.upper
            )
            when_false[symbol.name] = _Interval(
                current.lower, min(current.upper, boundary)
            )
        elif operator in {"<", "<="}:
            when_true[symbol.name] = _Interval(
                current.lower, min(current.upper, boundary)
            )
            when_false[symbol.name] = _Interval(
                max(current.lower, boundary), current.upper
            )
        return when_true, when_false

    def _add(self, left: _Interval, right: _Interval, context: str) -> _Interval:
        return self._check(
            _Interval(left.lower + right.lower, left.upper + right.upper), context
        )

    def _subtract(self, left: _Interval, right: _Interval, context: str) -> _Interval:
        return self._check(
            _Interval(left.lower - right.upper, left.upper - right.lower), context
        )

    def _multiply(self, left: _Interval, right: _Interval, context: str) -> _Interval:
        products = (
            left.lower * right.lower,
            left.lower * right.upper,
            left.upper * right.lower,
            left.upper * right.upper,
        )
        return self._check(_Interval(min(products), max(products)), context)

    def _power(self, base: _Interval, exponent: int, context: str) -> _Interval:
        if exponent == 0:
            return self._constant(1.0, context)
        if exponent % 2:
            result = _Interval(base.lower**exponent, base.upper**exponent)
        else:
            lower = (
                0.0
                if base.lower <= 0.0 <= base.upper
                else min(base.lower**exponent, base.upper**exponent)
            )
            result = _Interval(
                lower, max(base.lower**exponent, base.upper**exponent)
            )
        return self._check(result, context)

    def _divide(self, left: _Interval, right: _Interval, context: str) -> _Interval:
        if right.lower <= 0.0 <= right.upper:
            raise ControlCompilerError(
                f"{context}: divisor interval [{right.lower:.17g}, {right.upper:.17g}] "
                "does not exclude zero"
            )
        quotients = (
            left.lower / right.lower,
            left.lower / right.upper,
            left.upper / right.lower,
            left.upper / right.upper,
        )
        return self._check(_Interval(min(quotients), max(quotients)), context)

    def evaluate(self, source: str | Expression, context: str) -> _Interval | None:
        node = parse_expression(source) if isinstance(source, str) else source

        def visit(item: Expression, path: str) -> _Interval | None:
            if isinstance(item, Literal):
                return None if isinstance(item.value, bool) else self._constant(float(item.value), path)
            if isinstance(item, Symbol):
                if item.name == "pi":
                    return self._constant(math.pi, path)
                if item.name == "e":
                    return self._constant(math.e, path)
                try:
                    return self.environment[item.name]
                except KeyError as exc:
                    raise ControlCompilerError(
                        f"{path}: no fixed-point interval for symbol {item.name!r}"
                    ) from exc
            if isinstance(item, Unary):
                operand = visit(item.operand, f"{path}.{item.operator}")
                if item.operator == "not":
                    return None
                numeric = self._numeric(operand, path)
                if item.operator == "+":
                    return numeric
                if item.operator == "-":
                    return self._check(_Interval(-numeric.upper, -numeric.lower), path)
                raise ControlCompilerError(f"{path}: unsupported unary operator {item.operator!r}")
            if isinstance(item, Binary):
                left = visit(item.left, f"{path}.left")
                right = visit(item.right, f"{path}.right")
                if item.operator in {"and", "or"}:
                    return None
                left_numeric = self._numeric(left, path)
                right_numeric = self._numeric(right, path)
                if item.operator == "+":
                    return self._add(left_numeric, right_numeric, path)
                if item.operator == "-":
                    return self._subtract(left_numeric, right_numeric, path)
                if item.operator == "*":
                    return self._multiply(left_numeric, right_numeric, path)
                if item.operator == "/":
                    return self._divide(left_numeric, right_numeric, path)
                if item.operator == "**":
                    if not isinstance(item.right, Literal) or isinstance(item.right.value, bool):
                        raise ControlCompilerError(f"{path}: fixed-point powers require a literal exponent")
                    exponent = int(item.right.value)
                    result = self._constant(1.0, path)
                    for index in range(1, exponent + 1):
                        result = self._power(
                            left_numeric, index, f"{path}.power[{index}]"
                        )
                    return result
                raise ControlCompilerError(f"{path}: unsupported binary operator {item.operator!r}")
            if isinstance(item, Comparison):
                visit(item.left, f"{path}.left")
                visit(item.right, f"{path}.right")
                return None
            if isinstance(item, Conditional):
                visit(item.condition, f"{path}.condition")
                original = self.environment
                true_environment, false_environment = self._refined_environments(
                    item.condition
                )
                try:
                    self.environment = true_environment
                    when_true = visit(item.when_true, f"{path}.true")
                    self.environment = false_environment
                    when_false = visit(item.when_false, f"{path}.false")
                finally:
                    self.environment = original
                if when_true is None or when_false is None:
                    return None
                return self._check(
                    _Interval(
                        min(when_true.lower, when_false.lower),
                        max(when_true.upper, when_false.upper),
                    ),
                    path,
                )
            if isinstance(item, Call):
                if item.function == "where":
                    visit(item.arguments[0], f"{path}.where_condition")
                    original = self.environment
                    true_environment, false_environment = self._refined_environments(
                        item.arguments[0]
                    )
                    try:
                        self.environment = true_environment
                        left = visit(item.arguments[1], f"{path}.where_true")
                        self.environment = false_environment
                        right = visit(item.arguments[2], f"{path}.where_false")
                    finally:
                        self.environment = original
                    if left is None or right is None:
                        return None
                    return self._check(
                        _Interval(min(left.lower, right.lower), max(left.upper, right.upper)),
                        path,
                    )
                arguments = [
                    visit(argument, f"{path}.{item.function}[{index}]")
                    for index, argument in enumerate(item.arguments)
                ]
                numeric = [self._numeric(value, path) for value in arguments]
                if item.function == "abs":
                    source_interval = numeric[0]
                    maximum = max(abs(source_interval.lower), abs(source_interval.upper))
                    minimum = 0.0 if source_interval.lower <= 0.0 <= source_interval.upper else min(abs(source_interval.lower), abs(source_interval.upper))
                    return self._check(_Interval(minimum, maximum), path)
                if item.function == "sqrt":
                    if numeric[0].lower < 0.0:
                        raise ControlCompilerError(
                            f"{path}: sqrt argument interval [{numeric[0].lower:.17g}, "
                            f"{numeric[0].upper:.17g}] includes negative values"
                        )
                    return self._check(
                        _Interval(math.sqrt(numeric[0].lower), math.sqrt(numeric[0].upper)),
                        path,
                    )
                if item.function == "min":
                    return self._check(
                        _Interval(min(numeric[0].lower, numeric[1].lower), min(numeric[0].upper, numeric[1].upper)),
                        path,
                    )
                if item.function == "max":
                    return self._check(
                        _Interval(max(numeric[0].lower, numeric[1].lower), max(numeric[0].upper, numeric[1].upper)),
                        path,
                    )
                if item.function == "clip":
                    if numeric[1].upper > numeric[2].lower:
                        raise ControlCompilerError(
                            f"{path}: clip lower/upper intervals are not provably ordered"
                        )
                    return self._check(
                        _Interval(
                            min(
                                max(numeric[0].lower, numeric[1].lower),
                                numeric[2].lower,
                            ),
                            min(
                                max(numeric[0].upper, numeric[1].upper),
                                numeric[2].upper,
                            ),
                        ),
                        path,
                    )
                if item.function == "smooth_abs":
                    epsilon = numeric[1] if len(numeric) == 2 else self._constant(1e-12, path)
                    square = self._power(numeric[0], 2, f"{path}.square")
                    epsilon_square = self._power(epsilon, 2, f"{path}.epsilon_square")
                    summed = self._add(square, epsilon_square, f"{path}.sum")
                    if summed.lower < 0.0:
                        raise ControlCompilerError(f"{path}: smooth_abs radicand is not non-negative")
                    return self._check(_Interval(math.sqrt(max(0.0, summed.lower)), math.sqrt(summed.upper)), path)
                raise ControlCompilerError(
                    f"{path}: function {item.function!r} has no admitted fixed-point lowering"
                )
            raise ControlCompilerError(
                f"{path}: unsupported fixed-point expression node {type(item).__name__}"
            )

        return visit(node, context)

    def admit(self) -> None:
        self._admit_generated_constants()
        timed = [
            expression.path
            for expression in self.ir.expressions
            if expression.symbols & {"time", "time_in_mode"}
        ]
        if timed:
            raise ControlCompilerError(
                "fixed-point Verilog requires an explicit finite execution horizon "
                f"for time/time_in_mode expressions; unsupported paths={timed}"
            )
        for item in self.spec.derived:
            value = self.evaluate(item.expression, f"derived {item.name}")
            self.environment[f"derived.{item.name}"] = value
        if self.spec.emergency_when is not None:
            self.evaluate(self.spec.emergency_when, "emergency_when")
        for mode in self.spec.modes:
            for name, source in mode.outputs.items():
                self.evaluate(source, f"mode {mode.name}.outputs.{name}")
            for name, source in mode.updates.items():
                self.evaluate(source, f"mode {mode.name}.updates.{name}")
            for index, transition in enumerate(mode.transitions):
                self.evaluate(
                    transition.guard,
                    f"mode {mode.name}.transitions[{index}].guard",
                )


class _VerilogExpression:
    def __init__(self, ir: ControlIR, fixed: FixedPointFormat) -> None:
        self.ir = ir
        self.spec = ir.spec
        self.fixed = fixed
        self.parameters = {item.name: item for item in self.spec.parameters}

    def integer_literal(self, integer: int) -> str:
        if integer < 0:
            return f"-{self.fixed.total_bits}'sd{abs(integer)}"
        return f"{self.fixed.total_bits}'sd{integer}"

    def constant(self, value: float) -> str:
        return self.integer_literal(self.fixed.quantize(value))

    def lower_bound(self, value: float) -> str:
        return self.integer_literal(math.ceil(float(value) * self.fixed.scale))

    def upper_bound(self, value: float) -> str:
        return self.integer_literal(math.floor(float(value) * self.fixed.scale))

    def positive_limit(self, value: float) -> str:
        integer = math.floor(float(value) * self.fixed.scale)
        if integer <= 0 or integer >= self.fixed.maximum:
            raise ControlCompilerError(
                "positive fixed-point limit is not representable without loss"
            )
        return self.integer_literal(integer)

    def symbol(self, name: str) -> str:
        if name == "pi":
            return self.constant(3.141592653589793)
        if name == "e":
            return self.constant(2.718281828459045)
        if name == "time":
            return "time_q"
        if name == "time_in_mode":
            return "time_in_mode_q"
        if name == "dt":
            return "PERIOD_Q"
        parts = name.split(".")
        if len(parts) == 2 and parts[0] == "input":
            return f"explicit_{parts[1]}"
        if len(parts) == 2 and parts[0] == "output":
            return f"output_{parts[1]}_state"
        if len(parts) == 2 and parts[0] == "register":
            return f"register_{parts[1]}"
        if len(parts) == 2 and parts[0] == "parameter":
            try:
                value = self.parameters[parts[1]].default
            except KeyError as exc:
                raise ControlCompilerError(f"unknown parameter symbol {name!r}") from exc
            if isinstance(value, bool):
                return "1'b1" if value else "1'b0"
            return self.constant(value)
        if len(parts) == 2 and parts[0] == "derived":
            return f"derived_{parts[1]}"
        if len(parts) == 3 and parts[0] == "implicit":
            if parts[2] == "mean":
                return f"implicit_{parts[1]}_mean_next"
            if parts[2] == "variance":
                return f"implicit_{parts[1]}_variance_next"
            if parts[2] == "std":
                return f"q_sqrt(implicit_{parts[1]}_variance_next)"
        raise ControlCompilerError(
            f"cannot lower control symbol {name!r} to fixed-point Verilog"
        )

    def _power(self, base: str, exponent: Literal) -> str:
        if isinstance(exponent.value, bool) or int(exponent.value) != exponent.value:
            raise ControlCompilerError(
                "Verilog exponentiation requires a non-negative integer literal"
            )
        count = int(exponent.value)
        if not 0 <= count <= 8:
            raise ControlCompilerError(
                "Verilog exponentiation supports literal powers from 0 through 8"
            )
        if count == 0:
            return "ONE_Q"
        result = base
        for _ in range(count - 1):
            result = f"q_mul({result}, {base})"
        return result

    def render(self, source: str | Expression) -> str:
        node = parse_expression(source) if isinstance(source, str) else source
        if isinstance(node, Literal):
            if isinstance(node.value, bool):
                return "1'b1" if node.value else "1'b0"
            return self.constant(node.value)
        if isinstance(node, Symbol):
            return self.symbol(node.name)
        if isinstance(node, Unary):
            operand = self.render(node.operand)
            if node.operator == "not":
                return f"(!{operand})"
            if node.operator == "-":
                return f"q_neg({operand})"
            return operand
        if isinstance(node, Binary):
            left, right = self.render(node.left), self.render(node.right)
            if node.operator == "*":
                return f"q_mul({left}, {right})"
            if node.operator == "/":
                return f"q_div({left}, {right})"
            if node.operator == "**":
                if not isinstance(node.right, Literal):
                    raise ControlCompilerError(
                        "Verilog exponentiation requires a literal exponent"
                    )
                return self._power(left, node.right)
            if node.operator == "+":
                return f"q_add({left}, {right})"
            if node.operator == "-":
                return f"q_sub({left}, {right})"
            operator = {"and": "&&", "or": "||"}.get(node.operator, node.operator)
            return f"({left} {operator} {right})"
        if isinstance(node, Comparison):
            return f"({self.render(node.left)} {node.operator} {self.render(node.right)})"
        if isinstance(node, Conditional):
            return (
                f"({self.render(node.condition)} ? {self.render(node.when_true)} : "
                f"{self.render(node.when_false)})"
            )
        if isinstance(node, Call):
            args = [self.render(item) for item in node.arguments]
            if node.function == "abs":
                return f"q_abs({args[0]})"
            if node.function == "sqrt":
                return f"q_sqrt({args[0]})"
            if node.function == "min":
                return f"q_min({args[0]}, {args[1]})"
            if node.function == "max":
                return f"q_max({args[0]}, {args[1]})"
            if node.function == "clip":
                return f"q_clip({args[0]}, {args[1]}, {args[2]})"
            if node.function == "where":
                return f"({args[0]} ? {args[1]} : {args[2]})"
            if node.function == "smooth_abs":
                epsilon = args[1] if len(args) == 2 else self.constant(1e-12)
                return (
                    f"q_sqrt(q_add(q_mul({args[0]}, {args[0]}), "
                    f"q_mul({epsilon}, {epsilon})))"
                )
            if node.function == "der":
                raise ControlCompilerError("der() is not legal in controller code")
            raise ControlCompilerError(
                f"function {node.function!r} has no synthesizable fixed-point lowering; "
                "use algebraic primitives or a derived lookup-table input"
            )
        raise ControlCompilerError(
            f"cannot lower {type(node).__name__} to fixed-point Verilog"
        )


def _width(item_dtype: str, fixed: FixedPointFormat) -> str:
    return "" if item_dtype == "bool" else f"signed [{fixed.total_bits - 1}:0] "


def _declaration(kind: str, dtype: str, name: str, fixed: FixedPointFormat) -> str:
    return f"{kind} {_width(dtype, fixed)}{name};"


def _bounded(
    expression: str,
    lower: float | None,
    upper: float | None,
    render: _VerilogExpression,
) -> str:
    result = expression
    if lower is not None:
        result = f"q_max({result}, {render.lower_bound(lower)})"
    if upper is not None:
        result = f"q_min({result}, {render.upper_bound(upper)})"
    return result


def _closure_comment(ir: ControlIR) -> str | None:
    if ir.closure is None:
        return None
    payload = json.dumps(
        plain_compiler_data(ir.closure),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).replace("*/", "* /")
    return f"/* Resolved closure: {payload} */"


def _array_function(
    name: str, values: np.ndarray, renderer: _VerilogExpression
) -> list[str]:
    lines = [
        f"function signed [WIDTH-1:0] {name};",
        "    input integer index;",
        "    begin",
        "        case (index)",
    ]
    for index, value in enumerate(values.reshape(-1)):
        lines.append(f"            {index}: {name} = {renderer.constant(float(value))};")
    lines.extend(
        [
            f"            default: {name} = 0;",
            "        endcase",
            "    end",
            "endfunction",
            "",
        ]
    )
    return lines


def _observer_functions(ir: ControlIR, renderer: _VerilogExpression) -> list[str]:
    observer = ir.observer
    if observer is None:
        return []
    for label, values in (
        ("A", observer.A),
        ("B", observer.B),
        ("dynamics_bias", observer.dynamics_bias),
        ("C", observer.C),
        ("D", observer.D),
        ("measurement_bias", observer.measurement_bias),
        ("L", observer.L),
        ("M", observer.M),
        ("latent_bias", observer.latent_bias),
        ("initial_state", observer.initial_state),
        ("initial_covariance", observer.initial_covariance),
        ("process_covariance", observer.process_covariance),
        ("transition", observer.transition),
        ("discrete_input", observer.discrete_input),
        ("discrete_bias", observer.discrete_bias),
        ("discrete_process_covariance", observer.discrete_process_covariance),
        ("measurement_variance", observer.measurement_variance),
    ):
        for index, value in enumerate(values.reshape(-1)):
            quantized = renderer.fixed.quantize(float(value))
            if quantized in {renderer.fixed.minimum, renderer.fixed.maximum}:
                raise ControlCompilerError(
                    f"observer {label}[{index}] quantizes to a fixed-point rail "
                    "reserved for arithmetic faults"
                )
    for label, values in (
        ("initial_covariance", observer.initial_covariance),
        ("measurement_variance", observer.measurement_variance),
    ):
        for index, value in enumerate(values.reshape(-1)):
            if value > 0.0 and renderer.fixed.quantize(float(value)) == 0:
                raise ControlCompilerError(
                    f"observer {label}[{index}]={value:.17g} quantizes to zero in "
                    f"Q{renderer.fixed.total_bits - renderer.fixed.fractional_bits - 1}."
                    f"{renderer.fixed.fractional_bits}; increase precision"
                )
    q_eigenvalues = np.linalg.eigvalsh(observer.discrete_process_covariance)
    meaningful_q = q_eigenvalues[
        q_eigenvalues
        > 1e-12 * max(1.0, float(np.max(np.abs(q_eigenvalues))))
    ]
    if meaningful_q.size == 0:
        if np.any(observer.process_covariance != 0.0):
            raise ControlCompilerError(
                "observer process noise has no numerically meaningful discrete "
                "covariance mode"
            )
    elif any(
        renderer.fixed.quantize(float(value)) == 0 for value in meaningful_q
    ):
        raise ControlCompilerError(
            "a meaningful discrete process-covariance mode quantizes to zero; "
            "increase fixed-point precision"
        )
    for label, covariance in (
        ("initial covariance", observer.initial_covariance),
        ("discrete process covariance", observer.discrete_process_covariance),
    ):
        quantized_covariance = np.rint(
            covariance * renderer.fixed.scale
        ) / renderer.fixed.scale
        if not np.array_equal(quantized_covariance, quantized_covariance.T):
            raise ControlCompilerError(f"quantized {label} is not exactly symmetric")
        covariance_scale = max(1.0, float(np.max(np.abs(quantized_covariance))))
        if (
            float(np.min(np.linalg.eigvalsh(quantized_covariance)))
            < -1e-12 * covariance_scale
        ):
            raise ControlCompilerError(
                f"quantized {label} is not positive semidefinite"
            )
    arrays = (
        ("obs_transition", observer.transition),
        ("obs_discrete_input", observer.discrete_input),
        ("obs_discrete_bias", observer.discrete_bias),
        ("obs_c", observer.C),
        ("obs_d", observer.D),
        ("obs_measurement_bias", observer.measurement_bias),
        ("obs_l", observer.L),
        ("obs_m", observer.M),
        ("obs_latent_bias", observer.latent_bias),
        ("obs_discrete_q", observer.discrete_process_covariance),
        ("obs_r", observer.measurement_variance),
        ("obs_initial_state", observer.initial_state),
        ("obs_initial_covariance", observer.initial_covariance),
    )
    result: list[str] = []
    for name, values in arrays:
        result.extend(_array_function(name, values, renderer))
    return result


def _observer_declarations(ir: ControlIR, fixed: FixedPointFormat) -> list[str]:
    if ir.observer is None:
        return []
    q = f"reg signed [{fixed.total_bits - 1}:0]"
    return [
        f"{q} observer_state [0:OBS_NX-1];",
        f"{q} observer_state_next [0:OBS_NX-1];",
        f"{q} observer_covariance [0:(OBS_NX*OBS_NX)-1];",
        f"{q} observer_covariance_next [0:(OBS_NX*OBS_NX)-1];",
        f"{q} observer_input [0:OBS_NU_STORAGE-1];",
        f"{q} observer_measurement [0:OBS_NY_STORAGE-1];",
        f"{q} observer_x_work [0:OBS_NX-1];",
        f"{q} observer_x_temporary [0:OBS_NX-1];",
        f"{q} observer_p_work [0:(OBS_NX*OBS_NX)-1];",
        f"{q} observer_p_temporary [0:(OBS_NX*OBS_NX)-1];",
        f"{q} observer_residual [0:(OBS_NX*OBS_NX)-1];",
        f"{q} observer_gain [0:OBS_NX-1];",
        f"{q} observer_innovation;",
        f"{q} observer_innovation_variance;",
        "integer observer_i, observer_j, observer_k, observer_l, observer_measurement_index;",
        "integer observer_reset_i, observer_reset_j;",
    ]


def _observer_combinational(
    ir: ControlIR, renderer: _VerilogExpression
) -> list[str]:
    observer = ir.observer
    if observer is None:
        return []
    lines: list[str] = []
    for index, input_name in enumerate(observer.input_names):
        lines.append(f"    observer_input[{index}] = output_{input_name}_state;")
    for index, measurement_name in enumerate(observer.measurement_names):
        lines.append(
            f"    observer_measurement[{index}] = explicit_{measurement_name};"
        )
    lines.extend(
        [
            "    for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1) begin",
            "        observer_x_work[observer_i] = obs_discrete_bias(observer_i);",
            "        for (observer_j = 0; observer_j < OBS_NX; observer_j = observer_j + 1)",
            "            observer_x_work[observer_i] = q_add(observer_x_work[observer_i], q_mul(obs_transition(observer_i*OBS_NX+observer_j), observer_state[observer_j]));",
            "        for (observer_j = 0; observer_j < OBS_NU; observer_j = observer_j + 1)",
            "            observer_x_work[observer_i] = q_add(observer_x_work[observer_i], q_mul(obs_discrete_input(observer_i*OBS_NU+observer_j), observer_input[observer_j]));",
            "    end",
            "    for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1) begin",
            "        for (observer_j = 0; observer_j < OBS_NX; observer_j = observer_j + 1) begin",
            "            observer_p_work[observer_i*OBS_NX+observer_j] = obs_discrete_q(observer_i*OBS_NX+observer_j);",
            "            for (observer_k = 0; observer_k < OBS_NX; observer_k = observer_k + 1)",
            "                for (observer_l = 0; observer_l < OBS_NX; observer_l = observer_l + 1)",
            "                    observer_p_work[observer_i*OBS_NX+observer_j] = q_add(observer_p_work[observer_i*OBS_NX+observer_j], q_mul(q_mul(obs_transition(observer_i*OBS_NX+observer_k), observer_covariance[observer_k*OBS_NX+observer_l]), obs_transition(observer_j*OBS_NX+observer_l)));",
            "        end",
            "    end",
            "    for (observer_measurement_index = 0; observer_measurement_index < OBS_NY; observer_measurement_index = observer_measurement_index + 1) begin",
            "        if (!observer_failure) begin",
            "            observer_innovation = obs_measurement_bias(observer_measurement_index);",
            "            for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1)",
            "                observer_innovation = q_add(observer_innovation, q_mul(obs_c(observer_measurement_index*OBS_NX+observer_i), observer_x_work[observer_i]));",
            "            for (observer_i = 0; observer_i < OBS_NU; observer_i = observer_i + 1)",
            "                observer_innovation = q_add(observer_innovation, q_mul(obs_d(observer_measurement_index*OBS_NU+observer_i), observer_input[observer_i]));",
            "            observer_innovation = q_sub(observer_measurement[observer_measurement_index], observer_innovation);",
            "            observer_innovation_variance = obs_r(observer_measurement_index);",
            "            for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1)",
            "                for (observer_j = 0; observer_j < OBS_NX; observer_j = observer_j + 1)",
            "                    observer_innovation_variance = q_add(observer_innovation_variance, q_mul(q_mul(obs_c(observer_measurement_index*OBS_NX+observer_i), observer_p_work[observer_i*OBS_NX+observer_j]), obs_c(observer_measurement_index*OBS_NX+observer_j)));",
            "            if (observer_innovation_variance <= 0) begin",
            "                observer_failure = 1'b1;",
            "            end else begin",
            "                for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1) begin",
            "                    observer_gain[observer_i] = 0;",
            "                    for (observer_j = 0; observer_j < OBS_NX; observer_j = observer_j + 1)",
            "                        observer_gain[observer_i] = q_add(observer_gain[observer_i], q_mul(observer_p_work[observer_i*OBS_NX+observer_j], obs_c(observer_measurement_index*OBS_NX+observer_j)));",
            "                    observer_gain[observer_i] = q_div(observer_gain[observer_i], observer_innovation_variance);",
            "                    observer_x_temporary[observer_i] = q_add(observer_x_work[observer_i], q_mul(observer_gain[observer_i], observer_innovation));",
            "                end",
            "                for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1)",
            "                    for (observer_j = 0; observer_j < OBS_NX; observer_j = observer_j + 1)",
            "                        observer_residual[observer_i*OBS_NX+observer_j] = q_sub(((observer_i == observer_j) ? ONE_Q : 0), q_mul(observer_gain[observer_i], obs_c(observer_measurement_index*OBS_NX+observer_j)));",
            "                for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1) begin",
            "                    for (observer_j = 0; observer_j < OBS_NX; observer_j = observer_j + 1) begin",
            "                        observer_p_temporary[observer_i*OBS_NX+observer_j] = 0;",
            "                        for (observer_k = 0; observer_k < OBS_NX; observer_k = observer_k + 1)",
            "                            observer_p_temporary[observer_i*OBS_NX+observer_j] = q_add(observer_p_temporary[observer_i*OBS_NX+observer_j], q_mul(observer_residual[observer_i*OBS_NX+observer_k], observer_p_work[observer_k*OBS_NX+observer_j]));",
            "                    end",
            "                end",
            "                for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1) begin",
            "                    for (observer_j = 0; observer_j < OBS_NX; observer_j = observer_j + 1) begin",
            "                        observer_p_work[observer_i*OBS_NX+observer_j] = q_mul(q_mul(observer_gain[observer_i], obs_r(observer_measurement_index)), observer_gain[observer_j]);",
            "                        for (observer_k = 0; observer_k < OBS_NX; observer_k = observer_k + 1)",
            "                            observer_p_work[observer_i*OBS_NX+observer_j] = q_add(observer_p_work[observer_i*OBS_NX+observer_j], q_mul(observer_p_temporary[observer_i*OBS_NX+observer_k], observer_residual[observer_j*OBS_NX+observer_k]));",
            "                    end",
            "                    observer_x_work[observer_i] = observer_x_temporary[observer_i];",
            "                end",
            "            end",
            "        end",
            "    end",
            "    for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1) begin",
            "        observer_state_next[observer_i] = observer_x_work[observer_i];",
            "        for (observer_j = 0; observer_j < OBS_NX; observer_j = observer_j + 1)",
            "            observer_covariance_next[observer_i*OBS_NX+observer_j] = observer_p_work[observer_i*OBS_NX+observer_j];",
            "    end",
        ]
    )
    for latent_index, latent in enumerate(observer.latent_names):
        lines.extend(
            [
                f"    implicit_{latent}_mean_next = obs_latent_bias({latent_index});",
                f"    implicit_{latent}_variance_next = 0;",
                "    for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1) begin",
                f"        implicit_{latent}_mean_next = q_add(implicit_{latent}_mean_next, q_mul(obs_l({latent_index}*OBS_NX+observer_i), observer_state_next[observer_i]));",
                "        for (observer_j = 0; observer_j < OBS_NX; observer_j = observer_j + 1)",
                f"            implicit_{latent}_variance_next = q_add(implicit_{latent}_variance_next, q_mul(q_mul(obs_l({latent_index}*OBS_NX+observer_i), observer_covariance_next[observer_i*OBS_NX+observer_j]), obs_l({latent_index}*OBS_NX+observer_j)));",
                "    end",
                "    for (observer_i = 0; observer_i < OBS_NU; observer_i = observer_i + 1)",
                f"        implicit_{latent}_mean_next = q_add(implicit_{latent}_mean_next, q_mul(obs_m({latent_index}*OBS_NU+observer_i), observer_input[observer_i]));",
                f"    implicit_{latent}_mean_next = {_bounded(f'implicit_{latent}_mean_next', observer.latent_lower_bounds[latent_index], observer.latent_upper_bounds[latent_index], renderer)};",
                f"    implicit_{latent}_variance_next = q_max(implicit_{latent}_variance_next, 0);",
            ]
        )
    return lines


def _module(ir: ControlIR, fixed: FixedPointFormat) -> str:
    spec, name = ir.spec, ir.identifier
    _FixedPointAdmission(ir, fixed).admit()
    mode_symbols = [mode.name.upper() for mode in spec.modes]
    if len(mode_symbols) != len(set(mode_symbols)):
        raise ControlCompilerError(
            "controller mode names collide after deterministic Verilog symbol "
            f"normalization: {mode_symbols}"
        )
    signal_entries: list[tuple[str, str]] = [
        ("clock", "clk"),
        ("reset", "reset_n"),
        ("tick", "tick"),
        ("input error", "input_error"),
        ("observer error", "observer_error"),
        ("arithmetic error", "arithmetic_error"),
        ("mode register", "mode"),
        ("next mode", "mode_next"),
    ]
    signal_entries.extend(
        (f"explicit input {item.name}", f"explicit_{item.name}")
        for item in spec.explicit_inputs
    )
    signal_entries.extend(
        (f"output port {item.name}", f"output_{item.name}")
        for item in spec.outputs
    )
    signal_entries.extend(
        (f"output state {item.name}", f"output_{item.name}_state")
        for item in spec.outputs
    )
    signal_entries.extend(
        (f"output next {item.name}", f"output_{item.name}_next")
        for item in spec.outputs
    )
    signal_entries.extend(
        (f"register {item.name}", f"register_{item.name}")
        for item in spec.registers
    )
    signal_entries.extend(
        (f"derived value {item.name}", f"derived_{item.name}")
        for item in spec.derived
    )
    for item in spec.implicit_inputs:
        for suffix in ("mean", "variance"):
            signal_entries.extend(
                (
                    (
                        f"implicit input {item.name} {suffix}",
                        f"implicit_{item.name}_{suffix}",
                    ),
                    (
                        f"implicit input {item.name} next {suffix}",
                        f"implicit_{item.name}_{suffix}_next",
                    ),
                )
            )
    admit_target_symbol_table(
        "Verilog",
        {
            "module signals": tuple(signal_entries),
            "mode constants": (
                ("internal mode bit width", "MODE_BITS"),
                *tuple(
                    (f"mode {item.name}", f"MODE_{item.name.upper()}")
                    for item in spec.modes
                ),
            ),
        },
    )
    renderer = _VerilogExpression(ir, fixed)
    render = renderer.render
    mode_index = {mode.name: index for index, mode in enumerate(spec.modes)}
    mode_bits = max(1, (len(spec.modes) - 1).bit_length())
    ports = ["input wire clk", "input wire reset_n", "input wire tick"]
    for item in spec.explicit_inputs:
        ports.append(f"input wire {_width(item.dtype, fixed)}explicit_{item.name}")
    for item in spec.outputs:
        ports.append(f"output wire {_width(item.dtype, fixed)}output_{item.name}")
    ports.append("output reg input_error")
    ports.append("output reg observer_error")
    ports.append("output reg arithmetic_error")
    lines = [
        "// Generated complete synchronous controller (synthesizable Verilog-2001).",
        f"// Source: {ir.source_digest}",
        f"module {name} (",
    ]
    closure = _closure_comment(ir)
    if closure is not None:
        lines.insert(2, closure)
    lines.extend(
        f"    {port}{',' if index < len(ports) - 1 else ''}"
        for index, port in enumerate(ports)
    )
    lines.extend(
        [
            ");",
            "",
            f"localparam integer WIDTH = {fixed.total_bits};",
            f"localparam integer FRAC = {fixed.fractional_bits};",
            f"localparam signed [WIDTH-1:0] ONE_Q = {fixed.total_bits}'sd{fixed.scale};",
            f"localparam signed [WIDTH-1:0] PERIOD_Q = {renderer.constant(spec.period_s)};",
            "localparam signed [WIDTH-1:0] MAX_Q = {1'b0, {(WIDTH-1){1'b1}}};",
            "localparam signed [WIDTH-1:0] MIN_Q = {1'b1, {(WIDTH-1){1'b0}}};",
            f"localparam integer MODE_BITS = {mode_bits};",
        ]
    )
    if ir.observer is not None:
        lines.extend(
            [
                f"localparam integer OBS_NX = {len(ir.observer.state_names)};",
                f"localparam integer OBS_NU = {len(ir.observer.input_names)};",
                f"localparam integer OBS_NU_STORAGE = {max(1, len(ir.observer.input_names))};",
                f"localparam integer OBS_NY = {len(ir.observer.measurement_names)};",
                f"localparam integer OBS_NY_STORAGE = {max(1, len(ir.observer.measurement_names))};",
                f"localparam integer OBS_NZ = {len(ir.observer.latent_names)};",
            ]
        )
    for mode, index in mode_index.items():
        lines.append(
            f"localparam [MODE_BITS-1:0] MODE_{mode.upper()} = {mode_bits}'d{index};"
        )
    lines.extend(
        [
            "",
            "function q_fault;",
            "    input signed [WIDTH-1:0] value;",
            "    begin q_fault = (value == MAX_Q) || (value == MIN_Q); end",
            "endfunction",
            "",
            "function signed [WIDTH-1:0] q_add;",
            "    input signed [WIDTH-1:0] left;",
            "    input signed [WIDTH-1:0] right;",
            "    reg signed [WIDTH:0] wide;",
            "    begin",
            "        if (q_fault(left)) q_add = left;",
            "        else if (q_fault(right)) q_add = right;",
            "        else begin",
            "            wide = left;",
            "            wide = wide + right;",
            "            if (wide > MAX_Q) q_add = MAX_Q;",
            "            else if (wide < MIN_Q) q_add = MIN_Q;",
            "            else q_add = wide[WIDTH-1:0];",
            "        end",
            "    end",
            "endfunction",
            "",
            "function signed [WIDTH-1:0] q_sub;",
            "    input signed [WIDTH-1:0] left;",
            "    input signed [WIDTH-1:0] right;",
            "    reg signed [WIDTH:0] wide;",
            "    begin",
            "        if (q_fault(left)) q_sub = left;",
            "        else if (q_fault(right)) q_sub = right;",
            "        else begin",
            "            wide = left;",
            "            wide = wide - right;",
            "            if (wide > MAX_Q) q_sub = MAX_Q;",
            "            else if (wide < MIN_Q) q_sub = MIN_Q;",
            "            else q_sub = wide[WIDTH-1:0];",
            "        end",
            "    end",
            "endfunction",
            "",
            "function signed [WIDTH-1:0] q_neg;",
            "    input signed [WIDTH-1:0] value;",
            "    begin",
            "        if (q_fault(value)) q_neg = value;",
            "        else q_neg = -value;",
            "    end",
            "endfunction",
            "",
            "function signed [WIDTH-1:0] q_mul;",
            "    input signed [WIDTH-1:0] left;",
            "    input signed [WIDTH-1:0] right;",
            "    reg signed [(2*WIDTH)-1:0] product;",
            "    reg signed [(2*WIDTH)-1:0] scaled;",
            "    begin",
            "        if (q_fault(left)) q_mul = left;",
            "        else if (q_fault(right)) q_mul = right;",
            "        else begin",
            "            product = left * right;",
            "            scaled = product >>> FRAC;",
            "            if (scaled > MAX_Q) q_mul = MAX_Q;",
            "            else if (scaled < MIN_Q) q_mul = MIN_Q;",
            "            else q_mul = scaled[WIDTH-1:0];",
            "        end",
            "    end",
            "endfunction",
            "",
            "function signed [WIDTH-1:0] q_div;",
            "    input signed [WIDTH-1:0] numerator_value;",
            "    input signed [WIDTH-1:0] denominator;",
            "    reg signed [(2*WIDTH)-1:0] numerator;",
            "    reg signed [(2*WIDTH)-1:0] quotient;",
            "    begin",
            "        if (q_fault(numerator_value)) q_div = numerator_value;",
            "        else if (q_fault(denominator)) q_div = denominator;",
            "        else if (denominator == 0) q_div = (numerator_value < 0) ? MIN_Q : MAX_Q;",
            "        else begin",
            "            numerator = numerator_value;",
            "            numerator = numerator <<< FRAC;",
            "            quotient = numerator / denominator;",
            "            if (quotient > MAX_Q) q_div = MAX_Q;",
            "            else if (quotient < MIN_Q) q_div = MIN_Q;",
            "            else q_div = quotient[WIDTH-1:0];",
            "        end",
            "    end",
            "endfunction",
            "",
            "function signed [WIDTH-1:0] q_abs;",
            "    input signed [WIDTH-1:0] value;",
            "    begin q_abs = q_fault(value) ? value : ((value < 0) ? q_neg(value) : value); end",
            "endfunction",
            "",
            "function signed [WIDTH-1:0] q_min;",
            "    input signed [WIDTH-1:0] left;",
            "    input signed [WIDTH-1:0] right;",
            "    begin",
            "        if (q_fault(left)) q_min = left;",
            "        else if (q_fault(right)) q_min = right;",
            "        else q_min = (left < right) ? left : right;",
            "    end",
            "endfunction",
            "",
            "function signed [WIDTH-1:0] q_max;",
            "    input signed [WIDTH-1:0] left;",
            "    input signed [WIDTH-1:0] right;",
            "    begin",
            "        if (q_fault(left)) q_max = left;",
            "        else if (q_fault(right)) q_max = right;",
            "        else q_max = (left > right) ? left : right;",
            "    end",
            "endfunction",
            "",
            "function signed [WIDTH-1:0] q_clip;",
            "    input signed [WIDTH-1:0] value;",
            "    input signed [WIDTH-1:0] lower;",
            "    input signed [WIDTH-1:0] upper;",
            "    begin q_clip = q_min(q_max(value, lower), upper); end",
            "endfunction",
            "",
            "function signed [WIDTH-1:0] q_sqrt;",
            "    input signed [WIDTH-1:0] value;",
            "    integer iteration;",
            "    reg signed [WIDTH-1:0] guess;",
            "    reg signed [WIDTH-1:0] quotient;",
            "    reg signed [WIDTH:0] average_sum;",
            "    begin",
            "        if (q_fault(value)) begin",
            "            q_sqrt = value;",
            "        end else if (value < 0) begin",
            "            q_sqrt = MIN_Q;",
            "        end else if (value == 0) begin",
            "            q_sqrt = 0;",
            "        end else begin",
            "            guess = (value > ONE_Q) ? value : ONE_Q;",
            "            for (iteration = 0; iteration < WIDTH; iteration = iteration + 1) begin",
            "                quotient = q_div(value, guess);",
            "                if (q_fault(quotient)) guess = quotient;",
            "                else begin",
            "                    average_sum = guess;",
            "                    average_sum = average_sum + quotient;",
            "                    guess = average_sum >>> 1;",
            "                end",
            "            end",
            "            q_sqrt = guess;",
            "        end",
            "    end",
            "endfunction",
            "",
        ]
    )
    lines.extend(_observer_functions(ir, renderer))
    lines.extend(
        [
            "reg [MODE_BITS-1:0] mode, mode_next;",
            "reg emergency_next;",
            "reg inputs_valid;",
            "reg observer_failure;",
            "reg arithmetic_failure;",
        ]
    )
    for item in spec.outputs:
        lines.extend(
            [
                _declaration("reg", item.dtype, f"output_{item.name}_state", fixed),
                _declaration("reg", item.dtype, f"output_{item.name}_next", fixed),
                f"assign output_{item.name} = output_{item.name}_state;",
            ]
        )
    for item in spec.registers:
        lines.extend(
            [
                _declaration("reg", item.dtype, f"register_{item.name}", fixed),
                _declaration("reg", item.dtype, f"register_{item.name}_next", fixed),
            ]
        )
    for item in spec.implicit_inputs:
        for suffix in ("mean", "variance"):
            lines.extend(
                [
                    _declaration("reg", "real", f"implicit_{item.name}_{suffix}", fixed),
                    _declaration("reg", "real", f"implicit_{item.name}_{suffix}_next", fixed),
                ]
            )
    lines.extend(_observer_declarations(ir, fixed))
    for item in spec.derived:
        lines.append(_declaration("reg", item.dtype, f"derived_{item.name}", fixed))

    lines.extend(["", "always @* begin", "    mode_next = mode;", "    inputs_valid = 1'b1;", "    observer_failure = 1'b0;", "    arithmetic_failure = 1'b0;"])
    for item in spec.outputs:
        lines.append(f"    output_{item.name}_next = output_{item.name}_state;")
    for item in spec.registers:
        lines.append(f"    register_{item.name}_next = register_{item.name};")
    for item in spec.explicit_inputs:
        if item.dtype != "real":
            continue
        lines.append(
            f"    if (q_fault(explicit_{item.name})) arithmetic_failure = 1'b1;"
        )
        conditions: list[str] = []
        if item.bounds.lower is not None:
            conditions.append(
                f"explicit_{item.name} < {renderer.lower_bound(item.bounds.lower)}"
            )
        if item.bounds.upper is not None:
            conditions.append(
                f"explicit_{item.name} > {renderer.upper_bound(item.bounds.upper)}"
            )
        if conditions:
            lines.append(f"    if ({' || '.join(conditions)}) inputs_valid = 1'b0;")
    lines.extend(_observer_combinational(ir, renderer))

    for item in spec.derived:
        lines.append(f"    derived_{item.name} = {render(item.expression)};")
    emergency = "1'b0" if spec.emergency_when is None else render(spec.emergency_when)
    lines.extend([f"    emergency_next = {emergency};", "    case (mode)"])
    for mode in spec.modes:
        lines.append(f"        MODE_{mode.name.upper()}: begin")
        for output_name, expression in mode.outputs.items():
            lines.append(f"            output_{output_name}_next = {render(expression)};")
        for register_name, expression in mode.updates.items():
            lines.append(f"            register_{register_name}_next = {render(expression)};")
        for index, transition in enumerate(
            sorted(mode.transitions, key=lambda value: value.priority, reverse=True)
        ):
            keyword = "if" if index == 0 else "else if"
            lines.append(
                f"            {keyword} ({render(transition.guard)}) mode_next = MODE_{transition.target.upper()};"
            )
        lines.append("        end")
    lines.extend(
        [
            f"        default: mode_next = MODE_{spec.initial_mode.upper()};",
            "    endcase",
        ]
    )
    for item in spec.outputs:
        if item.emergency_value is not None:
            value = (
                "1'b1"
                if item.emergency_value is True
                else "1'b0"
                if item.emergency_value is False
                else renderer.constant(item.emergency_value)
            )
            lines.append(
                f"    if (emergency_next) output_{item.name}_next = {value};"
            )
        if item.dtype == "real":
            lines.append(
                f"    output_{item.name}_next = {_bounded(f'output_{item.name}_next', item.bounds.lower, item.bounds.upper, renderer)};"
            )
            if item.slew_rate is not None:
                delta = renderer.positive_limit(item.slew_rate * spec.period_s)
                lines.append(
                    f"    if (!emergency_next) output_{item.name}_next = q_clip(output_{item.name}_next, q_sub(output_{item.name}_state, {delta}), q_add(output_{item.name}_state, {delta}));"
                )
    for item in spec.registers:
        if item.dtype == "real":
            lines.append(
                f"    register_{item.name}_next = {_bounded(f'register_{item.name}_next', item.bounds.lower, item.bounds.upper, renderer)};"
            )
    for item in spec.outputs:
        if item.dtype == "real":
            lines.append(
                f"    if (q_fault(output_{item.name}_next)) arithmetic_failure = 1'b1;"
            )
    for item in spec.registers:
        if item.dtype == "real":
            lines.append(
                f"    if (q_fault(register_{item.name}_next)) arithmetic_failure = 1'b1;"
            )
    for item in spec.derived:
        if item.dtype == "real":
            lines.append(
                f"    if (q_fault(derived_{item.name})) arithmetic_failure = 1'b1;"
            )
    for item in spec.implicit_inputs:
        lines.extend(
            [
                f"    if (q_fault(implicit_{item.name}_mean_next)) arithmetic_failure = 1'b1;",
                f"    if (q_fault(implicit_{item.name}_variance_next)) arithmetic_failure = 1'b1;",
            ]
        )
    if ir.observer is not None:
        lines.extend(
            [
                "    for (observer_i = 0; observer_i < OBS_NX; observer_i = observer_i + 1) begin",
                "        if (q_fault(observer_state_next[observer_i])) arithmetic_failure = 1'b1;",
                "        for (observer_j = 0; observer_j < OBS_NX; observer_j = observer_j + 1)",
                "            if (q_fault(observer_covariance_next[observer_i*OBS_NX+observer_j])) arithmetic_failure = 1'b1;",
                "    end",
            ]
        )
    lines.extend(
        [
            "end",
            "",
            "always @(posedge clk) begin",
            "    if (!reset_n) begin",
            f"        mode <= MODE_{spec.initial_mode.upper()};",
            "        input_error <= 1'b0;",
            "        observer_error <= 1'b0;",
            "        arithmetic_error <= 1'b0;",
        ]
    )
    for item in spec.outputs:
        value = (
            "1'b1"
            if item.default is True
            else "1'b0"
            if item.default is False
            else renderer.constant(item.default)
        )
        lines.append(f"        output_{item.name}_state <= {value};")
    for item in spec.registers:
        value = (
            "1'b1"
            if item.initial is True
            else "1'b0"
            if item.initial is False
            else renderer.constant(item.initial)
        )
        lines.append(f"        register_{item.name} <= {value};")
    for item in spec.implicit_inputs:
        assert ir.observer is not None
        index = ir.observer.latent_names.index(item.name)
        output_defaults = {output.name: float(output.default) for output in spec.outputs}
        initial_input = np.asarray(
            [output_defaults[name] for name in ir.observer.input_names], dtype=float
        )
        initial_mean = (
            ir.observer.L[index] @ ir.observer.initial_state
            + ir.observer.M[index] @ initial_input
            + ir.observer.latent_bias[index]
        )
        initial_variance = (
            ir.observer.L[index]
            @ ir.observer.initial_covariance
            @ ir.observer.L[index]
        )
        if item.bounds.lower is not None:
            initial_mean = max(initial_mean, item.bounds.lower)
        if item.bounds.upper is not None:
            initial_mean = min(initial_mean, item.bounds.upper)
        lines.extend(
            [
                f"        implicit_{item.name}_mean <= {renderer.constant(initial_mean)};",
                f"        implicit_{item.name}_variance <= {renderer.constant(initial_variance)};",
            ]
        )
    if ir.observer is not None:
        lines.extend(
            [
                "        for (observer_reset_i = 0; observer_reset_i < OBS_NX; observer_reset_i = observer_reset_i + 1) begin",
                "            observer_state[observer_reset_i] <= obs_initial_state(observer_reset_i);",
                "            for (observer_reset_j = 0; observer_reset_j < OBS_NX; observer_reset_j = observer_reset_j + 1)",
                "                observer_covariance[observer_reset_i*OBS_NX+observer_reset_j] <= obs_initial_covariance(observer_reset_i*OBS_NX+observer_reset_j);",
                "        end",
            ]
        )
    lines.extend(
        [
            "    end else if (tick) begin",
            "        input_error <= !inputs_valid;",
            "        observer_error <= observer_failure;",
            "        arithmetic_error <= arithmetic_failure;",
            "        if (inputs_valid && !observer_failure && !arithmetic_failure) begin",
            "            mode <= mode_next;",
        ]
    )
    for item in spec.outputs:
        lines.append(f"            output_{item.name}_state <= output_{item.name}_next;")
    for item in spec.registers:
        lines.append(f"            register_{item.name} <= register_{item.name}_next;")
    for item in spec.implicit_inputs:
        lines.extend(
                [
                f"            implicit_{item.name}_mean <= implicit_{item.name}_mean_next;",
                f"            implicit_{item.name}_variance <= implicit_{item.name}_variance_next;",
                ]
            )
    if ir.observer is not None:
        lines.extend(
            [
                "            for (observer_reset_i = 0; observer_reset_i < OBS_NX; observer_reset_i = observer_reset_i + 1) begin",
                "                observer_state[observer_reset_i] <= observer_state_next[observer_reset_i];",
                "                for (observer_reset_j = 0; observer_reset_j < OBS_NX; observer_reset_j = observer_reset_j + 1)",
                "                    observer_covariance[observer_reset_i*OBS_NX+observer_reset_j] <= observer_covariance_next[observer_reset_i*OBS_NX+observer_reset_j];",
                "            end",
            ]
        )
    lines.extend(["        end", "    end", "end", "", "endmodule"])
    return "\n".join(lines) + "\n"


def generate_verilog(
    value: ControlSpec | ControlIR,
    *,
    identifier: str | None = None,
    fixed_point: FixedPointFormat | None = None,
) -> GeneratedArtifact:
    """Generate one complete synthesizable fixed-point controller module."""

    ir = as_ir(value, identifier=identifier)
    fixed = FixedPointFormat() if fixed_point is None else fixed_point
    if not isinstance(fixed, FixedPointFormat):
        raise TypeError("fixed_point must be a FixedPointFormat")
    return GeneratedArtifact(
        f"{ir.identifier}.v", "text/x-verilog", _module(ir, fixed)
    )


__all__ = ["generate_verilog"]
