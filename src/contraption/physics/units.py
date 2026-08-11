"""Small, strict SI unit and dimension system used by the PMDL validator.

The module intentionally implements only the deterministic algebra needed by
physical models.  It does not guess units, perform implicit conversions, or
accept Python expressions.  Unit strings are parsed by a tiny tokenizer with
the grammar ``product := atom (('*' | '/') atom)*`` and integer/rational
powers.  Angle is dimensionless in SI, but is tracked as a named unit so model
authors can still document reference conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Iterable, Iterator, Mapping


class UnitError(ValueError):
    """Raised when a unit is unknown, malformed, or dimensionally invalid."""


_AXES = ("mass", "length", "time", "current", "temperature", "amount", "luminous")


@dataclass(frozen=True, slots=True)
class Dimension:
    """The seven SI base dimensions, represented by exact rational powers."""

    exponents: tuple[Fraction, ...] = (Fraction(0),) * 7

    def __post_init__(self) -> None:
        values = tuple(Fraction(value) for value in self.exponents)
        if len(values) != len(_AXES):
            raise UnitError(f"a dimension requires {len(_AXES)} exponents")
        object.__setattr__(self, "exponents", values)

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, exponent: int | float | Fraction) -> "Dimension":
        power = Fraction(exponent)
        return Dimension(tuple(value * power for value in self.exponents))

    @property
    def is_dimensionless(self) -> bool:
        return all(value == 0 for value in self.exponents)

    def describe(self) -> str:
        if self.is_dimensionless:
            return "1"
        parts: list[str] = []
        for name, exponent in zip(_AXES, self.exponents):
            if exponent:
                suffix = "" if exponent == 1 else f"^{exponent}"
                parts.append(f"{name}{suffix}")
        return "*".join(parts)


DIMENSIONLESS = Dimension()
MASS = Dimension((1, 0, 0, 0, 0, 0, 0))
LENGTH = Dimension((0, 1, 0, 0, 0, 0, 0))
TIME = Dimension((0, 0, 1, 0, 0, 0, 0))
CURRENT = Dimension((0, 0, 0, 1, 0, 0, 0))
TEMPERATURE = Dimension((0, 0, 0, 0, 1, 0, 0))
AMOUNT = Dimension((0, 0, 0, 0, 0, 1, 0))
LUMINOUS_INTENSITY = Dimension((0, 0, 0, 0, 0, 0, 1))


@dataclass(frozen=True, slots=True)
class Unit:
    """A multiplicative unit relative to SI base units.

    Affine units are deliberately excluded because residual equations require
    unambiguous linear algebra.  Temperatures in PMDL therefore use kelvin.
    """

    symbol: str
    dimension: Dimension
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise UnitError("unit symbol may not be empty")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise UnitError("unit scale must be finite and positive")

    def compatible_with(self, other: "Unit") -> bool:
        return self.dimension == other.dimension

    def convert_value_to(self, value: float, target: "Unit") -> float:
        if not self.compatible_with(target):
            raise UnitError(f"cannot convert {self.symbol} to {target.symbol}")
        return float(value) * self.scale / target.scale

    def __mul__(self, other: "Unit") -> "Unit":
        return Unit(f"{self.symbol}*{other.symbol}", self.dimension * other.dimension, self.scale * other.scale)

    def __truediv__(self, other: "Unit") -> "Unit":
        return Unit(f"{self.symbol}/{other.symbol}", self.dimension / other.dimension, self.scale / other.scale)

    def __pow__(self, exponent: int | float | Fraction) -> "Unit":
        power = Fraction(exponent)
        return Unit(f"{self.symbol}^{power}", self.dimension**power, self.scale ** float(power))


def _d(m: int = 0, l: int = 0, t: int = 0, i: int = 0, k: int = 0, n: int = 0, j: int = 0) -> Dimension:
    return Dimension((m, l, t, i, k, n, j))


# Canonical and commonly useful engineering units.  Aliases all resolve to a
# canonical Unit, so dimensional checks are independent of spelling.
_UNIT_TABLE: dict[str, Unit] = {
    "1": Unit("1", DIMENSIONLESS),
    "rad": Unit("rad", DIMENSIONLESS),
    "deg": Unit("deg", DIMENSIONLESS, math.pi / 180.0),
    "%": Unit("%", DIMENSIONLESS, 0.01),
    "kg": Unit("kg", MASS),
    "g": Unit("g", MASS, 1e-3),
    "m": Unit("m", LENGTH),
    "cm": Unit("cm", LENGTH, 1e-2),
    "mm": Unit("mm", LENGTH, 1e-3),
    "um": Unit("um", LENGTH, 1e-6),
    "s": Unit("s", TIME),
    "ms": Unit("ms", TIME, 1e-3),
    "us": Unit("us", TIME, 1e-6),
    "min": Unit("min", TIME, 60.0),
    "A": Unit("A", CURRENT),
    "mA": Unit("mA", CURRENT, 1e-3),
    "K": Unit("K", TEMPERATURE),
    "mol": Unit("mol", AMOUNT),
    "cd": Unit("cd", LUMINOUS_INTENSITY),
    "Hz": Unit("Hz", DIMENSIONLESS / TIME),
    "N": Unit("N", MASS * LENGTH / (TIME**2)),
    "Pa": Unit("Pa", MASS / LENGTH / (TIME**2)),
    "J": Unit("J", MASS * (LENGTH**2) / (TIME**2)),
    "W": Unit("W", MASS * (LENGTH**2) / (TIME**3)),
    "C": Unit("C", CURRENT * TIME),
    "V": Unit("V", MASS * (LENGTH**2) / (TIME**3) / CURRENT),
    "ohm": Unit("ohm", MASS * (LENGTH**2) / (TIME**3) / (CURRENT**2)),
    "S": Unit("S", (TIME**3) * (CURRENT**2) / MASS / (LENGTH**2)),
    "F": Unit("F", (TIME**4) * (CURRENT**2) / MASS / (LENGTH**2)),
    "H": Unit("H", MASS * (LENGTH**2) / (TIME**2) / (CURRENT**2)),
    "Wb": Unit("Wb", MASS * (LENGTH**2) / (TIME**2) / CURRENT),
    "T": Unit("T", MASS / (TIME**2) / CURRENT),
}

_ALIASES = {
    "": "1",
    "dimensionless": "1",
    "meter": "m",
    "metre": "m",
    "second": "s",
    "ampere": "A",
    "amp": "A",
    "volt": "V",
    "coulomb": "C",
    "farad": "F",
    "henry": "H",
    "newton": "N",
    "joule": "J",
    "watt": "W",
    "radian": "rad",
    "Ohm": "ohm",
    "Ω": "ohm",
}

_TOKEN = re.compile(r"\s*([A-Za-z_%Ω][A-Za-z0-9_%Ω]*|1|\*\*|[*/^()+-]|\d+)\s*")


def known_units() -> Mapping[str, Unit]:
    """Return a read-only-by-convention snapshot of known unit symbols."""

    return dict(_UNIT_TABLE)


def register_unit(unit: Unit, aliases: Iterable[str] = ()) -> None:
    """Register an application unit explicitly.

    Registration is intentionally opt-in and rejects redefinition; production
    model loading should finish registration before parsing untrusted models.
    """

    if unit.symbol in _UNIT_TABLE:
        raise UnitError(f"unit {unit.symbol!r} is already registered")
    _UNIT_TABLE[unit.symbol] = unit
    for alias in aliases:
        if alias in _ALIASES or alias in _UNIT_TABLE:
            raise UnitError(f"unit alias {alias!r} is already registered")
        _ALIASES[alias] = unit.symbol


def _tokens(text: str) -> Iterator[str]:
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            raise UnitError(f"invalid unit syntax at character {position}: {text!r}")
        yield match.group(1)
        position = match.end()


class _UnitParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = list(_tokens(text))
        self.index = 0

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def peek_next(self) -> str | None:
        return self.tokens[self.index + 1] if self.index + 1 < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            wanted = expected or "token"
            raise UnitError(f"expected {wanted!r} in unit {self.text!r}")
        self.index += 1
        return token

    def parse(self) -> Unit:
        if not self.tokens:
            return _UNIT_TABLE["1"]
        result = self.product()
        if self.peek() is not None:
            raise UnitError(f"unexpected token {self.peek()!r} in unit {self.text!r}")
        return Unit(self.text.strip() or "1", result.dimension, result.scale)

    def product(self) -> Unit:
        result = self.factor()
        while self.peek() in ("*", "/"):
            operator = self.take()
            right = self.factor()
            result = result * right if operator == "*" else result / right
        return result

    def factor(self) -> Unit:
        if self.peek() == "(":
            self.take("(")
            result = self.product()
            self.take(")")
        else:
            symbol = self.take()
            symbol = _ALIASES.get(symbol, symbol)
            try:
                result = _UNIT_TABLE[symbol]
            except KeyError as exc:
                raise UnitError(f"unknown unit {symbol!r}") from exc
        if self.peek() in ("^", "**"):
            self.take()
            sign = 1
            if self.peek() in ("+", "-"):
                sign = -1 if self.take() == "-" else 1
            numerator = int(self.take()) * sign
            denominator = 1
            # A slash belongs to the exponent only when followed by an integer;
            # in ``m^2/s`` it is the enclosing product operator.
            if self.peek() == "/" and (self.peek_next() or "").isdigit():
                self.take("/")
                denominator = int(self.take())
            result = result ** Fraction(numerator, denominator)
        return result


def parse_unit(text: str | Unit | None) -> Unit:
    """Parse a unit expression without executing general-purpose code."""

    if isinstance(text, Unit):
        return text
    if text is None:
        text = "1"
    if not isinstance(text, str):
        raise UnitError(f"unit must be a string, got {type(text).__name__}")
    return _UnitParser(text).parse()


def require_compatible(left: str | Unit, right: str | Unit, context: str = "values") -> None:
    left_unit, right_unit = parse_unit(left), parse_unit(right)
    if left_unit.dimension != right_unit.dimension:
        raise UnitError(
            f"incompatible dimensions for {context}: {left_unit.symbol} "
            f"({left_unit.dimension.describe()}) vs {right_unit.symbol} "
            f"({right_unit.dimension.describe()})"
        )


@dataclass(frozen=True, slots=True)
class Quantity:
    """A scalar value with an explicit unit."""

    value: float
    unit: Unit

    @classmethod
    def of(cls, value: float, unit: str | Unit = "1") -> "Quantity":
        return cls(float(value), parse_unit(unit))

    def to(self, unit: str | Unit) -> "Quantity":
        target = parse_unit(unit)
        return Quantity(self.unit.convert_value_to(self.value, target), target)
