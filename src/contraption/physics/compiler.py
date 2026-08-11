"""Compile a constrained online-model IR into fixed-allocation C99.

The offline modeling language may contain descriptor systems and rich
probability distributions.  An onboard target needs a much smaller safety
boundary.  This module therefore accepts only continuous-time affine systems
or a fixed linearization over a declared validity envelope::

    x_dot = A x + B u + b
    y     = C x + D u + d

Uncertainty is retained as a covariance and propagated/conditioned by an
extended-Kalman-filter-compatible predict/update implementation.  For a
``linearized`` IR, the supplied matrices are the approved Jacobians at the
declared operating point; arbitrary target-side callbacks are never emitted.
All generated arrays have compile-time sizes and the generated runtime uses no
heap allocation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


class CompilerError(ValueError):
    """Base class for online compiler failures."""


class IRValidationError(CompilerError):
    """Raised when a model cannot safely enter the online subset."""


_C_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_STATES = 64
_MAX_INPUTS = 32
_MAX_MEASUREMENTS = 32


def _plain_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return converted
    if is_dataclass(value):
        converted = asdict(value)
        if isinstance(converted, Mapping):
            return converted
    raise IRValidationError("online model must be a mapping or data object with to_dict()")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise IRValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise IRValidationError(f"{label} must be finite")
    return result


def _names(values: Sequence[Any], label: str, maximum: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IRValidationError(f"{label} must be an array")
    if any(not isinstance(value, str) for value in values):
        raise IRValidationError(f"{label} entries must be strings")
    result = tuple(values)
    if not result or len(result) > maximum:
        raise IRValidationError(f"{label} must contain 1..{maximum} entries")
    if len(set(result)) != len(result):
        raise IRValidationError(f"{label} must be unique")
    for name in result:
        if not name or len(name) > 128:
            raise IRValidationError(f"invalid name in {label}: {name!r}")
    return result


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise IRValidationError(f"{label} must be a numeric array") from exc
    if result.shape != shape:
        raise IRValidationError(f"{label} has shape {result.shape}, expected {shape}")
    if not np.all(np.isfinite(result)):
        raise IRValidationError(f"{label} contains a non-finite value")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _symmetric_psd(
    value: Any,
    size: int,
    label: str,
    *,
    positive_definite: bool = False,
) -> np.ndarray:
    matrix = _array(value, (size, size), label)
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 1e-10 * scale
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=tolerance):
        raise IRValidationError(f"{label} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if positive_definite:
        if float(np.min(eigenvalues)) <= tolerance:
            raise IRValidationError(f"{label} must be positive definite")
    elif float(np.min(eigenvalues)) < -tolerance:
        raise IRValidationError(f"{label} must be positive semidefinite")
    return matrix


def _nested(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise IRValidationError(f"{key} must be an object")
    return value


@dataclass(frozen=True)
class OnlineModelIR:
    """Validated online affine/linearized model.

    Matrices use row-major mathematical shapes.  ``process_covariance`` is a
    continuous-time spectral covariance and is multiplied by ``dt`` during
    prediction.  Measurement covariance applies once per update.
    """

    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    measurement_names: tuple[str, ...]
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    process_covariance: np.ndarray
    measurement_covariance: np.ndarray
    initial_state: np.ndarray
    initial_covariance: np.ndarray
    dynamics_bias: np.ndarray
    measurement_bias: np.ndarray
    nominal_dt: float
    maximum_dt: float
    state_bounds: tuple[tuple[float | None, float | None], ...]
    kind: str = "linear"
    operating_point: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nx = len(self.state_names)
        nu = len(self.input_names)
        ny = len(self.measurement_names)
        _names(self.state_names, "state_names", _MAX_STATES)
        _names(self.input_names, "input_names", _MAX_INPUTS)
        _names(self.measurement_names, "measurement_names", _MAX_MEASUREMENTS)
        for attr, shape in (
            ("A", (nx, nx)),
            ("B", (nx, nu)),
            ("C", (ny, nx)),
            ("D", (ny, nu)),
            ("initial_state", (nx,)),
            ("dynamics_bias", (nx,)),
            ("measurement_bias", (ny,)),
        ):
            object.__setattr__(self, attr, _array(getattr(self, attr), shape, attr))
        object.__setattr__(
            self,
            "process_covariance",
            _symmetric_psd(self.process_covariance, nx, "process_covariance"),
        )
        object.__setattr__(
            self,
            "measurement_covariance",
            _symmetric_psd(
                self.measurement_covariance,
                ny,
                "measurement_covariance",
                positive_definite=True,
            ),
        )
        object.__setattr__(
            self,
            "initial_covariance",
            _symmetric_psd(self.initial_covariance, nx, "initial_covariance"),
        )
        nominal_dt = _finite(self.nominal_dt, "nominal_dt")
        maximum_dt = _finite(self.maximum_dt, "maximum_dt")
        if nominal_dt <= 0.0 or maximum_dt <= 0.0 or nominal_dt > maximum_dt:
            raise IRValidationError("require 0 < nominal_dt <= maximum_dt")
        object.__setattr__(self, "nominal_dt", nominal_dt)
        object.__setattr__(self, "maximum_dt", maximum_dt)
        if self.kind not in {"linear", "linearized"}:
            raise IRValidationError("online kind must be 'linear' or 'linearized'")
        if self.kind == "linearized" and not self.operating_point:
            raise IRValidationError("linearized models require an operating_point")
        if len(self.state_bounds) != nx:
            raise IRValidationError("state_bounds must have one pair per state")
        cleaned_bounds: list[tuple[float | None, float | None]] = []
        for index, pair in enumerate(self.state_bounds):
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise IRValidationError(f"state bound {index} must be [minimum, maximum]")
            low = None if pair[0] is None else _finite(pair[0], f"state bound {index} min")
            high = None if pair[1] is None else _finite(pair[1], f"state bound {index} max")
            if low is not None and high is not None and low > high:
                raise IRValidationError(f"state bound {index} is reversed")
            cleaned_bounds.append((low, high))
        object.__setattr__(self, "state_bounds", tuple(cleaned_bounds))
        if not isinstance(self.operating_point, Mapping) or not isinstance(self.metadata, Mapping):
            raise IRValidationError("operating_point and metadata must be objects")
        for key, value in self.operating_point.items():
            if not isinstance(key, str):
                raise IRValidationError("operating_point keys must be strings")
            _finite(value, f"operating_point.{key}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | Any) -> "OnlineModelIR":
        data = _plain_mapping(value)
        dynamics = _nested(data, "dynamics")
        measurement = _nested(data, "measurement")
        linearization = _nested(data, "linearization")

        def pick(*keys: str, sources: Sequence[Mapping[str, Any]] = ()) -> Any:
            for source in sources:
                for key in keys:
                    if key in source:
                        return source[key]
            for key in keys:
                if key in data:
                    return data[key]
            return None

        raw_a = pick("A", "state_matrix", sources=(dynamics, linearization))
        raw_b = pick("B", "input_matrix", sources=(dynamics, linearization))
        raw_c = pick("C", "H", "measurement_matrix", sources=(measurement, linearization))
        raw_d = pick("D", "feedthrough", sources=(measurement, linearization))
        if raw_a is None or raw_b is None or raw_c is None:
            raise IRValidationError("online IR requires A, B, and C/H matrices")
        try:
            a_probe = np.asarray(raw_a, dtype=np.float64)
            b_probe = np.asarray(raw_b, dtype=np.float64)
            c_probe = np.asarray(raw_c, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise IRValidationError("A, B, and C/H must be rectangular numeric arrays") from exc
        if a_probe.ndim != 2 or a_probe.shape[0] != a_probe.shape[1] or a_probe.shape[0] == 0:
            raise IRValidationError("A must be a non-empty square matrix")
        if b_probe.ndim != 2 or b_probe.shape[0] != a_probe.shape[0] or b_probe.shape[1] == 0:
            raise IRValidationError("B must have shape [states, positive inputs]")
        if c_probe.ndim != 2 or c_probe.shape[1] != a_probe.shape[0] or c_probe.shape[0] == 0:
            raise IRValidationError("C/H must have shape [positive measurements, states]")
        nx, nu, ny = a_probe.shape[0], b_probe.shape[1], c_probe.shape[0]
        state_names = data.get("state_names", [f"x{index}" for index in range(nx)])
        input_names = data.get("input_names", [f"u{index}" for index in range(nu)])
        measurement_names = data.get(
            "measurement_names",
            data.get("output_names", [f"y{index}" for index in range(ny)]),
        )
        bounds_data = data.get("state_bounds", [[None, None] for _ in range(nx)])
        if isinstance(bounds_data, Mapping):
            bounds_data = [bounds_data.get(name, [None, None]) for name in state_names]
        q = pick("process_covariance", "Q", sources=(dynamics,))
        r = pick("measurement_covariance", "R", sources=(measurement,))
        if q is None or r is None:
            raise IRValidationError(
                "online IR requires process_covariance/Q and measurement_covariance/R"
            )
        d_value = np.zeros((ny, nu)) if raw_d is None else raw_d
        return cls(
            state_names=_names(state_names, "state_names", _MAX_STATES),
            input_names=_names(input_names, "input_names", _MAX_INPUTS),
            measurement_names=_names(
                measurement_names, "measurement_names", _MAX_MEASUREMENTS
            ),
            A=raw_a,
            B=raw_b,
            C=raw_c,
            D=d_value,
            process_covariance=q,
            measurement_covariance=r,
            initial_state=data.get("initial_state", data.get("x0", np.zeros(nx))),
            initial_covariance=data.get("initial_covariance", data.get("P0", np.eye(nx))),
            dynamics_bias=pick("bias", "dynamics_bias", sources=(dynamics,))
            if pick("bias", "dynamics_bias", sources=(dynamics,)) is not None
            else np.zeros(nx),
            measurement_bias=pick("bias", "measurement_bias", sources=(measurement,))
            if pick("bias", "measurement_bias", sources=(measurement,)) is not None
            else np.zeros(ny),
            nominal_dt=data.get("nominal_dt", data.get("dt", 0.01)),
            maximum_dt=data.get("maximum_dt", data.get("max_dt", data.get("dt", 0.01))),
            state_bounds=tuple(bounds_data),
            kind=data.get("kind", linearization.get("kind", "linear")),
            operating_point=data.get(
                "operating_point", linearization.get("operating_point", {})
            ),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_data(cls, value: Mapping[str, Any] | Any) -> "OnlineModelIR":
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "state_names": list(self.state_names),
            "input_names": list(self.input_names),
            "measurement_names": list(self.measurement_names),
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "C": self.C.tolist(),
            "D": self.D.tolist(),
            "dynamics_bias": self.dynamics_bias.tolist(),
            "measurement_bias": self.measurement_bias.tolist(),
            "process_covariance": self.process_covariance.tolist(),
            "measurement_covariance": self.measurement_covariance.tolist(),
            "initial_state": self.initial_state.tolist(),
            "initial_covariance": self.initial_covariance.tolist(),
            "nominal_dt": self.nominal_dt,
            "maximum_dt": self.maximum_dt,
            "state_bounds": [list(pair) for pair in self.state_bounds],
            "operating_point": dict(self.operating_point),
            "metadata": dict(self.metadata),
        }


def _model_name(value: str) -> str:
    if not isinstance(value, str) or not _C_IDENTIFIER.fullmatch(value):
        raise IRValidationError("model_name must be a portable C identifier")
    if len(value) > 48:
        raise IRValidationError("model_name must be at most 48 characters")
    return value


def _c_number(value: float) -> str:
    value = 0.0 if value == 0.0 else float(value)
    return f"{value:.17g}"


def _c_values(array: np.ndarray) -> str:
    return ", ".join(_c_number(value) for value in array.reshape(-1))


def _header(ir: OnlineModelIR, prefix: str) -> str:
    upper = prefix.upper()
    nx, nu, ny = len(ir.state_names), len(ir.input_names), len(ir.measurement_names)
    return f"""/* Generated by contraption.physics.compiler; do not hand edit. */
#ifndef {upper}_ONLINE_MODEL_H
#define {upper}_ONLINE_MODEL_H

#include <stddef.h>

#define {upper}_NX {nx}
#define {upper}_NU {nu}
#define {upper}_NY {ny}
#define {upper}_STATE_COUNT {upper}_NX
#define {upper}_INPUT_COUNT {upper}_NU
#define {upper}_MEASUREMENT_COUNT {upper}_NY

#ifdef __cplusplus
extern \"C\" {{
#endif

typedef struct {{
    double x[{upper}_NX];
    double covariance[{upper}_NX * {upper}_NX];
}} {prefix}_filter_t;

void {prefix}_init({prefix}_filter_t *filter);
int {prefix}_predict({prefix}_filter_t *filter,
                     const double input[{upper}_NU], double dt);
int {prefix}_update({prefix}_filter_t *filter,
                    const double input[{upper}_NU],
                    const double measurement[{upper}_NY]);
void {prefix}_output(const {prefix}_filter_t *filter,
                     const double input[{upper}_NU],
                     double output[{upper}_NY]);

#ifdef __cplusplus
}}
#endif
#endif
"""


def _source(ir: OnlineModelIR, prefix: str) -> str:
    upper = prefix.upper()
    nx, nu, ny = len(ir.state_names), len(ir.input_names), len(ir.measurement_names)
    clamps: list[str] = []
    for index, (low, high) in enumerate(ir.state_bounds):
        if low is not None:
            clamps.append(
                f"    if (filter->x[{index}] < {_c_number(low)}) filter->x[{index}] = {_c_number(low)};"
            )
        if high is not None:
            clamps.append(
                f"    if (filter->x[{index}] > {_c_number(high)}) filter->x[{index}] = {_c_number(high)};"
            )
    clamp_code = "\n".join(clamps) if clamps else "    /* No state clamps declared. */"
    return f"""/* Generated fixed-allocation C99 affine simulator and EKF.
 * State covariance uses row-major storage. No heap allocation is performed.
 */
#include \"{prefix}.h\"

#include <math.h>
#include <string.h>

static const double MODEL_A[{upper}_NX * {upper}_NX] = {{ {_c_values(ir.A)} }};
static const double MODEL_B[{upper}_NX * {upper}_NU] = {{ {_c_values(ir.B)} }};
static const double MODEL_C[{upper}_NY * {upper}_NX] = {{ {_c_values(ir.C)} }};
static const double MODEL_D[{upper}_NY * {upper}_NU] = {{ {_c_values(ir.D)} }};
static const double MODEL_BIAS[{upper}_NX] = {{ {_c_values(ir.dynamics_bias)} }};
static const double MODEL_MEAS_BIAS[{upper}_NY] = {{ {_c_values(ir.measurement_bias)} }};
static const double MODEL_Q[{upper}_NX * {upper}_NX] = {{ {_c_values(ir.process_covariance)} }};
static const double MODEL_R[{upper}_NY * {upper}_NY] = {{ {_c_values(ir.measurement_covariance)} }};
static const double MODEL_X0[{upper}_NX] = {{ {_c_values(ir.initial_state)} }};
static const double MODEL_P0[{upper}_NX * {upper}_NX] = {{ {_c_values(ir.initial_covariance)} }};
static const double MODEL_MAX_DT = {_c_number(ir.maximum_dt)};

static int model_inverse(const double source[{upper}_NY * {upper}_NY],
                         double inverse[{upper}_NY * {upper}_NY]) {{
    double augmented[{upper}_NY][2 * {upper}_NY];
    size_t row, column, pivot;
    for (row = 0; row < {upper}_NY; ++row) {{
        for (column = 0; column < {upper}_NY; ++column) {{
            augmented[row][column] = source[row * {upper}_NY + column];
            augmented[row][column + {upper}_NY] = row == column ? 1.0 : 0.0;
        }}
    }}
    for (column = 0; column < {upper}_NY; ++column) {{
        pivot = column;
        for (row = column + 1; row < {upper}_NY; ++row) {{
            if (fabs(augmented[row][column]) > fabs(augmented[pivot][column])) {{
                pivot = row;
            }}
        }}
        if (!isfinite(augmented[pivot][column]) ||
            fabs(augmented[pivot][column]) < 1e-15) return -1;
        if (pivot != column) {{
            size_t item;
            for (item = 0; item < 2 * {upper}_NY; ++item) {{
                double temporary = augmented[column][item];
                augmented[column][item] = augmented[pivot][item];
                augmented[pivot][item] = temporary;
            }}
        }}
        {{
            const double divisor = augmented[column][column];
            size_t item;
            for (item = 0; item < 2 * {upper}_NY; ++item) {{
                augmented[column][item] /= divisor;
            }}
        }}
        for (row = 0; row < {upper}_NY; ++row) {{
            if (row != column) {{
                const double factor = augmented[row][column];
                size_t item;
                for (item = 0; item < 2 * {upper}_NY; ++item) {{
                    augmented[row][item] -= factor * augmented[column][item];
                }}
            }}
        }}
    }}
    for (row = 0; row < {upper}_NY; ++row) {{
        for (column = 0; column < {upper}_NY; ++column) {{
            inverse[row * {upper}_NY + column] = augmented[row][column + {upper}_NY];
        }}
    }}
    return 0;
}}

static void model_clamp_state({prefix}_filter_t *filter) {{
{clamp_code}
}}

void {prefix}_init({prefix}_filter_t *filter) {{
    if (filter == NULL) return;
    memcpy(filter->x, MODEL_X0, sizeof(MODEL_X0));
    memcpy(filter->covariance, MODEL_P0, sizeof(MODEL_P0));
    model_clamp_state(filter);
}}

int {prefix}_predict({prefix}_filter_t *filter,
                     const double input[{upper}_NU], double dt) {{
    double predicted[{upper}_NX] = {{0.0}};
    double transition[{upper}_NX * {upper}_NX] = {{0.0}};
    double temporary[{upper}_NX * {upper}_NX] = {{0.0}};
    double covariance[{upper}_NX * {upper}_NX] = {{0.0}};
    size_t row, column, inner;
    if (filter == NULL || input == NULL || !isfinite(dt) || dt <= 0.0 || dt > MODEL_MAX_DT) return -1;
    for (row = 0; row < {upper}_NX; ++row) {{
        double derivative = MODEL_BIAS[row];
        for (column = 0; column < {upper}_NX; ++column) {{
            derivative += MODEL_A[row * {upper}_NX + column] * filter->x[column];
            transition[row * {upper}_NX + column] =
                (row == column ? 1.0 : 0.0) + dt * MODEL_A[row * {upper}_NX + column];
        }}
        for (column = 0; column < {upper}_NU; ++column) {{
            derivative += MODEL_B[row * {upper}_NU + column] * input[column];
        }}
        predicted[row] = filter->x[row] + dt * derivative;
        if (!isfinite(predicted[row])) return -2;
    }}
    for (row = 0; row < {upper}_NX; ++row) {{
        for (column = 0; column < {upper}_NX; ++column) {{
            for (inner = 0; inner < {upper}_NX; ++inner) {{
                temporary[row * {upper}_NX + column] +=
                    transition[row * {upper}_NX + inner] *
                    filter->covariance[inner * {upper}_NX + column];
            }}
        }}
    }}
    for (row = 0; row < {upper}_NX; ++row) {{
        for (column = 0; column < {upper}_NX; ++column) {{
            for (inner = 0; inner < {upper}_NX; ++inner) {{
                covariance[row * {upper}_NX + column] +=
                    temporary[row * {upper}_NX + inner] *
                    transition[column * {upper}_NX + inner];
            }}
            covariance[row * {upper}_NX + column] += dt * MODEL_Q[row * {upper}_NX + column];
        }}
    }}
    memcpy(filter->x, predicted, sizeof(predicted));
    for (row = 0; row < {upper}_NX; ++row) {{
        for (column = 0; column < {upper}_NX; ++column) {{
            filter->covariance[row * {upper}_NX + column] = 0.5 *
                (covariance[row * {upper}_NX + column] + covariance[column * {upper}_NX + row]);
        }}
    }}
    model_clamp_state(filter);
    return 0;
}}

void {prefix}_output(const {prefix}_filter_t *filter,
                     const double input[{upper}_NU],
                     double output[{upper}_NY]) {{
    size_t row, column;
    if (filter == NULL || input == NULL || output == NULL) return;
    for (row = 0; row < {upper}_NY; ++row) {{
        output[row] = MODEL_MEAS_BIAS[row];
        for (column = 0; column < {upper}_NX; ++column) {{
            output[row] += MODEL_C[row * {upper}_NX + column] * filter->x[column];
        }}
        for (column = 0; column < {upper}_NU; ++column) {{
            output[row] += MODEL_D[row * {upper}_NU + column] * input[column];
        }}
    }}
}}

int {prefix}_update({prefix}_filter_t *filter,
                    const double input[{upper}_NU],
                    const double measurement[{upper}_NY]) {{
    double predicted[{upper}_NY] = {{0.0}};
    double innovation[{upper}_NY] = {{0.0}};
    double pc_transpose[{upper}_NX * {upper}_NY] = {{0.0}};
    double innovation_cov[{upper}_NY * {upper}_NY] = {{0.0}};
    double innovation_inverse[{upper}_NY * {upper}_NY] = {{0.0}};
    double gain[{upper}_NX * {upper}_NY] = {{0.0}};
    double identity_minus_gain_c[{upper}_NX * {upper}_NX] = {{0.0}};
    double temporary[{upper}_NX * {upper}_NX] = {{0.0}};
    double covariance[{upper}_NX * {upper}_NX] = {{0.0}};
    size_t row, column, inner;
    if (filter == NULL || input == NULL || measurement == NULL) return -1;
    {prefix}_output(filter, input, predicted);
    for (row = 0; row < {upper}_NY; ++row) {{
        if (!isfinite(measurement[row])) return -1;
        innovation[row] = measurement[row] - predicted[row];
    }}
    for (row = 0; row < {upper}_NX; ++row) {{
        for (column = 0; column < {upper}_NY; ++column) {{
            for (inner = 0; inner < {upper}_NX; ++inner) {{
                pc_transpose[row * {upper}_NY + column] +=
                    filter->covariance[row * {upper}_NX + inner] *
                    MODEL_C[column * {upper}_NX + inner];
            }}
        }}
    }}
    for (row = 0; row < {upper}_NY; ++row) {{
        for (column = 0; column < {upper}_NY; ++column) {{
            innovation_cov[row * {upper}_NY + column] = MODEL_R[row * {upper}_NY + column];
            for (inner = 0; inner < {upper}_NX; ++inner) {{
                innovation_cov[row * {upper}_NY + column] +=
                    MODEL_C[row * {upper}_NX + inner] *
                    pc_transpose[inner * {upper}_NY + column];
            }}
        }}
    }}
    if (model_inverse(innovation_cov, innovation_inverse) != 0) return -2;
    for (row = 0; row < {upper}_NX; ++row) {{
        for (column = 0; column < {upper}_NY; ++column) {{
            for (inner = 0; inner < {upper}_NY; ++inner) {{
                gain[row * {upper}_NY + column] +=
                    pc_transpose[row * {upper}_NY + inner] *
                    innovation_inverse[inner * {upper}_NY + column];
            }}
            filter->x[row] += gain[row * {upper}_NY + column] * innovation[column];
        }}
    }}
    for (row = 0; row < {upper}_NX; ++row) {{
        for (column = 0; column < {upper}_NX; ++column) {{
            identity_minus_gain_c[row * {upper}_NX + column] = row == column ? 1.0 : 0.0;
            for (inner = 0; inner < {upper}_NY; ++inner) {{
                identity_minus_gain_c[row * {upper}_NX + column] -=
                    gain[row * {upper}_NY + inner] * MODEL_C[inner * {upper}_NX + column];
            }}
        }}
    }}
    /* Joseph covariance update preserves symmetry/positive semidefiniteness. */
    for (row = 0; row < {upper}_NX; ++row) {{
        for (column = 0; column < {upper}_NX; ++column) {{
            for (inner = 0; inner < {upper}_NX; ++inner) {{
                temporary[row * {upper}_NX + column] +=
                    identity_minus_gain_c[row * {upper}_NX + inner] *
                    filter->covariance[inner * {upper}_NX + column];
            }}
        }}
    }}
    for (row = 0; row < {upper}_NX; ++row) {{
        for (column = 0; column < {upper}_NX; ++column) {{
            for (inner = 0; inner < {upper}_NX; ++inner) {{
                covariance[row * {upper}_NX + column] +=
                    temporary[row * {upper}_NX + inner] *
                    identity_minus_gain_c[column * {upper}_NX + inner];
            }}
            for (inner = 0; inner < {upper}_NY; ++inner) {{
                size_t second;
                for (second = 0; second < {upper}_NY; ++second) {{
                    covariance[row * {upper}_NX + column] +=
                        gain[row * {upper}_NY + inner] *
                        MODEL_R[inner * {upper}_NY + second] *
                        gain[column * {upper}_NY + second];
                }}
            }}
        }}
    }}
    for (row = 0; row < {upper}_NX; ++row) {{
        for (column = 0; column < {upper}_NX; ++column) {{
            filter->covariance[row * {upper}_NX + column] = 0.5 *
                (covariance[row * {upper}_NX + column] + covariance[column * {upper}_NX + row]);
        }}
    }}
    model_clamp_state(filter);
    return 0;
}}
"""


@dataclass(frozen=True)
class SyntaxCheckResult:
    ok: bool
    compiler: str | None
    command: tuple[str, ...] = ()
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class CompilationArtifact:
    """In-memory generated files with deterministic disk emission."""

    model_name: str
    header: str
    source: str
    manifest: Mapping[str, Any]

    @property
    def manifest_json(self) -> str:
        return json.dumps(self.manifest, indent=2, sort_keys=True) + "\n"

    @property
    def files(self) -> Mapping[str, str]:
        return {
            f"{self.model_name}.h": self.header,
            f"{self.model_name}.c": self.source,
            f"{self.model_name}.manifest.json": self.manifest_json,
        }

    def write(self, output_directory: str | Path) -> Mapping[str, Path]:
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for filename, contents in self.files.items():
            path = destination / filename
            path.write_text(contents, encoding="utf-8")
            written[filename] = path
        return written

    def syntax_check(self, compiler: str | None = None) -> SyntaxCheckResult:
        return syntax_check(self, compiler)


def _manifest(ir: OnlineModelIR, prefix: str, header: str, source: str) -> dict[str, Any]:
    canonical_ir = json.dumps(ir.to_dict(), sort_keys=True, separators=(",", ":"))
    return {
        "schema": "contraption.online-model-manifest/v1",
        "model_name": prefix,
        "source_ir_sha256": hashlib.sha256(canonical_ir.encode("utf-8")).hexdigest(),
        "generated_files": {
            "header": f"{prefix}.h",
            "source": f"{prefix}.c",
            "header_sha256": hashlib.sha256(header.encode("utf-8")).hexdigest(),
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        },
        "dimensions": {
            "states": len(ir.state_names),
            "inputs": len(ir.input_names),
            "measurements": len(ir.measurement_names),
        },
        "names": {
            "states": list(ir.state_names),
            "inputs": list(ir.input_names),
            "measurements": list(ir.measurement_names),
        },
        "numeric_contract": {
            "scalar": "IEEE-754 double",
            "layout": "row-major",
            "allocation": "fixed/stack-or-struct; no heap",
            "integration": "forward Euler",
            "filter": "extended Kalman filter with fixed approved Jacobians",
            "covariance_update": "Joseph form",
            "process_covariance": "continuous-time spectral covariance scaled by dt",
        },
        "validity": {
            "kind": ir.kind,
            "nominal_dt": ir.nominal_dt,
            "maximum_dt": ir.maximum_dt,
            "state_bounds": [list(pair) for pair in ir.state_bounds],
            "operating_point": dict(ir.operating_point),
        },
        "model": ir.to_dict(),
    }


class OnlineCompiler:
    """Compiler facade accepting :class:`OnlineModelIR` or a plain mapping."""

    def compile(
        self,
        model: OnlineModelIR | Mapping[str, Any] | Any,
        output_directory: str | Path | None = None,
        *,
        model_name: str = "contraption_model",
        check_syntax: bool = False,
        compiler: str | None = None,
    ) -> CompilationArtifact:
        ir = model if isinstance(model, OnlineModelIR) else OnlineModelIR.from_dict(model)
        prefix = _model_name(model_name)
        header = _header(ir, prefix)
        source = _source(ir, prefix)
        artifact = CompilationArtifact(
            prefix,
            header,
            source,
            _manifest(ir, prefix, header, source),
        )
        if output_directory is not None:
            artifact.write(output_directory)
        if check_syntax:
            result = artifact.syntax_check(compiler)
            if result.compiler is None:
                raise CompilerError("no C99 compiler found for requested syntax check")
            if not result.ok:
                raise CompilerError(f"generated C99 failed syntax check: {result.stderr}")
        return artifact

    def compile_contraption(
        self,
        resolved: Any,
        output_directory: str | Path | None = None,
        *,
        operating_state: Mapping[str, float] | Sequence[float] | None = None,
        operating_controls: Mapping[str, float] | Sequence[float] | None = None,
        operating_parameters: Mapping[str, float] | None = None,
        operating_time: float = 0.0,
        model_name: str = "contraption_model",
        check_syntax: bool = False,
        compiler: str | None = None,
        **options: Any,
    ) -> CompilationArtifact:
        """Compile only a canonical catalog-resolved PMDL assembly."""

        return compile_resolved_assembly(
            resolved,
            output_directory,
            operating_state=operating_state,
            operating_controls=operating_controls,
            operating_parameters=operating_parameters,
            operating_time=operating_time,
            model_name=model_name,
            check_syntax=check_syntax,
            compiler=compiler,
            **options,
        )


def compile_online_model(
    model: OnlineModelIR | Mapping[str, Any] | Any,
    output_directory: str | Path | None = None,
    *,
    model_name: str = "contraption_model",
    check_syntax: bool = False,
    compiler: str | None = None,
) -> CompilationArtifact:
    """Functional wrapper around :class:`OnlineCompiler`."""

    return OnlineCompiler().compile(
        model,
        output_directory,
        model_name=model_name,
        check_syntax=check_syntax,
        compiler=compiler,
    )


_CANONICAL_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _required_closure_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _CANONICAL_HASH.fullmatch(value) is None:
        raise IRValidationError(
            f"{label} must be 'sha256:' followed by 64 lowercase hex digits"
        )
    return value


def _controller_provenance(value: Any) -> dict[str, str] | None:
    """Verify and expose the controller identity retained by resolution."""

    from .controls import ControlProgram

    reference = value.specification.controller
    program = value.controller
    if reference is None:
        if program is not None:
            raise IRValidationError(
                "resolved controller has no canonical contraption reference"
            )
        return None
    if not isinstance(reference, Mapping):
        raise IRValidationError("resolved controller reference must be a mapping")
    try:
        result = {
            name: str(reference[name]) for name in ("id", "version", "sha256")
        }
    except KeyError as exc:
        raise IRValidationError(
            f"resolved controller reference is missing {exc.args[0]!r}"
        ) from exc
    if not isinstance(program, ControlProgram):
        raise IRValidationError(
            "resolved controller reference did not retain a parsed ControlProgram"
        )
    if result["id"] != program.name or result["version"] != program.version:
        raise IRValidationError(
            "resolved controller identity/version differs from its canonical reference"
        )
    expected = _required_closure_hash(result["sha256"], "controller sha256")
    payload = json.dumps(
        program.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    actual = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise IRValidationError(
            "resolved controller content hash no longer matches its canonical "
            f"reference: expected={expected}, actual={actual}"
        )
    return result


def _resolved_pmdl_system(value: Any) -> tuple[Any, str, str, dict[str, str] | None]:
    """Accept only the complete canonical resolution product."""

    from .assembly import AssembledPMDLSystem
    from .resolved import ResolvedAssembly

    if isinstance(value, OnlineModelIR):
        raise IRValidationError(
            "compile_resolved_assembly refuses authored OnlineModelIR; pass the "
            "canonical ResolvedAssembly"
        )
    if not isinstance(value, ResolvedAssembly):
        if isinstance(value, AssembledPMDLSystem):
            raise IRValidationError(
                "compile_resolved_assembly refuses a bare AssembledPMDLSystem because "
                "it loses the physical/catalog/controller closure; pass ResolvedAssembly"
            )
        raise IRValidationError(
            "compile_resolved_assembly accepts only ResolvedAssembly; authored "
            "mappings, OnlineModelIR, and partial projections are forbidden"
        )
    system = value.system
    assembly_sha256 = _required_closure_hash(
        value.assembly_sha256, "resolved assembly_sha256"
    )
    if not isinstance(system, AssembledPMDLSystem):
        raise IRValidationError(
            "ResolvedAssembly.system must be an AssembledPMDLSystem"
        )
    if system.assembly_sha256 != assembly_sha256:
        raise IRValidationError(
            "resolved physical/PMDL assembly hash mismatch: physical="
            f"{assembly_sha256}, PMDL={system.assembly_sha256}"
        )
    controller = _controller_provenance(value)

    pmdl_sha256 = _required_closure_hash(
        getattr(system, "pmdl_sha256", None), "pmdl_sha256"
    )
    diagnostics = getattr(system, "diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise IRValidationError("assembled system diagnostics must be a mapping")
    if diagnostics.get("assembly_sha256") != assembly_sha256:
        raise IRValidationError(
            "assembled diagnostics assembly_sha256 does not match canonical identity"
        )
    if diagnostics.get("pmdl_sha256") != pmdl_sha256:
        raise IRValidationError(
            "assembled diagnostics pmdl_sha256 does not match PMDL closure identity"
        )
    return system, assembly_sha256, pmdl_sha256, controller


def _point_vector(value: Any, names: Sequence[str], label: str) -> np.ndarray:
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(names))
        missing = sorted(set(names) - set(value))
        if unknown or missing:
            raise IRValidationError(
                f"{label} keys must exactly match declared names; "
                f"missing={missing}, unknown={unknown}"
            )
        raw = [value[name] for name in names]
    else:
        if isinstance(value, (str, bytes)):
            raise IRValidationError(f"{label} must be a numeric vector or mapping")
        raw = value
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise IRValidationError(f"{label} must be numeric") from exc
    if result.shape != (len(names),):
        raise IRValidationError(
            f"{label} has shape {result.shape}, expected {(len(names),)}"
        )
    if not np.all(np.isfinite(result)):
        raise IRValidationError(f"{label} contains a non-finite value")
    return np.array(result, dtype=np.float64, copy=True)


def _finite_difference_matrix(
    function: Any,
    point: np.ndarray,
    output_size: int,
    *,
    relative_step: float,
    label: str,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for index in range(point.size):
        step = relative_step * max(1.0, abs(float(point[index])))
        positive = np.array(point, copy=True)
        negative = np.array(point, copy=True)
        positive[index] += step
        negative[index] -= step
        above = np.asarray(function(positive), dtype=np.float64)
        below = np.asarray(function(negative), dtype=np.float64)
        if above.shape != (output_size,) or below.shape != (output_size,):
            raise IRValidationError(
                f"{label} evaluation returned an unexpected residual shape"
            )
        if not np.all(np.isfinite(above)) or not np.all(np.isfinite(below)):
            raise IRValidationError(
                f"{label} finite difference produced a non-finite residual "
                f"for column {index}"
            )
        columns.append((above - below) / (2.0 * step))
    if not columns:
        return np.empty((output_size, 0), dtype=np.float64)
    return np.stack(columns, axis=1)


def compile_resolved_assembly(
    resolved: Any,
    output_directory: str | Path | None = None,
    *,
    operating_state: Mapping[str, float] | Sequence[float] | None = None,
    operating_controls: Mapping[str, float] | Sequence[float] | None = None,
    operating_parameters: Mapping[str, float] | None = None,
    operating_time: float = 0.0,
    model_name: str = "contraption_model",
    nominal_dt: float | None = None,
    maximum_dt: float | None = None,
    process_covariance: Any | None = None,
    measurement_covariance: Any | None = None,
    initial_covariance: Any | None = None,
    relative_step: float = 1e-6,
    newton_tolerance: float = 1e-10,
    newton_max_iterations: int = 20,
    maximum_condition_number: float = 1e12,
    expected_assembly_sha256: str | None = None,
    expected_pmdl_sha256: str | None = None,
    check_syntax: bool = False,
    compiler: str | None = None,
) -> CompilationArtifact:
    """Derive target C99 directly from a canonical assembled PMDL DAE.

    The differential-state dynamics are obtained from the implicit-function
    theorem.  With differential state ``x``, algebraics ``a``, and
    ``q=[xdot,a]``, the operating point solves ``F(x,q,u)=0``.  Central finite
    differences form ``G=dF/dq``, ``H=dF/dx``, and ``J=dF/du``; the first
    ``len(x)`` rows of ``-G^-1 [H,J]`` are the approved local ``A`` and ``B``.
    No caller-authored online matrices are accepted by this entry point.
    """

    from .backend import NumpyBackend
    from .resolved import ResolutionError

    system, assembly_sha256, pmdl_sha256, controller = _resolved_pmdl_system(resolved)
    try:
        dynamics_record = resolved.dynamics_completeness
    except ResolutionError as exc:
        raise IRValidationError(
            "resolved assembly lacks a valid mandatory dynamics_completeness record"
        ) from exc
    dynamics_completeness = dynamics_record.to_dict()
    if expected_assembly_sha256 is not None:
        expected = _required_closure_hash(
            expected_assembly_sha256, "expected_assembly_sha256"
        )
        if expected != assembly_sha256:
            raise IRValidationError(
                f"assembly hash mismatch: resolved={assembly_sha256}, expected={expected}"
            )
    if expected_pmdl_sha256 is not None:
        expected = _required_closure_hash(
            expected_pmdl_sha256, "expected_pmdl_sha256"
        )
        if expected != pmdl_sha256:
            raise IRValidationError(
                f"PMDL hash mismatch: resolved={pmdl_sha256}, expected={expected}"
            )

    step = _finite(relative_step, "relative_step")
    tolerance = _finite(newton_tolerance, "newton_tolerance")
    condition_limit = _finite(
        maximum_condition_number, "maximum_condition_number"
    )
    if step <= 0.0 or tolerance <= 0.0 or condition_limit <= 1.0:
        raise IRValidationError(
            "relative_step/newton_tolerance must be positive and "
            "maximum_condition_number must exceed one"
        )
    if isinstance(newton_max_iterations, bool) or not isinstance(
        newton_max_iterations, int
    ) or newton_max_iterations <= 0:
        raise IRValidationError("newton_max_iterations must be a positive integer")
    time_value = _finite(operating_time, "operating_time")

    state_names = tuple(str(name) for name in system.state_names)
    differential_names = tuple(
        str(name) for name in system.differential_state_names
    )
    if not differential_names:
        raise IRValidationError(
            "resolved assembly has no differential PMDL states to compile"
        )
    if len(differential_names) > _MAX_STATES:
        raise IRValidationError(
            f"resolved assembly has {len(differential_names)} differential states; "
            f"C99 limit is {_MAX_STATES}"
        )
    differential_indices = tuple(state_names.index(name) for name in differential_names)
    differential_set = set(differential_indices)
    algebraic_indices = tuple(
        index for index in range(len(state_names)) if index not in differential_set
    )
    algebraic_names = tuple(state_names[index] for index in algebraic_indices)
    residual_names = tuple(str(name) for name in system.residual_names)
    if len(residual_names) != len(state_names):
        raise IRValidationError(
            "assembled DAE must remain square at compilation: "
            f"residuals={len(residual_names)}, unknowns={len(state_names)}"
        )

    initial_full = _point_vector(
        system.initial_state, state_names, "assembled initial_state"
    )
    if operating_state is None:
        full_guess = initial_full
        x_operating = initial_full[list(differential_indices)]
    elif isinstance(operating_state, Mapping) and set(operating_state) == set(state_names):
        full_guess = _point_vector(
            operating_state, state_names, "operating_state"
        )
        x_operating = full_guess[list(differential_indices)]
    else:
        x_operating = _point_vector(
            operating_state, differential_names, "operating_state"
        )
        full_guess = np.array(initial_full, copy=True)
        full_guess[list(differential_indices)] = x_operating

    control_names = tuple(str(name) for name in system.control_names)
    if not control_names:
        raise IRValidationError(
            "existing fixed-allocation C99 generator requires at least one control input"
        )
    if len(control_names) > _MAX_INPUTS:
        raise IRValidationError(
            f"resolved assembly has {len(control_names)} controls; C99 limit is {_MAX_INPUTS}"
        )
    defaults = dict(system.control_defaults)
    if operating_controls is None:
        missing = sorted(set(control_names) - set(defaults))
        if missing:
            raise IRValidationError(
                "operating_controls is required because controls lack defaults: "
                f"{missing}"
            )
        u_operating = _point_vector(defaults, control_names, "control defaults")
    elif isinstance(operating_controls, Mapping):
        unknown = sorted(set(operating_controls) - set(control_names))
        if unknown:
            raise IRValidationError(
                f"operating_controls contains unknown controls: {unknown}"
            )
        merged_controls = dict(defaults)
        merged_controls.update(operating_controls)
        missing = sorted(set(control_names) - set(merged_controls))
        if missing:
            raise IRValidationError(
                f"operating_controls is missing controls without defaults: {missing}"
            )
        u_operating = _point_vector(
            merged_controls, control_names, "operating_controls"
        )
    else:
        u_operating = _point_vector(
            operating_controls, control_names, "operating_controls"
        )

    parameter_names = tuple(str(name) for name in system.default_parameters)
    parameter_values = dict(system.default_parameters)
    if operating_parameters is not None:
        if not isinstance(operating_parameters, Mapping):
            raise IRValidationError("operating_parameters must be a mapping")
        unknown = sorted(set(operating_parameters) - set(parameter_names))
        if unknown:
            raise IRValidationError(
                f"operating_parameters contains unknown parameters: {unknown}"
            )
        parameter_values.update(operating_parameters)
    parameter_vector = _point_vector(
        parameter_values, parameter_names, "operating_parameters"
    ) if parameter_names else np.empty((0,), dtype=np.float64)
    parameter_values = {
        name: float(parameter_vector[index])
        for index, name in enumerate(parameter_names)
    }
    for name, value in parameter_values.items():
        low, high = system.parameter_bounds.get(name, (None, None))
        if (low is not None and value < low) or (high is not None and value > high):
            raise IRValidationError(
                f"operating parameter {name!r}={value} is outside bounds [{low}, {high}]"
            )

    validity_ranges = dict(getattr(system.validity, "ranges", {}))
    state_bounds: list[tuple[float | None, float | None]] = []
    for index, name in enumerate(differential_names):
        bounds = validity_ranges.get(name)
        pair = (
            (None, None)
            if bounds is None
            else (getattr(bounds, "lower", None), getattr(bounds, "upper", None))
        )
        state_bounds.append(pair)
        if (pair[0] is not None and x_operating[index] < pair[0]) or (
            pair[1] is not None and x_operating[index] > pair[1]
        ):
            raise IRValidationError(
                f"operating state {name!r}={x_operating[index]} is outside "
                f"validity range [{pair[0]}, {pair[1]}]"
            )

    backend = NumpyBackend()
    parameter_batch = {
        name: np.asarray([value], dtype=np.float64)
        for name, value in parameter_values.items()
    }

    nx = len(differential_names)
    na = len(algebraic_names)
    equation_count = len(residual_names)

    def evaluate(x_value: np.ndarray, q_value: np.ndarray, u_value: np.ndarray) -> np.ndarray:
        state = np.zeros((1, len(state_names)), dtype=np.float64)
        state_derivative = np.zeros_like(state)
        state[0, list(differential_indices)] = x_value
        if na:
            state[0, list(algebraic_indices)] = q_value[nx:]
        state_derivative[0, list(differential_indices)] = q_value[:nx]
        controls = {
            name: np.asarray([u_value[index]], dtype=np.float64)
            for index, name in enumerate(control_names)
        }
        try:
            result = system.residual(
                time_value,
                state,
                state_derivative,
                parameter_batch,
                controls,
                backend,
            )
        except Exception as exc:
            raise IRValidationError(
                f"assembled PMDL residual evaluation failed during C99 derivation: {exc}"
            ) from exc
        array = np.asarray(result, dtype=np.float64)
        if array.shape != (1, equation_count):
            raise IRValidationError(
                "assembled PMDL residual returned shape "
                f"{array.shape}, expected {(1, equation_count)}"
            )
        vector = array[0]
        if not np.all(np.isfinite(vector)):
            bad = int(np.flatnonzero(~np.isfinite(vector))[0])
            raise IRValidationError(
                f"assembled residual {residual_names[bad]!r} is non-finite at the operating point"
            )
        return vector

    q_operating = np.concatenate(
        [np.zeros(nx, dtype=np.float64), full_guess[list(algebraic_indices)]]
    )
    iterations = 0
    for iteration in range(newton_max_iterations):
        iterations = iteration + 1
        value = evaluate(x_operating, q_operating, u_operating)
        if float(np.max(np.abs(value))) <= tolerance:
            break
        g_iteration = _finite_difference_matrix(
            lambda point: evaluate(x_operating, point, u_operating),
            q_operating,
            equation_count,
            relative_step=step,
            label="dF/dq",
        )
        rank = int(np.linalg.matrix_rank(g_iteration))
        if rank != equation_count:
            worst = int(np.argmax(np.abs(value)))
            raise IRValidationError(
                "DAE operating-point solve has singular dF/d[xdot,a]: "
                f"rank={rank}/{equation_count}, worst_residual="
                f"{residual_names[worst]!r} ({value[worst]:.17g})"
            )
        condition = float(np.linalg.cond(g_iteration))
        if not math.isfinite(condition) or condition > condition_limit:
            raise IRValidationError(
                "DAE operating-point Jacobian is ill-conditioned: "
                f"condition={condition:.17g}, limit={condition_limit:.17g}"
            )
        try:
            update = np.linalg.solve(g_iteration, -value)
        except np.linalg.LinAlgError as exc:
            raise IRValidationError(
                f"DAE operating-point linear solve failed: {exc}"
            ) from exc
        if not np.all(np.isfinite(update)):
            raise IRValidationError("DAE operating-point Newton update is non-finite")
        q_operating += update
    final_residual = evaluate(x_operating, q_operating, u_operating)
    worst_residual = float(np.max(np.abs(final_residual)))
    if worst_residual > max(tolerance * 10.0, 1e-9):
        worst = int(np.argmax(np.abs(final_residual)))
        raise IRValidationError(
            "DAE operating-point solve did not converge after "
            f"{iterations} iteration(s); worst_residual={residual_names[worst]!r} "
            f"({final_residual[worst]:.17g})"
        )

    g_matrix = _finite_difference_matrix(
        lambda point: evaluate(x_operating, point, u_operating),
        q_operating,
        equation_count,
        relative_step=step,
        label="dF/dq",
    )
    rank = int(np.linalg.matrix_rank(g_matrix))
    if g_matrix.shape != (equation_count, equation_count) or rank != equation_count:
        raise IRValidationError(
            "DAE implicit-function Jacobian must be square and full rank: "
            f"shape={g_matrix.shape}, rank={rank}/{equation_count}"
        )
    condition = float(np.linalg.cond(g_matrix))
    if not math.isfinite(condition) or condition > condition_limit:
        raise IRValidationError(
            "DAE implicit-function Jacobian is ill-conditioned: "
            f"condition={condition:.17g}, limit={condition_limit:.17g}"
        )
    h_matrix = _finite_difference_matrix(
        lambda point: evaluate(point, q_operating, u_operating),
        x_operating,
        equation_count,
        relative_step=step,
        label="dF/dx",
    )
    j_matrix = _finite_difference_matrix(
        lambda point: evaluate(x_operating, q_operating, point),
        u_operating,
        equation_count,
        relative_step=step,
        label="dF/du",
    )
    try:
        sensitivity = -np.linalg.solve(
            g_matrix, np.concatenate([h_matrix, j_matrix], axis=1)
        )
    except np.linalg.LinAlgError as exc:
        raise IRValidationError(
            f"DAE implicit-function sensitivity solve failed: {exc}"
        ) from exc
    if not np.all(np.isfinite(sensitivity)):
        raise IRValidationError("DAE linearization contains non-finite sensitivities")
    a_matrix = sensitivity[:nx, :nx]
    b_matrix = sensitivity[:nx, nx:]
    xdot_operating = q_operating[:nx]
    dynamics_bias = xdot_operating - a_matrix @ x_operating - b_matrix @ u_operating

    declared_maximum = getattr(system.validity, "max_timestep", None)
    if declared_maximum is None and maximum_dt is None:
        raise IRValidationError(
            "assembled PMDL validity does not declare max_timestep; maximum_dt is required"
        )
    max_dt = _finite(
        declared_maximum if maximum_dt is None else maximum_dt, "maximum_dt"
    )
    if max_dt <= 0.0:
        raise IRValidationError("maximum_dt must be positive")
    if declared_maximum is not None and max_dt > float(declared_maximum):
        raise IRValidationError(
            f"maximum_dt {max_dt} exceeds assembled validity limit {declared_maximum}"
        )
    chosen_nominal = min(0.01, max_dt) if nominal_dt is None else _finite(
        nominal_dt, "nominal_dt"
    )
    if chosen_nominal <= 0.0 or chosen_nominal > max_dt:
        raise IRValidationError("require 0 < nominal_dt <= maximum_dt")

    if nx > _MAX_MEASUREMENTS:
        raise IRValidationError(
            "identity differential-state observation exceeds existing C99 measurement "
            f"limit {_MAX_MEASUREMENTS}"
        )
    q_covariance = (
        np.zeros((nx, nx), dtype=np.float64)
        if process_covariance is None
        else process_covariance
    )
    r_covariance = (
        np.eye(nx, dtype=np.float64) * 1e-9
        if measurement_covariance is None
        else measurement_covariance
    )
    p0_covariance = (
        np.zeros((nx, nx), dtype=np.float64)
        if initial_covariance is None
        else initial_covariance
    )
    operating_point = {"time": time_value}
    operating_point.update(
        {f"state:{name}": float(x_operating[index]) for index, name in enumerate(differential_names)}
    )
    operating_point.update(
        {f"control:{name}": float(u_operating[index]) for index, name in enumerate(control_names)}
    )
    controller_execution = (
        {
            "emitted": False,
            "contract": "no_controller_declared",
            "reason": "the resolved assembly does not declare a ControlProgram",
        }
        if controller is None
        else {
            "emitted": False,
            "contract": "canonical_control_program_runtime_supplies_resolved_control_sources",
            "reason": "this artifact contains only DAE-derived dynamics and estimator code",
        }
    )
    metadata = {
        "source": "canonical_resolved_pmdl_assembly",
        "assembly_sha256": assembly_sha256,
        "pmdl_sha256": pmdl_sha256,
        "controller": controller,
        "controller_execution": controller_execution,
        "dynamics_completeness": dynamics_completeness,
        "dae_linearization": {
            "method": "implicit_function_central_difference",
            "equation_count": equation_count,
            "differential_state_names": list(differential_names),
            "algebraic_names": list(algebraic_names),
            "operating_state_derivative": xdot_operating.tolist(),
            "operating_algebraics": q_operating[nx:].tolist(),
            "dF_dq_rank": rank,
            "dF_dq_condition": condition,
            "relative_step": step,
            "newton_tolerance": tolerance,
            "newton_iterations": iterations,
            "residual_max": worst_residual,
        },
        "measurement_contract": {
            "kind": "identity_differential_state_projection",
            "default_covariance_is_numerical_regularization": measurement_covariance is None,
        },
    }
    ir = OnlineModelIR(
        state_names=differential_names,
        input_names=control_names,
        measurement_names=differential_names,
        A=a_matrix,
        B=b_matrix,
        C=np.eye(nx, dtype=np.float64),
        D=np.zeros((nx, len(control_names)), dtype=np.float64),
        process_covariance=q_covariance,
        measurement_covariance=r_covariance,
        initial_state=x_operating,
        initial_covariance=p0_covariance,
        dynamics_bias=dynamics_bias,
        measurement_bias=np.zeros(nx, dtype=np.float64),
        nominal_dt=chosen_nominal,
        maximum_dt=max_dt,
        state_bounds=tuple(state_bounds),
        kind="linearized",
        operating_point=operating_point,
        metadata=metadata,
    )

    prefix = _model_name(model_name)
    controller_comment = (
        "none"
        if controller is None
        else f"{controller['id']}@{controller['version']} ({controller['sha256']})"
    )
    controller_execution_comment = (
        " * Controller execution: NOT APPLICABLE; no controller is declared.\n"
        if controller is None
        else " * Controller execution: NOT EMITTED; canonical controller outputs must "
        "supply the resolved control-source inputs.\n"
    )
    dynamics_comment = (
        " * Dynamics completeness: "
        + dynamics_record.status.upper()
        + "; open gates: "
        + (
            ", ".join(gate.id for gate in dynamics_record.open_gates)
            if dynamics_record.open_gates
            else "none"
        )
        + ".\n"
    )
    identity_comment = (
        "/* Canonical assembly: " + assembly_sha256 + "\n"
        " * PMDL closure: " + pmdl_sha256 + "\n"
        " * Controller: " + controller_comment + "\n"
        + controller_execution_comment
        + dynamics_comment
        + " */\n"
    )
    header = identity_comment + _header(ir, prefix)
    source = identity_comment + _source(ir, prefix)
    manifest = _manifest(ir, prefix, header, source)
    manifest["assembly_sha256"] = assembly_sha256
    manifest["pmdl_sha256"] = pmdl_sha256
    manifest["controller"] = controller
    manifest["controller_execution"] = dict(metadata["controller_execution"])
    manifest["dynamics_completeness"] = dynamics_completeness
    manifest["derivation"] = dict(metadata["dae_linearization"])
    artifact = CompilationArtifact(prefix, header, source, manifest)
    if output_directory is not None:
        artifact.write(output_directory)
    if check_syntax:
        result = artifact.syntax_check(compiler)
        if result.compiler is None:
            raise CompilerError("no C99 compiler found for requested syntax check")
        if not result.ok:
            raise CompilerError(f"generated C99 failed syntax check: {result.stderr}")
    return artifact


def compile_contraption(
    resolved: Any,
    output_directory: str | Path | None = None,
    *,
    operating_state: Mapping[str, float] | Sequence[float] | None = None,
    operating_controls: Mapping[str, float] | Sequence[float] | None = None,
    operating_parameters: Mapping[str, float] | None = None,
    operating_time: float = 0.0,
    model_name: str = "contraption_model",
    check_syntax: bool = False,
    compiler: str | None = None,
    **options: Any,
) -> CompilationArtifact:
    """Compile a contraption from its canonical resolved PMDL assembly.

    Raw contraption mappings, hand-authored online matrices, and reviewed
    aggregate abstractions are deliberately outside this API.  Every emitted
    artifact is therefore bound to the exact physical/catalog/model closure by
    ``assembly_sha256`` and to the composed PMDL closure by ``pmdl_sha256``.
    """

    return compile_resolved_assembly(
        resolved,
        output_directory,
        operating_state=operating_state,
        operating_controls=operating_controls,
        operating_parameters=operating_parameters,
        operating_time=operating_time,
        model_name=model_name,
        check_syntax=check_syntax,
        compiler=compiler,
        **options,
    )


def syntax_check(
    artifact: CompilationArtifact, compiler: str | None = None
) -> SyntaxCheckResult:
    """Ask an available GCC/Clang-compatible compiler to parse generated C99."""

    selected = compiler
    if selected is None:
        selected = next(
            (candidate for candidate in ("cc", "gcc", "clang") if shutil.which(candidate)),
            None,
        )
    if selected is None:
        return SyntaxCheckResult(False, None)
    executable = shutil.which(selected) or selected
    with tempfile.TemporaryDirectory(prefix="contraption-c99-") as temporary:
        paths = artifact.write(temporary)
        command = (
            executable,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            str(paths[f"{artifact.model_name}.c"]),
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return SyntaxCheckResult(
            completed.returncode == 0,
            executable,
            command,
            completed.stdout,
            completed.stderr,
        )


__all__ = [
    "CompilationArtifact",
    "CompilerError",
    "IRValidationError",
    "OnlineCompiler",
    "OnlineModelIR",
    "SyntaxCheckResult",
    "compile_contraption",
    "compile_online_model",
    "compile_resolved_assembly",
    "syntax_check",
]
