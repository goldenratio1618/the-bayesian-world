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


_ASSEMBLY_COVERAGE_SCHEMA = "contraption.online-assembly-coverage/v1"


def _coverage_names(
    values: Any, label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IRValidationError(f"{label} must be an array")
    result = tuple(str(value) for value in values)
    if not allow_empty and not result:
        raise IRValidationError(f"{label} may not be empty")
    if any(not value for value in result):
        raise IRValidationError(f"{label} entries may not be empty")
    if len(set(result)) != len(result):
        raise IRValidationError(f"{label} entries must be unique")
    return result


@dataclass(frozen=True)
class AssemblyCoverage:
    """Reviewed declaration of the exact contraption scope represented by an IR.

    Coverage is deliberately stronger than a list of admitted model names: it
    binds component instances to model references and binds the reviewed IR to
    a canonical hash of every connection kind, domain, and endpoint.  A review
    record is mandatory when no full validated model registry is supplied.
    """

    component_ids: tuple[str, ...]
    connection_ids: tuple[str, ...]
    component_models: Mapping[str, str]
    topology_sha256: str
    review: Mapping[str, Any] | None = None
    schema: str = _ASSEMBLY_COVERAGE_SCHEMA

    @classmethod
    def from_dict(cls, value: Any) -> "AssemblyCoverage":
        if not isinstance(value, Mapping):
            raise IRValidationError("metadata.assembly_coverage must be an object")
        if any(not isinstance(key, str) for key in value):
            raise IRValidationError(
                "metadata.assembly_coverage field names must be strings"
            )
        allowed = {
            "schema",
            "component_ids",
            "connection_ids",
            "component_models",
            "topology_sha256",
            "review",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise IRValidationError(
                "metadata.assembly_coverage has unknown fields: " + ", ".join(unknown)
            )
        required = allowed - {"review"}
        missing = sorted(required - set(value))
        if missing:
            raise IRValidationError(
                "metadata.assembly_coverage is missing fields: " + ", ".join(missing)
            )
        schema = value.get("schema")
        if schema != _ASSEMBLY_COVERAGE_SCHEMA:
            raise IRValidationError(
                f"metadata.assembly_coverage.schema must be {_ASSEMBLY_COVERAGE_SCHEMA!r}"
            )
        component_ids = _coverage_names(
            value.get("component_ids"), "metadata.assembly_coverage.component_ids"
        )
        connection_ids = _coverage_names(
            value.get("connection_ids"),
            "metadata.assembly_coverage.connection_ids",
            allow_empty=True,
        )
        component_models = value.get("component_models")
        if not isinstance(component_models, Mapping):
            raise IRValidationError(
                "metadata.assembly_coverage.component_models must be an object"
            )
        cleaned_models: dict[str, str] = {}
        for key, model_reference in component_models.items():
            if not isinstance(key, str) or not key:
                raise IRValidationError(
                    "metadata.assembly_coverage.component_models keys must be non-empty strings"
                )
            if not isinstance(model_reference, str) or not model_reference:
                raise IRValidationError(
                    f"metadata.assembly_coverage.component_models.{key} must be a non-empty string"
                )
            cleaned_models[key] = model_reference
        if set(cleaned_models) != set(component_ids):
            raise IRValidationError(
                "metadata.assembly_coverage.component_models keys must exactly match component_ids"
            )
        topology_sha256 = value.get("topology_sha256")
        if not isinstance(topology_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", topology_sha256
        ):
            raise IRValidationError(
                "metadata.assembly_coverage.topology_sha256 must be a lowercase SHA-256 digest"
            )
        review = value.get("review")
        if review is not None and not isinstance(review, Mapping):
            raise IRValidationError("metadata.assembly_coverage.review must be an object")
        return cls(
            component_ids,
            connection_ids,
            cleaned_models,
            topology_sha256,
            None if review is None else dict(review),
            schema,
        )

    def require_reviewed_abstraction(self) -> Mapping[str, Any]:
        """Validate the explicit human/vendor abstraction boundary."""

        if self.review is None:
            raise IRValidationError(
                "a full validated model registry or metadata.assembly_coverage.review is required"
            )
        if any(not isinstance(key, str) for key in self.review):
            raise IRValidationError(
                "metadata.assembly_coverage.review field names must be strings"
            )
        allowed = {
            "review_id",
            "reviewed_by",
            "basis",
            "component_contracts_reviewed",
            "ports_and_connections_reviewed",
            "assembled_ir_coverage_reviewed",
            "limitations",
        }
        unknown = sorted(set(self.review) - allowed)
        missing = sorted(allowed - set(self.review))
        if unknown:
            raise IRValidationError(
                "metadata.assembly_coverage.review has unknown fields: "
                + ", ".join(unknown)
            )
        if missing:
            raise IRValidationError(
                "metadata.assembly_coverage.review is missing fields: "
                + ", ".join(missing)
            )
        for key in ("review_id", "reviewed_by", "basis"):
            if not isinstance(self.review.get(key), str) or not self.review[key].strip():
                raise IRValidationError(
                    f"metadata.assembly_coverage.review.{key} must be a non-empty string"
                )
        for key in (
            "component_contracts_reviewed",
            "ports_and_connections_reviewed",
            "assembled_ir_coverage_reviewed",
        ):
            if self.review.get(key) is not True:
                raise IRValidationError(
                    f"metadata.assembly_coverage.review.{key} must be true"
                )
        limitations = self.review.get("limitations")
        if (
            isinstance(limitations, (str, bytes))
            or not isinstance(limitations, Sequence)
            or not limitations
            or any(not isinstance(item, str) or not item.strip() for item in limitations)
        ):
            raise IRValidationError(
                "metadata.assembly_coverage.review.limitations must be a non-empty string array"
            )
        return self.review


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
    return f"""/* Generated by contraption.compiler; do not hand edit. */
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
        specification: Mapping[str, Any] | Any,
        model_registry: Mapping[str, Any] | None = None,
        assembled_system: OnlineModelIR | Mapping[str, Any] | Any | None = None,
        output_directory: str | Path | None = None,
        *,
        operating_point: Mapping[str, float] | None = None,
        model_name: str = "contraption_model",
        check_syntax: bool = False,
        compiler: str | None = None,
    ) -> CompilationArtifact:
        """Validate a contraption compilation scope, then compile its assembly."""

        return compile_contraption(
            specification,
            model_registry,
            assembled_system,
            output_directory,
            operating_point=operating_point,
            model_name=model_name,
            check_syntax=check_syntax,
            compiler=compiler,
            _compiler=self,
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


def _records(value: Any, label: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        records: list[Mapping[str, Any]] = []
        for key in sorted(value, key=str):
            item = dict(_plain_mapping(value[key]))
            item.setdefault("id", str(key))
            records.append(item)
        return records
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_mapping(item) for item in value]
    raise IRValidationError(f"{label} must be an array or object")


def _component_model_reference(component: Mapping[str, Any]) -> str:
    model = component.get("model", component.get("model_id"))
    if isinstance(model, Mapping):
        for key in ("id", "name", "model_id", "qualified_name"):
            if key in model:
                reference = model[key]
                if isinstance(reference, str) and reference:
                    return reference
                break
    if isinstance(model, str) and model:
        return model
    raise IRValidationError(
        f"component {component.get('id', '?')!r} lacks an unambiguous model reference"
    )


def _admission_mapping(value: Any) -> Mapping[str, Any] | None:
    try:
        data = _plain_mapping(value)
    except IRValidationError:
        return None
    admission = data.get("online_admission", data.get("online_compilation"))
    metadata = data.get("metadata", {})
    if admission is None and isinstance(metadata, Mapping):
        admission = metadata.get("online_admission", metadata.get("online_compilation"))
    if isinstance(admission, bool):
        return {"admitted": admission, "kind": "linear"}
    return admission if isinstance(admission, Mapping) else None


def _connection_endpoints(
    connection: Mapping[str, Any], *, label: str = "connection"
) -> tuple[tuple[str, str], ...]:
    endpoints = connection.get("endpoints")
    if not isinstance(endpoints, Sequence) or isinstance(endpoints, (str, bytes)):
        for first, second in (("from", "to"), ("source", "target"), ("a", "b")):
            if first in connection or second in connection:
                endpoints = [connection.get(first), connection.get(second)]
                break
    if not isinstance(endpoints, Sequence) or isinstance(endpoints, (str, bytes)):
        raise IRValidationError(f"{label}.endpoints must be an array")
    if len(endpoints) < 2:
        raise IRValidationError(f"{label} must have at least two endpoints")
    result: list[tuple[str, str]] = []
    for index, endpoint in enumerate(endpoints):
        if isinstance(endpoint, str):
            try:
                component, port = endpoint.rsplit(".", 1)
            except ValueError as exc:
                raise IRValidationError(
                    f"{label}.endpoints[{index}] must be 'component.port'"
                ) from exc
        elif isinstance(endpoint, Mapping):
            component = endpoint.get("component", endpoint.get("component_id"))
            port = endpoint.get("port", endpoint.get("port_id"))
        else:
            raise IRValidationError(
                f"{label}.endpoints[{index}] must be a string or object"
            )
        if not isinstance(component, str) or not component:
            raise IRValidationError(
                f"{label}.endpoints[{index}] has no component identifier"
            )
        if not isinstance(port, str) or not port:
            raise IRValidationError(f"{label}.endpoints[{index}] has no port identifier")
        result.append((component, port))
    return tuple(result)


def _connection_component_ids(connection: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(component for component, _ in _connection_endpoints(connection))


def _topology_payload(
    components: Sequence[Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    component_rows = []
    for component in components:
        component_id = str(component.get("id", component.get("name", "")))
        component_rows.append(
            {"id": component_id, "model": _component_model_reference(component)}
        )
    connection_rows = []
    for index, connection in enumerate(connections):
        connection_id = connection.get("id")
        if not isinstance(connection_id, str) or not connection_id:
            raise IRValidationError(
                f"contraption.connections[{index}] needs a non-empty identifier"
            )
        kind = connection.get("kind")
        if kind not in {"power", "signal", "attachment", "constraint"}:
            raise IRValidationError(
                f"connection {connection_id!r} has unsupported kind {kind!r}"
            )
        endpoints = _connection_endpoints(
            connection, label=f"contraption.connections[{index}]"
        )
        domain = connection.get("domain")
        if domain is not None and (not isinstance(domain, str) or not domain):
            raise IRValidationError(
                f"connection {connection_id!r} domain must be a non-empty string or null"
            )
        connection_rows.append(
            {
                "id": connection_id,
                "kind": kind,
                "domain": domain,
                "endpoints": sorted(
                    (
                        {"component": component, "port": port}
                        for component, port in endpoints
                    ),
                    key=lambda item: (item["component"], item["port"]),
                ),
            }
        )
    return {
        "components": sorted(component_rows, key=lambda item: item["id"]),
        "connections": sorted(connection_rows, key=lambda item: item["id"]),
    }


def _topology_sha256(
    components: Sequence[Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]],
) -> str:
    canonical = json.dumps(
        _topology_payload(components, connections),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _scope_mismatch(label: str, expected: Sequence[str], actual: Sequence[str]) -> str | None:
    expected_set = set(expected)
    actual_set = set(actual)
    if expected_set == actual_set and len(expected) == len(actual):
        return None
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    return f"{label} mismatch; missing={missing}, extra={extra}"


def _validate_assembly_coverage(
    metadata: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]],
    component_ids: Sequence[str],
    connection_ids: Sequence[str],
    model_references: Sequence[str],
) -> AssemblyCoverage:
    raw_coverage = metadata.get("assembly_coverage")
    if raw_coverage is None:
        legacy = metadata.get("admitted_models")
        suffix = (
            "; metadata.admitted_models is only a string list and is not an admission contract"
            if legacy is not None
            else ""
        )
        raise IRValidationError(
            "assembled IR metadata must include explicit assembly_coverage" + suffix
        )
    coverage = AssemblyCoverage.from_dict(raw_coverage)
    for message in (
        _scope_mismatch("assembly component_ids", component_ids, coverage.component_ids),
        _scope_mismatch("assembly connection_ids", connection_ids, coverage.connection_ids),
    ):
        if message is not None:
            raise IRValidationError(message)
    expected_models = dict(zip(component_ids, model_references, strict=True))
    if dict(coverage.component_models) != expected_models:
        mismatches = sorted(
            component_id
            for component_id in set(expected_models) | set(coverage.component_models)
            if expected_models.get(component_id) != coverage.component_models.get(component_id)
        )
        raise IRValidationError(
            "assembly component_models do not match the contraption for: "
            + ", ".join(mismatches)
        )
    expected_hash = _topology_sha256(components, connections)
    if coverage.topology_sha256 != expected_hash:
        raise IRValidationError(
            "assembly topology_sha256 does not match the contraption topology"
        )
    return coverage


def _validate_registry_scope(
    specification: Mapping[str, Any] | Any,
    components: Sequence[Mapping[str, Any]],
    model_registry: Mapping[str, Any],
) -> None:
    """Validate full PMDL models, their admission records, and composition."""

    from .specs import ContraptionSpec, ModelSpec, SpecError
    from .validation import validate_contraption, validate_model

    normalized: dict[str, ModelSpec] = {}
    for component in components:
        reference = _component_model_reference(component)
        if reference in normalized:
            continue
        try:
            candidate = model_registry[reference]
        except KeyError as exc:
            raise IRValidationError(
                f"model {reference!r} is absent from the supplied model registry"
            ) from exc
        try:
            model = (
                candidate
                if isinstance(candidate, ModelSpec)
                else ModelSpec.from_dict(_plain_mapping(candidate))
            )
        except (IRValidationError, SpecError) as exc:
            raise IRValidationError(
                f"registry entry {reference!r} is not a valid full PMDL model: {exc}"
            ) from exc
        if model.id != reference:
            raise IRValidationError(
                f"registry key {reference!r} contains model {model.id!r}"
            )
        report = validate_model(model)
        if not report.valid:
            details = "; ".join(
                f"[{issue.code}] {issue.path}: {issue.message}"
                for issue in report.errors
            )
            raise IRValidationError(
                f"model {reference!r} failed PMDL validation: {details}"
            )
        admission = _admission_mapping(model)
        if admission is None or admission.get("admitted") is not True:
            raise IRValidationError(
                f"model {reference!r} has no affirmative online_admission contract"
            )
        kind = str(admission.get("kind", admission.get("model_kind", ""))).lower()
        if kind not in {"linear", "affine", "linearized", "linearizable"}:
            raise IRValidationError(
                f"model {reference!r} has unsupported online kind {kind!r}"
            )
        mechanics = str(
            admission.get("mechanics", admission.get("mechanical_fidelity", "rigid_body"))
        ).lower()
        if mechanics in {"nonrigid", "non-rigid", "flexible", "deformable"}:
            raise IRValidationError(
                f"model {reference!r} mixes non-rigid mechanics into the Phase 1 online scope"
            )
        normalized[reference] = model

    try:
        strict_spec = (
            specification
            if isinstance(specification, ContraptionSpec)
            else ContraptionSpec.from_dict(_plain_mapping(specification))
        )
    except (IRValidationError, SpecError) as exc:
        raise IRValidationError(
            f"contraption schema validation failed before compilation: {exc}"
        ) from exc
    composition = validate_contraption(strict_spec, normalized)
    if not composition.valid:
        details = "; ".join(
            f"[{issue.code}] {issue.path}: {issue.message}"
            for issue in composition.errors
        )
        raise IRValidationError(
            f"contraption compatibility validation failed: {details}"
        )


def compile_contraption(
    specification: Mapping[str, Any] | Any,
    model_registry: Mapping[str, Any] | None = None,
    assembled_system: OnlineModelIR | Mapping[str, Any] | Any | None = None,
    output_directory: str | Path | None = None,
    *,
    operating_point: Mapping[str, float] | None = None,
    model_name: str = "contraption_model",
    check_syntax: bool = False,
    compiler: str | None = None,
    _compiler: OnlineCompiler | None = None,
) -> CompilationArtifact:
    """Compile a validated contraption and its assembled linearization.

    Network assembly/linearization is kept behind an explicit trusted adapter:
    pass a data-only assembled IR, or an object implementing
    ``assemble_online_ir(specification, operating_point=...)``.  The assembled
    IR must carry ``metadata.assembly_coverage`` binding it to the exact
    component instances, model references, connection identifiers, and hashed
    endpoint topology.  Admission then requires either a full validated PMDL
    registry or an explicit reviewed-abstraction record in that coverage.
    Unstructured metadata strings and component-local claims are never
    sufficient for target-side execution.
    """

    spec = _plain_mapping(specification)
    components = _records(spec.get("components"), "components")
    connections = _records(spec.get("connections"), "connections")
    if not components:
        raise IRValidationError("contraption compilation requires components")
    component_ids = [item.get("id", item.get("name")) for item in components]
    if (
        any(not isinstance(value, str) or not value for value in component_ids)
        or len(set(component_ids)) != len(component_ids)
    ):
        raise IRValidationError("contraption components need unique non-empty identifiers")
    connection_ids = [item.get("id") for item in connections]
    if (
        any(not isinstance(value, str) or not value for value in connection_ids)
        or len(set(connection_ids)) != len(connection_ids)
    ):
        raise IRValidationError("contraption connections need unique non-empty identifiers")
    scope_models = [_component_model_reference(component) for component in components]

    known = set(component_ids)
    for index, connection in enumerate(connections):
        connection_id = connection_ids[index]
        endpoint_ids = _connection_component_ids(connection)
        endpoints = _connection_endpoints(
            connection, label=f"contraption.connections[{index}]"
        )
        if any(value not in known for value in endpoint_ids):
            raise IRValidationError(
                f"connection {connection_id!r} contains an unknown component endpoint"
            )
        if len(set(endpoints)) != len(endpoints):
            raise IRValidationError(
                f"connection {connection_id!r} repeats an endpoint"
            )
        kind = connection.get("kind")
        if kind not in {"power", "signal", "attachment", "constraint"}:
            raise IRValidationError(
                f"connection {connection_id!r} has unsupported kind {kind!r}"
            )
        connection_metadata = connection.get("metadata", {})
        if isinstance(connection_metadata, Mapping) and connection_metadata.get("online_supported") is False:
            raise IRValidationError(
                f"connection {connection_id!r} is explicitly excluded from online compilation"
            )

    # Locate the assembly before admission because its coverage contract binds
    # the numeric IR to the complete contraption scope.
    assembly = assembled_system
    spec_metadata = spec.get("metadata", {})
    spec_metadata = spec_metadata if isinstance(spec_metadata, Mapping) else {}
    if assembly is None:
        assembly = spec.get(
            "online_model",
            spec.get(
                "online_ir",
                spec_metadata.get("online_model", spec_metadata.get("online_ir")),
            ),
        )
    if assembly is None:
        raise IRValidationError(
            "contraption requires a reviewed assembled_system/online_ir; component matrices are not composed implicitly"
        )
    if not isinstance(assembly, (OnlineModelIR, Mapping)):
        adapter = getattr(assembly, "assemble_online_ir", None)
        if not callable(adapter):
            adapter = getattr(assembly, "linearize", None)
        if not callable(adapter):
            raise IRValidationError(
                "assembled_system must be data-only IR or a trusted assemble_online_ir adapter"
            )
        try:
            assembly = adapter(specification, operating_point=operating_point or {})
        except TypeError:
            # Support compact trusted adapters whose documented signature takes
            # only the operating point.
            assembly = adapter(operating_point=operating_point or {})
    ir_data = assembly.to_dict() if isinstance(assembly, OnlineModelIR) else dict(_plain_mapping(assembly))
    assembly_metadata = ir_data.get("metadata", {})
    if not isinstance(assembly_metadata, Mapping):
        raise IRValidationError("assembled IR metadata must be an object")
    coverage = _validate_assembly_coverage(
        assembly_metadata,
        components,
        connections,
        component_ids,
        connection_ids,
        scope_models,
    )

    if model_registry is None:
        review = coverage.require_reviewed_abstraction()
        validation_level = "reviewed_abstraction"
        admission_summary = (
            "reviewed abstraction coverage; referenced PMDL models were not registry-validated"
        )
    else:
        if not isinstance(model_registry, Mapping):
            raise IRValidationError("model_registry must implement Mapping")
        _validate_registry_scope(specification, components, model_registry)
        review = None
        validation_level = "validated_model_registry"
        admission_summary = "all referenced models validated through explicit registry"

    if operating_point is not None:
        ir_data["operating_point"] = {
            str(key): _finite(value, f"operating_point.{key}")
            for key, value in operating_point.items()
        }
        if ir_data.get("kind", "linear") == "linear":
            ir_data["kind"] = "linearized"
    scope = {
        "contraption_id": str(spec.get("id", spec.get("name", "contraption"))),
        "component_ids": component_ids,
        "connection_ids": connection_ids,
        "model_references": scope_models,
        "topology_sha256": coverage.topology_sha256,
        "operating_point": dict(operating_point or ir_data.get("operating_point", {})),
        "validation_level": validation_level,
        "models_registry_validated": model_registry is not None,
        "admission": admission_summary,
    }
    if review is not None:
        scope["review"] = {
            "review_id": review["review_id"],
            "reviewed_by": review["reviewed_by"],
            "basis": review["basis"],
            "limitations": list(review["limitations"]),
        }
    ir_metadata = dict(assembly_metadata)
    ir_metadata["contraption_scope"] = scope
    ir_data["metadata"] = ir_metadata
    ir = OnlineModelIR.from_dict(ir_data)
    selected_compiler = _compiler or OnlineCompiler()
    artifact = selected_compiler.compile(
        ir,
        None,
        model_name=model_name,
        check_syntax=check_syntax,
        compiler=compiler,
    )
    manifest = dict(artifact.manifest)
    manifest["contraption_scope"] = scope
    final_artifact = CompilationArtifact(
        artifact.model_name, artifact.header, artifact.source, manifest
    )
    if output_directory is not None:
        final_artifact.write(output_directory)
    return final_artifact


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


# Compatibility aliases with explicit semantics.
Compiler = OnlineCompiler
CompileResult = CompilationArtifact
compile_c99 = compile_online_model


__all__ = [
    "AssemblyCoverage",
    "CompilationArtifact",
    "CompileResult",
    "Compiler",
    "CompilerError",
    "IRValidationError",
    "OnlineCompiler",
    "OnlineModelIR",
    "SyntaxCheckResult",
    "compile_c99",
    "compile_contraption",
    "compile_online_model",
    "syntax_check",
]
