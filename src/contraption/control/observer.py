"""Plant-derived coupled affine observers for ``control-1`` programs.

The observer is never authored as matrices.  It is derived from the canonical
resolved PMDL descriptor system at its initial operating point.  Differential
states form the internal Kalman state; algebraic sensor and latent quantities
are projected with the same implicit-function sensitivities used to derive the
local dynamics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from ..physics.backend import NumpyBackend
from ..physics.dsl import Binary, Call, Literal, Symbol, Unary
from ..physics.specs import FrozenDict
from .specs import ControlSpec, control_digest


class ObserverDerivationError(ValueError):
    """A resolved PMDL/controller closure cannot admit a local observer."""


_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ObserverDerivationError(
            f"{label} must be 'sha256:' followed by 64 lowercase hex digits"
        )
    return value


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ObserverDerivationError(f"{label} must be numeric") from exc
    if result.shape != shape:
        raise ObserverDerivationError(
            f"{label} has shape {result.shape}, expected {shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ObserverDerivationError(f"{label} contains a non-finite value")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _names(value: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ObserverDerivationError(f"{label} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ObserverDerivationError(f"{label} must be unique")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(_freeze_json(item) for item in value.tolist())
    if isinstance(value, Mapping):
        return FrozenDict(
            (str(key), _freeze_json(item)) for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _control_structural_targets(
    system: Any,
    control_name: str,
    target_variables: Sequence[str],
) -> tuple[str, ...]:
    """Return observer variables structurally reachable from one PMDL control.

    Reachability is deliberately expression-structural, not derivative-based:
    a relation containing ``u ** 2`` remains dependent on ``u`` even at the
    zero operating point.
    """

    state_names = tuple(str(name) for name in getattr(system, "state_names", ()))
    state_index = {name: index for index, name in enumerate(state_names)}
    layouts = getattr(system, "_layouts", None)
    network_equations = getattr(system, "_network_equations", None)
    if layouts is None or network_equations is None:
        raise ObserverDerivationError(
            "assembled PMDL system lacks structural relation metadata required "
            f"to qualify unowned control {control_name!r}"
        )

    equation_dependencies: list[frozenset[int]] = []
    equation_controls: list[str | None] = []
    for layout in layouts:
        unknown_indices = dict(getattr(layout, "unknown_indices", {}))
        derivative_indices = dict(getattr(layout, "derivative_indices", {}))
        for _, expression in getattr(layout, "relations", ()):
            related: set[int] = set()
            for symbol in expression.variables():
                if symbol in unknown_indices:
                    related.add(int(unknown_indices[symbol]))
                if symbol in derivative_indices:
                    related.add(int(derivative_indices[symbol]))
            equation_dependencies.append(frozenset(related))
            equation_controls.append(None)
    for equation in network_equations:
        dependencies = getattr(equation, "dependencies", None)
        if dependencies is None:
            dependencies = frozenset(
                int(index)
                for index, coefficient in getattr(equation, "terms", ())
                if float(coefficient) != 0.0
            )
        equation_dependencies.append(frozenset(int(index) for index in dependencies))
        source = getattr(equation, "control_source", None)
        equation_controls.append(None if source is None else str(source))

    seeds = {
        index
        for index, source in enumerate(equation_controls)
        if source == control_name
    }
    if not seeds:
        raise ObserverDerivationError(
            f"unowned PMDL control {control_name!r} has no structural drive equation"
        )
    equations_by_variable: dict[int, set[int]] = {}
    for equation_index, dependencies in enumerate(equation_dependencies):
        for variable_index in dependencies:
            equations_by_variable.setdefault(variable_index, set()).add(equation_index)

    reachable_equations = set(seeds)
    reachable_variables: set[int] = set()
    pending = list(seeds)
    while pending:
        equation_index = pending.pop()
        for variable_index in equation_dependencies[equation_index]:
            if variable_index in reachable_variables:
                continue
            reachable_variables.add(variable_index)
            for neighbor in equations_by_variable.get(variable_index, ()):
                if neighbor not in reachable_equations:
                    reachable_equations.add(neighbor)
                    pending.append(neighbor)

    return tuple(
        name
        for name in target_variables
        if name in state_index and state_index[name] in reachable_variables
    )


def _reject_time_dependent_relations(system: Any) -> None:
    layouts = getattr(system, "_layouts", None)
    if layouts is None:
        raise ObserverDerivationError(
            "assembled PMDL system lacks relation metadata for time-dependence admission"
        )
    for layout in layouts:
        for relation_name, expression in getattr(layout, "relations", ()):
            if "t" in expression.variables():
                component = getattr(getattr(layout, "component", None), "id", "?")
                raise ObserverDerivationError(
                    "time-dependent PMDL relation "
                    f"{component}.{relation_name} cannot be represented by a "
                    "time-invariant local affine observer"
                )


def _matrix_exponential(matrix: np.ndarray) -> np.ndarray:
    """Higham scaling-and-squaring Padé(13) matrix exponential."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ObserverDerivationError("matrix exponential requires a square matrix")
    size = value.shape[0]
    if size == 0:
        return np.empty((0, 0), dtype=np.float64)
    theta_13 = 5.371920351148152
    norm = float(np.linalg.norm(value, 1))
    scale_power = (
        0
        if norm <= theta_13
        else max(0, int(math.ceil(math.log2(norm / theta_13))))
    )
    scaled = value / float(2**scale_power)
    identity = np.eye(size, dtype=np.float64)
    a2 = scaled @ scaled
    a4 = a2 @ a2
    a6 = a4 @ a2
    coefficients = (
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    )
    u = scaled @ (
        a6
        @ (
            coefficients[13] * a6
            + coefficients[11] * a4
            + coefficients[9] * a2
        )
        + coefficients[7] * a6
        + coefficients[5] * a4
        + coefficients[3] * a2
        + coefficients[1] * identity
    )
    v = (
        a6
        @ (
            coefficients[12] * a6
            + coefficients[10] * a4
            + coefficients[8] * a2
        )
        + coefficients[6] * a6
        + coefficients[4] * a4
        + coefficients[2] * a2
        + coefficients[0] * identity
    )
    try:
        result = np.linalg.solve(v - u, v + u)
    except np.linalg.LinAlgError as exc:
        raise ObserverDerivationError(
            "exact observer discretization matrix exponential failed"
        ) from exc
    for _ in range(scale_power):
        result = result @ result
    if not np.all(np.isfinite(result)):
        raise ObserverDerivationError(
            "exact observer discretization produced non-finite values"
        )
    return result


def _exact_discretization(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    bias: np.ndarray,
    process_covariance: np.ndarray,
    period_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, nu = b_matrix.shape
    affine = np.zeros((nx + nu + 1, nx + nu + 1), dtype=np.float64)
    affine[:nx, :nx] = a_matrix
    affine[:nx, nx : nx + nu] = b_matrix
    affine[:nx, -1] = bias
    affine_exponential = _matrix_exponential(period_s * affine)
    transition = affine_exponential[:nx, :nx]
    discrete_input = affine_exponential[:nx, nx : nx + nu]
    discrete_bias = affine_exponential[:nx, -1]

    van_loan = np.zeros((2 * nx, 2 * nx), dtype=np.float64)
    van_loan[:nx, :nx] = a_matrix
    van_loan[:nx, nx:] = process_covariance
    van_loan[nx:, nx:] = -a_matrix.T
    covariance_exponential = _matrix_exponential(period_s * van_loan)
    discrete_covariance = covariance_exponential[:nx, nx:] @ transition.T
    discrete_covariance = 0.5 * (
        discrete_covariance + discrete_covariance.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(discrete_covariance)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(discrete_covariance))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ObserverDerivationError(
            "exact discrete process covariance is not positive semidefinite"
        )
    discrete_covariance = (
        eigenvectors * np.maximum(eigenvalues, 0.0)[None, :]
    ) @ eigenvectors.T
    cleanup_tolerance = 1e-14 * max(
        1.0, float(np.max(np.abs(discrete_covariance)))
    )
    discrete_covariance[
        np.abs(discrete_covariance) < cleanup_tolerance
    ] = 0.0
    return transition, discrete_input, discrete_bias, discrete_covariance


@dataclass(frozen=True, slots=True)
class ObservabilityDiagnostic:
    """Exact local observability-rank result for one requested latent."""

    implicit_input: str
    variable: str
    observable: bool
    rank: int
    augmented_rank: int
    state_dimension: int
    measurement_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "implicit_input": self.implicit_input,
            "variable": self.variable,
            "observable": self.observable,
            "rank": self.rank,
            "augmented_rank": self.augmented_rank,
            "state_dimension": self.state_dimension,
            "measurement_names": list(self.measurement_names),
        }


@dataclass(frozen=True, slots=True)
class AffineObserverModel:
    """Immutable local PMDL observer shared by runtime and code generators."""

    controller_id: str
    controller_digest: str
    controller_link_digest: str
    assembly_sha256: str
    pmdl_sha256: str
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    plant_input_names: tuple[str, ...]
    measurement_names: tuple[str, ...]
    measurement_variables: tuple[str, ...]
    latent_names: tuple[str, ...]
    latent_variables: tuple[str, ...]
    A: np.ndarray
    B: np.ndarray
    dynamics_bias: np.ndarray
    C: np.ndarray
    D: np.ndarray
    measurement_bias: np.ndarray
    L: np.ndarray
    M: np.ndarray
    latent_bias: np.ndarray
    process_covariance: np.ndarray
    transition: np.ndarray
    discrete_input: np.ndarray
    discrete_bias: np.ndarray
    discrete_process_covariance: np.ndarray
    measurement_variance: np.ndarray
    initial_state: np.ndarray
    initial_covariance: np.ndarray
    latent_lower_bounds: tuple[float | None, ...]
    latent_upper_bounds: tuple[float | None, ...]
    period_s: float
    operating_point: Mapping[str, Any]
    derivation: Mapping[str, Any]
    validity: Mapping[str, Any]
    dynamics_completeness: Mapping[str, Any]

    def __post_init__(self) -> None:
        _hash(self.controller_digest, "controller_digest")
        _hash(self.controller_link_digest, "controller_link_digest")
        _hash(self.assembly_sha256, "assembly_sha256")
        _hash(self.pmdl_sha256, "pmdl_sha256")
        states = _names(self.state_names, "state_names")
        inputs = _names(self.input_names, "input_names")
        plant_inputs = _names(self.plant_input_names, "plant_input_names")
        measurements = _names(self.measurement_names, "measurement_names")
        measurement_variables = _names(
            self.measurement_variables, "measurement_variables"
        )
        latents = _names(self.latent_names, "latent_names")
        latent_variables = _names(self.latent_variables, "latent_variables")
        if not states or not latents:
            raise ObserverDerivationError(
                "an affine observer requires differential states and latent outputs"
            )
        if len(inputs) != len(plant_inputs):
            raise ObserverDerivationError(
                "controller and PMDL input manifests must have the same length"
            )
        if len(measurements) != len(measurement_variables):
            raise ObserverDerivationError(
                "measurement names and PMDL variables must have the same length"
            )
        if len(latents) != len(latent_variables):
            raise ObserverDerivationError(
                "latent names and PMDL variables must have the same length"
            )
        nx, nu, ny, nz = len(states), len(inputs), len(measurements), len(latents)
        for name, shape in (
            ("A", (nx, nx)),
            ("B", (nx, nu)),
            ("dynamics_bias", (nx,)),
            ("C", (ny, nx)),
            ("D", (ny, nu)),
            ("measurement_bias", (ny,)),
            ("L", (nz, nx)),
            ("M", (nz, nu)),
            ("latent_bias", (nz,)),
            ("process_covariance", (nx, nx)),
            ("transition", (nx, nx)),
            ("discrete_input", (nx, nu)),
            ("discrete_bias", (nx,)),
            ("discrete_process_covariance", (nx, nx)),
            ("measurement_variance", (ny,)),
            ("initial_state", (nx,)),
            ("initial_covariance", (nx, nx)),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        for name in (
            "process_covariance",
            "discrete_process_covariance",
            "initial_covariance",
        ):
            matrix = getattr(self, name)
            scale = max(1.0, float(np.max(np.abs(matrix))))
            if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10 * scale):
                raise ObserverDerivationError(f"{name} must be symmetric")
            if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-10 * scale:
                raise ObserverDerivationError(f"{name} must be positive semidefinite")
        if ny and np.any(self.measurement_variance <= 0.0):
            raise ObserverDerivationError("measurement variances must be positive")
        if len(self.latent_lower_bounds) != nz or len(self.latent_upper_bounds) != nz:
            raise ObserverDerivationError("latent bounds must cover every latent")
        if not math.isfinite(self.period_s) or self.period_s <= 0.0:
            raise ObserverDerivationError("period_s must be positive and finite")
        for name in (
            "operating_point",
            "derivation",
            "validity",
            "dynamics_completeness",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ObserverDerivationError(f"{name} must be a mapping")
            object.__setattr__(self, name, _freeze_json(value))
        qualification = self.derivation.get("sampled_local_linearity_evidence")
        if not isinstance(qualification, Mapping):
            raise ObserverDerivationError(
                "observer derivation must contain sampled local-linearity evidence"
            )
        for field in (
            "kind",
            "sample_radius_relative",
            "maximum_sampled_remainder",
            "observed_sampled_remainder",
            "sample_count",
            "scope",
        ):
            if field not in qualification:
                raise ObserverDerivationError(
                    f"observer derivation qualification is missing {field!r}"
                )

    @property
    def transition_matrix(self) -> np.ndarray:
        return self.transition

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @property
    def observability(self) -> tuple[ObservabilityDiagnostic, ...]:
        return observability_diagnostics(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "contraption-affine-observer/v1",
            "controller_id": self.controller_id,
            "controller_digest": self.controller_digest,
            "controller_link_digest": self.controller_link_digest,
            "assembly_sha256": self.assembly_sha256,
            "pmdl_sha256": self.pmdl_sha256,
            "state_names": list(self.state_names),
            "input_names": list(self.input_names),
            "plant_input_names": list(self.plant_input_names),
            "measurement_names": list(self.measurement_names),
            "measurement_variables": list(self.measurement_variables),
            "latent_names": list(self.latent_names),
            "latent_variables": list(self.latent_variables),
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "dynamics_bias": self.dynamics_bias.tolist(),
            "C": self.C.tolist(),
            "D": self.D.tolist(),
            "measurement_bias": self.measurement_bias.tolist(),
            "L": self.L.tolist(),
            "M": self.M.tolist(),
            "latent_bias": self.latent_bias.tolist(),
            "process_covariance": self.process_covariance.tolist(),
            "transition": self.transition.tolist(),
            "discrete_input": self.discrete_input.tolist(),
            "discrete_bias": self.discrete_bias.tolist(),
            "discrete_process_covariance": self.discrete_process_covariance.tolist(),
            "measurement_variance": self.measurement_variance.tolist(),
            "initial_state": self.initial_state.tolist(),
            "initial_covariance": self.initial_covariance.tolist(),
            "latent_lower_bounds": list(self.latent_lower_bounds),
            "latent_upper_bounds": list(self.latent_upper_bounds),
            "period_s": self.period_s,
            "operating_point": _jsonable(self.operating_point),
            "derivation": _jsonable(self.derivation),
            "validity": _jsonable(self.validity),
            "dynamics_completeness": _jsonable(self.dynamics_completeness),
        }


def observability_diagnostics(
    observer: AffineObserverModel, *, tolerance: float | None = None
) -> tuple[ObservabilityDiagnostic, ...]:
    """Return row-space observability admission for each latent projection."""

    if not isinstance(observer, AffineObserverModel):
        raise TypeError("observability_diagnostics requires an AffineObserverModel")
    nx = len(observer.state_names)
    if tolerance is not None and (
        not math.isfinite(tolerance) or tolerance < 0.0
    ):
        raise ValueError("observability tolerance must be finite and non-negative")
    if len(observer.measurement_names):
        blocks: list[np.ndarray] = []
        power = np.eye(nx, dtype=np.float64)
        transition = observer.transition_matrix
        for _ in range(nx):
            blocks.append(observer.C @ power)
            power = power @ transition
        matrix = np.concatenate(blocks, axis=0)
    else:
        matrix = np.empty((0, nx), dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix, tol=tolerance))
    diagnostics: list[ObservabilityDiagnostic] = []
    for index, name in enumerate(observer.latent_names):
        augmented = np.concatenate([matrix, observer.L[index : index + 1]], axis=0)
        augmented_rank = int(np.linalg.matrix_rank(augmented, tol=tolerance))
        diagnostics.append(
            ObservabilityDiagnostic(
                name,
                observer.latent_variables[index],
                augmented_rank == rank,
                rank,
                augmented_rank,
                nx,
                observer.measurement_names,
            )
        )
    return tuple(diagnostics)


def _point_vector(value: Any, names: Sequence[str], label: str) -> np.ndarray:
    if isinstance(value, Mapping):
        missing = sorted(set(names) - set(value))
        unknown = sorted(set(value) - set(names))
        if missing or unknown:
            raise ObserverDerivationError(
                f"{label} keys must exactly match names; missing={missing}, unknown={unknown}"
            )
        value = [value[name] for name in names]
    return _array(value, (len(names),), label)


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
            raise ObserverDerivationError(
                f"{label} evaluation returned an unexpected residual shape"
            )
        if not np.all(np.isfinite(above)) or not np.all(np.isfinite(below)):
            raise ObserverDerivationError(
                f"{label} finite difference produced a non-finite residual at column {index}"
            )
        columns.append((above - below) / (2.0 * step))
    if not columns:
        return np.empty((output_size, 0), dtype=np.float64)
    return np.stack(columns, axis=1)


def _covariance_from_projection(
    projection: np.ndarray, variances: np.ndarray, label: str
) -> np.ndarray:
    """Construct a PSD state covariance matching requested latent variances.

    Rank-dependent latent aliases are permitted when their requested diagonal
    variances are mutually consistent.  The non-negative rank-one weights are
    solved in the latent projection basis and checked exactly after recovery.
    """

    nx = projection.shape[1]
    if np.all(variances == 0.0):
        return np.zeros((nx, nx), dtype=np.float64)
    gram_squared = (projection @ projection.T) ** 2
    weights = np.linalg.lstsq(gram_squared, variances, rcond=None)[0]
    scale = max(1.0, float(np.max(variances)))
    if np.any(weights < -1e-10 * scale):
        raise ObserverDerivationError(
            f"{label} cannot be represented by a positive-semidefinite coupled "
            "state covariance for the requested latent projections"
        )
    weights = np.maximum(weights, 0.0)
    covariance = projection.T @ np.diag(weights) @ projection
    recovered = np.einsum("ij,jk,ik->i", projection, covariance, projection)
    if not np.allclose(recovered, variances, rtol=1e-8, atol=1e-10 * scale):
        raise ObserverDerivationError(
            f"{label} is inconsistent for linearly dependent latent bindings; "
            f"requested={variances.tolist()}, realizable={recovered.tolist()}"
        )
    return 0.5 * (covariance + covariance.T)


def _affine_degree(node: Any, dynamic_symbols: set[str]) -> int:
    """Return 0 for constant, 1 for affine, and 2 for nonlinear/unproven."""

    if isinstance(node, Literal):
        return 0
    if isinstance(node, Symbol):
        return int(node.name in dynamic_symbols)
    if isinstance(node, Unary):
        return _affine_degree(node.operand, dynamic_symbols)
    if isinstance(node, Binary):
        left = _affine_degree(node.left, dynamic_symbols)
        right = _affine_degree(node.right, dynamic_symbols)
        if node.operator in {"+", "-"}:
            return min(2, max(left, right))
        if node.operator == "*":
            return min(2, left + right)
        if node.operator == "/":
            return left if right == 0 else 2
        if node.operator == "**":
            if isinstance(node.right, Literal) and node.right.value in {0, 1}:
                return 0 if node.right.value == 0 else left
            return 2 if left else 0
        return 2
    if isinstance(node, Call) and node.function == "der" and len(node.arguments) == 1:
        return _affine_degree(node.arguments[0], dynamic_symbols)
    return 2


def _classify_pmdl_affinity(system: Any) -> tuple[bool, tuple[str, ...]]:
    nonlinear: list[str] = []
    layouts = getattr(system, "_layouts", None)
    if layouts is None:
        return False, ("<unavailable PMDL relation layout>",)
    for layout in layouts:
        dynamic = set(layout.unknown_indices) | set(layout.derivative_indices)
        for relation_name, expression in layout.relations:
            if _affine_degree(expression, dynamic) > 1:
                nonlinear.append(f"{layout.component.id}.{relation_name}")
    return not nonlinear, tuple(nonlinear)


def _nonlinear_coordinate_sets(
    system: Any,
    differential_indices: tuple[int, ...],
    algebraic_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Return every dynamic-coordinate set in a nonlinear residual row."""

    layouts = getattr(system, "_layouts", None)
    if layouts is None:
        raise ObserverDerivationError(
            "PMDL relation layout is required for coupled nonlinear evidence"
        )
    nx = len(differential_indices)
    differential_positions = {
        global_index: position
        for position, global_index in enumerate(differential_indices)
    }
    algebraic_positions = {
        global_index: position
        for position, global_index in enumerate(algebraic_indices)
    }
    coordinate_sets: set[tuple[int, ...]] = set()
    for layout in layouts:
        dynamic = set(layout.unknown_indices) | set(layout.derivative_indices)
        for _, expression in layout.relations:
            if _affine_degree(expression, dynamic) <= 1:
                continue
            coordinates: set[int] = set()
            for symbol in expression.variables():
                if symbol in layout.unknown_indices:
                    global_index = int(layout.unknown_indices[symbol])
                    if global_index in differential_positions:
                        coordinates.add(differential_positions[global_index])
                    elif global_index in algebraic_positions:
                        coordinates.add(
                            2 * nx + algebraic_positions[global_index]
                        )
                if symbol in layout.derivative_indices:
                    global_index = int(layout.derivative_indices[symbol])
                    try:
                        coordinates.add(nx + differential_positions[global_index])
                    except KeyError as exc:
                        raise ObserverDerivationError(
                            "PMDL relation differentiates a non-differential state"
                        ) from exc
            ordered = sorted(coordinates)
            if len(ordered) >= 2:
                coordinate_sets.add(tuple(ordered))
    return tuple(sorted(coordinate_sets))


def _projection(
    variable: str,
    *,
    state_names: tuple[str, ...],
    differential_indices: tuple[int, ...],
    algebraic_indices: tuple[int, ...],
    sensitivity: np.ndarray,
    x_operating: np.ndarray,
    q_operating: np.ndarray,
    selected_control_indices: tuple[int, ...],
    selected_controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    nx = len(differential_indices)
    nu = len(selected_control_indices)
    try:
        global_index = state_names.index(variable)
    except ValueError as exc:
        raise ObserverDerivationError(
            f"PMDL projection references missing variable {variable!r}"
        ) from exc
    if global_index in differential_indices:
        row = np.zeros(nx, dtype=np.float64)
        row[differential_indices.index(global_index)] = 1.0
        return row, np.zeros(nu, dtype=np.float64), 0.0
    algebraic_position = algebraic_indices.index(global_index)
    q_row = nx + algebraic_position
    state_row = np.array(sensitivity[q_row, :nx], copy=True)
    input_row = np.array(
        sensitivity[q_row, nx + np.asarray(selected_control_indices, dtype=int)],
        copy=True,
    ) if nu else np.empty((0,), dtype=np.float64)
    value = float(q_operating[q_row])
    bias = value - state_row @ x_operating - input_row @ selected_controls
    return state_row, input_row, float(bias)


def derive_affine_observer(
    system: Any,
    spec: ControlSpec,
    *,
    explicit_bindings: Mapping[str, Any],
    implicit_bindings: Mapping[str, Any],
    output_bindings: Mapping[str, str],
    assembly_sha256: str,
    pmdl_sha256: str,
    controller_link_digest: str,
    dynamics_completeness: Mapping[str, Any],
) -> AffineObserverModel:
    """Derive one controller observer from a canonical assembled PMDL DAE."""

    if not isinstance(spec, ControlSpec) or not spec.implicit_inputs:
        raise ObserverDerivationError(
            "derive_affine_observer requires a ControlSpec with implicit inputs"
        )
    if spec.observer is None:
        raise ObserverDerivationError("implicit controller lacks observer admission")
    if not isinstance(dynamics_completeness, Mapping):
        raise ObserverDerivationError(
            "implicit controller requires a validated dynamics_completeness record"
        )
    gate_values = dynamics_completeness.get("gates")
    if not isinstance(gate_values, (tuple, list)):
        raise ObserverDerivationError(
            "dynamics_completeness.gates must be an array for observer admission"
        )
    open_gates = tuple(
        str(gate.get("id"))
        for gate in gate_values
        if isinstance(gate, Mapping) and gate.get("status") == "open"
    )
    status = dynamics_completeness.get("status")
    if status not in {"complete", "incomplete"}:
        raise ObserverDerivationError(
            "dynamics_completeness.status must be complete or incomplete"
        )
    if (status == "complete") != (not open_gates):
        raise ObserverDerivationError(
            "dynamics_completeness status/open gates are internally inconsistent"
        )
    acknowledged = spec.observer.acknowledged_open_gates
    if set(acknowledged) != set(open_gates) or len(acknowledged) != len(open_gates):
        raise ObserverDerivationError(
            "observer acknowledged_open_gates must exactly match the canonical "
            f"open gates; missing={sorted(set(open_gates) - set(acknowledged))}, "
            f"extra={sorted(set(acknowledged) - set(open_gates))}"
        )
    _hash(assembly_sha256, "assembly_sha256")
    _hash(pmdl_sha256, "pmdl_sha256")
    _hash(controller_link_digest, "controller_link_digest")
    if getattr(system, "assembly_sha256", None) != assembly_sha256:
        raise ObserverDerivationError("PMDL and physical assembly hashes differ")
    if getattr(system, "pmdl_sha256", None) != pmdl_sha256:
        raise ObserverDerivationError("PMDL closure hash differs from resolved context")
    _reject_time_dependent_relations(system)

    state_names = tuple(str(name) for name in system.state_names)
    differential_names = tuple(str(name) for name in system.differential_state_names)
    if not differential_names:
        raise ObserverDerivationError("resolved PMDL system has no differential states")
    differential_indices = tuple(state_names.index(name) for name in differential_names)
    differential_set = set(differential_indices)
    algebraic_indices = tuple(
        index for index in range(len(state_names)) if index not in differential_set
    )
    residual_names = tuple(str(name) for name in system.residual_names)
    if len(residual_names) != len(state_names):
        raise ObserverDerivationError(
            "assembled PMDL DAE must be square for observer derivation"
        )

    plant_control_names = tuple(str(name) for name in system.control_names)
    defaults = dict(system.control_defaults)
    missing_defaults = sorted(set(plant_control_names) - set(defaults))
    if missing_defaults:
        raise ObserverDerivationError(
            "observer operating point requires defaults for PMDL controls: "
            f"{missing_defaults}"
        )
    u_operating = _point_vector(defaults, plant_control_names, "PMDL control defaults")
    input_names = tuple(
        item.name for item in spec.outputs if item.name in output_bindings
    )
    if set(output_bindings) != set(input_names):
        raise ObserverDerivationError(
            "plant output bindings must name a unique subset of controller outputs"
        )
    non_real_plant_outputs = [
        item.name
        for item in spec.outputs
        if item.name in output_bindings and item.dtype != "real"
    ]
    if non_real_plant_outputs:
        raise ObserverDerivationError(
            "affine observer plant inputs must be real-valued controller outputs: "
            f"{non_real_plant_outputs}"
        )
    try:
        bound_plant_inputs = tuple(output_bindings[name] for name in input_names)
    except KeyError as exc:
        raise ObserverDerivationError(
            f"controller output binding is missing {exc.args[0]!r}"
        ) from exc
    if len(set(bound_plant_inputs)) != len(bound_plant_inputs):
        raise ObserverDerivationError("controller outputs must bind distinct PMDL controls")
    try:
        selected_control_indices = tuple(
            plant_control_names.index(name) for name in bound_plant_inputs
        )
    except ValueError as exc:
        raise ObserverDerivationError(
            "controller output binding does not name a resolved PMDL control"
        ) from exc
    selected_controls = u_operating[list(selected_control_indices)]

    measurement_specs = tuple(
        item
        for item in spec.explicit_inputs
        if item.source == "sensor" and item.dtype == "real"
    )
    measurement_names = tuple(item.name for item in measurement_specs)
    try:
        measurement_variables = tuple(
            str(explicit_bindings[name].state_name) for name in measurement_names
        )
    except (KeyError, AttributeError) as exc:
        raise ObserverDerivationError(
            "every sensor input must have an exact resolved PMDL variable"
        ) from exc
    if any(name == "None" for name in measurement_variables):
        raise ObserverDerivationError("sensor binding resolved without a PMDL variable")
    latent_names = tuple(item.name for item in spec.implicit_inputs)
    try:
        latent_variables = tuple(
            str(implicit_bindings[name].state_name) for name in latent_names
        )
    except (KeyError, AttributeError) as exc:
        raise ObserverDerivationError(
            "every implicit input must bind an exact resolved PMDL variable"
        ) from exc
    if any(name == "None" for name in latent_variables):
        raise ObserverDerivationError("implicit binding resolved without a PMDL variable")

    full_initial = _point_vector(system.initial_state, state_names, "PMDL initial_state")
    x_operating = full_initial[list(differential_indices)]
    nx = len(differential_names)
    na = len(algebraic_indices)
    equation_count = len(residual_names)
    parameter_values = {
        name: np.asarray([value], dtype=np.float64)
        for name, value in system.default_parameters.items()
    }
    backend = NumpyBackend()

    def evaluate(
        x_value: np.ndarray, q_value: np.ndarray, u_value: np.ndarray
    ) -> np.ndarray:
        state = np.zeros((1, len(state_names)), dtype=np.float64)
        derivative = np.zeros_like(state)
        state[0, list(differential_indices)] = x_value
        if na:
            state[0, list(algebraic_indices)] = q_value[nx:]
        derivative[0, list(differential_indices)] = q_value[:nx]
        controls = {
            name: np.asarray([u_value[index]], dtype=np.float64)
            for index, name in enumerate(plant_control_names)
        }
        try:
            result = system.residual(
                0.0, state, derivative, parameter_values, controls, backend
            )
        except Exception as exc:
            raise ObserverDerivationError(
                f"PMDL residual evaluation failed during observer derivation: {exc}"
            ) from exc
        array = np.asarray(result, dtype=np.float64)
        if array.shape != (1, equation_count):
            raise ObserverDerivationError(
                f"PMDL residual shape {array.shape} is not {(1, equation_count)}"
            )
        if not np.all(np.isfinite(array)):
            raise ObserverDerivationError(
                "PMDL residual is non-finite at the observer operating point"
            )
        return array[0]

    settings = spec.observer
    q_operating = np.concatenate(
        [np.zeros(nx, dtype=np.float64), full_initial[list(algebraic_indices)]]
    )
    iterations = 0
    for iteration in range(settings.newton_max_iterations):
        iterations = iteration + 1
        value = evaluate(x_operating, q_operating, u_operating)
        if float(np.max(np.abs(value))) <= settings.newton_tolerance:
            break
        jacobian = _finite_difference_matrix(
            lambda point: evaluate(x_operating, point, u_operating),
            q_operating,
            equation_count,
            relative_step=settings.relative_step,
            label="dF/d[xdot,a]",
        )
        rank = int(np.linalg.matrix_rank(jacobian))
        condition = float(np.linalg.cond(jacobian))
        if rank != equation_count:
            raise ObserverDerivationError(
                "DAE operating-point dF/d[xdot,a] is singular: "
                f"rank={rank}/{equation_count}"
            )
        if not math.isfinite(condition) or condition > settings.maximum_condition_number:
            raise ObserverDerivationError(
                "DAE operating-point Jacobian is ill-conditioned: "
                f"condition={condition:.17g}, limit={settings.maximum_condition_number:.17g}"
            )
        q_operating = q_operating + np.linalg.solve(jacobian, -value)
    final_residual = evaluate(x_operating, q_operating, u_operating)
    residual_max = float(np.max(np.abs(final_residual)))
    if residual_max > max(settings.newton_tolerance * 10.0, 1e-9):
        worst = int(np.argmax(np.abs(final_residual)))
        raise ObserverDerivationError(
            "DAE operating-point solve did not converge; worst residual "
            f"{residual_names[worst]!r}={final_residual[worst]:.17g}"
        )

    validity_ranges = getattr(getattr(system, "validity", None), "ranges", {})
    operating_values = {
        name: float(full_initial[index]) for index, name in enumerate(state_names)
    }
    for index, global_index in enumerate(algebraic_indices):
        operating_values[state_names[global_index]] = float(q_operating[nx + index])
    operating_values["t"] = 0.0
    for variable, bounds in validity_ranges.items():
        if variable not in operating_values:
            continue
        value = operating_values[variable]
        if not bounds.contains(value):
            raise ObserverDerivationError(
                f"observer operating point {variable}={value:.17g} is outside PMDL "
                f"validity range [{bounds.lower}, {bounds.upper}]"
            )

    g_matrix = _finite_difference_matrix(
        lambda point: evaluate(x_operating, point, u_operating),
        q_operating,
        equation_count,
        relative_step=settings.relative_step,
        label="dF/d[xdot,a]",
    )
    rank = int(np.linalg.matrix_rank(g_matrix))
    condition = float(np.linalg.cond(g_matrix))
    if rank != equation_count:
        raise ObserverDerivationError(
            f"DAE implicit Jacobian is singular: rank={rank}/{equation_count}"
        )
    if not math.isfinite(condition) or condition > settings.maximum_condition_number:
        raise ObserverDerivationError(
            "DAE implicit Jacobian exceeds condition limit: "
            f"{condition:.17g} > {settings.maximum_condition_number:.17g}"
        )
    h_matrix = _finite_difference_matrix(
        lambda point: evaluate(point, q_operating, u_operating),
        x_operating,
        equation_count,
        relative_step=settings.relative_step,
        label="dF/dx",
    )
    j_matrix = _finite_difference_matrix(
        lambda point: evaluate(x_operating, q_operating, point),
        u_operating,
        equation_count,
        relative_step=settings.relative_step,
        label="dF/du",
    )
    sensitivity = -np.linalg.solve(
        g_matrix, np.concatenate([h_matrix, j_matrix], axis=1)
    )
    if not np.all(np.isfinite(sensitivity)):
        raise ObserverDerivationError("DAE sensitivities contain non-finite values")

    qualification_radius = settings.sample_radius_relative
    qualification_remainders: list[float] = []
    combined_operating = np.concatenate([x_operating, q_operating, u_operating])
    steps = qualification_radius * np.maximum(1.0, np.abs(combined_operating))

    def qualify_candidate(candidate: np.ndarray) -> None:
        x_candidate = candidate[:nx]
        q_candidate = candidate[nx : nx + len(q_operating)]
        u_candidate = candidate[nx + len(q_operating) :]
        actual = evaluate(x_candidate, q_candidate, u_candidate)
        dx = x_candidate - x_operating
        dq = q_candidate - q_operating
        du = u_candidate - u_operating
        linear = final_residual + h_matrix @ dx + g_matrix @ dq + j_matrix @ du
        scale = max(1.0, float(np.max(np.abs(actual))))
        qualification_remainders.append(
            float(np.max(np.abs(actual - linear))) / scale
        )

    # Axis samples alone falsely certify cross terms such as x*y at (0, 0),
    # while axes+pairs still miss x*y*z.  For each nonlinear residual relation,
    # exercise every signed joint corner of its complete structural coordinate
    # set.  This is explicitly sampled derivation evidence, not a global bound.
    for first in range(len(combined_operating)):
        for first_direction in (-1.0, 1.0):
            candidate = np.array(combined_operating, copy=True)
            candidate[first] += first_direction * steps[first]
            qualify_candidate(candidate)
    nonlinear_sets = _nonlinear_coordinate_sets(
        system, differential_indices, algebraic_indices
    )
    for coordinates in nonlinear_sets:
        for direction_bits in range(1 << len(coordinates)):
            candidate = np.array(combined_operating, copy=True)
            for bit, coordinate in enumerate(coordinates):
                direction = 1.0 if direction_bits & (1 << bit) else -1.0
                candidate[coordinate] += direction * steps[coordinate]
            qualify_candidate(candidate)
    sampled_remainder = max(qualification_remainders, default=0.0)
    if sampled_remainder > settings.maximum_sampled_remainder:
        raise ObserverDerivationError(
            "local affine qualification sampled remainder "
            f"{sampled_remainder:.17g} exceeds admitted maximum "
            f"{settings.maximum_sampled_remainder:.17g}"
        )

    owned_control_indices = set(selected_control_indices)
    unowned_input_proof: list[dict[str, Any]] = []
    projection_variables = measurement_variables + latent_variables
    structural_targets = differential_names + projection_variables
    for control_index, control_name in enumerate(plant_control_names):
        if control_index in owned_control_indices:
            continue
        reachable_targets = _control_structural_targets(
            system, control_name, structural_targets
        )
        if reachable_targets:
            raise ObserverDerivationError(
                f"unowned PMDL control {control_name!r} is structurally connected to "
                f"observer state/projection variables {list(reachable_targets)}; bind it "
                "explicitly to this controller or redesign the observer boundary"
            )
        values = list(sensitivity[:nx, nx + control_index])
        for variable in projection_variables:
            global_index = state_names.index(variable)
            if global_index in algebraic_indices:
                algebraic_position = algebraic_indices.index(global_index)
                values.append(
                    float(sensitivity[nx + algebraic_position, nx + control_index])
                )
            else:
                values.append(0.0)
        maximum = max((abs(float(value)) for value in values), default=0.0)
        unowned_input_proof.append(
            {
                "plant_input": control_name,
                "structurally_isolated": True,
                "reachable_observer_variables": [],
                "maximum_local_sensitivity_supplement": maximum,
            }
        )
        if maximum > 1e-10:
            raise ObserverDerivationError(
                f"unowned PMDL control {control_name!r} passed structural isolation but "
                f"has nonzero local sensitivity {maximum:.17g}; refusing inconsistent "
                "observer admission"
            )

    selected = np.asarray(selected_control_indices, dtype=int)
    a_matrix = sensitivity[:nx, :nx]
    b_matrix = sensitivity[:nx, nx + selected] if len(selected) else np.empty((nx, 0))
    xdot_operating = q_operating[:nx]
    dynamics_bias = xdot_operating - a_matrix @ x_operating - b_matrix @ selected_controls

    measurement_projection = tuple(
        _projection(
            variable,
            state_names=state_names,
            differential_indices=differential_indices,
            algebraic_indices=algebraic_indices,
            sensitivity=sensitivity,
            x_operating=x_operating,
            q_operating=q_operating,
            selected_control_indices=selected_control_indices,
            selected_controls=selected_controls,
        )
        for variable in measurement_variables
    )
    latent_projection = tuple(
        _projection(
            variable,
            state_names=state_names,
            differential_indices=differential_indices,
            algebraic_indices=algebraic_indices,
            sensitivity=sensitivity,
            x_operating=x_operating,
            q_operating=q_operating,
            selected_control_indices=selected_control_indices,
            selected_controls=selected_controls,
        )
        for variable in latent_variables
    )
    c_matrix = np.stack([item[0] for item in measurement_projection]) if measurement_projection else np.empty((0, nx))
    d_matrix = np.stack([item[1] for item in measurement_projection]) if measurement_projection else np.empty((0, len(input_names)))
    measurement_bias = np.asarray([item[2] for item in measurement_projection])
    l_matrix = np.stack([item[0] for item in latent_projection])
    m_matrix = np.stack([item[1] for item in latent_projection])
    latent_bias = np.asarray([item[2] for item in latent_projection])

    latent_operating = l_matrix @ x_operating + m_matrix @ selected_controls + latent_bias
    initial_state = np.array(x_operating, copy=True)
    for index, item in enumerate(spec.implicit_inputs):
        if not item.bounds.contains(float(latent_operating[index])):
            raise ObserverDerivationError(
                f"PMDL operating-point latent {item.name!r}={latent_operating[index]:.17g} "
                "is outside its declared controller bounds"
            )
    initial_variance = np.asarray(
        [item.initial_variance for item in spec.implicit_inputs], dtype=np.float64
    )
    process_variance = np.asarray(
        [item.process_variance_per_s for item in spec.implicit_inputs],
        dtype=np.float64,
    )
    initial_covariance = _covariance_from_projection(
        l_matrix, initial_variance, "implicit initial_variance"
    )
    process_covariance = _covariance_from_projection(
        l_matrix, process_variance, "implicit process_variance_per_s"
    )
    (
        transition,
        discrete_input,
        discrete_bias,
        discrete_process_covariance,
    ) = _exact_discretization(
        a_matrix,
        b_matrix,
        dynamics_bias,
        process_covariance,
        spec.period_s,
    )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if not math.isfinite(spectral_radius):
        raise ObserverDerivationError(
            "exact discrete observer transition has non-finite spectral radius"
        )
    measurement_variance = np.asarray(
        [float(item.measurement_variance) for item in measurement_specs],
        dtype=np.float64,
    )

    validity = (
        system.validity.to_dict()
        if hasattr(system.validity, "to_dict")
        else _jsonable(system.validity)
    )
    operating_point = {
        "time_s": 0.0,
        "differential_state": {
            name: float(x_operating[index])
            for index, name in enumerate(differential_names)
        },
        "state_derivative": {
            name: float(xdot_operating[index])
            for index, name in enumerate(differential_names)
        },
        "algebraic": {
            state_names[global_index]: float(q_operating[nx + index])
            for index, global_index in enumerate(algebraic_indices)
        },
        "plant_controls": {
            name: float(u_operating[index])
            for index, name in enumerate(plant_control_names)
        },
    }
    is_affine, nonlinear_relations = _classify_pmdl_affinity(system)
    derivation = {
        "kind": "local_affine_linearization",
        "approximation": not is_affine,
        "pmdl_dynamics_classification": "affine" if is_affine else "nonlinear",
        "nonlinear_relations": list(nonlinear_relations),
        "nonlinear_approximation_approved": not is_affine,
        "method": "implicit_function_central_difference",
        "equation_count": equation_count,
        "differential_state_count": nx,
        "algebraic_count": na,
        "dF_dq_rank": rank,
        "dF_dq_condition": condition,
        "relative_step": settings.relative_step,
        "sampled_local_linearity_evidence": {
            "kind": "symmetric_axis_and_structural_joint_corner_residual_remainder",
            "sample_radius_relative": settings.sample_radius_relative,
            "maximum_sampled_remainder": settings.maximum_sampled_remainder,
            "observed_sampled_remainder": sampled_remainder,
            "sample_count": len(qualification_remainders),
            "structural_joint_set_count": len(nonlinear_sets),
            "online_trust_region_enforced": False,
            "scope": "derivation_time_sampled_residual_only_not_a_runtime_validity_bound",
        },
        "newton_tolerance": settings.newton_tolerance,
        "newton_iterations": iterations,
        "residual_max": residual_max,
        "discretization": "exact_zero_order_hold_matrix_exponential",
        "discrete_transition_spectral_radius": spectral_radius,
        "discrete_transition_stable": spectral_radius <= 1.0 + 1e-10,
        "sensor_projection": "implicit_function_state_and_control_sensitivity",
        "latent_projection": "implicit_function_state_and_control_sensitivity",
        "open_gate_admission": {
            "assembly_status": status,
            "acknowledged_open_gates": list(acknowledged),
        },
        "unowned_input_structural_isolation": unowned_input_proof,
    }
    return AffineObserverModel(
        spec.id,
        control_digest(spec),
        controller_link_digest,
        assembly_sha256,
        pmdl_sha256,
        differential_names,
        input_names,
        bound_plant_inputs,
        measurement_names,
        measurement_variables,
        latent_names,
        latent_variables,
        a_matrix,
        b_matrix,
        dynamics_bias,
        c_matrix,
        d_matrix,
        measurement_bias,
        l_matrix,
        m_matrix,
        latent_bias,
        process_covariance,
        transition,
        discrete_input,
        discrete_bias,
        discrete_process_covariance,
        measurement_variance,
        initial_state,
        initial_covariance,
        tuple(item.bounds.lower for item in spec.implicit_inputs),
        tuple(item.bounds.upper for item in spec.implicit_inputs),
        spec.period_s,
        operating_point,
        derivation,
        validity,
        dynamics_completeness,
    )


__all__ = [
    "AffineObserverModel",
    "ObservabilityDiagnostic",
    "ObserverDerivationError",
    "derive_affine_observer",
    "observability_diagnostics",
]
