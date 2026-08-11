"""Fail-closed composition of PMDL component models into one descriptor system.

The assembler is deliberately a compiler, not another physical model.  It
namespaces every component variable, retains every component residual, and
adds only the typed equations implied by contraption connections:

* electrical power nets share effort and conserve signed flow;
* mechanical power/attachment nets share signed flow and conserve effort;
* directed signal nets copy their one output into every input; and
* control bindings constrain their target input to a declared control source.

The resulting :class:`AssembledPMDLSystem` implements the structural residual
protocol consumed by :func:`contraption.physics.simulator.simulate`, so the same object
runs on NumPy and Torch.  Unsupported PMDL or connection semantics, incomplete
port coverage, equation-count mismatches, and structural singularities are
rejected before integration rather than being approximated silently.

Physical-only attachments are retained for a geometry/kinematics compiler.
They must be explicit: ``connector_bindings`` maps a fully-qualified physical
connector (``"component.connector"``) to a local PMDL power-port name, or to
``None`` when it is intentionally kinematic-only.  An attachment whose
endpoints are partly behavioral and partly kinematic is invalid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

import numpy as np

from .backend import Array, Backend
from .dsl import Expression, parse_expression
from .simulator import _evaluate_backend_expression
from .specs import (
    BoundsSpec,
    FrozenDict,
    ModelSpec,
    PortRef,
    PowerPortSpec,
    SignalPortSpec,
    ValiditySpec,
)
from .units import UnitError, parse_unit
from .validation import validate_contraption_structure, validate_model


_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


class AssemblyError(ValueError):
    """Base class for deterministic contraption-assembly failures."""


class UnsupportedAssemblySemanticsError(AssemblyError, NotImplementedError):
    """A declared PMDL or connection behavior is not implemented safely."""


class AssemblyBalanceError(AssemblyError):
    """The assembled residual system is not square or structurally solvable."""


class NetworkInvariantError(AssemblyError):
    """An accepted state violates one or more connection equations."""


@dataclass(frozen=True, slots=True)
class AssemblyBalance:
    """Inspectable equation/unknown accounting for one resolved assembly."""

    unknown_count: int
    equation_count: int
    component_equations: tuple[tuple[str, int], ...]
    connection_equations: tuple[tuple[str, int], ...]
    unknown_names: tuple[str, ...]
    equation_names: tuple[str, ...]
    structural_rank: int

    @property
    def square(self) -> bool:
        return self.unknown_count == self.equation_count

    @property
    def structurally_full_rank(self) -> bool:
        return self.square and self.structural_rank == self.unknown_count


@dataclass(frozen=True, slots=True)
class _PowerEndpoint:
    connector: str
    component: str
    port: PowerPortSpec
    effort_index: int
    flow_index: int
    sign: float


@dataclass(frozen=True, slots=True)
class _SignalEndpoint:
    connector: str
    component: str
    port: SignalPortSpec
    index: int


@dataclass(frozen=True, slots=True)
class _LinearEquation:
    name: str
    terms: tuple[tuple[int, float], ...]
    control_source: str | None = None
    control_scale: float = 1.0

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(index for index, coefficient in self.terms if coefficient != 0.0)


@dataclass(frozen=True, slots=True)
class _ComponentLayout:
    component: _ComponentInput
    model: ModelSpec
    unknown_indices: Mapping[str, int]
    derivative_indices: Mapping[str, int]
    parameter_names: Mapping[str, str]
    relations: tuple[tuple[str, Expression], ...]


@dataclass(frozen=True, slots=True)
class _ComponentInput:
    """Minimal resolved-component protocol consumed by the assembler."""

    id: str
    parameters: Mapping[str, Any]
    model_reference: str | None


def _qualified(component: str, symbol: str) -> str:
    return f"{component}.{symbol}"


def _record_field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _reference_name(reference: Any) -> str:
    name = _record_field(reference, "port")
    if name is None:
        name = _record_field(reference, "connector")
    if not isinstance(name, str) or not name:
        raise AssemblyError(
            "connection endpoint must expose a non-empty port or connector name"
        )
    return name


def _reference_component(reference: Any) -> str:
    component = _record_field(reference, "component")
    if not isinstance(component, str) or not component:
        raise AssemblyError(
            "connection endpoint must expose a non-empty component name"
        )
    return component


def _connector_key(reference: Any) -> str:
    return _qualified(_reference_component(reference), _reference_name(reference))


def _component_input(record: Any) -> _ComponentInput:
    component_id = _record_field(record, "id")
    if not isinstance(component_id, str) or not component_id:
        raise AssemblyError("resolved component id must be a non-empty string")
    parameters = _record_field(record, "parameters", {})
    if not isinstance(parameters, Mapping):
        raise AssemblyError(
            f"resolved component {component_id!r} parameters must be an object"
        )
    model_reference = _record_field(record, "model_id")
    if model_reference is not None and (
        not isinstance(model_reference, str) or not model_reference
    ):
        raise AssemblyError(
            f"resolved component {component_id!r} model_id must be a non-empty string"
        )
    return _ComponentInput(component_id, parameters, model_reference)


def _numeric(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AssemblyError(f"{context} must be finite")
    return result


def _orientation_sign(port: PowerPortSpec, context: str) -> float:
    if port.orientation == "into_component":
        return 1.0
    if port.orientation == "out_of_component":
        return -1.0
    raise UnsupportedAssemblySemanticsError(
        f"{context} uses orientation={port.orientation!r}; assembled conservation "
        "requires an explicit into_component or out_of_component sign"
    )


def _reject_unsupported_model_semantics(component: _ComponentInput, model: ModelSpec) -> None:
    prefix = f"component {component.id!r} model {model.id!r}"
    if model.modes:
        raise UnsupportedAssemblySemanticsError(
            f"{prefix} declares discrete modes; assembled mode transitions/resets are not implemented"
        )
    if model.initialization.constraints:
        raise UnsupportedAssemblySemanticsError(
            f"{prefix} declares {len(model.initialization.constraints)} initialization "
            "constraint(s); a network consistent-initialization solver is required"
        )
    if model.initialization.strategy != "consistent":
        raise UnsupportedAssemblySemanticsError(
            f"{prefix} uses unsupported initialization strategy "
            f"{model.initialization.strategy!r}"
        )
    if len(model.fidelity_levels) > 1:
        names = [level.name for level in model.fidelity_levels]
        raise UnsupportedAssemblySemanticsError(
            f"{prefix} declares multiple fidelity levels {names}; no component-level "
            "fidelity selection was supplied"
        )
    if model.fidelity_levels:
        level = model.fidelity_levels[0]
        relation_names = {relation.name for relation in model.relations}
        if set(level.active_relations) != relation_names or level.parameter_overrides:
            raise UnsupportedAssemblySemanticsError(
                f"{prefix} fidelity {level.name!r} changes relations or parameters; "
                "fidelity execution is not implemented"
            )
    for port in model.signal_ports:
        if port.shape:
            raise UnsupportedAssemblySemanticsError(
                f"{prefix} signal port {port.name!r} has shape {port.shape}; the PMDL "
                "scalar residual assembler does not silently flatten vector signals"
            )
        if port.dtype not in {"float32", "float64"}:
            raise UnsupportedAssemblySemanticsError(
                f"{prefix} signal port {port.name!r} has dtype {port.dtype!r}; "
                "discrete/bool signals belong in the control runtime"
            )


def _parameter_value(
    component: _ComponentInput,
    parameter: Any,
) -> tuple[float, Mapping[str, Any] | None, str | None]:
    raw = component.parameters.get(parameter.name, parameter.default)
    uncertainty: Mapping[str, Any] | None = None
    correlation_group: str | None = getattr(parameter.uncertainty, "correlation_group", None)
    converted_unit_scale = False
    if isinstance(raw, Mapping):
        if "value" not in raw:
            raise AssemblyError(
                f"component {component.id!r} parameter {parameter.name!r} must contain numeric value"
            )
        value = _numeric(raw["value"], f"component {component.id!r} parameter {parameter.name!r}.value")
        override_unit = raw.get("unit")
        if override_unit is not None:
            if not isinstance(override_unit, str):
                raise AssemblyError(
                    f"component {component.id!r} parameter {parameter.name!r}.unit must be a string"
                )
            try:
                source_unit = parse_unit(override_unit)
                target_unit = parse_unit(parameter.unit)
                converted_unit_scale = source_unit.scale != target_unit.scale
                value = source_unit.convert_value_to(value, target_unit)
            except UnitError as exc:
                raise AssemblyError(
                    f"component {component.id!r} parameter {parameter.name!r} unit mismatch: {exc}"
                ) from exc
        raw_uncertainty = raw.get("uncertainty")
        if raw_uncertainty is not None:
            if not isinstance(raw_uncertainty, Mapping):
                raise AssemblyError(
                    f"component {component.id!r} parameter {parameter.name!r}.uncertainty must be an object"
                )
            if converted_unit_scale:
                raise UnsupportedAssemblySemanticsError(
                    f"component {component.id!r} parameter {parameter.name!r} supplies uncertainty "
                    f"in override unit {override_unit!r}, whose scale differs from model unit "
                    f"{parameter.unit!r}; uncertainty-unit conversion is not declared"
                )
            uncertainty = dict(raw_uncertainty)
            group = uncertainty.pop("correlation_group", correlation_group)
            correlation_group = None if group is None else str(group)
    else:
        value = _numeric(raw, f"component {component.id!r} parameter {parameter.name!r}")

    if not parameter.bounds.contains(value):
        raise AssemblyError(
            f"component {component.id!r} parameter {parameter.name!r}={value!r} is outside "
            f"model bounds [{parameter.bounds.lower!r}, {parameter.bounds.upper!r}]"
        )
    if uncertainty is None and parameter.uncertainty.distribution != "fixed":
        uncertainty = {
            "distribution": parameter.uncertainty.distribution,
            "parameters": dict(parameter.uncertainty.parameters),
        }
    return value, uncertainty, correlation_group


def _maximum_matching(dependencies: Sequence[frozenset[int]], unknown_count: int) -> tuple[int, dict[int, int]]:
    """Return structural rank and unknown->equation matching."""

    match: dict[int, int] = {}

    def augment(equation: int, seen: set[int]) -> bool:
        for unknown in sorted(dependencies[equation]):
            if unknown in seen:
                continue
            seen.add(unknown)
            owner = match.get(unknown)
            if owner is None or augment(owner, seen):
                match[unknown] = equation
                return True
        return False

    rank = 0
    for equation in range(len(dependencies)):
        if augment(equation, set()):
            rank += 1
    return min(rank, unknown_count), match


def _diagnostic_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


class AssembledPMDLSystem:
    """A namespaced, square PMDL descriptor network.

    Instances are constructed by :func:`assemble_contraption`; their public
    attributes intentionally match the simulator's structural protocol.
    """

    def __init__(
        self,
        *,
        specification: Any,
        layouts: Sequence[_ComponentLayout],
        state_names: Sequence[str],
        initial_state: Sequence[float],
        differential_state_names: Sequence[str],
        residual_names: Sequence[str],
        component_equation_count: Mapping[str, int],
        network_equations: Sequence[_LinearEquation],
        connection_equation_count: Mapping[str, int],
        default_parameters: Mapping[str, float],
        parameter_bounds: Mapping[str, tuple[float | None, float | None]],
        parameter_uncertainty: Mapping[str, Mapping[str, Any]],
        correlated_uncertainty: Sequence[str],
        control_names: Sequence[str],
        control_defaults: Mapping[str, float],
        control_bounds: Mapping[str, tuple[float | None, float | None]],
        control_slew_rates: Mapping[str, float],
        validity: ValiditySpec,
        dependencies: Sequence[frozenset[int]],
        kinematic_connection_ids: Sequence[str],
        connector_bindings: Mapping[str, str | None],
        assembly_sha256: str,
        pmdl_sha256: str,
        canonical_assembly_sha256: str | None,
    ) -> None:
        self.specification = specification
        self._layouts = tuple(layouts)
        self.component_models = MappingProxyType(
            {layout.component.id: layout.model for layout in self._layouts}
        )
        self.connector_bindings = MappingProxyType(dict(connector_bindings))
        self.state_names = tuple(state_names)
        self.initial_state = tuple(initial_state)
        self.differential_state_names = tuple(differential_state_names)
        differential_set = set(self.differential_state_names)
        self.differential_state_indices = tuple(
            index for index, name in enumerate(self.state_names) if name in differential_set
        )
        self.algebraic_names = tuple(
            name for name in self.state_names if name not in differential_set
        )
        self.algebraic_indices = tuple(
            index for index, name in enumerate(self.state_names) if name not in differential_set
        )
        self.residual_names = tuple(residual_names)
        self._network_equations = tuple(network_equations)
        self.network_residual_names = tuple(equation.name for equation in network_equations)
        self.default_parameters = dict(default_parameters)
        self.parameter_bounds = dict(parameter_bounds)
        self.parameter_uncertainty = dict(parameter_uncertainty)
        self._correlated_uncertainty = tuple(correlated_uncertainty)
        self.control_names = tuple(control_names)
        self.control_defaults = dict(control_defaults)
        self.control_bounds = MappingProxyType(dict(control_bounds))
        self.control_slew_rates = MappingProxyType(dict(control_slew_rates))
        self.validity = validity
        self.kinematic_connection_ids = tuple(kinematic_connection_ids)
        self.assembly_sha256 = assembly_sha256
        self.pmdl_sha256 = pmdl_sha256
        self.canonical_assembly_sha256 = canonical_assembly_sha256
        structural_rank, _ = _maximum_matching(dependencies, len(self.state_names))
        self.balance = AssemblyBalance(
            len(self.state_names),
            len(self.residual_names),
            tuple(component_equation_count.items()),
            tuple(connection_equation_count.items()),
            self.state_names,
            self.residual_names,
            structural_rank,
        )
        self.diagnostics = MappingProxyType(
            {
                "assembly_sha256": self.assembly_sha256,
                "pmdl_sha256": self.pmdl_sha256,
                "canonical_identity": self.canonical_assembly_sha256 is not None,
                "unknown_count": self.balance.unknown_count,
                "equation_count": self.balance.equation_count,
                "structural_rank": self.balance.structural_rank,
                "kinematic_connection_ids": self.kinematic_connection_ids,
            }
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.default_parameters)

    def _control_value(
        self,
        name: str,
        controls: Mapping[str, Array],
        backend: Backend,
        batch_size: int,
    ) -> Array:
        if name in controls:
            raw = controls[name]
        elif name in self.control_defaults:
            raw = self.control_defaults[name]
        else:
            raise KeyError(
                f"assembled contraption requires control source {name!r}; no runtime value "
                "or explicit settings.default was supplied"
            )
        value = backend.asarray(raw)
        if len(value.shape) == 0:
            value = backend.broadcast_to(value, (batch_size,))
        elif tuple(value.shape) != (batch_size,):
            raise ValueError(
                f"assembled control {name!r} must be scalar or [sample], got shape {tuple(value.shape)}"
            )
        diagnostic = _diagnostic_array(value).reshape(-1)
        nonfinite = np.flatnonzero(~np.isfinite(diagnostic))
        if nonfinite.size:
            sample = int(nonfinite[0])
            raise ValueError(
                f"assembled control {name!r} is non-finite for sample={sample}: "
                f"{diagnostic[sample]!r}"
            )
        lower, upper = self.control_bounds.get(name, (None, None))
        invalid = np.zeros(diagnostic.shape, dtype=bool)
        if lower is not None:
            invalid |= diagnostic < lower
        if upper is not None:
            invalid |= diagnostic > upper
        failing = np.flatnonzero(invalid)
        if failing.size:
            sample = int(failing[0])
            raise ValueError(
                f"assembled control {name!r} violates declared bounds for sample={sample}: "
                f"value={float(diagnostic[sample]):.17g}, allowed="
                f"[{('-inf' if lower is None else format(lower, '.17g'))}, "
                f"{('inf' if upper is None else format(upper, '.17g'))}]"
            )
        return value

    @staticmethod
    def _residual_column(value: Any, backend: Backend, batch_size: int, name: str) -> Array:
        result = backend.asarray(value)
        if len(result.shape) == 0:
            result = backend.broadcast_to(result, (batch_size,))
        if tuple(result.shape) != (batch_size,):
            raise ValueError(
                f"assembled residual {name!r} must produce [sample], got {tuple(result.shape)}"
            )
        return result

    def _linear_value(
        self,
        equation: _LinearEquation,
        state: Array,
        controls: Mapping[str, Array],
        backend: Backend,
    ) -> Array:
        batch_size = int(state.shape[0])
        value = backend.zeros((batch_size,))
        for index, coefficient in equation.terms:
            value = value + coefficient * state[:, index]
        if equation.control_source is not None:
            value = value - equation.control_scale * self._control_value(
                equation.control_source, controls, backend, batch_size
            )
        return value

    def residual(
        self,
        t: Any,
        state: Array,
        state_derivative: Array,
        parameters: Mapping[str, Array],
        controls: Mapping[str, Array],
        backend: Backend,
    ) -> Array:
        if len(state.shape) != 2 or int(state.shape[-1]) != len(self.state_names):
            raise ValueError(
                f"assembled state must have shape [sample,{len(self.state_names)}], "
                f"got {tuple(state.shape)}"
            )
        if tuple(state_derivative.shape) != tuple(state.shape):
            raise ValueError("assembled state_derivative shape must match state")
        unknown_controls = sorted(set(controls) - set(self.control_names))
        if unknown_controls:
            raise KeyError(
                f"assembled contraption received unknown control source(s) {unknown_controls}; "
                f"declared sources are {list(self.control_names)}"
            )
        batch_size = int(state.shape[0])
        values: list[Array] = []
        for layout in self._layouts:
            environment: dict[str, Any] = {"t": t}
            for name, index in layout.unknown_indices.items():
                environment[name] = state[:, index]
            for name, index in layout.derivative_indices.items():
                environment[name] = state_derivative[:, index]
            for local_name, global_name in layout.parameter_names.items():
                raw = parameters.get(global_name, self.default_parameters[global_name])
                environment[local_name] = raw
            for relation_name, expression in layout.relations:
                qualified = _qualified(layout.component.id, relation_name)
                result = _evaluate_backend_expression(expression, environment, backend)
                values.append(self._residual_column(result, backend, batch_size, qualified))
        for equation in self._network_equations:
            values.append(self._linear_value(equation, state, controls, backend))
        if not values:
            return backend.zeros((batch_size, 0))
        return backend.stack(values, axis=-1)

    def backward_euler_jacobian(
        self,
        t: Any,
        state: Array,
        state_derivative: Array,
        parameters: Mapping[str, Array],
        controls: Mapping[str, Array],
        dt: Any,
        backend: Backend,
    ) -> Array:
        """Evaluate a sparse-aware backward-Euler Jacobian.

        Component residuals only depend on symbols local to that component.
        Differentiating each local expression over its actual dependency set
        avoids the generic simulator's O(global_unknowns * all_relations)
        residual reevaluation.  Network rows are exactly linear and inserted
        analytically.  This is backend-native and preserves Torch graphs.
        """

        if len(state.shape) != 2 or int(state.shape[-1]) != len(self.state_names):
            raise ValueError(
                f"assembled state must have shape [sample,{len(self.state_names)}], "
                f"got {tuple(state.shape)}"
            )
        batch_size = int(state.shape[0])
        dimension = len(self.state_names)
        zero = backend.zeros((batch_size,))
        one = backend.asarray(1.0)
        dt_value = backend.asarray(dt)
        rows: list[Array] = []

        for layout in self._layouts:
            base_environment: dict[str, Any] = {"t": t}
            for name, index in layout.unknown_indices.items():
                base_environment[name] = state[:, index]
            for name, index in layout.derivative_indices.items():
                base_environment[name] = state_derivative[:, index]
            for local_name, global_name in layout.parameter_names.items():
                base_environment[local_name] = parameters.get(
                    global_name, self.default_parameters[global_name]
                )

            for relation_name, expression in layout.relations:
                variables = expression.variables()
                affected: dict[int, tuple[list[str], list[str]]] = {}
                for symbol in variables:
                    if symbol in layout.unknown_indices:
                        index = layout.unknown_indices[symbol]
                        affected.setdefault(index, ([], []))[0].append(symbol)
                    if symbol in layout.derivative_indices:
                        index = layout.derivative_indices[symbol]
                        affected.setdefault(index, ([], []))[1].append(symbol)
                derivatives: dict[int, Array] = {}
                for index, (state_symbols, derivative_symbols) in affected.items():
                    step = 1e-6 * backend.maximum(backend.abs(state[:, index]), one)
                    positive = dict(base_environment)
                    negative = dict(base_environment)
                    for symbol in state_symbols:
                        positive[symbol] = base_environment[symbol] + step
                        negative[symbol] = base_environment[symbol] - step
                    derivative_step = step / dt_value
                    for symbol in derivative_symbols:
                        positive[symbol] = base_environment[symbol] + derivative_step
                        negative[symbol] = base_environment[symbol] - derivative_step
                    above = _evaluate_backend_expression(expression, positive, backend)
                    below = _evaluate_backend_expression(expression, negative, backend)
                    qualified = _qualified(layout.component.id, relation_name)
                    above = self._residual_column(
                        above, backend, batch_size, qualified
                    )
                    below = self._residual_column(
                        below, backend, batch_size, qualified
                    )
                    derivatives[index] = (above - below) / (2.0 * step)
                rows.append(
                    backend.stack(
                        [derivatives.get(index, zero) for index in range(dimension)],
                        axis=-1,
                    )
                )

        for equation in self._network_equations:
            coefficients = dict(equation.terms)
            rows.append(
                backend.stack(
                    [
                        backend.broadcast_to(
                            backend.asarray(coefficients[index]), (batch_size,)
                        )
                        if index in coefficients
                        else zero
                        for index in range(dimension)
                    ],
                    axis=-1,
                )
            )
        if len(rows) != dimension:
            raise AssemblyBalanceError(
                f"Jacobian row count {len(rows)} does not match state dimension {dimension}"
            )
        return backend.stack(rows, axis=1)

    def consistent_initial_state(
        self,
        t: Any,
        state: Array,
        parameters: Mapping[str, Array],
        controls: Mapping[str, Array],
        backend: Backend,
        *,
        tolerance: float = 1e-9,
        max_iterations: int = 16,
    ) -> Array:
        """Solve algebraics and initial derivatives while holding declared states fixed.

        The solve variables are ``[xdot_differential, algebraics]``.  This is
        the square consistent-initialization problem for an index-1 assembly;
        a singular or nonconvergent solve fails before the first frame is
        emitted, so viewers and metrics never receive a physically impossible
        zero-filled algebraic snapshot.
        """

        if tolerance <= 0 or not math.isfinite(tolerance):
            raise ValueError("consistent initialization tolerance must be finite and positive")
        if max_iterations < 1:
            raise ValueError("consistent initialization max_iterations must be positive")
        if len(state.shape) != 2 or int(state.shape[-1]) != len(self.state_names):
            raise ValueError(
                f"assembled initial state must have shape [sample,{len(self.state_names)}]"
            )
        batch_size = int(state.shape[0])
        differential = self.differential_state_indices
        algebraic = self.algebraic_indices
        dimension = len(self.state_names)
        if len(differential) + len(algebraic) != dimension:
            raise AssemblyBalanceError("initialization layout does not cover every unknown")
        differential_position = {index: offset for offset, index in enumerate(differential)}
        algebraic_position = {
            index: len(differential) + offset for offset, index in enumerate(algebraic)
        }
        guess = backend.zeros((batch_size, dimension))
        # Preserve any caller-supplied algebraic guess while derivative guesses start at zero.
        if algebraic:
            guess = backend.stack(
                [
                    backend.zeros((batch_size,))
                    if index < len(differential)
                    else state[:, algebraic[index - len(differential)]]
                    for index in range(dimension)
                ],
                axis=-1,
            )

        def unpack(candidate: Array) -> tuple[Array, Array]:
            resolved_state = backend.stack(
                [
                    state[:, index]
                    if index in differential_position
                    else candidate[:, algebraic_position[index]]
                    for index in range(dimension)
                ],
                axis=-1,
            )
            derivative = backend.stack(
                [
                    candidate[:, differential_position[index]]
                    if index in differential_position
                    else backend.zeros((batch_size,))
                    for index in range(dimension)
                ],
                axis=-1,
            )
            return resolved_state, derivative

        def residual(candidate: Array) -> Array:
            resolved_state, derivative = unpack(candidate)
            return self.residual(
                t, resolved_state, derivative, parameters, controls, backend
            )

        def diagnostic(value: Array) -> np.ndarray:
            if getattr(backend, "is_torch", False):
                return np.asarray(value.detach().cpu().numpy())
            return np.asarray(value)

        for iteration in range(max_iterations):
            value = residual(guess)
            value_np = diagnostic(value)
            if not np.all(np.isfinite(value_np)):
                location = tuple(int(item) for item in np.argwhere(~np.isfinite(value_np))[0])
                raise AssemblyError(
                    f"consistent initialization residual is non-finite at {location}"
                )
            columns: list[Array] = []
            for index in range(dimension):
                basis = backend.stack(
                    [backend.asarray(1.0 if item == index else 0.0) for item in range(dimension)]
                )
                step = 1e-6 * backend.maximum(
                    backend.abs(guess[:, index]), backend.asarray(1.0)
                )
                perturbation = step[:, None] * basis[None, :]
                columns.append(
                    (residual(guess + perturbation) - residual(guess - perturbation))
                    / (2.0 * step[:, None])
                )
            jacobian = backend.stack(columns, axis=-1)
            jacobian_np = diagnostic(jacobian)
            ranks = np.asarray(np.linalg.matrix_rank(jacobian_np))
            singular = np.flatnonzero(ranks < dimension)
            if singular.size:
                sample = int(singular[0])
                raise AssemblyBalanceError(
                    "consistent initialization Jacobian is singular for "
                    f"sample={sample}: rank={int(ranks[sample])}/{dimension}; "
                    f"residual_max={float(np.max(np.abs(value_np[sample]))):.17g}"
                )
            update = backend.solve(jacobian, -value[..., None])[..., 0]
            update_np = diagnostic(update)
            if not np.all(np.isfinite(update_np)):
                raise AssemblyError("consistent initialization Newton update is non-finite")
            guess = guess + update
            if float(np.max(np.abs(update_np))) <= tolerance:
                break
        final = residual(guess)
        final_np = diagnostic(final)
        maximum = float(np.max(np.abs(final_np)))
        if not np.all(np.isfinite(final_np)) or maximum > max(10.0 * tolerance, 1e-8):
            sample, equation = np.unravel_index(
                int(np.argmax(np.abs(final_np))), final_np.shape
            )
            raise AssemblyError(
                "consistent initialization did not converge: "
                f"sample={sample}, equation={self.residual_names[equation]!r}, "
                f"absolute_residual={abs(float(final_np[sample, equation])):.17g}"
            )
        resolved_state, _derivative = unpack(guess)
        return resolved_state

    def network_residuals(
        self,
        state: Array,
        controls: Mapping[str, Array] | None,
        backend: Backend,
    ) -> Mapping[str, Array]:
        """Evaluate only connection/control invariants for an accepted state."""

        controls = {} if controls is None else controls
        if len(state.shape) != 2 or int(state.shape[-1]) != len(self.state_names):
            raise ValueError(
                f"assembled state must have shape [sample,{len(self.state_names)}], "
                f"got {tuple(state.shape)}"
            )
        return {
            equation.name: self._linear_value(equation, state, controls, backend)
            for equation in self._network_equations
        }

    def require_network_invariants(
        self,
        state: Array,
        controls: Mapping[str, Array] | None,
        backend: Backend,
        *,
        tolerance: float = 1e-8,
        time: float | None = None,
    ) -> Mapping[str, Array]:
        """Raise with the worst named net equation when coupling drifts."""

        if tolerance < 0 or not math.isfinite(tolerance):
            raise ValueError("network invariant tolerance must be finite and nonnegative")
        values = self.network_residuals(state, controls, backend)
        worst_name: str | None = None
        worst_value = -1.0
        worst_sample = 0
        for name, value in values.items():
            diagnostic = _diagnostic_array(value).reshape(-1)
            invalid = np.flatnonzero(~np.isfinite(diagnostic))
            if invalid.size:
                sample = int(invalid[0])
                raise NetworkInvariantError(
                    f"network invariant {name!r} is non-finite"
                    f"{'' if time is None else f' at time={time:.17g}'} for sample={sample}: "
                    f"{diagnostic[sample]!r}"
                )
            if diagnostic.size:
                sample = int(np.argmax(np.abs(diagnostic)))
                magnitude = float(abs(diagnostic[sample]))
                if magnitude > worst_value:
                    worst_name, worst_value, worst_sample = name, magnitude, sample
        if worst_name is not None and worst_value > tolerance:
            raise NetworkInvariantError(
                f"network invariant {worst_name!r} violated"
                f"{'' if time is None else f' at time={time:.17g}'} for sample={worst_sample}; "
                f"absolute residual={worst_value:.17g}, tolerance={tolerance:.17g}"
            )
        return values


def assemble_contraption(
    specification: Any,
    model_registry: Mapping[str, ModelSpec] | None = None,
    *,
    component_models: Mapping[str, ModelSpec | str] | None = None,
    connector_bindings: Mapping[str, str | None] | None = None,
    canonical_assembly_sha256: str | None = None,
) -> AssembledPMDLSystem:
    """Resolve and validate a contraption as one PMDL descriptor system.

    ``component_models`` resolves PMDL behavior by contraption component ID.
    Values may be exact :class:`ModelSpec` objects or IDs in ``model_registry``.
    When absent, each resolved component's ``model_id`` is used.

    ``connector_bindings`` binds typed physical connectors to PMDL behavior
    ports.  Missing bindings are never
    guessed: a connector may use a same-named PMDL port directly, or it must be
    explicitly mapped to a local port/``None``.

    ``canonical_assembly_sha256`` is the optional full physical-closure digest
    produced by the catalog/geometry resolver.  It becomes the shared artifact
    identity; the assembler's independently reproducible PMDL closure digest
    remains available as ``system.pmdl_sha256``.
    """

    if canonical_assembly_sha256 is not None and (
        not isinstance(canonical_assembly_sha256, str)
        or _SHA256_PATTERN.fullmatch(canonical_assembly_sha256) is None
    ):
        raise AssemblyError(
            "canonical_assembly_sha256 must be 'sha256:' followed by 64 lowercase hex digits"
        )

    if all(
        hasattr(specification, name)
        for name in ("components", "connections", "controls", "to_dict")
    ):
        spec = specification
    else:
        raise TypeError(
            "specification must be a resolved contraption record with "
            "components/connections/controls/to_dict"
        )
    structure = validate_contraption_structure(spec)
    if not structure.valid:
        structure.require_valid()
    if model_registry is None:
        model_registry = {}
    elif not isinstance(model_registry, Mapping):
        raise TypeError("model_registry must implement Mapping")
    if component_models is not None and not isinstance(component_models, Mapping):
        raise TypeError("component_models must implement Mapping")
    bindings = {} if connector_bindings is None else dict(connector_bindings)
    invalid_bindings = sorted(
        (
            key
            for key, value in bindings.items()
            if not isinstance(key, str)
            or not key
            or (value is not None and (not isinstance(value, str) or not value))
        ),
        key=str,
    )
    if invalid_bindings:
        raise AssemblyError(
            "connector_bindings keys must be non-empty strings and values must be "
            f"non-empty PMDL port names or None; invalid keys={invalid_bindings}"
        )
    unknown_binding_keys = sorted(
        set(bindings)
        - {
            _connector_key(endpoint)
            for connection in spec.connections
            for endpoint in connection.endpoints
        }
        - {_connector_key(control.target) for control in spec.controls}
    )
    if unknown_binding_keys:
        raise AssemblyError(
            f"connector_bindings contains connector(s) absent from the contraption: {unknown_binding_keys}"
        )

    resolved_components = tuple(_component_input(component) for component in spec.components)
    component_ids = {component.id for component in resolved_components}
    if len(component_ids) != len(resolved_components):
        raise AssemblyError("resolved component IDs must be unique")
    if component_models is not None:
        supplied_ids = set(component_models)
        missing_ids = sorted(component_ids - supplied_ids)
        extra_ids = sorted(supplied_ids - component_ids)
        if missing_ids or extra_ids:
            raise AssemblyError(
                "component_models keys must exactly match component instance IDs; "
                f"missing={missing_ids}, extra={extra_ids}"
            )

    models: dict[str, ModelSpec] = {}
    for component in resolved_components:
        resolved: ModelSpec | str | None
        if component_models is not None:
            resolved = component_models[component.id]
        else:
            resolved = component.model_reference
        if isinstance(resolved, ModelSpec):
            model = resolved
        elif isinstance(resolved, str):
            try:
                model = model_registry[resolved]
            except KeyError as exc:
                raise AssemblyError(
                    f"component {component.id!r} resolves to unregistered model {resolved!r}"
                ) from exc
        elif resolved is None:
            raise AssemblyError(
                f"component {component.id!r} has no resolved PMDL model; supply "
                "component_models keyed by instance ID or a resolved model_id"
            )
        else:
            raise AssemblyError(
                f"component_models[{component.id!r}] must be a ModelSpec or model ID string"
            )
        if not isinstance(model, ModelSpec):
            raise AssemblyError(
                f"resolved model for component {component.id!r} must be a ModelSpec, "
                f"got {type(model).__name__}"
            )
        report = validate_model(model)
        if not report.valid:
            details = "; ".join(
                f"[{issue.code}] {issue.path}: {issue.message}" for issue in report.errors
            )
            raise AssemblyError(
                f"component {component.id!r} model {model.id!r} is invalid: {details}"
            )
        _reject_unsupported_model_semantics(component, model)
        unknown_overrides = sorted(set(component.parameters) - set(model.parameter_names))
        if unknown_overrides:
            raise AssemblyError(
                f"component {component.id!r} has unknown parameter override(s) {unknown_overrides}; "
                f"model parameters are {list(model.parameter_names)}"
            )
        models[component.id] = model

    state_names: list[str] = []
    initial_state: list[float] = []
    differential_state_names: list[str] = []
    layouts: list[_ComponentLayout] = []
    power_ports: dict[tuple[str, str], _PowerEndpoint] = {}
    signal_ports: dict[tuple[str, str], _SignalEndpoint] = {}
    default_parameters: dict[str, float] = {}
    parameter_bounds: dict[str, tuple[float | None, float | None]] = {}
    parameter_uncertainty: dict[str, Mapping[str, Any]] = {}
    correlated_uncertainty: list[str] = []
    validity_ranges: dict[str, BoundsSpec] = {}
    validity_assumptions: list[str] = []
    maximum_timesteps: list[float] = []
    component_equation_count: dict[str, int] = {}
    dependencies: list[frozenset[int]] = []
    residual_names: list[str] = []

    def add_unknown(name: str, initial: float) -> int:
        if name in state_names:
            raise AssemblyError(f"assembled unknown {name!r} is duplicated")
        index = len(state_names)
        state_names.append(name)
        initial_state.append(float(initial))
        return index

    for component in resolved_components:
        model = models[component.id]
        local_unknowns: dict[str, int] = {}
        derivative_indices: dict[str, int] = {}
        for state in model.states:
            index = add_unknown(_qualified(component.id, state.name), state.initial)
            local_unknowns[state.name] = index
            derivative_indices[state.derivative or f"{state.name}_dot"] = index
            derivative_indices[f"{state.name}_dot"] = index
            differential_state_names.append(state_names[index])
        for algebraic in model.algebraics:
            local_unknowns[algebraic.name] = add_unknown(
                _qualified(component.id, algebraic.name), algebraic.initial
            )
        for port in model.power_ports:
            effort_index = add_unknown(_qualified(component.id, port.effort), 0.0)
            flow_index = add_unknown(_qualified(component.id, port.flow), 0.0)
            local_unknowns[port.effort] = effort_index
            local_unknowns[port.flow] = flow_index
            power_ports[(component.id, port.name)] = _PowerEndpoint(
                _qualified(component.id, port.name),
                component.id,
                port,
                effort_index,
                flow_index,
                _orientation_sign(port, f"component {component.id!r} port {port.name!r}"),
            )
        for port in model.signal_ports:
            index = add_unknown(_qualified(component.id, port.name), 0.0)
            local_unknowns[port.name] = index
            signal_ports[(component.id, port.name)] = _SignalEndpoint(
                _qualified(component.id, port.name), component.id, port, index
            )

        local_parameters: dict[str, str] = {}
        for parameter in model.parameters:
            global_name = _qualified(component.id, parameter.name)
            value, uncertainty, correlation_group = _parameter_value(component, parameter)
            local_parameters[parameter.name] = global_name
            default_parameters[global_name] = value
            parameter_bounds[global_name] = (parameter.bounds.lower, parameter.bounds.upper)
            if uncertainty is not None:
                parameter_uncertainty[global_name] = uncertainty
            if correlation_group is not None:
                correlated_uncertainty.append(f"{global_name}:{correlation_group}")

        relations = tuple(
            (relation.name, parse_expression(relation.expression))
            for relation in model.relations
        )
        layout = _ComponentLayout(
            component, model, local_unknowns, derivative_indices, local_parameters, relations
        )
        layouts.append(layout)
        component_equation_count[component.id] = len(relations)
        for relation_name, expression in relations:
            residual_names.append(_qualified(component.id, relation_name))
            related: set[int] = set()
            for symbol in expression.variables():
                if symbol in local_unknowns:
                    related.add(local_unknowns[symbol])
                if symbol in derivative_indices:
                    related.add(derivative_indices[symbol])
            dependencies.append(frozenset(related))

        for symbol, bounds in model.validity.ranges.items():
            if symbol == "t":
                global_symbol = "t"
            elif symbol in local_unknowns:
                global_symbol = state_names[local_unknowns[symbol]]
            elif symbol in local_parameters:
                global_symbol = local_parameters[symbol]
            else:  # validate_model should make this unreachable.
                raise AssemblyError(
                    f"component {component.id!r} validity symbol {symbol!r} cannot be namespaced"
                )
            previous = validity_ranges.get(global_symbol)
            if previous is None:
                validity_ranges[global_symbol] = bounds
            else:
                lower_values = [value for value in (previous.lower, bounds.lower) if value is not None]
                upper_values = [value for value in (previous.upper, bounds.upper) if value is not None]
                lower = max(lower_values) if lower_values else None
                upper = min(upper_values) if upper_values else None
                if lower is not None and upper is not None and lower > upper:
                    raise AssemblyError(
                        f"assembled validity ranges for {global_symbol!r} have empty intersection"
                    )
                validity_ranges[global_symbol] = BoundsSpec(lower, upper)
        validity_assumptions.extend(
            f"{component.id}: {assumption}" for assumption in model.validity.assumptions
        )
        if model.validity.max_timestep is not None:
            maximum_timesteps.append(model.validity.max_timestep)

    def binding_for(reference: PortRef) -> str | None | object:
        key = _connector_key(reference)
        if key in bindings:
            return bindings[key]
        return _reference_name(reference)

    missing = object()

    def resolve_power(reference: PortRef, *, allow_kinematic: bool) -> _PowerEndpoint | None:
        raw_binding = binding_for(reference)
        if raw_binding is None:
            if allow_kinematic:
                return None
            raise AssemblyError(
                f"connector {_connector_key(reference)!r} is explicitly kinematic-only but a power port is required"
            )
        if not isinstance(raw_binding, str):
            raise AssemblyError(
                f"connector {_connector_key(reference)!r} has a non-string PMDL power-port binding"
            )
        component_id = _reference_component(reference)
        endpoint = power_ports.get((component_id, raw_binding), missing)
        if endpoint is missing:
            if _connector_key(reference) not in bindings and allow_kinematic:
                raise AssemblyError(
                    f"attachment connector {_connector_key(reference)!r} has no same-named PMDL power port; "
                    "map it explicitly to a behavior port or None in connector_bindings"
                )
            raise AssemblyError(
                f"connector {_connector_key(reference)!r} maps to missing PMDL power port "
                f"{raw_binding!r} on component {component_id!r}"
            )
        if not isinstance(endpoint, _PowerEndpoint):  # Defensive against invalid registries.
            raise AssemblyError(
                f"connector {_connector_key(reference)!r} did not resolve to a PMDL power port"
            )
        return endpoint

    def resolve_signal(reference: PortRef) -> _SignalEndpoint:
        raw_binding = binding_for(reference)
        if raw_binding is None:
            raise AssemblyError(
                f"connector {_connector_key(reference)!r} is kinematic-only but a signal port is required"
            )
        if not isinstance(raw_binding, str):
            raise AssemblyError(
                f"connector {_connector_key(reference)!r} has a non-string PMDL signal-port binding"
            )
        component_id = _reference_component(reference)
        endpoint = signal_ports.get((component_id, raw_binding))
        if endpoint is None:
            raise AssemblyError(
                f"connector {_connector_key(reference)!r} maps to missing PMDL signal port "
                f"{raw_binding!r} on component {component_id!r}"
            )
        return endpoint

    network_equations: list[_LinearEquation] = []
    connection_equation_count: dict[str, int] = {}
    used_power_indices: dict[tuple[int, int], str] = {}
    used_signal_connectors: dict[str, str] = {}
    driven_signal_indices: dict[int, str] = {}
    kinematic_connection_ids: list[str] = []

    def add_network_equation(equation: _LinearEquation) -> None:
        network_equations.append(equation)
        residual_names.append(equation.name)
        dependencies.append(equation.dependencies)

    for connection in spec.connections:
        before = len(network_equations)
        if connection.kind == "constraint":
            raise UnsupportedAssemblySemanticsError(
                f"connection {connection.id!r} uses kind='constraint'; its mathematical semantics are not declared"
            )
        if connection.kind == "signal":
            endpoints = [resolve_signal(reference) for reference in connection.endpoints]
            for endpoint in endpoints:
                previous = used_signal_connectors.get(endpoint.connector)
                if previous is not None:
                    raise AssemblyError(
                        f"signal connector {endpoint.connector!r} occurs in both {previous!r} and {connection.id!r}"
                    )
                used_signal_connectors[endpoint.connector] = connection.id
            outputs = [endpoint for endpoint in endpoints if endpoint.port.direction == "output"]
            inputs = [endpoint for endpoint in endpoints if endpoint.port.direction == "input"]
            if len(outputs) != 1 or not inputs:
                raise AssemblyError(
                    f"signal connection {connection.id!r} requires exactly one output and at least one input; "
                    f"found outputs={len(outputs)}, inputs={len(inputs)}"
                )
            dimensions = {parse_unit(endpoint.port.unit).dimension for endpoint in endpoints}
            if len(dimensions) != 1:
                raise AssemblyError(
                    f"signal connection {connection.id!r} has incompatible endpoint units"
                )
            source = outputs[0]
            for sink in inputs:
                if sink.index in driven_signal_indices:
                    raise AssemblyError(
                        f"signal input {state_names[sink.index]!r} has multiple drivers: "
                        f"{driven_signal_indices[sink.index]!r} and {connection.id!r}"
                    )
                driven_signal_indices[sink.index] = connection.id
                add_network_equation(
                    _LinearEquation(
                        f"connection.{connection.id}.signal.{sink.connector}",
                        (
                            (sink.index, parse_unit(sink.port.unit).scale),
                            (source.index, -parse_unit(source.port.unit).scale),
                        ),
                    )
                )
        elif connection.kind in {"power", "attachment"}:
            allow_kinematic = connection.kind == "attachment"
            if allow_kinematic and connection.domain not in {
                None,
                "mechanical",
                "rigid_mechanical",
            }:
                raise AssemblyError(
                    f"attachment {connection.id!r} must use a mechanical domain, "
                    f"got {connection.domain!r}"
                )
            resolved = [
                resolve_power(reference, allow_kinematic=allow_kinematic)
                for reference in connection.endpoints
            ]
            behavioral = [endpoint for endpoint in resolved if endpoint is not None]
            if allow_kinematic and not behavioral:
                kinematic_connection_ids.append(connection.id)
                connection_equation_count[connection.id] = 0
                continue
            if len(behavioral) != len(resolved):
                raise AssemblyError(
                    f"attachment {connection.id!r} mixes PMDL mechanical power endpoints with "
                    "kinematic-only endpoints; bind all endpoints behaviorally or all to None"
                )
            endpoints = behavioral
            if not endpoints:  # ConnectionSpec requires endpoints; retain a fail-closed guard.
                raise AssemblyError(
                    f"connection {connection.id!r} has no behavioral PMDL endpoints"
                )
            for endpoint in endpoints:
                key = (endpoint.effort_index, endpoint.flow_index)
                previous = used_power_indices.get(key)
                if previous is not None:
                    raise AssemblyError(
                        f"power port {endpoint.connector!r} occurs in both {previous!r} and {connection.id!r}"
                    )
                used_power_indices[key] = connection.id
            domains = {endpoint.port.domain for endpoint in endpoints}
            if len(domains) != 1:
                raise AssemblyError(
                    f"connection {connection.id!r} mixes power domains {sorted(domains)}"
                )
            domain = next(iter(domains))
            if connection.domain is not None:
                declared = "mechanical" if connection.domain == "rigid_mechanical" else connection.domain
                if declared != domain:
                    raise AssemblyError(
                        f"connection {connection.id!r} declares domain {connection.domain!r} "
                        f"but its PMDL ports use {domain!r}"
                    )
            effort_dimensions = {parse_unit(endpoint.port.effort_unit).dimension for endpoint in endpoints}
            flow_dimensions = {parse_unit(endpoint.port.flow_unit).dimension for endpoint in endpoints}
            if len(effort_dimensions) != 1 or len(flow_dimensions) != 1:
                raise AssemblyError(
                    f"connection {connection.id!r} has incompatible effort/flow units"
                )
            first = endpoints[0]
            if domain == "electrical":
                if connection.kind == "attachment":
                    raise AssemblyError(
                        f"attachment {connection.id!r} resolves to electrical PMDL ports"
                    )
                for endpoint in endpoints[1:]:
                    add_network_equation(
                        _LinearEquation(
                            f"connection.{connection.id}.effort.{endpoint.connector}",
                            (
                                (
                                    endpoint.effort_index,
                                    parse_unit(endpoint.port.effort_unit).scale,
                                ),
                                (
                                    first.effort_index,
                                    -parse_unit(first.port.effort_unit).scale,
                                ),
                            ),
                        )
                    )
                add_network_equation(
                    _LinearEquation(
                        f"connection.{connection.id}.flow_conservation",
                        tuple(
                            (
                                endpoint.flow_index,
                                endpoint.sign * parse_unit(endpoint.port.flow_unit).scale,
                            )
                            for endpoint in endpoints
                        ),
                    )
                )
            elif domain == "mechanical":
                for endpoint in endpoints[1:]:
                    add_network_equation(
                        _LinearEquation(
                            f"connection.{connection.id}.flow.{endpoint.connector}",
                            (
                                (
                                    endpoint.flow_index,
                                    endpoint.sign * parse_unit(endpoint.port.flow_unit).scale,
                                ),
                                (
                                    first.flow_index,
                                    -first.sign * parse_unit(first.port.flow_unit).scale,
                                ),
                            ),
                        )
                    )
                add_network_equation(
                    _LinearEquation(
                        f"connection.{connection.id}.effort_conservation",
                        tuple(
                            (
                                endpoint.effort_index,
                                endpoint.sign * parse_unit(endpoint.port.effort_unit).scale,
                            )
                            for endpoint in endpoints
                        ),
                    )
                )
            else:
                raise UnsupportedAssemblySemanticsError(
                    f"connection {connection.id!r} uses unsupported power domain {domain!r}; "
                    "declare its junction semantics before assembly"
                )
        else:  # Strict ConnectionSpec currently prevents this, retain a hard guard.
            raise UnsupportedAssemblySemanticsError(
                f"connection {connection.id!r} has unsupported kind {connection.kind!r}"
            )
        connection_equation_count[connection.id] = len(network_equations) - before

    control_names: list[str] = []
    control_defaults: dict[str, float] = {}
    control_bounds: dict[str, tuple[float | None, float | None]] = {}
    control_slew_rates: dict[str, float] = {}
    control_source_units: dict[str, tuple[Any, float]] = {}
    for control in spec.controls:
        endpoint = resolve_signal(control.target)
        if endpoint.port.direction != "input":
            raise AssemblyError(
                f"control {control.id!r} targets output signal {endpoint.connector!r}"
            )
        if endpoint.connector in used_signal_connectors:
            raise AssemblyError(
                f"signal input {endpoint.connector!r} is driven by both connection "
                f"{used_signal_connectors[endpoint.connector]!r} and control {control.id!r}"
            )
        if endpoint.index in driven_signal_indices:
            raise AssemblyError(
                f"signal input {state_names[endpoint.index]!r} has multiple drivers"
            )
        driven_signal_indices[endpoint.index] = control.id
        unknown_settings = sorted(
            set(control.settings)
            - {"unit", "default", "minimum", "maximum", "slew_per_second"}
        )
        if unknown_settings:
            raise UnsupportedAssemblySemanticsError(
                f"control {control.id!r} declares unsupported setting(s) "
                f"{unknown_settings}; refusing to ignore requested control behavior"
            )
        target_unit = parse_unit(endpoint.port.unit)
        raw_control_unit = control.settings.get("unit", endpoint.port.unit)
        if not isinstance(raw_control_unit, str):
            raise AssemblyError(
                f"control {control.id!r} settings.unit must be a unit string"
            )
        try:
            control_unit = parse_unit(raw_control_unit)
        except UnitError as exc:
            raise AssemblyError(
                f"control {control.id!r} has invalid settings.unit {raw_control_unit!r}: {exc}"
            ) from exc
        if control_unit.dimension != target_unit.dimension:
            raise AssemblyError(
                f"control {control.id!r} unit {raw_control_unit!r} is incompatible with "
                f"target {endpoint.connector!r} unit {endpoint.port.unit!r}"
            )
        source_unit = (control_unit.dimension, control_unit.scale)
        previous_unit = control_source_units.get(control.source)
        if previous_unit is not None and previous_unit != source_unit:
            raise AssemblyError(
                f"control source {control.source!r} is reused with inconsistent unit scales"
            )
        control_source_units[control.source] = source_unit
        if control.source not in control_names:
            control_names.append(control.source)
        lower = (
            _numeric(
                control.settings["minimum"],
                f"control {control.id!r} settings.minimum",
            )
            if "minimum" in control.settings
            else None
        )
        upper = (
            _numeric(
                control.settings["maximum"],
                f"control {control.id!r} settings.maximum",
            )
            if "maximum" in control.settings
            else None
        )
        if lower is not None and upper is not None and lower > upper:
            raise AssemblyError(
                f"control {control.id!r} settings.minimum exceeds settings.maximum"
            )
        bounds = (lower, upper)
        previous_bounds = control_bounds.get(control.source)
        if previous_bounds is not None and previous_bounds != bounds:
            raise AssemblyError(
                f"control source {control.source!r} is reused with inconsistent bounds "
                f"{previous_bounds} and {bounds}"
            )
        control_bounds[control.source] = bounds
        if "slew_per_second" in control.settings:
            slew_rate = _numeric(
                control.settings["slew_per_second"],
                f"control {control.id!r} settings.slew_per_second",
            )
            if slew_rate <= 0.0:
                raise AssemblyError(
                    f"control {control.id!r} settings.slew_per_second must be positive"
                )
            previous_slew = control_slew_rates.get(control.source)
            if previous_slew is not None and previous_slew != slew_rate:
                raise AssemblyError(
                    f"control source {control.source!r} is reused with inconsistent "
                    f"slew rates {previous_slew} and {slew_rate}"
                )
            control_slew_rates[control.source] = slew_rate
        if "default" in control.settings:
            default = _numeric(control.settings["default"], f"control {control.id!r} settings.default")
            if (lower is not None and default < lower) or (
                upper is not None and default > upper
            ):
                raise AssemblyError(
                    f"control {control.id!r} settings.default={default:.17g} is outside "
                    f"declared bounds {bounds}"
                )
            previous = control_defaults.get(control.source)
            if previous is not None and previous != default:
                raise AssemblyError(
                    f"control source {control.source!r} has conflicting defaults {previous} and {default}"
                )
            control_defaults[control.source] = default
        add_network_equation(
            _LinearEquation(
                f"control.{control.id}",
                ((endpoint.index, target_unit.scale),),
                control.source,
                control_unit.scale,
            )
        )
        connection_equation_count[f"control:{control.id}"] = 1

    unconnected_power = sorted(
        endpoint.connector
        for endpoint in power_ports.values()
        if (endpoint.effort_index, endpoint.flow_index) not in used_power_indices
    )
    if unconnected_power:
        raise AssemblyError(
            "every PMDL power port requires exactly one explicit network; unconnected ports="
            f"{unconnected_power}"
        )
    undriven_inputs = sorted(
        state_names[endpoint.index]
        for endpoint in signal_ports.values()
        if endpoint.port.direction == "input" and endpoint.index not in driven_signal_indices
    )
    if undriven_inputs:
        raise AssemblyError(
            "every PMDL signal input requires exactly one signal connection or control binding; "
            f"undriven inputs={undriven_inputs}"
        )

    unknown_count = len(state_names)
    equation_count = len(residual_names)
    if unknown_count == 0:
        raise AssemblyBalanceError("assembled contraption has no executable PMDL unknowns")
    if equation_count != unknown_count:
        direction = "underdetermined" if equation_count < unknown_count else "overdetermined"
        delta = abs(unknown_count - equation_count)
        raise AssemblyBalanceError(
            f"assembled PMDL system is {direction}: equations={equation_count}, "
            f"unknowns={unknown_count}, difference={delta}; component_equations="
            f"{component_equation_count}, connection_equations={connection_equation_count}"
        )
    structural_rank, matching = _maximum_matching(dependencies, unknown_count)
    if structural_rank != unknown_count:
        unmatched_unknowns = [
            state_names[index] for index in range(unknown_count) if index not in matching
        ]
        matched_equations = set(matching.values())
        unmatched_equations = [
            residual_names[index]
            for index in range(equation_count)
            if index not in matched_equations
        ]
        raise AssemblyBalanceError(
            f"assembled PMDL system is square but structurally singular: structural_rank="
            f"{structural_rank}/{unknown_count}; unmatched_unknowns={unmatched_unknowns}; "
            f"unmatched_equations={unmatched_equations}"
        )

    maximum_timestep = min(maximum_timesteps) if maximum_timesteps else None
    validity = ValiditySpec(
        FrozenDict(validity_ranges), tuple(validity_assumptions), maximum_timestep
    )
    hash_payload = {
        "schema": "contraption.resolved-pmdl-assembly/v1",
        "specification": spec.to_dict(),
        "models": {
            component.id: models[component.id].to_dict() for component in resolved_components
        },
        "connector_bindings": {key: bindings[key] for key in sorted(bindings)},
        "state_names": state_names,
        "differential_state_names": differential_state_names,
        "residual_names": residual_names,
        "network_equations": [
            {
                "name": equation.name,
                "terms": [
                    {"unknown": state_names[index], "coefficient": coefficient}
                    for index, coefficient in equation.terms
                ],
                "control_source": equation.control_source,
                "control_scale": equation.control_scale,
            }
            for equation in network_equations
        ],
        "control_names": control_names,
        "control_bounds": control_bounds,
        "control_slew_rates": control_slew_rates,
    }
    encoded = json.dumps(
        hash_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    pmdl_sha256 = "sha256:" + hashlib.sha256(encoded).hexdigest()
    assembly_sha256 = canonical_assembly_sha256 or pmdl_sha256
    return AssembledPMDLSystem(
        specification=spec,
        layouts=layouts,
        state_names=state_names,
        initial_state=initial_state,
        differential_state_names=differential_state_names,
        residual_names=residual_names,
        component_equation_count=component_equation_count,
        network_equations=network_equations,
        connection_equation_count=connection_equation_count,
        default_parameters=default_parameters,
        parameter_bounds=parameter_bounds,
        parameter_uncertainty=parameter_uncertainty,
        correlated_uncertainty=correlated_uncertainty,
        control_names=control_names,
        control_defaults=control_defaults,
        control_bounds=control_bounds,
        control_slew_rates=control_slew_rates,
        validity=validity,
        dependencies=dependencies,
        kinematic_connection_ids=kinematic_connection_ids,
        connector_bindings=bindings,
        assembly_sha256=assembly_sha256,
        pmdl_sha256=pmdl_sha256,
        canonical_assembly_sha256=canonical_assembly_sha256,
    )


__all__ = [
    "AssembledPMDLSystem",
    "AssemblyBalance",
    "AssemblyBalanceError",
    "AssemblyError",
    "NetworkInvariantError",
    "UnsupportedAssemblySemanticsError",
    "assemble_contraption",
]
