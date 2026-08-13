"""Resolve one closed ``contraption-4`` bundle into physical and PMDL views.

The resolver accepts parsed, hash-verified controller and verification
artifacts. Filesystem access belongs to :mod:`contraption.loading`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..catalog.instantiations import PartInstantiationRegistry
from ..control import ControlSpec, OutputSpec, control_digest
from ..verification import VerificationProgram

from .assembly import AssembledPMDLSystem, AssemblyError, assemble_contraption
from .physical import (
    PhysicalAssemblySpec,
    PhysicalComponentInstance,
    ResolvedPartRegistry,
    ResolvedPartSpec,
    PlanarRootStateBindingSpec,
    PhysicalSpecError,
    ResolvedPhysicalAssembly,
    TransformSpec,
    resolve_physical_assembly,
    split_state_reference,
)
from .specs import (
    ActuatorBindingSpec,
    ConnectionSpec,
    ContraptionSpec,
    ControllerLinkSpec,
    ControllerOutputBindingSpec,
    ExplicitInputBindingSpec,
    FrozenDict,
    ModelSpec,
    PortRef,
    VerificationLinkSpec,
)
from .units import UnitError, parse_unit
from .validation import validate_model

if TYPE_CHECKING:
    from ..control.observer import AffineObserverModel
    from .simulator import SimulationResult


class ResolutionError(ValueError):
    """The canonical assembly closure is missing, stale, or incompatible."""



def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ResolutionError(f"{context} must be an object with string keys")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResolutionError(f"{context} must be an array")
    return value


def _keys(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
    required: set[str] = frozenset(),
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ResolutionError(f"unknown {context} field(s): {', '.join(unknown)}")
    if missing:
        raise ResolutionError(f"missing {context} field(s): {', '.join(missing)}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResolutionError(f"{context} must be a non-empty string")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResolutionError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ResolutionError(f"{context} must be finite")
    return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical(value: Any, context: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                _plain(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ResolutionError(f"{context} must be finite JSON-compatible data: {exc}") from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict((str(key), _freeze(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ResolvedComponent:
    id: str
    part: str
    model_id: str
    parameters: FrozenDict[Any]
    condition: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "part": self.part,
            "model_id": self.model_id,
            "parameters": dict(self.parameters),
            "condition": self.condition,
        }


@dataclass(frozen=True, slots=True)
class DynamicsCompletenessGate:
    """One hash-bound model-fidelity gate that downstream claims must retain."""

    id: str
    status: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "status": self.status, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class DynamicsCompletenessRecord:
    """Strict canonical record of intentionally incomplete assembly dynamics."""

    schema: str
    status: str
    modeled_scope: str
    parameter_basis: FrozenDict[Any]
    gates: tuple[DynamicsCompletenessGate, ...]

    @property
    def open_gates(self) -> tuple[DynamicsCompletenessGate, ...]:
        return tuple(gate for gate in self.gates if gate.status == "open")

    @property
    def complete(self) -> bool:
        return self.status == "complete" and not self.open_gates

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "modeled_scope": self.modeled_scope,
            "parameter_basis": _plain(self.parameter_basis),
            "gates": [gate.to_dict() for gate in self.gates],
        }


def _parse_dynamics_completeness(
    metadata: Mapping[str, Any],
) -> DynamicsCompletenessRecord | None:
    value = metadata.get("dynamics_completeness")
    if value is None:
        return None
    raw = _mapping(value, "contraption.metadata.dynamics_completeness")
    required = {"schema", "status", "modeled_scope", "parameter_basis", "gates"}
    _keys(raw, required, "contraption.metadata.dynamics_completeness", required)
    schema = _text(
        raw["schema"], "contraption.metadata.dynamics_completeness.schema"
    )
    if schema != "contraption.dynamics-completeness/v1":
        raise ResolutionError(
            "contraption.metadata.dynamics_completeness.schema must be "
            "'contraption.dynamics-completeness/v1'"
        )
    status = _text(
        raw["status"], "contraption.metadata.dynamics_completeness.status"
    )
    if status not in {"complete", "incomplete"}:
        raise ResolutionError(
            "contraption.metadata.dynamics_completeness.status must be "
            "'complete' or 'incomplete'"
        )
    modeled_scope = _text(
        raw["modeled_scope"],
        "contraption.metadata.dynamics_completeness.modeled_scope",
    )
    parameter_basis = _mapping(
        raw["parameter_basis"],
        "contraption.metadata.dynamics_completeness.parameter_basis",
    )
    gate_values = _sequence(
        raw["gates"], "contraption.metadata.dynamics_completeness.gates"
    )
    gates: list[DynamicsCompletenessGate] = []
    for index, value in enumerate(gate_values):
        context = f"contraption.metadata.dynamics_completeness.gates[{index}]"
        gate = _mapping(value, context)
        fields = {"id", "status", "reason"}
        _keys(gate, fields, context, fields)
        gate_status = _text(gate["status"], f"{context}.status")
        if gate_status not in {"open", "closed"}:
            raise ResolutionError(f"{context}.status must be 'open' or 'closed'")
        gates.append(
            DynamicsCompletenessGate(
                _text(gate["id"], f"{context}.id"),
                gate_status,
                _text(gate["reason"], f"{context}.reason"),
            )
        )
    duplicate_ids = sorted(
        {gate.id for gate in gates if sum(item.id == gate.id for item in gates) > 1}
    )
    if duplicate_ids:
        raise ResolutionError(
            "contraption.metadata.dynamics_completeness has duplicate gate id(s): "
            + ", ".join(duplicate_ids)
        )
    open_gates = tuple(gate for gate in gates if gate.status == "open")
    if status == "complete" and open_gates:
        raise ResolutionError(
            "dynamics_completeness status='complete' may not retain open gates"
        )
    if status == "incomplete" and not open_gates:
        raise ResolutionError(
            "dynamics_completeness status='incomplete' requires at least one open gate"
        )
    return DynamicsCompletenessRecord(
        schema,
        status,
        modeled_scope,
        _freeze(_canonical(parameter_basis, "dynamics completeness parameter_basis")),
        tuple(gates),
    )


@dataclass(frozen=True, slots=True)
class ResolvedContraptionRecord:
    """Narrow typed protocol accepted by the PMDL assembly compiler."""

    format: str
    id: str
    name: str
    version: str
    components: tuple[ResolvedComponent, ...]
    connections: tuple[ConnectionSpec, ...]
    actuators: tuple[ActuatorBindingSpec, ...]
    controllers: tuple[ControllerLinkSpec, ...]
    verifications: tuple[VerificationLinkSpec, ...]
    environment: FrozenDict[Any]
    metadata: FrozenDict[Any]
    physical_root: FrozenDict[Any]
    source: FrozenDict[Any]

    def to_dict(self) -> dict[str, Any]:
        # Return the exact normalized source.  It is the provenance record, not
        # a second model assembled from selected fields.
        return _canonical(dict(self.source), "resolved contraption source")


@dataclass(frozen=True, slots=True)
class ResolvedExplicitInputBinding:
    """One controller input wired to an external pin or exact PMDL state."""

    kind: str
    source: str
    state_name: str | None = None
    state_index: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedImplicitInputBinding:
    """One controller latent bound to an exact namespaced PMDL variable."""

    source: str
    state_name: str
    state_index: int | None = None


@dataclass(frozen=True, slots=True)
class ResolvedControllerOutputBinding:
    """One controller output exposed as a plant signal or external pin."""

    kind: Literal["signal", "external"]
    source: str
    state_name: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"signal", "external"}:
            raise ResolutionError(
                "resolved controller output kind must be 'signal' or 'external'"
            )
        if not isinstance(self.source, str) or not self.source:
            raise ResolutionError(
                "resolved controller output source must be a non-empty string"
            )
        if self.kind == "signal":
            if not isinstance(self.state_name, str) or not self.state_name:
                raise ResolutionError(
                    "resolved signal output requires a non-empty PMDL state_name"
                )
        elif self.state_name is not None:
            raise ResolutionError(
                "resolved external output may not declare a PMDL state_name"
            )


@dataclass(frozen=True, slots=True)
class ResolvedController:
    """A parsed controller plus fully resolved runtime wiring."""

    id: str
    spec: ControlSpec
    explicit_input_bindings: FrozenDict[ResolvedExplicitInputBinding]
    implicit_input_bindings: FrozenDict[ResolvedImplicitInputBinding]
    output_bindings: FrozenDict[ResolvedControllerOutputBinding]
    controller_link_digest: str
    observer: "AffineObserverModel | None" = None

    @property
    def plant_output_bindings(self) -> FrozenDict[str]:
        """Map only PMDL-driving outputs to assembled plant control sources."""

        return FrozenDict(
            (name, binding.source)
            for name, binding in self.output_bindings.items()
            if binding.kind == "signal"
        )


@dataclass(frozen=True, slots=True)
class ResolvedVerificationInputBinding:
    """One verification input wired to an exact PMDL state coordinate."""

    source: str
    state_name: str
    state_index: int


@dataclass(frozen=True, slots=True)
class ResolvedVerification:
    """A parsed verification program plus its observable wiring."""

    id: str
    spec: VerificationProgram
    input_bindings: FrozenDict[ResolvedVerificationInputBinding]


@dataclass(frozen=True, slots=True)
class ResolvedAssembly:
    """One verified assembly closure and its physical/behavior projections."""

    specification: ResolvedContraptionRecord
    parts: ResolvedPartRegistry
    component_models: FrozenDict[ModelSpec]
    connector_bindings: FrozenDict[str | None]
    controllers: FrozenDict[ResolvedController]
    verifications: FrozenDict[ResolvedVerification]
    physical: ResolvedPhysicalAssembly
    system: AssembledPMDLSystem

    @property
    def assembly_sha256(self) -> str:
        return self.physical.assembly_sha256

    @property
    def scene(self) -> Mapping[str, Any]:
        return self.physical.scene

    @property
    def dynamics_completeness(self) -> DynamicsCompletenessRecord | None:
        """Return the already-validated hash-bound dynamics-fidelity record."""

        return _parse_dynamics_completeness(self.specification.metadata)

    def resolve_signal_state(
        self,
        reference: PortRef | str,
        *,
        direction: str,
        system: AssembledPMDLSystem | None = None,
        context: str = "signal binding",
    ) -> tuple[str, int]:
        """Re-resolve one canonical connector to its exact PMDL state coordinate."""

        components = {item.id: item for item in self.specification.components}
        endpoint = _resolve_signal_endpoint(
            reference,
            direction=direction,
            components=components,
            parts=self.parts,
            models=self.component_models,
            context=context,
        )
        active_system = self.system if system is None else system
        try:
            index = active_system.state_names.index(endpoint.state_name)
        except ValueError as exc:
            raise ResolutionError(
                f"{context} resolved to missing PMDL state {endpoint.state_name!r}"
            ) from exc
        return endpoint.state_name, index

    def attest_pmdl_system(self) -> AssembledPMDLSystem:
        """Independently rebuild and attest the executable PMDL projection.

        Compiler provenance may not trust the mutable assembled-system object or
        its stored digest.  The canonical resolved source, exact-hash component
        models, connector projection, and physical assembly identity are compiled
        again, then every executable semantic field is compared before the fresh
        system is returned.
        """

        try:
            canonical = assemble_contraption(
                self.specification,
                self.component_models,
                component_models=self.component_models,
                connector_bindings=self.connector_bindings,
                canonical_assembly_sha256=self.physical.assembly_sha256,
            )
        except AssemblyError as exc:
            raise ResolutionError(
                f"canonical PMDL reassembly failed during provenance attestation: {exc}"
            ) from exc

        actual = self.system

        def equation_payload(system: AssembledPMDLSystem) -> tuple[Any, ...]:
            return tuple(
                (
                    equation.name,
                    tuple(equation.terms),
                    equation.control_source,
                    equation.control_scale,
                )
                for equation in system._network_equations
            )

        def layout_payload(system: AssembledPMDLSystem) -> tuple[Any, ...]:
            return tuple(
                (
                    layout.component.id,
                    _plain(layout.component.parameters),
                    layout.component.model_reference,
                    layout.model.to_dict(),
                    tuple(layout.unknown_indices.items()),
                    tuple(layout.derivative_indices.items()),
                    tuple(layout.parameter_names.items()),
                    tuple((name, repr(expression)) for name, expression in layout.relations),
                    tuple(layout.process_noise_channels),
                    tuple(
                        (
                            increment.target_index,
                            increment.target_name,
                            repr(increment.expression),
                        )
                        for increment in layout.process_noise_increments
                    ),
                )
                for layout in system._layouts
            )

        fields = {
            "specification": lambda value: value.specification.to_dict(),
            "layouts": layout_payload,
            "component_models": lambda value: {
                name: model.to_dict() for name, model in value.component_models.items()
            },
            "connector_bindings": lambda value: dict(value.connector_bindings),
            "state_names": lambda value: tuple(value.state_names),
            "initial_state": lambda value: tuple(value.initial_state),
            "differential_state_names": lambda value: tuple(value.differential_state_names),
            "differential_state_indices": lambda value: tuple(value.differential_state_indices),
            "algebraic_names": lambda value: tuple(value.algebraic_names),
            "algebraic_indices": lambda value: tuple(value.algebraic_indices),
            "residual_names": lambda value: tuple(value.residual_names),
            "network_equations": equation_payload,
            "network_residual_names": lambda value: tuple(value.network_residual_names),
            "default_parameters": lambda value: dict(value.default_parameters),
            "parameter_bounds": lambda value: dict(value.parameter_bounds),
            "parameter_uncertainty": lambda value: _plain(value.parameter_uncertainty),
            "correlated_uncertainty": lambda value: tuple(value._correlated_uncertainty),
            "control_names": lambda value: tuple(value.control_names),
            "control_defaults": lambda value: dict(value.control_defaults),
            "control_bounds": lambda value: dict(value.control_bounds),
            "control_slew_rates": lambda value: dict(value.control_slew_rates),
            "validity": lambda value: value.validity.to_dict(),
            "process_noise_channel_names": lambda value: tuple(value.process_noise_channel_names),
            "process_noise_seed_policy": lambda value: value.process_noise_seed_policy,
            "process_noise_reproducibility": lambda value: value.process_noise_reproducibility,
            "has_process_noise": lambda value: value.has_process_noise,
            "kinematic_connection_ids": lambda value: tuple(value.kinematic_connection_ids),
            "assembly_sha256": lambda value: value.assembly_sha256,
            "pmdl_sha256": lambda value: value.pmdl_sha256,
            "canonical_assembly_sha256": lambda value: value.canonical_assembly_sha256,
            "balance": lambda value: value.balance,
        }
        mismatches = [
            name
            for name, projection in fields.items()
            if projection(actual) != projection(canonical)
        ]
        if mismatches:
            raise ResolutionError(
                "resolved PMDL system differs from independent canonical reassembly; "
                "mismatched semantic field(s): " + ", ".join(mismatches)
            )
        return canonical

    def with_configuration(
        self,
        *,
        root_pose: TransformSpec | Mapping[str, Any] | None = None,
        joint_coordinates: Mapping[str, float] | None = None,
    ) -> ResolvedPhysicalAssembly:
        return self.physical.with_configuration(
            root_pose=root_pose,
            joint_coordinates=joint_coordinates,
        )

    def configuration_from_state(
        self,
        state: Mapping[str, Any] | Sequence[Any] | Any,
        *,
        state_names: Sequence[str] | None = None,
    ) -> ResolvedPhysicalAssembly:
        """Resolve one exact PMDL state vector into its physical configuration."""

        required = _required_physical_state_references(self.physical)
        if isinstance(state, Mapping):
            if state_names is not None:
                raise ResolutionError(
                    "state_names must be omitted when state is a namespaced mapping"
                )
            values = _mapping(state, "physical configuration state")
            missing = sorted(required - set(values))
            if missing:
                raise ResolutionError(
                    "physical configuration state is missing binding(s): "
                    + ", ".join(missing)
                )
            vector_names = tuple(self.system.state_names)
            unknown = sorted(set(values) - set(vector_names))
            if unknown:
                raise ResolutionError(
                    "physical configuration state contains unknown PMDL state(s): "
                    + ", ".join(unknown)
                )
            vector = np.asarray(
                [
                    float(values.get(name, self.system.initial_state[index]))
                    for index, name in enumerate(vector_names)
                ],
                dtype=float,
            )
        else:
            vector_names = (
                tuple(self.system.state_names)
                if state_names is None
                else tuple(state_names)
            )
            if vector_names != tuple(self.system.state_names):
                raise ResolutionError(
                    "state_names must exactly match the resolved PMDL system"
                )
            vector = _numpy_array(state, "physical configuration state")
            if vector.ndim != 1 or vector.shape[0] != len(vector_names):
                raise ResolutionError(
                    "physical configuration state must be a one-dimensional vector "
                    f"with {len(vector_names)} values"
                )
        if not np.all(np.isfinite(vector)):
            raise ResolutionError("physical configuration state contains non-finite values")
        indices = {name: index for index, name in enumerate(vector_names)}
        missing = sorted(required - set(indices))
        if missing:
            raise ResolutionError(
                "physical configuration state_names are missing binding(s): "
                + ", ".join(missing)
            )
        samples = vector.reshape((1, 1, -1))
        try:
            return self._configuration_from_result_sample(samples, indices, 0, 0)
        except (PhysicalSpecError, UnitError, ValueError) as exc:
            raise ResolutionError(
                f"physical configuration reconstructed from state is invalid: {exc}"
            ) from exc

    def validate_simulation_state(
        self,
        state: Mapping[str, Any] | Sequence[Any] | Any,
        *,
        state_names: Sequence[str] | None = None,
        sample_index: int | None = None,
        step_index: int | None = None,
        time_s: float | None = None,
        require_initial_configuration: bool = False,
    ) -> ResolvedPhysicalAssembly | tuple[ResolvedPhysicalAssembly, ...]:
        """Validate one accepted simulator state, or a batch of sample states.

        A one-dimensional vector or namespaced mapping returns one resolved
        configuration.  A two-dimensional ``[sample,state]`` array validates
        every sample and returns a tuple in sample-axis order.  Callers should
        set ``require_initial_configuration`` for simulator t0 admission so a
        mutated initial state cannot diverge from the hash-bound static scene.
        """

        if sample_index is not None and (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
        ):
            raise ResolutionError("sample_index must be a non-negative integer")
        if step_index is not None and (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or step_index < 0
        ):
            raise ResolutionError("step_index must be a non-negative integer")
        if time_s is not None:
            if isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
                raise ResolutionError("time_s must be a finite non-negative number")
            time_s = float(time_s)
            if not math.isfinite(time_s) or time_s < 0.0:
                raise ResolutionError("time_s must be a finite non-negative number")
        if not isinstance(require_initial_configuration, bool):
            raise ResolutionError("require_initial_configuration must be boolean")

        is_mapping = isinstance(state, Mapping)
        if is_mapping:
            rows: tuple[Any, ...] = (state,)
            batch = False
        else:
            array = _numpy_array(state, "simulation state")
            if array.ndim == 1:
                rows = (array,)
                batch = False
            elif array.ndim == 2:
                if array.shape[0] < 1:
                    raise ResolutionError(
                        "batched simulation state must contain at least one sample"
                    )
                if sample_index is not None:
                    raise ResolutionError(
                        "sample_index must be omitted for a batched simulation state; "
                        "the array sample axis supplies exact sample indices"
                    )
                rows = tuple(array[index] for index in range(array.shape[0]))
                batch = True
            else:
                raise ResolutionError(
                    "simulation state must be a namespaced mapping, one-dimensional "
                    "state vector, or two-dimensional [sample,state] array"
                )

        configurations: list[ResolvedPhysicalAssembly] = []
        for row_index, row in enumerate(rows):
            exact_sample_index = row_index if batch else sample_index
            context_fields: list[str] = []
            if exact_sample_index is not None:
                context_fields.append(f"sample={exact_sample_index}")
            if step_index is not None:
                context_fields.append(f"step_index={step_index}")
            if time_s is not None:
                context_fields.append(f"time_s={time_s:.17g}")
            context = ", ".join(context_fields) if context_fields else "unindexed state"
            try:
                configured = self.configuration_from_state(
                    row,
                    state_names=state_names,
                )
                if require_initial_configuration and (
                    configured.body_poses != self.physical.body_poses
                    or configured.connector_poses != self.physical.connector_poses
                ):
                    raise PhysicalSpecError(
                        "initial physical configuration does not equal the "
                        "hash-bound static assembly scene"
                    )
            except (ResolutionError, PhysicalSpecError, UnitError, ValueError) as exc:
                raise ResolutionError(
                    f"physical boundary validation failed at {context}: {exc}"
                ) from exc
            configurations.append(configured)

        if batch:
            return tuple(configurations)
        return configurations[0]

    def validate_simulation_step(
        self,
        *,
        step_index: int,
        time_s: float,
        state: Any,
        state_names: Sequence[str],
        backend: Any,
    ) -> None:
        """Simulator hook that rejects each accepted batch before publication."""

        del backend  # ``_numpy_array`` safely detaches and copies backend tensors.
        self.validate_simulation_state(
            state,
            state_names=state_names,
            step_index=step_index,
            time_s=time_s,
            require_initial_configuration=step_index == 0,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "assembly_sha256": self.assembly_sha256,
            "pmdl_sha256": self.system.pmdl_sha256,
            "contraption_id": self.specification.id,
            "component_count": len(self.specification.components),
            "connection_count": len(self.specification.connections),
            "state_count": len(self.system.state_names),
            "differential_state_count": len(self.system.differential_state_names),
            "equation_count": len(self.system.residual_names),
            "kinematic_connection_ids": list(self.system.kinematic_connection_ids),
            "controller_ids": list(self.controllers),
            "verification_ids": list(self.verifications),
        }

    def _configuration_from_result_sample(
        self,
        samples: np.ndarray,
        state_indices: Mapping[str, int],
        sample_index: int,
        time_index: int,
    ) -> ResolvedPhysicalAssembly:
        root_pose = self.physical._root_pose
        binding = self.physical.root_state_binding
        if binding is not None:
            x = _state_sample_in_unit(
                self,
                binding.x,
                samples[sample_index, time_index, state_indices[binding.x]],
                "m",
            )
            y = _state_sample_in_unit(
                self,
                binding.y,
                samples[sample_index, time_index, state_indices[binding.y]],
                "m",
            )
            yaw = _state_sample_in_unit(
                self,
                binding.yaw,
                samples[sample_index, time_index, state_indices[binding.yaw]],
                "rad",
                angle=True,
            )
            root_pose = root_pose.with_planar_coordinates(x_m=x, y_m=y, yaw_rad=yaw)

        coordinates: dict[str, float] = {}
        for attachment in self.physical.attachments:
            if attachment.joint.kind != "revolute":
                continue
            assert attachment.joint.coordinate is not None
            coordinate = attachment.joint.coordinate
            coordinates[coordinate] = _attachment_coordinate_from_sample(
                self,
                attachment,
                samples,
                state_indices,
                sample_index,
                time_index,
            )
        return self.physical.with_configuration(
            root_pose=root_pose,
            joint_coordinates=coordinates,
        )

    def _validated_result_arrays(
        self, result: "SimulationResult"
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        from .simulator import SimulationResult

        if not isinstance(result, SimulationResult):
            raise ResolutionError(
                "physical pose reconstruction requires an actual SimulationResult "
                "with per-sample trajectories; distribution means are not accepted"
            )
        metadata = _mapping(result.metadata, "simulation result metadata")
        if metadata.get("assembly_sha256") != self.assembly_sha256:
            raise ResolutionError(
                "simulation result assembly_sha256 is missing or stale: expected "
                f"{self.assembly_sha256}, got {metadata.get('assembly_sha256')!r}"
            )
        if metadata.get("pmdl_sha256") != self.system.pmdl_sha256:
            raise ResolutionError(
                "simulation result pmdl_sha256 is missing or stale: expected "
                f"{self.system.pmdl_sha256}, got {metadata.get('pmdl_sha256')!r}"
            )
        state_names = tuple(result.state_names)
        if state_names != tuple(self.system.state_names):
            raise ResolutionError(
                "simulation result state_names do not exactly match the resolved PMDL "
                "system; reordered, missing, or extra states are refused"
            )
        if len(set(state_names)) != len(state_names):
            raise ResolutionError("simulation result state_names must be unique")
        samples = _numpy_array(result.samples, "simulation result samples")
        times = _numpy_array(result.time, "simulation result time")
        if samples.ndim != 3:
            raise ResolutionError(
                "simulation result samples must have shape [sample,time,state], got "
                f"rank {samples.ndim}"
            )
        if times.ndim != 1:
            raise ResolutionError("simulation result time must be one-dimensional")
        if samples.shape[0] < 1 or samples.shape[1] < 1:
            raise ResolutionError("simulation result must contain samples and time points")
        if samples.shape[1] != times.shape[0]:
            raise ResolutionError(
                "simulation result sample time dimension does not match result.time"
            )
        if samples.shape[2] != len(state_names):
            raise ResolutionError(
                "simulation result sample state dimension does not match state_names"
            )
        if not np.all(np.isfinite(samples)):
            raise ResolutionError("simulation result samples contain non-finite values")
        if not np.all(np.isfinite(times)):
            raise ResolutionError("simulation result time contains non-finite values")
        if abs(float(times[0])) > 1e-12 or np.any(np.diff(times) <= 0.0):
            raise ResolutionError(
                "simulation result time must start at zero and increase strictly"
            )
        declared_samples = metadata.get("sample_count")
        if (
            isinstance(declared_samples, bool)
            or not isinstance(declared_samples, int)
            or declared_samples != samples.shape[0]
        ):
            raise ResolutionError(
                "simulation result metadata.sample_count does not match the actual "
                f"sample axis ({samples.shape[0]})"
            )
        state_indices = {name: index for index, name in enumerate(state_names)}
        required = _required_physical_state_references(self.physical)
        missing = sorted(required - set(state_indices))
        if missing:
            raise ResolutionError(
                "simulation result is missing physical state binding(s): "
                + ", ".join(missing)
            )
        return samples, times, state_indices

    def validate_simulation_result(self, result: "SimulationResult") -> None:
        """Validate every sample/time against all physical connection constraints."""

        samples, times, state_indices = self._validated_result_arrays(result)
        for sample_index in range(samples.shape[0]):
            for time_index, time_s in enumerate(times):
                try:
                    configured = self._configuration_from_result_sample(
                        samples, state_indices, sample_index, time_index
                    )
                    if time_index == 0 and (
                        configured.body_poses != self.physical.body_poses
                        or configured.connector_poses != self.physical.connector_poses
                    ):
                        raise PhysicalSpecError(
                            "initial physical configuration does not equal the "
                            "hash-bound static assembly scene"
                        )
                except (PhysicalSpecError, UnitError, ValueError) as exc:
                    raise ResolutionError(
                        "physical boundary validation failed at "
                        f"sample={sample_index}, time_index={time_index}, "
                        f"time_s={float(time_s):.17g}: {exc}"
                    ) from exc

    def body_pose_frames(
        self, result: "SimulationResult", *, sample_index: int
    ) -> FrozenDict[Any]:
        """Return exact resolver-produced pose frames for one actual sample."""

        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ResolutionError("sample_index must be an integer")
        samples, times, state_indices = self._validated_result_arrays(result)
        if sample_index < 0 or sample_index >= samples.shape[0]:
            raise ResolutionError(
                f"sample_index {sample_index} is outside [0, {samples.shape[0]})"
            )
        # Admission is deliberately ensemble-wide.  Rendering one selected
        # path must not conceal another sample that violates a hard constraint.
        self.validate_simulation_result(result)
        frames: list[dict[str, Any]] = []
        for time_index, time_s in enumerate(times):
            configured = self._configuration_from_result_sample(
                samples, state_indices, sample_index, time_index
            )
            frames.append(
                {
                    "time_s": float(time_s),
                    "body_poses": {
                        key: configured.body_poses[key].to_dict()
                        for key in sorted(configured.body_poses)
                    },
                    "connector_poses": {
                        key: configured.connector_poses[key].to_dict()
                        for key in sorted(configured.connector_poses)
                    },
                }
            )
        static_bodies = {
            key: self.physical.body_poses[key].to_dict()
            for key in sorted(self.physical.body_poses)
        }
        static_connectors = {
            key: self.physical.connector_poses[key].to_dict()
            for key in sorted(self.physical.connector_poses)
        }
        if (
            frames[0]["body_poses"] != static_bodies
            or frames[0]["connector_poses"] != static_connectors
        ):
            raise ResolutionError(
                f"simulation sample {sample_index} initial physical configuration "
                "does not equal the hash-bound static scene"
            )
        return _freeze(
            {
                "assembly_sha256": self.assembly_sha256,
                "frames": frames,
            }
        )


def _numpy_array(value: Any, context: str) -> np.ndarray:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ResolutionError(f"{context} cannot be read as a numeric array: {exc}") from exc


def _required_physical_state_references(
    physical: ResolvedPhysicalAssembly,
) -> set[str]:
    result: set[str] = set()
    if physical.root_state_binding is not None:
        result.update(
            (
                physical.root_state_binding.x,
                physical.root_state_binding.y,
                physical.root_state_binding.yaw,
            )
        )
    result.update(
        binding.state
        for attachment in physical.attachments
        if attachment.joint.kind == "revolute"
        for binding in attachment.joint.coordinate_bindings
    )
    for component in physical.components:
        part = physical.parts[component.part]
        result.update(
            f"{component.id}.{connector.kinematics.state}"
            for connector in part.connectors
            if connector.kinematics is not None
        )
    return result


def _state_spec_for_reference(
    component_models: Mapping[str, ModelSpec], reference: str
) -> Any:
    component_id, state_name = split_state_reference(reference)
    try:
        model = component_models[component_id]
    except KeyError as exc:
        raise ResolutionError(
            f"physical state binding {reference!r} references unknown component "
            f"{component_id!r}"
        ) from exc
    states = {state.name: state for state in model.states}
    try:
        return states[state_name]
    except KeyError as exc:
        raise ResolutionError(
            f"physical state binding {reference!r} references missing PMDL state "
            f"{state_name!r} on component {component_id!r}"
        ) from exc


def _validate_state_unit(reference: str, unit: str, target: str, *, angle: bool) -> None:
    try:
        source_unit = parse_unit(unit)
        target_unit = parse_unit(target)
    except UnitError as exc:
        raise ResolutionError(
            f"physical state binding {reference!r} has invalid unit {unit!r}: {exc}"
        ) from exc
    if not source_unit.compatible_with(target_unit):
        raise ResolutionError(
            f"physical state binding {reference!r} unit {unit!r} is incompatible "
            f"with required unit {target!r}"
        )
    if angle and unit not in {"rad", "radian", "deg"}:
        raise ResolutionError(
            f"angular physical state binding {reference!r} must explicitly use "
            f"rad or deg, got {unit!r}"
        )


def _state_value_in_unit(
    component_models: Mapping[str, ModelSpec],
    reference: str,
    value: Any,
    target: str,
    *,
    angle: bool = False,
) -> float:
    state = _state_spec_for_reference(component_models, reference)
    _validate_state_unit(reference, state.unit, target, angle=angle)
    return _convert_unit_value(
        _number(value, f"physical state {reference!r} value"),
        state.unit,
        target,
        f"physical state binding {reference!r}",
    )


def _state_sample_in_unit(
    assembly: ResolvedAssembly,
    reference: str,
    value: Any,
    target: str,
    *,
    angle: bool = False,
) -> float:
    return _state_value_in_unit(
        assembly.component_models,
        reference,
        value,
        target,
        angle=angle,
    )


def _attachment_coordinate_from_sample(
    assembly: ResolvedAssembly,
    attachment: Any,
    samples: np.ndarray,
    state_indices: Mapping[str, int],
    sample_index: int,
    time_index: int,
) -> float:
    if (
        attachment.joint.kind != "revolute"
        or not attachment.joint.coordinate_bindings
    ):
        raise PhysicalSpecError(
            f"attachment {attachment.id!r} has no revolute coordinate bindings"
        )
    state_values: list[tuple[str, float, float]] = []
    for binding in attachment.joint.coordinate_bindings:
        state_angle = _state_sample_in_unit(
            assembly,
            binding.state,
            samples[sample_index, time_index, state_indices[binding.state]],
            "rad",
            angle=True,
        )
        joint_angle = state_angle + binding.joint_angle_at_state_zero_rad
        state_values.append((binding.state, state_angle, joint_angle))
    primary_state, primary_value, primary_joint_angle = state_values[0]
    for state, value, joint_angle in state_values[1:]:
        error = math.atan2(
            math.sin(joint_angle - primary_joint_angle),
            math.cos(joint_angle - primary_joint_angle),
        )
        if abs(error) > 1e-9:
            raise PhysicalSpecError(
                f"revolute attachment {attachment.id!r} holonomic coordinate "
                f"mismatch: primary {primary_state}={primary_value:.17g} rad "
                f"implies joint_angle={primary_joint_angle:.17g} rad, but "
                f"{state}={value:.17g} rad implies joint_angle={joint_angle:.17g} "
                f"rad; wrapped_error_rad={error:.17g}, tolerance=1e-9"
            )
    return primary_value


def _validate_physical_state_bindings(
    physical: ResolvedPhysicalAssembly,
    component_models: Mapping[str, ModelSpec],
) -> dict[str, float]:
    """Validate PMDL-backed root/joint frames and return initial radians."""

    binding = physical.root_state_binding
    if binding is not None:
        expected_x = _state_value_in_unit(
            component_models,
            binding.x,
            _state_spec_for_reference(component_models, binding.x).initial,
            "m",
        )
        expected_y = _state_value_in_unit(
            component_models,
            binding.y,
            _state_spec_for_reference(component_models, binding.y).initial,
            "m",
        )
        expected_yaw = _state_value_in_unit(
            component_models,
            binding.yaw,
            _state_spec_for_reference(component_models, binding.yaw).initial,
            "rad",
            angle=True,
        )
        _roll, _pitch, pose_yaw = physical._root_pose.roll_pitch_yaw()
        pose_x, pose_y, _pose_z = physical._root_pose.translation_m
        for label, actual, expected, unit in (
            ("x", pose_x, expected_x, "m"),
            ("y", pose_y, expected_y, "m"),
            ("yaw", pose_yaw, expected_yaw, "rad"),
        ):
            delta = (
                math.atan2(math.sin(actual - expected), math.cos(actual - expected))
                if label == "yaw"
                else actual - expected
            )
            if abs(delta) > 1e-9:
                raise ResolutionError(
                    f"physical root pose {label}={actual:.17g} {unit} disagrees "
                    f"with initial PMDL state binding value={expected:.17g} {unit}"
                )

    component_ids = {component.id for component in physical.components}
    resolved_parts = {
        component.id: physical.parts[component.part]
        for component in physical.components
    }
    initial_coordinates: dict[str, float] = {}
    for attachment in physical.attachments:
        if attachment.joint.kind != "revolute":
            continue
        assert attachment.joint.coordinate is not None
        endpoint_components = {
            attachment.parent.component,
            attachment.child.component,
        }
        expected_bindings: set[str] = set()
        for endpoint in (attachment.parent, attachment.child):
            connector = resolved_parts[endpoint.component].connector_map[
                endpoint.connector
            ]
            if connector.joint_coordinate_state is not None:
                expected_bindings.add(
                    f"{endpoint.component}.{connector.joint_coordinate_state}"
                )
        declared_bindings = {
            binding.state for binding in attachment.joint.coordinate_bindings
        }
        if declared_bindings != expected_bindings:
            raise ResolutionError(
                f"revolute attachment {attachment.id!r} coordinate-binding "
                f"coverage mismatch; missing={sorted(expected_bindings - declared_bindings)}, "
                f"extra={sorted(declared_bindings - expected_bindings)}"
            )
        initial_joint_angles: list[tuple[str, float, float]] = []
        for coordinate_binding in attachment.joint.coordinate_bindings:
            reference = coordinate_binding.state
            coordinate_component, _state_name = split_state_reference(reference)
            if (
                coordinate_component not in component_ids
                or coordinate_component not in endpoint_components
            ):
                raise ResolutionError(
                    f"revolute attachment {attachment.id!r} coordinate binding "
                    f"{reference!r} must reference a PMDL state on an endpoint"
                )
            state = _state_spec_for_reference(component_models, reference)
            state_initial = _state_value_in_unit(
                component_models,
                reference,
                state.initial,
                "rad",
                angle=True,
            )
            initial_joint_angles.append(
                (
                    reference,
                    state_initial,
                    state_initial
                    + coordinate_binding.joint_angle_at_state_zero_rad,
                )
            )
        primary_reference, primary_initial, primary_joint_angle = initial_joint_angles[0]
        for reference, state_initial, joint_angle in initial_joint_angles[1:]:
            error = math.atan2(
                math.sin(joint_angle - primary_joint_angle),
                math.cos(joint_angle - primary_joint_angle),
            )
            if abs(error) > 1e-9:
                raise ResolutionError(
                    f"revolute attachment {attachment.id!r} initial holonomic "
                    f"coordinate mismatch: {primary_reference}={primary_initial:.17g} "
                    f"rad implies {primary_joint_angle:.17g} rad, but "
                    f"{reference}={state_initial:.17g} rad implies "
                    f"{joint_angle:.17g} rad"
                )
        initial_coordinates[primary_reference] = primary_initial

    attachment_coordinates = {
        attachment.joint.coordinate_bindings[0].state
        for attachment in physical.attachments
        if attachment.joint.kind == "revolute"
        and attachment.joint.coordinate_bindings
    }
    for component in physical.components:
        part = physical.parts[component.part]
        for connector in part.connectors:
            if connector.kinematics is None:
                continue
            reference = f"{component.id}.{connector.kinematics.state}"
            _state_spec_for_reference(component_models, reference)
            _validate_state_unit(
                reference,
                _state_spec_for_reference(component_models, reference).unit,
                "rad",
                angle=True,
            )
            if reference not in attachment_coordinates:
                raise ResolutionError(
                    f"connector {component.id}.{connector.id} counter_rotation state "
                    f"{reference!r} is not the coordinate of a revolute attachment"
                )
    return initial_coordinates


def _parse_component(
    value: Mapping[str, Any],
    index: int,
    instantiations: PartInstantiationRegistry,
) -> ResolvedComponent:
    context = f"contraption.components[{index}]"
    _keys(
        value,
        {"id", "instantiation"},
        context,
        {"id", "instantiation"},
    )
    identifier = _text(value["id"], f"{context}.id")
    instantiation_id = _text(
        value["instantiation"], f"{context}.instantiation"
    )
    try:
        instantiation = instantiations[instantiation_id]
    except KeyError as exc:
        raise ResolutionError(
            f"component {identifier!r} references missing model instantiation "
            f"{instantiation_id!r}"
        ) from exc
    return ResolvedComponent(
        identifier,
        instantiation_id,
        instantiation.model_instance.model.id,
        instantiation.parameters,
        instantiation.model_instance.condition,
    )


def _model_digest(model: ModelSpec) -> str:
    return "sha256:" + hashlib.sha256(model.to_json().encode("utf-8")).hexdigest()


def _convert_unit_value(
    value: float, source_unit: str, target_unit: str, context: str
) -> float:
    try:
        return parse_unit(source_unit).convert_value_to(value, parse_unit(target_unit))
    except UnitError as exc:
        raise ResolutionError(f"{context} unit mismatch: {exc}") from exc


def _validate_part_parameter_bindings(
    part: ResolvedPartSpec, model: ModelSpec
) -> None:
    parameters = {parameter.name: parameter for parameter in model.parameters}
    for binding in part.parameter_bindings:
        try:
            parameter = parameters[binding.model_parameter]
        except KeyError as exc:
            raise ResolutionError(
                f"part {part.id!r} physical binding references missing PMDL "
                f"parameter {binding.model_parameter!r}"
            ) from exc
        measured = part.measure_parameter(binding)
        default = _convert_unit_value(
            parameter.default,
            parameter.unit,
            binding.unit,
            f"part {part.id!r} parameter {parameter.name!r}",
        )
        error = abs(default - measured)
        if error > binding.absolute_tolerance:
            raise ResolutionError(
                f"part {part.id!r} PMDL parameter {parameter.name!r} default="
                f"{default:.17g} {binding.unit} disagrees with physical measure="
                f"{measured:.17g} {binding.unit}; absolute_error={error:.17g}, "
                f"tolerance={binding.absolute_tolerance:.17g}"
            )
        uncertainty = parameter.uncertainty
        if (
            uncertainty.distribution != "fixed"
            or bool(uncertainty.parameters)
            or uncertainty.correlation_group is not None
        ):
            raise ResolutionError(
                f"part {part.id!r} geometry-bound parameter "
                f"{parameter.name!r} may not declare independent PMDL uncertainty; "
                "sample-specific geometry is not represented"
            )


def _validate_component_parameter_bindings(
    component: ResolvedComponent,
    part: ResolvedPartSpec,
    model: ModelSpec,
) -> None:
    parameters = {parameter.name: parameter for parameter in model.parameters}
    for binding in part.parameter_bindings:
        parameter = parameters[binding.model_parameter]
        raw = component.parameters.get(parameter.name, parameter.default)
        source_unit = parameter.unit
        if isinstance(raw, Mapping):
            if "value" not in raw:
                raise ResolutionError(
                    f"component {component.id!r} parameter {parameter.name!r} "
                    "override must contain numeric value"
                )
            value = _number(
                raw["value"],
                f"component {component.id!r} parameter {parameter.name!r}.value",
            )
            if raw.get("unit") is not None:
                source_unit = _text(
                    raw["unit"],
                    f"component {component.id!r} parameter {parameter.name!r}.unit",
                )
            override_uncertainty = raw.get("uncertainty")
            if override_uncertainty not in (None, {}):
                if not isinstance(override_uncertainty, Mapping) or bool(
                    override_uncertainty
                ):
                    raise ResolutionError(
                        f"component {component.id!r} geometry-bound parameter "
                        f"{parameter.name!r} may not declare independent uncertainty"
                    )
        else:
            value = _number(
                raw, f"component {component.id!r} parameter {parameter.name!r}"
            )
        value = _convert_unit_value(
            value,
            source_unit,
            binding.unit,
            f"component {component.id!r} parameter {parameter.name!r}",
        )
        measured = part.measure_parameter(binding)
        error = abs(value - measured)
        if error > binding.absolute_tolerance:
            raise ResolutionError(
                f"component {component.id!r} parameter {parameter.name!r}="
                f"{value:.17g} {binding.unit} disagrees with part physical "
                f"measure={measured:.17g} {binding.unit}; absolute_error="
                f"{error:.17g}, tolerance={binding.absolute_tolerance:.17g}"
            )


def _verify_part_model(
    part: ResolvedPartSpec,
    model_registry: Mapping[str, ModelSpec],
) -> ModelSpec:
    try:
        model = model_registry[part.model.id]
    except KeyError as exc:
        raise ResolutionError(
            f"part {part.id!r} references unregistered PMDL model {part.model.id!r}"
        ) from exc
    if not isinstance(model, ModelSpec):
        raise ResolutionError(
            f"registry entry {part.model.id!r} must be a parsed ModelSpec"
        )
    if model.id != part.model.id or model.version != part.model.version:
        raise ResolutionError(
            f"part {part.id!r} model identity/version mismatch: part="
            f"{part.model.id}@{part.model.version}, registry={model.id}@{model.version}"
        )
    digest = _model_digest(model)
    if digest != part.model.sha256:
        raise ResolutionError(
            f"part {part.id!r} PMDL content hash mismatch: expected "
            f"{part.model.sha256}, got {digest}; stale or substituted model refused"
        )
    report = validate_model(model)
    if not report.valid:
        details = "; ".join(
            f"[{issue.code}] {issue.path}: {issue.message}" for issue in report.errors
        )
        raise ResolutionError(f"part {part.id!r} PMDL is invalid: {details}")
    _validate_part_parameter_bindings(part, model)

    declared_power = {port.name: port for port in model.power_ports}
    declared_signal = {port.name: port for port in model.signal_ports}
    mapped = {
        connector.model_port: connector
        for connector in part.connectors
        if connector.model_port is not None
    }
    expected = set(declared_power) | set(declared_signal)
    if set(mapped) != expected:
        raise ResolutionError(
            f"part {part.id!r} connector/model-port coverage mismatch; "
            f"missing={sorted(expected - set(mapped))}, extra={sorted(set(mapped) - expected)}"
        )
    for port_name, connector in mapped.items():
        if port_name in declared_power:
            if connector.domain != declared_power[port_name].domain:
                raise ResolutionError(
                    f"part {part.id!r} connector {connector.id!r} domain "
                    f"{connector.domain!r} does not match PMDL power port {port_name!r} "
                    f"domain {declared_power[port_name].domain!r}"
                )
        elif connector.domain != "signal":
            raise ResolutionError(
                f"part {part.id!r} signal connector {connector.id!r} must use "
                f"domain 'signal', got {connector.domain!r}"
            )
    state_map = {state.name: state for state in model.states}
    for connector in part.connectors:
        if connector.joint_coordinate_state is None:
            continue
        if connector.model_port is None:
            raise ResolutionError(
                f"part {part.id!r} connector {connector.id!r} binds joint "
                "coordinate state but has model_port:null"
            )
        try:
            state = state_map[connector.joint_coordinate_state]
        except KeyError as exc:
            raise ResolutionError(
                f"part {part.id!r} connector {connector.id!r} binds missing "
                f"PMDL angle state {connector.joint_coordinate_state!r}"
            ) from exc
        _validate_state_unit(
            f"{part.id}.{state.name}", state.unit, "rad", angle=True
        )
    return model


@dataclass(frozen=True, slots=True)
class _ResolvedSignalEndpoint:
    reference: PortRef
    model_port: Any
    state_name: str


def _resolve_signal_endpoint(
    reference: PortRef | str,
    *,
    direction: str,
    components: Mapping[str, ResolvedComponent],
    parts: ResolvedPartRegistry,
    models: Mapping[str, ModelSpec],
    context: str,
) -> _ResolvedSignalEndpoint:
    try:
        endpoint = reference if isinstance(reference, PortRef) else PortRef.from_dict(reference)
    except Exception as exc:
        raise ResolutionError(f"{context} must name a component connector: {exc}") from exc
    try:
        component = components[endpoint.component]
        connector = parts[component.part].connector_map[endpoint.port]
    except KeyError as exc:
        raise ResolutionError(
            f"{context} references missing connector {endpoint.component}.{endpoint.port}"
        ) from exc
    if connector.model_port is None:
        raise ResolutionError(
            f"{context} connector {endpoint.component}.{endpoint.port} has no PMDL port binding"
        )
    ports = {port.name: port for port in models[endpoint.component].signal_ports}
    try:
        model_port = ports[connector.model_port]
    except KeyError as exc:
        raise ResolutionError(
            f"{context} connector {endpoint.component}.{endpoint.port} does not bind a PMDL signal port"
        ) from exc
    if model_port.direction != direction:
        raise ResolutionError(
            f"{context} must bind a PMDL signal {direction}, got {model_port.direction!r}"
        )
    if model_port.shape:
        raise ResolutionError(f"{context} must bind a scalar PMDL signal")
    return _ResolvedSignalEndpoint(
        endpoint, model_port, f"{endpoint.component}.{model_port.name}"
    )


def _validate_control_dtype(control_dtype: str, pmdl_dtype: str, context: str) -> None:
    compatible = (
        control_dtype == "real" and pmdl_dtype in {"float32", "float64"}
    ) or (control_dtype == "bool" and pmdl_dtype == "bool")
    if not compatible:
        raise ResolutionError(
            f"{context} dtype {control_dtype!r} is incompatible with PMDL dtype {pmdl_dtype!r}"
        )


def _validate_units(
    authored: str,
    pmdl: str,
    context: str,
    *,
    exact_scale: bool,
) -> None:
    try:
        authored_unit = parse_unit(authored)
        pmdl_unit = parse_unit(pmdl)
    except UnitError as exc:
        raise ResolutionError(f"{context} has invalid units: {exc}") from exc
    if authored_unit.dimension != pmdl_unit.dimension or (
        exact_scale and authored_unit.scale != pmdl_unit.scale
    ):
        qualifier = " and scale" if exact_scale else ""
        raise ResolutionError(
            f"{context} unit {authored!r} must match PMDL unit {pmdl!r} dimension{qualifier}"
        )


def _output_settings(output: OutputSpec) -> FrozenDict[Any]:
    settings: dict[str, Any] = {"unit": output.unit, "default": output.default}
    if output.bounds.lower is not None:
        settings["minimum"] = output.bounds.lower
    if output.bounds.upper is not None:
        settings["maximum"] = output.bounds.upper
    if output.slew_rate is not None:
        settings["slew_per_second"] = output.slew_rate
    return FrozenDict(settings)


def _prepare_runtime_wiring(
    specification: ContraptionSpec,
    controller_specs: Mapping[str, ControlSpec],
    *,
    components: Mapping[str, ResolvedComponent],
    parts: ResolvedPartRegistry,
    models: Mapping[str, ModelSpec],
) -> tuple[
    tuple[ActuatorBindingSpec, ...],
    FrozenDict[ResolvedController],
]:
    expected = {link.id for link in specification.controllers}
    supplied = set(controller_specs)
    if supplied != expected:
        raise ResolutionError(
            "controller artifact registry must exactly cover contraption links; "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )

    actuators: list[ActuatorBindingSpec] = []
    target_drivers: dict[str, str] = {}
    output_sources: dict[str, str] = {}
    actuator_ids: set[str] = set()

    def claim_output_source(source: str, driver_id: str) -> None:
        previous = output_sources.get(source)
        if previous is not None:
            raise ResolutionError(
                f"controller output source {source!r} is exposed by both "
                f"{previous!r} and {driver_id!r}"
            )
        output_sources[source] = driver_id

    for actuator in specification.actuators:
        if not actuator.external:
            raise ResolutionError(
                f"top-level actuator {actuator.id!r} must declare external:true; "
                "controller outputs are derived from controllers[].outputs"
            )
        endpoint = _resolve_signal_endpoint(
            actuator.target,
            direction="input",
            components=components,
            parts=parts,
            models=models,
            context=f"actuator {actuator.id!r}",
        )
        previous = target_drivers.get(endpoint.state_name)
        if previous is not None:
            raise ResolutionError(
                f"PMDL signal input {endpoint.state_name!r} is driven by both {previous!r} "
                f"and actuator {actuator.id!r}"
            )
        if actuator.id in actuator_ids:
            raise ResolutionError(f"duplicate actuator id {actuator.id!r}")
        target_drivers[endpoint.state_name] = actuator.id
        actuator_ids.add(actuator.id)
        actuators.append(actuator)

    controllers: dict[str, ResolvedController] = {}
    for link in specification.controllers:
        program = controller_specs[link.id]
        if not isinstance(program, ControlSpec):
            raise ResolutionError(
                f"controller artifact {link.id!r} must be a parsed ControlSpec"
            )
        if program.id != link.id:
            raise ResolutionError(
                f"controller link id {link.id!r} does not match artifact id {program.id!r}"
            )
        actual_digest = control_digest(program)
        if actual_digest != link.program.sha256:
            raise ResolutionError(
                f"controller artifact {link.id!r} canonical content hash mismatch: "
                f"expected {link.program.sha256}, got {actual_digest}"
            )
        explicit = {item.name: item for item in program.explicit_inputs}
        authored_inputs = dict(link.explicit_inputs)
        if set(authored_inputs) != set(explicit):
            raise ResolutionError(
                f"controller {link.id!r} explicit input coverage mismatch; "
                f"missing={sorted(set(explicit) - set(authored_inputs))}, "
                f"extra={sorted(set(authored_inputs) - set(explicit))}"
            )
        resolved_inputs: dict[str, ResolvedExplicitInputBinding] = {}
        for name, input_spec in explicit.items():
            binding = authored_inputs[name]
            assert isinstance(binding, ExplicitInputBindingSpec)
            context = f"controller {link.id!r} input {name!r}"
            if input_spec.source == "sensor":
                if binding.signal is None:
                    raise ResolutionError(f"{context} requires a signal binding")
                endpoint = _resolve_signal_endpoint(
                    binding.signal,
                    direction="output",
                    components=components,
                    parts=parts,
                    models=models,
                    context=context,
                )
                _validate_control_dtype(input_spec.dtype, endpoint.model_port.dtype, context)
                _validate_units(
                    input_spec.unit,
                    endpoint.model_port.unit,
                    context,
                    exact_scale=True,
                )
                resolved_inputs[name] = ResolvedExplicitInputBinding(
                    "sensor", binding.signal, endpoint.state_name, None
                )
            else:
                if binding.external is None:
                    raise ResolutionError(f"{context} requires an external binding")
                resolved_inputs[name] = ResolvedExplicitInputBinding(
                    "external", binding.external
                )

        implicit = {item.name: item for item in program.implicit_inputs}
        authored_implicit = dict(link.implicit_inputs)
        if set(authored_implicit) != set(implicit):
            raise ResolutionError(
                f"controller {link.id!r} implicit input coverage mismatch; "
                f"missing={sorted(set(implicit) - set(authored_implicit))}, "
                f"extra={sorted(set(authored_implicit) - set(implicit))}"
            )
        resolved_implicit = {
            name: ResolvedImplicitInputBinding(source, source)
            for name, source in authored_implicit.items()
        }

        outputs = {item.name: item for item in program.outputs}
        authored_outputs = dict(link.outputs)
        if set(authored_outputs) != set(outputs):
            raise ResolutionError(
                f"controller {link.id!r} output coverage mismatch; "
                f"missing={sorted(set(outputs) - set(authored_outputs))}, "
                f"extra={sorted(set(authored_outputs) - set(outputs))}"
            )
        output_bindings: dict[str, ResolvedControllerOutputBinding] = {}
        for name, output in outputs.items():
            binding = authored_outputs[name]
            assert isinstance(binding, ControllerOutputBindingSpec)
            driver_id = f"{link.id}.{name}"
            if binding.external is not None:
                claim_output_source(binding.external, driver_id)
                output_bindings[name] = ResolvedControllerOutputBinding(
                    "external", binding.external
                )
                continue
            assert binding.signal is not None
            target = binding.signal
            endpoint = _resolve_signal_endpoint(
                target,
                direction="input",
                components=components,
                parts=parts,
                models=models,
                context=f"controller {link.id!r} output {name!r}",
            )
            _validate_control_dtype(
                output.dtype,
                endpoint.model_port.dtype,
                f"controller {link.id!r} output {name!r}",
            )
            _validate_units(
                output.unit,
                endpoint.model_port.unit,
                f"controller {link.id!r} output {name!r}",
                exact_scale=False,
            )
            actuator_id = driver_id
            claim_output_source(actuator_id, driver_id)
            previous = target_drivers.get(endpoint.state_name)
            if previous is not None:
                raise ResolutionError(
                    f"PMDL signal input {endpoint.state_name!r} is driven by both "
                    f"{previous!r} and controller output {actuator_id!r}"
                )
            if actuator_id in actuator_ids:
                raise ResolutionError(f"duplicate actuator id {actuator_id!r}")
            source = actuator_id
            actuators.append(
                ActuatorBindingSpec(
                    actuator_id,
                    source,
                    PortRef.from_dict(target),
                    _output_settings(output),
                    False,
                )
            )
            target_drivers[endpoint.state_name] = actuator_id
            actuator_ids.add(actuator_id)
            output_bindings[name] = ResolvedControllerOutputBinding(
                "signal", source, endpoint.state_name
            )
        controllers[link.id] = ResolvedController(
            link.id,
            program,
            FrozenDict(resolved_inputs),
            FrozenDict(resolved_implicit),
            FrozenDict(output_bindings),
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    link.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        )
    return tuple(actuators), FrozenDict(controllers)


def _index_controllers(
    controllers: Mapping[str, ResolvedController],
    state_names: Sequence[str],
    models: Mapping[str, ModelSpec],
) -> FrozenDict[ResolvedController]:
    indices = {name: index for index, name in enumerate(state_names)}
    result: dict[str, ResolvedController] = {}
    for identifier, controller in controllers.items():
        bindings: dict[str, ResolvedExplicitInputBinding] = {}
        for name, binding in controller.explicit_input_bindings.items():
            if binding.kind == "external":
                bindings[name] = binding
                continue
            assert binding.state_name is not None
            try:
                index = indices[binding.state_name]
            except KeyError as exc:
                raise ResolutionError(
                    f"controller {identifier!r} sensor {name!r} resolved to missing PMDL state "
                    f"{binding.state_name!r}"
                ) from exc
            bindings[name] = ResolvedExplicitInputBinding(
                binding.kind, binding.source, binding.state_name, index
            )
        implicit_bindings: dict[str, ResolvedImplicitInputBinding] = {}
        implicit_specs = {item.name: item for item in controller.spec.implicit_inputs}
        for name, binding in controller.implicit_input_bindings.items():
            try:
                index = indices[binding.state_name]
            except KeyError as exc:
                raise ResolutionError(
                    f"controller {identifier!r} implicit input {name!r} binds missing "
                    f"PMDL variable {binding.state_name!r}"
                ) from exc
            if "." not in binding.state_name:
                raise ResolutionError(
                    f"controller {identifier!r} implicit input {name!r} must use an "
                    "exact namespaced PMDL variable"
                )
            component_id, local_name = binding.state_name.rsplit(".", 1)
            try:
                model = models[component_id]
            except KeyError as exc:
                raise ResolutionError(
                    f"controller {identifier!r} implicit input {name!r} names missing "
                    f"component {component_id!r}"
                ) from exc
            pmdl_unit = _observable_unit(
                model,
                local_name,
                f"controller {identifier!r} implicit input {name!r}",
            )
            _validate_units(
                implicit_specs[name].unit,
                pmdl_unit,
                f"controller {identifier!r} implicit input {name!r}",
                exact_scale=True,
            )
            implicit_bindings[name] = ResolvedImplicitInputBinding(
                binding.source, binding.state_name, index
            )
        result[identifier] = ResolvedController(
            controller.id,
            controller.spec,
            FrozenDict(bindings),
            FrozenDict(implicit_bindings),
            controller.output_bindings,
            controller.controller_link_digest,
            controller.observer,
        )
    return FrozenDict(result)


def _observable_unit(model: ModelSpec, local_name: str, context: str) -> str:
    matches: list[str] = []
    matches.extend(item.unit for item in model.states if item.name == local_name)
    matches.extend(item.unit for item in model.algebraics if item.name == local_name)
    matches.extend(item.unit for item in model.signal_ports if item.name == local_name)
    for port in model.power_ports:
        if port.effort == local_name:
            matches.append(port.effort_unit)
        if port.flow == local_name:
            matches.append(port.flow_unit)
    if len(matches) != 1:
        raise ResolutionError(f"{context} does not identify one PMDL scalar observable")
    return matches[0]


def _resolve_verifications(
    specification: ContraptionSpec,
    verification_specs: Mapping[str, VerificationProgram],
    *,
    components: Mapping[str, ResolvedComponent],
    parts: ResolvedPartRegistry,
    models: Mapping[str, ModelSpec],
    state_names: Sequence[str],
) -> FrozenDict[ResolvedVerification]:
    expected = {link.id for link in specification.verifications}
    supplied = set(verification_specs)
    if supplied != expected:
        raise ResolutionError(
            "verification artifact registry must exactly cover contraption links; "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )
    indices = {name: index for index, name in enumerate(state_names)}
    result: dict[str, ResolvedVerification] = {}
    for link in specification.verifications:
        program = verification_specs[link.id]
        if not isinstance(program, VerificationProgram):
            raise ResolutionError(
                f"verification artifact {link.id!r} must be a parsed VerificationProgram"
            )
        if program.id != link.id:
            raise ResolutionError(
                f"verification link id {link.id!r} does not match artifact id {program.id!r}"
            )
        if program.sha256 != link.program.sha256:
            raise ResolutionError(
                f"verification artifact {link.id!r} canonical content hash mismatch: "
                f"expected {link.program.sha256}, got {program.sha256}"
            )
        inputs = {item.name: item for item in program.inputs}
        authored = dict(link.inputs)
        if set(authored) != set(inputs):
            raise ResolutionError(
                f"verification {link.id!r} input coverage mismatch; "
                f"missing={sorted(set(inputs) - set(authored))}, "
                f"extra={sorted(set(authored) - set(inputs))}"
            )
        bindings: dict[str, ResolvedVerificationInputBinding] = {}
        for name, input_spec in inputs.items():
            source = authored[name]
            context = f"verification {link.id!r} input {name!r}"
            state_name = source
            try:
                ref = PortRef.from_dict(source)
                component = components[ref.component]
                connector = parts[component.part].connector_map.get(ref.port)
            except Exception:
                connector = None
            if connector is not None:
                endpoint = _resolve_signal_endpoint(
                    source,
                    direction="output",
                    components=components,
                    parts=parts,
                    models=models,
                    context=context,
                )
                state_name = endpoint.state_name
            try:
                state_index = indices[state_name]
            except KeyError as exc:
                raise ResolutionError(
                    f"{context} references unknown PMDL state {state_name!r}"
                ) from exc
            component_id, local_name = state_name.rsplit(".", 1)
            source_unit = _observable_unit(models[component_id], local_name, context)
            _validate_units(input_spec.unit, source_unit, context, exact_scale=True)
            bindings[name] = ResolvedVerificationInputBinding(
                source, state_name, state_index
            )
        result[link.id] = ResolvedVerification(
            link.id, program, FrozenDict(bindings)
        )
    return FrozenDict(result)


def resolve_assembly(
    specification: ContraptionSpec,
    instantiations: PartInstantiationRegistry,
    model_registry: Mapping[str, ModelSpec],
    *,
    controller_specs: Mapping[str, ControlSpec] | None = None,
    verification_specs: Mapping[str, VerificationProgram] | None = None,
    joint_coordinates: Mapping[str, float] | None = None,
) -> ResolvedAssembly:
    """Resolve a strict contraption-4 closure with plural artifact wiring."""

    if not isinstance(specification, ContraptionSpec):
        raise TypeError("specification must be a parsed ContraptionSpec")
    if not isinstance(instantiations, PartInstantiationRegistry):
        raise TypeError("instantiations must be a PartInstantiationRegistry")
    if not isinstance(model_registry, Mapping):
        raise TypeError("model_registry must implement Mapping")
    instantiations.validate_models(model_registry)
    canonical_source = _canonical(specification.to_dict(), "contraption")
    part_registry = instantiations.resolved_parts
    components = tuple(
        _parse_component(item.to_dict(), index, instantiations)
        for index, item in enumerate(specification.components)
    )
    if len({item.id for item in components}) != len(components):
        raise ResolutionError("contraption component identifiers must be unique")
    connections = specification.connections
    duplicate_connection_ids = sorted(
        {
            connection.id
            for connection in connections
            if sum(item.id == connection.id for item in connections) > 1
        }
    )
    if duplicate_connection_ids:
        raise ResolutionError(
            "contraption connection identifiers must be unique; duplicates="
            f"{duplicate_connection_ids}"
        )
    for index, connection in enumerate(connections):
        if connection.metadata:
            raise ResolutionError(
                f"contraption.connections[{index}].metadata must be empty; "
                "connection semantics require typed fields rather than opaque metadata"
            )
    component_by_id = {item.id: item for item in components}
    used_part_ids = {item.part for item in components}
    verified_models: dict[str, ModelSpec] = {}
    part_models = {
        part_id: _verify_part_model(part_registry[part_id], model_registry)
        for part_id in sorted(used_part_ids)
    }
    for component in components:
        verified_models[component.id] = part_models[component.part]
        _validate_component_parameter_bindings(
            component, part_registry[component.part], verified_models[component.id]
        )

    actuators, controller_drafts = _prepare_runtime_wiring(
        specification,
        {} if controller_specs is None else controller_specs,
        components=component_by_id,
        parts=part_registry,
        models=verified_models,
    )
    metadata = _mapping(specification.metadata, "contraption.metadata")
    dynamics_record = _parse_dynamics_completeness(metadata)
    if dynamics_record is None:
        raise ResolutionError(
            "contraption requires metadata.dynamics_completeness"
        )
    record = ResolvedContraptionRecord(
        "contraption-4",
        specification.id,
        specification.name,
        specification.version,
        components,
        connections,
        actuators,
        specification.controllers,
        specification.verifications,
        _freeze(_canonical(specification.environment, "contraption.environment")),
        _freeze(_canonical(metadata, "contraption.metadata")),
        _freeze(_canonical(specification.physical_root, "contraption.physical_root")),
        _freeze(canonical_source),
    )

    zero_coordinates: dict[str, float] = {}
    for connection in connections:
        joint = connection.joint
        if joint is not None and joint.kind == "revolute":
            assert joint.coordinate is not None
            zero_coordinates[joint.coordinate] = 0.0
    try:
        physical_source = dict(canonical_source)
        physical_source["components"] = [item.to_dict() for item in components]
        frozen_physical_source = _freeze(
            _canonical(physical_source, "physical assembly source")
        )
        physical = resolve_physical_assembly(
            PhysicalAssemblySpec(
                specification.id,
                tuple(
                    PhysicalComponentInstance(component.id, component.part)
                    for component in components
                ),
                connections,
                record.physical_root,
                frozen_physical_source,
            ),
            part_registry,
            zero_coordinates,
        )
    except PhysicalSpecError as exc:
        raise ResolutionError(f"physical assembly resolution failed: {exc}") from exc
    initial_coordinates = _validate_physical_state_bindings(physical, verified_models)
    if joint_coordinates is not None:
        provided = dict(joint_coordinates)
        if set(provided) != set(initial_coordinates):
            raise ResolutionError(
                "initial joint_coordinates must exactly cover PMDL-backed revolute states; "
                f"missing={sorted(set(initial_coordinates) - set(provided))}, "
                f"extra={sorted(set(provided) - set(initial_coordinates))}"
            )
        for name, expected_value in initial_coordinates.items():
            actual = _number(provided[name], f"joint_coordinates.{name}")
            if abs(actual - expected_value) > 1e-9:
                raise ResolutionError(
                    f"initial joint coordinate {name!r} disagrees with PMDL state"
                )
    try:
        physical = physical.with_configuration(joint_coordinates=initial_coordinates)
    except PhysicalSpecError as exc:
        raise ResolutionError(
            f"initial physical configuration validation failed: {exc}"
        ) from exc

    connector_bindings: dict[str, str | None] = {}
    referenced = {
        f"{endpoint.component}.{endpoint.port}"
        for connection in connections
        for endpoint in connection.endpoints
    } | {f"{actuator.target.component}.{actuator.target.port}" for actuator in actuators}
    for key in sorted(referenced):
        component_id, connector_id = key.rsplit(".", 1)
        try:
            component = component_by_id[component_id]
            connector = part_registry[component.part].connector_map[connector_id]
        except KeyError as exc:
            raise ResolutionError(f"unresolved connector binding {key!r}") from exc
        connector_bindings[key] = connector.model_port
    try:
        system = assemble_contraption(
            record,
            model_registry,
            component_models=verified_models,
            connector_bindings=connector_bindings,
            canonical_assembly_sha256=physical.assembly_sha256,
        )
    except AssemblyError as exc:
        raise ResolutionError(f"PMDL assembly failed: {exc}") from exc
    if system.assembly_sha256 != physical.assembly_sha256:
        raise ResolutionError(
            "internal assembly hash mismatch between physical and PMDL projections"
        )
    indexed_controllers = _index_controllers(
        controller_drafts, system.state_names, verified_models
    )
    controllers_with_observers: dict[str, ResolvedController] = {}
    for identifier, controller in indexed_controllers.items():
        observer = None
        if controller.spec.implicit_inputs:
            if dynamics_record is None:
                raise ResolutionError(
                    f"controller {identifier!r} uses implicit inputs but the contraption "
                    "does not declare dynamics_completeness"
                )
            from ..control.observer import (
                ObserverDerivationError,
                derive_affine_observer,
            )

            try:
                observer = derive_affine_observer(
                    system,
                    controller.spec,
                    explicit_bindings=controller.explicit_input_bindings,
                    implicit_bindings=controller.implicit_input_bindings,
                    output_bindings=controller.plant_output_bindings,
                    assembly_sha256=physical.assembly_sha256,
                    pmdl_sha256=system.pmdl_sha256,
                    controller_link_digest=controller.controller_link_digest,
                    dynamics_completeness=dynamics_record.to_dict(),
                )
            except ObserverDerivationError as exc:
                raise ResolutionError(
                    f"controller {identifier!r} observer derivation failed: {exc}"
                ) from exc
        controllers_with_observers[identifier] = ResolvedController(
            controller.id,
            controller.spec,
            controller.explicit_input_bindings,
            controller.implicit_input_bindings,
            controller.output_bindings,
            controller.controller_link_digest,
            observer,
        )
    controllers = FrozenDict(controllers_with_observers)
    verifications = _resolve_verifications(
        specification,
        {} if verification_specs is None else verification_specs,
        components=component_by_id,
        parts=part_registry,
        models=verified_models,
        state_names=system.state_names,
    )
    return ResolvedAssembly(
        record,
        part_registry,
        FrozenDict(verified_models),
        FrozenDict(connector_bindings),
        controllers,
        verifications,
        physical,
        system,
    )


def resolve_body_pose_frames(
    assembly: ResolvedAssembly,
    result: "SimulationResult",
    *,
    sample_index: int,
) -> FrozenDict[Any]:
    """Functional spelling of :meth:`ResolvedAssembly.body_pose_frames`."""

    if not isinstance(assembly, ResolvedAssembly):
        raise TypeError("assembly must be a ResolvedAssembly")
    return assembly.body_pose_frames(result, sample_index=sample_index)


__all__ = [
    "DynamicsCompletenessGate",
    "DynamicsCompletenessRecord",
    "ResolutionError",
    "ResolvedAssembly",
    "ResolvedComponent",
    "ResolvedContraptionRecord",
    "ResolvedController",
    "ResolvedControllerOutputBinding",
    "ResolvedExplicitInputBinding",
    "ResolvedVerification",
    "ResolvedVerificationInputBinding",
    "resolve_assembly",
    "resolve_body_pose_frames",
]
