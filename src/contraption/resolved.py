"""Resolve one contraption source into the canonical physical and PMDL assembly.

``contraption-2`` instances name component packages, never independent model or
geometry records.  A package owns the exact PMDL content hash, solid geometry,
and connector-to-model-port bindings.  Resolution verifies the complete
closure and produces one :class:`ResolvedAssembly` consumed by simulation,
visualization, build planning, and compilation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from .assembly import AssembledPMDLSystem, AssemblyError, assemble_contraption
from .controls import ControlProgram
from .physical import (
    ComponentPackageRegistry,
    ComponentPackageSpec,
    PlanarRootStateBindingSpec,
    PhysicalSpecError,
    ResolvedPhysicalAssembly,
    TransformSpec,
    resolve_physical_assembly,
    split_state_reference,
)
from .specs import ControlBindingSpec, FrozenDict, ModelSpec, PortRef
from .units import UnitError, parse_unit
from .validation import validate_model

if TYPE_CHECKING:
    from .simulator import SimulationResult


class ResolutionError(ValueError):
    """The canonical assembly closure is missing, stale, or incompatible."""


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


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
    package: str
    model_id: str
    parameters: FrozenDict[Any]
    condition: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "package": self.package,
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
class ResolvedConnection:
    id: str
    kind: str
    endpoints: tuple[PortRef, ...]
    domain: str | None
    joint: FrozenDict[Any]
    metadata: FrozenDict[Any]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "endpoints": [endpoint.to_dict() for endpoint in self.endpoints],
            "domain": self.domain,
            "metadata": dict(self.metadata),
        }
        if self.kind == "attachment":
            result["joint"] = dict(self.joint)
        return result


@dataclass(frozen=True, slots=True)
class ResolvedContraptionRecord:
    """Narrow typed protocol accepted by the PMDL assembly compiler."""

    format: str
    id: str
    name: str
    version: str
    components: tuple[ResolvedComponent, ...]
    connections: tuple[ResolvedConnection, ...]
    controls: tuple[ControlBindingSpec, ...]
    environment: FrozenDict[Any]
    metadata: FrozenDict[Any]
    physical_root: FrozenDict[Any]
    controller: FrozenDict[Any] | None
    source: FrozenDict[Any]

    def to_dict(self) -> dict[str, Any]:
        # Return the exact normalized source.  It is the provenance record, not
        # a second model assembled from selected fields.
        return _canonical(dict(self.source), "resolved contraption source")


@dataclass(frozen=True, slots=True)
class ResolvedAssembly:
    """One verified assembly closure and its physical/behavior projections."""

    specification: ResolvedContraptionRecord
    packages: ComponentPackageRegistry
    component_models: FrozenDict[ModelSpec]
    connector_bindings: FrozenDict[str | None]
    controller: ControlProgram | None
    physical: ResolvedPhysicalAssembly
    system: AssembledPMDLSystem

    @property
    def assembly_sha256(self) -> str:
        return self.physical.assembly_sha256

    @property
    def scene(self) -> Mapping[str, Any]:
        return self.physical.scene

    @property
    def controller_output_bindings(self) -> FrozenDict[str]:
        """Hash-bound controller-output to assembled-control-source mapping."""

        if self.specification.controller is None:
            return FrozenDict()
        bindings = self.specification.controller["output_bindings"]
        assert isinstance(bindings, FrozenDict)
        return bindings

    @property
    def controller_telemetry_outputs(self) -> tuple[str, ...]:
        """Controller outputs intentionally retained as non-actuator telemetry."""

        if self.specification.controller is None:
            return ()
        values = self.specification.controller["telemetry_outputs"]
        assert isinstance(values, tuple)
        return values

    @property
    def dynamics_completeness(self) -> DynamicsCompletenessRecord:
        """Return the already-validated hash-bound dynamics-fidelity record."""

        result = _parse_dynamics_completeness(self.specification.metadata)
        if result is None:  # Resolution requires it; retain a corruption guard.
            raise ResolutionError(
                "resolved contraption lost mandatory metadata.dynamics_completeness"
            )
        return result

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
            "controller": None if self.controller is None else self.controller.name,
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
            if attachment.kind != "revolute":
                continue
            assert attachment.coordinate is not None
            coordinate = attachment.coordinate
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
        if attachment.kind == "revolute"
        for binding in attachment.coordinate_bindings
    )
    for component in physical.components:
        package = physical.packages[component.package]
        result.update(
            f"{component.id}.{connector.kinematics.state}"
            for connector in package.connectors
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
    if attachment.kind != "revolute" or not attachment.coordinate_bindings:
        raise PhysicalSpecError(
            f"attachment {attachment.id!r} has no revolute coordinate bindings"
        )
    state_values: list[tuple[str, float, float]] = []
    for binding in attachment.coordinate_bindings:
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
    component_packages = {
        component.id: physical.packages[component.package]
        for component in physical.components
    }
    initial_coordinates: dict[str, float] = {}
    for attachment in physical.attachments:
        if attachment.kind != "revolute":
            continue
        assert attachment.coordinate is not None
        endpoint_components = {
            attachment.parent.component,
            attachment.child.component,
        }
        expected_bindings: set[str] = set()
        for endpoint in (attachment.parent, attachment.child):
            connector = component_packages[endpoint.component].connector_map[
                endpoint.connector
            ]
            if connector.joint_coordinate_state is not None:
                expected_bindings.add(
                    f"{endpoint.component}.{connector.joint_coordinate_state}"
                )
        declared_bindings = {
            binding.state for binding in attachment.coordinate_bindings
        }
        if declared_bindings != expected_bindings:
            raise ResolutionError(
                f"revolute attachment {attachment.id!r} coordinate-binding "
                f"coverage mismatch; missing={sorted(expected_bindings - declared_bindings)}, "
                f"extra={sorted(declared_bindings - expected_bindings)}"
            )
        initial_joint_angles: list[tuple[str, float, float]] = []
        for coordinate_binding in attachment.coordinate_bindings:
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
        attachment.coordinate_bindings[0].state
        for attachment in physical.attachments
        if attachment.kind == "revolute" and attachment.coordinate_bindings
    }
    for component in physical.components:
        package = physical.packages[component.package]
        for connector in package.connectors:
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
    packages: ComponentPackageRegistry,
) -> ResolvedComponent:
    context = f"contraption.components[{index}]"
    _keys(
        value,
        {"id", "package", "parameters", "condition"},
        context,
        {"id", "package"},
    )
    identifier = _text(value["id"], f"{context}.id")
    package_id = _text(value["package"], f"{context}.package")
    try:
        package = packages[package_id]
    except KeyError as exc:
        raise ResolutionError(
            f"component {identifier!r} references missing package {package_id!r}"
        ) from exc
    parameters = _mapping(value.get("parameters", {}), f"{context}.parameters")
    normalized_parameters = _canonical(parameters, f"{context}.parameters")
    return ResolvedComponent(
        identifier,
        package_id,
        package.model.id,
        _freeze(normalized_parameters),
        _text(value.get("condition", "unverified"), f"{context}.condition"),
    )


def _parse_connection(value: Mapping[str, Any], index: int) -> ResolvedConnection:
    context = f"contraption.connections[{index}]"
    kind = _text(value.get("kind"), f"{context}.kind")
    allowed = {"id", "kind", "endpoints", "domain", "metadata"}
    if kind == "attachment":
        allowed.add("joint")
    _keys(value, allowed, context, {"id", "kind", "endpoints"})
    endpoints = tuple(
        PortRef.from_dict(endpoint)
        for endpoint in _sequence(value["endpoints"], f"{context}.endpoints")
    )
    if len(endpoints) < 2:
        raise ResolutionError(f"{context} requires at least two endpoints")
    if kind == "attachment" and len(endpoints) != 2:
        raise ResolutionError(f"{context} attachment must be pairwise")
    domain = value.get("domain")
    if domain is not None:
        domain = _text(domain, f"{context}.domain")
    joint = _mapping(value.get("joint", {}), f"{context}.joint")
    metadata = _mapping(value.get("metadata", {}), f"{context}.metadata")
    if metadata:
        raise ResolutionError(
            f"{context}.metadata must be empty; connection semantics require "
            "typed fields rather than opaque metadata"
        )
    return ResolvedConnection(
        _text(value["id"], f"{context}.id"),
        kind,
        endpoints,
        domain,
        _freeze(_canonical(joint, f"{context}.joint")),
        _freeze(_canonical(metadata, f"{context}.metadata")),
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


def _validate_package_parameter_bindings(
    package: ComponentPackageSpec, model: ModelSpec
) -> None:
    parameters = {parameter.name: parameter for parameter in model.parameters}
    for binding in package.parameter_bindings:
        try:
            parameter = parameters[binding.model_parameter]
        except KeyError as exc:
            raise ResolutionError(
                f"package {package.id!r} physical binding references missing PMDL "
                f"parameter {binding.model_parameter!r}"
            ) from exc
        measured = package.measure_parameter(binding)
        default = _convert_unit_value(
            parameter.default,
            parameter.unit,
            binding.unit,
            f"package {package.id!r} parameter {parameter.name!r}",
        )
        error = abs(default - measured)
        if error > binding.absolute_tolerance:
            raise ResolutionError(
                f"package {package.id!r} PMDL parameter {parameter.name!r} default="
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
                f"package {package.id!r} geometry-bound parameter "
                f"{parameter.name!r} may not declare independent PMDL uncertainty; "
                "sample-specific geometry is not represented"
            )


def _validate_component_parameter_bindings(
    component: ResolvedComponent,
    package: ComponentPackageSpec,
    model: ModelSpec,
) -> None:
    parameters = {parameter.name: parameter for parameter in model.parameters}
    for binding in package.parameter_bindings:
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
        measured = package.measure_parameter(binding)
        error = abs(value - measured)
        if error > binding.absolute_tolerance:
            raise ResolutionError(
                f"component {component.id!r} parameter {parameter.name!r}="
                f"{value:.17g} {binding.unit} disagrees with package physical "
                f"measure={measured:.17g} {binding.unit}; absolute_error="
                f"{error:.17g}, tolerance={binding.absolute_tolerance:.17g}"
            )


def _verify_package_model(
    package: ComponentPackageSpec,
    model_registry: Mapping[str, ModelSpec],
) -> ModelSpec:
    try:
        model = model_registry[package.model.id]
    except KeyError as exc:
        raise ResolutionError(
            f"package {package.id!r} references unregistered PMDL model {package.model.id!r}"
        ) from exc
    if not isinstance(model, ModelSpec):
        raise ResolutionError(
            f"registry entry {package.model.id!r} must be a parsed ModelSpec"
        )
    if model.id != package.model.id or model.version != package.model.version:
        raise ResolutionError(
            f"package {package.id!r} model identity/version mismatch: package="
            f"{package.model.id}@{package.model.version}, registry={model.id}@{model.version}"
        )
    digest = _model_digest(model)
    if digest != package.model.sha256:
        raise ResolutionError(
            f"package {package.id!r} PMDL content hash mismatch: expected "
            f"{package.model.sha256}, got {digest}; stale or substituted model refused"
        )
    report = validate_model(model)
    if not report.valid:
        details = "; ".join(
            f"[{issue.code}] {issue.path}: {issue.message}" for issue in report.errors
        )
        raise ResolutionError(f"package {package.id!r} PMDL is invalid: {details}")
    _validate_package_parameter_bindings(package, model)

    declared_power = {port.name: port for port in model.power_ports}
    declared_signal = {port.name: port for port in model.signal_ports}
    mapped = {
        connector.model_port: connector
        for connector in package.connectors
        if connector.model_port is not None
    }
    expected = set(declared_power) | set(declared_signal)
    if set(mapped) != expected:
        raise ResolutionError(
            f"package {package.id!r} connector/model-port coverage mismatch; "
            f"missing={sorted(expected - set(mapped))}, extra={sorted(set(mapped) - expected)}"
        )
    for port_name, connector in mapped.items():
        if port_name in declared_power:
            if connector.domain != declared_power[port_name].domain:
                raise ResolutionError(
                    f"package {package.id!r} connector {connector.id!r} domain "
                    f"{connector.domain!r} does not match PMDL power port {port_name!r} "
                    f"domain {declared_power[port_name].domain!r}"
                )
        elif connector.domain != "signal":
            raise ResolutionError(
                f"package {package.id!r} signal connector {connector.id!r} must use "
                f"domain 'signal', got {connector.domain!r}"
            )
    state_map = {state.name: state for state in model.states}
    for connector in package.connectors:
        if connector.joint_coordinate_state is None:
            continue
        if connector.model_port is None:
            raise ResolutionError(
                f"package {package.id!r} connector {connector.id!r} binds joint "
                "coordinate state but has model_port:null"
            )
        try:
            state = state_map[connector.joint_coordinate_state]
        except KeyError as exc:
            raise ResolutionError(
                f"package {package.id!r} connector {connector.id!r} binds missing "
                f"PMDL angle state {connector.joint_coordinate_state!r}"
            ) from exc
        _validate_state_unit(
            f"{package.id}.{state.name}", state.unit, "rad", angle=True
        )
    return model


def _package_registry(
    packages: ComponentPackageRegistry
    | Mapping[str, ComponentPackageSpec | Mapping[str, Any]]
    | Sequence[ComponentPackageSpec],
) -> ComponentPackageRegistry:
    if isinstance(packages, ComponentPackageRegistry):
        return packages
    if isinstance(packages, Mapping):
        values: list[ComponentPackageSpec] = []
        for key, value in packages.items():
            package = (
                value
                if isinstance(value, ComponentPackageSpec)
                else ComponentPackageSpec.from_dict(_mapping(value, f"packages.{key}"))
            )
            if key != package.id:
                raise ResolutionError(
                    f"package registry key {key!r} does not match id {package.id!r}"
                )
            values.append(package)
        return ComponentPackageRegistry(values)
    return ComponentPackageRegistry(packages)


def _controller_digest(program: ControlProgram) -> str:
    payload = json.dumps(
        _canonical(program.to_dict(), "control program"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _resolve_controller(
    raw_reference: Mapping[str, Any] | None,
    control_programs: Mapping[str, ControlProgram] | None,
    controls: Sequence[ControlBindingSpec],
) -> ControlProgram | None:
    if raw_reference is None:
        return None
    reference = _mapping(raw_reference, "contraption.controller")
    names = {"id", "version", "sha256", "output_bindings", "telemetry_outputs"}
    _keys(reference, names, "contraption.controller", names)
    identifier = _text(reference["id"], "contraption.controller.id")
    version = _text(reference["version"], "contraption.controller.version")
    expected_digest = _text(reference["sha256"], "contraption.controller.sha256")
    if _SHA256.fullmatch(expected_digest) is None:
        raise ResolutionError(
            "contraption.controller.sha256 must be 'sha256:' followed by 64 "
            "lowercase hexadecimal characters"
        )
    if control_programs is None:
        raise ResolutionError(
            f"contraption references controller {identifier!r}, but no parsed "
            "control-program registry was supplied"
        )
    try:
        program = control_programs[identifier]
    except KeyError as exc:
        raise ResolutionError(
            f"contraption references missing controller {identifier!r}"
        ) from exc
    if not isinstance(program, ControlProgram):
        raise ResolutionError(
            f"controller registry entry {identifier!r} must be a parsed ControlProgram"
        )
    if program.name != identifier or program.version != version:
        raise ResolutionError(
            f"controller identity/version mismatch: reference={identifier}@{version}, "
            f"registry={program.name}@{program.version}"
        )
    actual_digest = _controller_digest(program)
    if actual_digest != expected_digest:
        raise ResolutionError(
            f"controller {identifier!r} content hash mismatch: expected "
            f"{expected_digest}, got {actual_digest}; stale or substituted controller refused"
        )

    raw_bindings = _mapping(
        reference["output_bindings"], "contraption.controller.output_bindings"
    )
    output_bindings = {
        _text(name, "controller output binding name"): _text(
            target, f"controller output binding {name!r} target"
        )
        for name, target in raw_bindings.items()
    }
    telemetry_values = _sequence(
        reference["telemetry_outputs"], "contraption.controller.telemetry_outputs"
    )
    telemetry_outputs = tuple(
        _text(value, f"contraption.controller.telemetry_outputs[{index}]")
        for index, value in enumerate(telemetry_values)
    )
    if len(set(telemetry_outputs)) != len(telemetry_outputs):
        raise ResolutionError("contraption.controller.telemetry_outputs must be unique")
    overlap = sorted(set(output_bindings) & set(telemetry_outputs))
    if overlap:
        raise ResolutionError(
            "controller outputs may not be both actuator-bound and telemetry-only: "
            + ", ".join(overlap)
        )
    program_outputs = {output.name: output for output in program.outputs}
    covered_outputs = set(output_bindings) | set(telemetry_outputs)
    if covered_outputs != set(program_outputs):
        raise ResolutionError(
            f"controller output coverage mismatch; missing="
            f"{sorted(set(program_outputs) - covered_outputs)}, extra="
            f"{sorted(covered_outputs - set(program_outputs))}"
        )
    controls_by_source = {control.source: control for control in controls}
    if len(controls_by_source) != len(controls):
        raise ResolutionError("contraption control binding sources must be unique")
    bound_sources = list(output_bindings.values())
    if len(set(bound_sources)) != len(bound_sources):
        raise ResolutionError("controller actuator outputs must bind distinct control sources")
    if set(bound_sources) != set(controls_by_source):
        raise ResolutionError(
            f"controller/contraption control-source coverage mismatch; unbound="
            f"{sorted(set(controls_by_source) - set(bound_sources))}, unknown="
            f"{sorted(set(bound_sources) - set(controls_by_source))}"
        )
    for output_name, source in output_bindings.items():
        output = program_outputs[output_name]
        settings = controls_by_source[source].settings
        for setting_name, output_value in (
            ("default", output.default),
            ("minimum", output.minimum),
            ("maximum", output.maximum),
        ):
            if setting_name not in settings:
                raise ResolutionError(
                    f"control source {source!r} lacks {setting_name!r} required by "
                    f"controller output {output_name!r}"
                )
            setting_value = settings[setting_name]
            if output_value is None or setting_value != output_value:
                raise ResolutionError(
                    f"controller output {output_name!r} {setting_name}="
                    f"{output_value!r} disagrees with control source {source!r} "
                    f"setting={setting_value!r}"
                )
        setting_unit = _text(
            settings.get("unit"), f"control source {source!r} unit"
        )
        try:
            output_unit = parse_unit(output.unit)
            bound_unit = parse_unit(setting_unit)
        except UnitError as exc:
            raise ResolutionError(
                f"controller output {output_name!r} unit is invalid: {exc}"
            ) from exc
        if (
            output_unit.dimension != bound_unit.dimension
            or output_unit.scale != bound_unit.scale
        ):
            raise ResolutionError(
                f"controller output {output_name!r} unit {output.unit!r} disagrees "
                f"with control source {source!r} unit {setting_unit!r}"
            )
    return program


def resolve_assembly(
    specification: Mapping[str, Any] | Any,
    packages: ComponentPackageRegistry
    | Mapping[str, ComponentPackageSpec | Mapping[str, Any]]
    | Sequence[ComponentPackageSpec],
    model_registry: Mapping[str, ModelSpec],
    *,
    joint_coordinates: Mapping[str, float] | None = None,
    control_programs: Mapping[str, ControlProgram] | None = None,
) -> ResolvedAssembly:
    """Verify and compile a strict ``contraption-2`` assembly closure."""

    raw_value = specification.to_dict() if hasattr(specification, "to_dict") else specification
    raw = _mapping(raw_value, "contraption")
    allowed = {
        "format",
        "id",
        "name",
        "version",
        "physical_root",
        "components",
        "connections",
        "controls",
        "controller",
        "environment",
        "metadata",
    }
    required = {"format", "id", "name", "version", "physical_root", "components"}
    _keys(raw, allowed, "contraption", required)
    if raw["format"] != "contraption-2":
        raise ResolutionError(
            f"canonical assembly requires format 'contraption-2', got {raw['format']!r}"
        )
    canonical_source = _canonical(raw, "contraption")
    package_registry = _package_registry(packages)
    components = tuple(
        _parse_component(_mapping(value, f"contraption.components[{index}]"), index, package_registry)
        for index, value in enumerate(
            _sequence(raw["components"], "contraption.components")
        )
    )
    component_ids = [component.id for component in components]
    if len(set(component_ids)) != len(component_ids):
        raise ResolutionError("contraption component identifiers must be unique")
    connections = tuple(
        _parse_connection(_mapping(value, f"contraption.connections[{index}]"), index)
        for index, value in enumerate(
            _sequence(raw.get("connections", []), "contraption.connections")
        )
    )
    if len({connection.id for connection in connections}) != len(connections):
        raise ResolutionError("contraption connection identifiers must be unique")
    controls = tuple(
        ControlBindingSpec.from_dict(value)
        for value in _sequence(raw.get("controls", []), "contraption.controls")
    )
    environment = _mapping(raw.get("environment", {}), "contraption.environment")
    metadata = _mapping(raw.get("metadata", {}), "contraption.metadata")
    if _parse_dynamics_completeness(metadata) is None:
        raise ResolutionError(
            "contraption-2 requires metadata.dynamics_completeness so omitted "
            "physical interactions cannot be silent"
        )
    raw_controller_value = raw.get("controller")
    raw_controller = (
        None
        if raw_controller_value is None
        else _mapping(raw_controller_value, "contraption.controller")
    )
    controller = _resolve_controller(raw_controller, control_programs, controls)
    root = _mapping(raw["physical_root"], "contraption.physical_root")
    record = ResolvedContraptionRecord(
        "contraption-2",
        _text(raw["id"], "contraption.id"),
        _text(raw["name"], "contraption.name"),
        _text(raw["version"], "contraption.version"),
        components,
        connections,
        controls,
        _freeze(_canonical(environment, "contraption.environment")),
        _freeze(_canonical(metadata, "contraption.metadata")),
        _freeze(_canonical(root, "contraption.physical_root")),
        None
        if raw_controller is None
        else _freeze(_canonical(raw_controller, "contraption.controller")),
        _freeze(canonical_source),
    )

    used_package_ids = {component.package for component in components}
    unused_packages = sorted(set(package_registry) - used_package_ids)
    if unused_packages:
        # An explicit registry may be larger than one device.  Unused entries do
        # not enter the closure; this is informational, not a simulation omission.
        pass
    verified_models: dict[str, ModelSpec] = {}
    package_models: dict[str, ModelSpec] = {}
    for package_id in sorted(used_package_ids):
        package_models[package_id] = _verify_package_model(
            package_registry[package_id], model_registry
        )
    for component in components:
        verified_models[component.id] = package_models[component.package]
        _validate_component_parameter_bindings(
            component,
            package_registry[component.package],
            verified_models[component.id],
        )

    zero_coordinates = {
        connection.joint["coordinate"]: 0.0
        for connection in connections
        if connection.kind == "attachment"
        and connection.joint.get("kind") == "revolute"
    }
    try:
        physical = resolve_physical_assembly(
            canonical_source,
            package_registry,
            zero_coordinates,
        )
    except PhysicalSpecError as exc:
        raise ResolutionError(f"physical assembly resolution failed: {exc}") from exc

    initial_coordinates = _validate_physical_state_bindings(
        physical, verified_models
    )
    if joint_coordinates is not None:
        provided = dict(joint_coordinates)
        if set(provided) != set(initial_coordinates):
            raise ResolutionError(
                "initial joint_coordinates must exactly cover PMDL-backed revolute "
                f"states; missing={sorted(set(initial_coordinates) - set(provided))}, "
                f"extra={sorted(set(provided) - set(initial_coordinates))}"
            )
        for name, expected in initial_coordinates.items():
            actual = _number(provided[name], f"joint_coordinates.{name}")
            if abs(actual - expected) > 1e-9:
                raise ResolutionError(
                    f"initial joint coordinate {name!r}={actual:.17g} rad disagrees "
                    f"with PMDL state initial={expected:.17g} rad"
                )
    try:
        physical = physical.with_configuration(
            joint_coordinates=initial_coordinates
        )
    except PhysicalSpecError as exc:
        raise ResolutionError(
            f"initial physical configuration validation failed: {exc}"
        ) from exc

    connector_bindings: dict[str, str | None] = {}
    component_by_id = {component.id: component for component in components}
    referenced_connectors = {
        f"{endpoint.component}.{endpoint.port}"
        for connection in connections
        for endpoint in connection.endpoints
    } | {f"{control.target.component}.{control.target.port}" for control in controls}
    for key in sorted(referenced_connectors):
        component_id, connector_id = key.rsplit(".", 1)
        try:
            component = component_by_id[component_id]
            connector = package_registry[component.package].connector_map[connector_id]
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
        raise ResolutionError("internal assembly hash mismatch between physical and PMDL projections")
    return ResolvedAssembly(
        record,
        package_registry,
        FrozenDict(verified_models),
        FrozenDict(connector_bindings),
        controller,
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
    "ResolvedConnection",
    "ResolvedContraptionRecord",
    "resolve_assembly",
    "resolve_body_pose_frames",
]
