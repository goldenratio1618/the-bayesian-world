"""Build planning derived only from a canonical :class:`ResolvedAssembly`.

The build planner is a consumer of the same part closure used by the PMDL
simulator and physical resolver.  It does not parse a second contraption
schema, consult display metadata, or invent routing, fasteners, ratings, or
placements.  Connector and body poses are exact projections of the resolved
assembly; facts which are not present in that closure remain explicit release
gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..control import control_digest
from ..physics.physical import TransformSpec
from ..physics.resolved import ResolutionError, ResolvedAssembly


class BuildInstructionError(ValueError):
    """The canonical closure cannot produce a truthful build plan."""


def _pose_dict(pose: TransformSpec) -> dict[str, list[float]]:
    return {
        "translation_m": [float(value) for value in pose.translation_m],
        "rotation_quaternion_wxyz": [
            float(value) for value in pose.rotation_quaternion_wxyz
        ],
    }


def _distance(left: TransformSpec, right: TransformSpec) -> float:
    return math.sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left.translation_m, right.translation_m, strict=True)
        )
    )


def _minimum_spanning_length(poses: Sequence[TransformSpec]) -> float:
    """Return a geometric lower bound without choosing a physical wire route."""

    if len(poses) < 2:
        return 0.0
    visited = {0}
    total = 0.0
    while len(visited) < len(poses):
        candidate: tuple[float, int] | None = None
        for source in visited:
            for target in range(len(poses)):
                if target in visited:
                    continue
                edge = (_distance(poses[source], poses[target]), target)
                if candidate is None or edge < candidate:
                    candidate = edge
        if candidate is None:  # pragma: no cover - guarded by the loop invariant.
            raise BuildInstructionError("could not derive connector distance lower bound")
        total += candidate[0]
        visited.add(candidate[1])
    return total


@dataclass(frozen=True, slots=True)
class BOMItem:
    part_id: str
    model_id: str
    model_version: str
    quantity: int
    component_ids: tuple[str, ...]
    provenance_kind: str
    provenance_source: str
    provenance_reference: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlacementInstruction:
    body: str
    component_id: str
    part_id: str
    world_pose: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MechanicalInstruction:
    connection_id: str
    parent_connector: str
    child_connector: str
    joint_kind: str
    behavior_binding: str
    coordinate: str | None
    zero_angle_rad: float
    coordinate_bindings: tuple[Mapping[str, Any], ...]
    parent_world_pose: Mapping[str, Any]
    child_world_pose: Mapping[str, Any]
    connector_provenance: Mapping[str, Mapping[str, Any]]
    connector_joint_coordinate_states: Mapping[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WiringInstruction:
    connection_id: str
    kind: str
    domain: str
    endpoints: tuple[str, ...]
    connector_world_poses: Mapping[str, Mapping[str, Any]]
    connector_provenance: Mapping[str, Mapping[str, Any]]
    nonspatial_endpoints: tuple[str, ...]
    straight_line_lower_bound_m: float
    routed_length_m: None = None
    conductor_specification: None = None
    protection: None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelConnection:
    """A non-assembly PMDL connection retained in the build record."""

    connection_id: str
    kind: str
    domain: str
    endpoints: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AssemblyStep:
    number: int
    title: str
    instruction: str
    verification: str
    source_connection_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """Hash-bound, deterministic projection of one resolved assembly closure."""

    contraption_id: str
    assembly_sha256: str
    pmdl_sha256: str
    controllers: tuple[Mapping[str, str], ...]
    bill_of_materials: tuple[BOMItem, ...]
    placements: tuple[PlacementInstruction, ...]
    mechanical: tuple[MechanicalInstruction, ...]
    wiring: tuple[WiringInstruction, ...]
    model_connections: tuple[ModelConnection, ...]
    steps: tuple[AssemblyStep, ...]
    safety_notes: tuple[str, ...]
    unresolved: tuple[str, ...]
    schema: str = "contraption.build-plan/v3"

    def __post_init__(self) -> None:
        pattern = re.compile(r"sha256:[0-9a-f]{64}\Z")
        if pattern.fullmatch(self.assembly_sha256) is None:
            raise BuildInstructionError("build plan has an invalid assembly_sha256")
        if pattern.fullmatch(self.pmdl_sha256) is None:
            raise BuildInstructionError("build plan has an invalid pmdl_sha256")
        ids: set[str] = set()
        for controller in self.controllers:
            if set(controller) != {"id", "version", "sha256"}:
                raise BuildInstructionError(
                    "build plan controller provenance must contain exactly "
                    "id/version/sha256"
                )
            if not all(
                isinstance(controller[name], str) and controller[name]
                for name in ("id", "version")
            ):
                raise BuildInstructionError(
                    "build plan controller id/version must be non-empty strings"
                )
            if controller["id"] in ids:
                raise BuildInstructionError("build plan controller ids must be unique")
            ids.add(controller["id"])
            if pattern.fullmatch(str(controller["sha256"])) is None:
                raise BuildInstructionError(
                    "build plan controller provenance has an invalid sha256"
                )

    @property
    def bom(self) -> tuple[BOMItem, ...]:
        return self.bill_of_materials

    @property
    def build_ready(self) -> bool:
        return not self.unresolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contraption_id": self.contraption_id,
            "assembly_sha256": self.assembly_sha256,
            "pmdl_sha256": self.pmdl_sha256,
            "controllers": [dict(item) for item in self.controllers],
            "build_ready": self.build_ready,
            "bill_of_materials": [item.to_dict() for item in self.bill_of_materials],
            "placements": [item.to_dict() for item in self.placements],
            "mechanical": [item.to_dict() for item in self.mechanical],
            "wiring": [item.to_dict() for item in self.wiring],
            "model_connections": [item.to_dict() for item in self.model_connections],
            "steps": [item.to_dict() for item in self.steps],
            "safety_notes": list(self.safety_notes),
            "unresolved": list(self.unresolved),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=indent, ensure_ascii=False, allow_nan=False
        ) + "\n"

    def to_markdown(self) -> str:
        lines = [
            f"# Build plan: {self.contraption_id}",
            "",
            f"Assembly closure: `{self.assembly_sha256}`  ",
            f"PMDL closure: `{self.pmdl_sha256}`  ",
            "Controllers: "
            + (
                "none"
                if not self.controllers
                else ", ".join(
                    f"`{item['id']}@{item['version']}` (`{item['sha256']}`)"
                    for item in self.controllers
                )
            )
            + "  ",
            f"Build ready: **{'yes' if self.build_ready else 'no'}**",
            "",
            "This document is a projection of the canonical resolved assembly. "
            "It does not add geometry, routing, fasteners, or ratings.",
            "",
            "## Safety gate",
            "",
            *[f"- {note}" for note in self.safety_notes],
            "",
            "## Bill of materials",
            "",
            "| Part | Model | Qty | Components | Provenance |",
            "|---|---|---:|---|---|",
        ]
        for item in self.bill_of_materials:
            reference = (
                f" ({item.provenance_reference})" if item.provenance_reference else ""
            )
            lines.append(
                f"| {item.part_id} | {item.model_id}@{item.model_version} | "
                f"{item.quantity} | {', '.join(item.component_ids)} | "
                f"{item.provenance_kind}: {item.provenance_source}{reference} |"
            )
        lines.extend(["", "## Resolved body placement", ""])
        for item in self.placements:
            position = item.world_pose["translation_m"]
            quaternion = item.world_pose["rotation_quaternion_wxyz"]
            lines.append(
                f"- `{item.body}`: position `{position}` m, quaternion wxyz `{quaternion}`."
            )
        lines.extend(["", "## Mechanical assembly", ""])
        for item in self.mechanical:
            position = item.parent_world_pose["translation_m"]
            coordinate = f", coordinate `{item.coordinate}`" if item.coordinate else ""
            aliases = (
                ", bound states "
                + ", ".join(
                    f"`{binding['state']}`@{binding['joint_angle_at_state_zero_rad']} rad"
                    for binding in item.coordinate_bindings
                )
                if item.coordinate_bindings
                else ""
            )
            lines.append(
                f"- `{item.connection_id}`: {item.joint_kind} {item.parent_connector} → "
                f"{item.child_connector} at `{position}` m "
                f"({item.behavior_binding}{coordinate}{aliases})."
            )
        lines.extend(["", "## Wiring/net schedule", ""])
        for item in self.wiring:
            virtual = (
                f" Nonspatial model boundary: {', '.join(item.nonspatial_endpoints)}."
                if item.nonspatial_endpoints
                else ""
            )
            lines.append(
                f"- `{item.connection_id}` ({item.kind}/{item.domain}): "
                f"{', '.join(item.endpoints)}. Geometric lower bound "
                f"{item.straight_line_lower_bound_m:.6g} m; routed length unresolved."
                f"{virtual}"
            )
        lines.extend(["", "## Ordered procedure", ""])
        for step in self.steps:
            lines.extend(
                [
                    f"### {step.number}. {step.title}",
                    "",
                    step.instruction,
                    "",
                    f"Verification: {step.verification}",
                    "",
                ]
            )
        lines.extend(["## Unresolved before construction", ""])
        lines.extend([f"- {item}" for item in self.unresolved] or ["- None."])
        return "\n".join(lines).rstrip() + "\n"

    def write(self, destination: str | Path) -> Mapping[str, Path]:
        path = Path(destination)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".json":
                path.write_text(self.to_json(), encoding="utf-8")
            elif path.suffix.lower() in {".md", ".markdown"}:
                path.write_text(self.to_markdown(), encoding="utf-8")
            else:
                raise BuildInstructionError(
                    "build-plan output must be a directory, .json, or .md path"
                )
            return {path.name: path}
        path.mkdir(parents=True, exist_ok=True)
        markdown = path / "BUILD_INSTRUCTIONS.md"
        machine = path / "build-plan.json"
        markdown.write_text(self.to_markdown(), encoding="utf-8")
        machine.write_text(self.to_json(), encoding="utf-8")
        return {markdown.name: markdown, machine.name: machine}


def _require_resolved(value: Any) -> ResolvedAssembly:
    if not isinstance(value, ResolvedAssembly):
        raise BuildInstructionError(
            "build instructions require a ResolvedAssembly; raw contraption JSON and "
            "independent placement/build metadata are not accepted"
        )
    if value.system.assembly_sha256 != value.assembly_sha256:
        raise BuildInstructionError(
            "physical and PMDL assembly hashes differ; refusing a mixed-representation plan"
        )
    return value


def _connector_key(endpoint: Any) -> str:
    return f"{endpoint.component}.{endpoint.port}"


def _provenance_dict(value: Any) -> dict[str, Any]:
    return {
        "kind": value.kind,
        "source": value.source,
        "reference": value.reference,
    }


def _controller_identities(
    assembly: ResolvedAssembly,
) -> tuple[dict[str, str], ...]:
    """Project immutable controller identities into a build artifact.

    Wiring remains in the canonical assembly hash and is not copied into the
    build plan as an alternate controller representation.
    """

    declared = {item.id: item for item in assembly.specification.controllers}
    if set(declared) != set(assembly.controllers):
        raise BuildInstructionError(
            "resolved controllers differ from canonical contraption links"
        )
    result: list[dict[str, str]] = []
    for controller_id, controller in sorted(assembly.controllers.items()):
        expected = declared[controller_id].program.sha256
        actual = control_digest(controller.spec)
        if actual != expected:
            raise BuildInstructionError(
                f"controller {controller_id!r} hash differs from its canonical link"
            )
        result.append(
            {
                "id": controller_id,
                "version": controller.spec.version,
                "sha256": actual,
            }
        )
    return tuple(result)


def _bom(assembly: ResolvedAssembly) -> tuple[BOMItem, ...]:
    groups: dict[str, list[str]] = {}
    for component in (
        *assembly.specification.components,
        *assembly.physical.world_objects,
    ):
        groups.setdefault(component.part, []).append(component.id)
    result: list[BOMItem] = []
    for part_id in sorted(groups):
        part = assembly.parts[part_id]
        result.append(
            BOMItem(
                part.id,
                part.model.id,
                part.model.version,
                len(groups[part_id]),
                tuple(sorted(groups[part_id])),
                part.provenance.kind,
                part.provenance.source,
                part.provenance.reference,
            )
        )
    return tuple(result)


def generate_build_instructions(
    assembly: ResolvedAssembly,
    output: str | Path | None = None,
) -> BuildPlan:
    """Generate a build record without introducing another object model."""

    resolved = _require_resolved(assembly)
    component_by_id = {
        component.id: component
        for component in (
            *resolved.specification.components,
            *resolved.physical.world_objects,
        )
    }
    placements: list[PlacementInstruction] = []
    for body_key in sorted(resolved.physical.body_poses):
        component_id, separator, _body_id = body_key.partition("/")
        if not separator or component_id not in component_by_id:
            raise BuildInstructionError(
                f"resolved body key {body_key!r} does not identify a canonical component"
            )
        placements.append(
            PlacementInstruction(
                body_key,
                component_id,
                component_by_id[component_id].part,
                _pose_dict(resolved.physical.body_poses[body_key]),
            )
        )

    mechanical: list[MechanicalInstruction] = []
    wiring: list[WiringInstruction] = []
    model_connections: list[ModelConnection] = []
    unresolved: set[str] = set()
    try:
        dynamics_completeness = resolved.dynamics_completeness
    except ResolutionError as exc:
        raise BuildInstructionError(
            "resolved assembly lacks a valid mandatory dynamics_completeness record"
        ) from exc
    if dynamics_completeness is None:
        raise BuildInstructionError(
            "resolved assembly lacks a valid mandatory dynamics_completeness record"
        )
    for gate in dynamics_completeness.open_gates:
        unresolved.add(
            f"dynamics completeness gate {gate.id}: {gate.reason}"
        )
    attachment_by_id = {
        attachment.id: attachment for attachment in resolved.physical.attachments
    }
    for connection in resolved.specification.connections:
        endpoint_keys = tuple(_connector_key(endpoint) for endpoint in connection.endpoints)
        if connection.kind == "attachment":
            try:
                attachment = attachment_by_id[connection.id]
                parent_pose = resolved.physical.connector_poses[endpoint_keys[0]]
                child_pose = resolved.physical.connector_poses[endpoint_keys[1]]
            except KeyError as exc:
                raise BuildInstructionError(
                    f"attachment {connection.id!r} is missing a resolved connector pose"
                ) from exc
            translation_error = _distance(parent_pose, child_pose)
            angular_error = parent_pose.angular_distance(child_pose)
            if translation_error > 1e-9 or angular_error > 1e-9:
                raise BuildInstructionError(
                    f"attachment {connection.id!r} connector frames are not coincident: "
                    f"translation_error_m={translation_error:.17g}, "
                    f"angular_error_rad={angular_error:.17g}"
                )
            mechanical.append(
                MechanicalInstruction(
                    connection.id,
                    endpoint_keys[0],
                    endpoint_keys[1],
                    attachment.joint.kind,
                    attachment.joint.behavior_binding,
                    attachment.joint.coordinate,
                    attachment.joint.zero_angle_rad,
                    tuple(
                        binding.to_dict()
                        for binding in attachment.joint.coordinate_bindings
                    ),
                    _pose_dict(parent_pose),
                    _pose_dict(child_pose),
                    {
                        endpoint_keys[0]: _provenance_dict(
                            resolved.parts[
                                component_by_id[connection.endpoints[0].component].part
                            ].connector_map[connection.endpoints[0].port].provenance
                        ),
                        endpoint_keys[1]: _provenance_dict(
                            resolved.parts[
                                component_by_id[connection.endpoints[1].component].part
                            ].connector_map[connection.endpoints[1].port].provenance
                        ),
                    },
                    {
                        endpoint_keys[0]: resolved.parts[
                            component_by_id[
                                connection.endpoints[0].component
                            ].part
                        ].connector_map[
                            connection.endpoints[0].port
                        ].joint_coordinate_state,
                        endpoint_keys[1]: resolved.parts[
                            component_by_id[
                                connection.endpoints[1].component
                            ].part
                        ].connector_map[
                            connection.endpoints[1].port
                        ].joint_coordinate_state,
                    },
                )
            )
            if attachment.joint.kind == "fixed":
                unresolved.add(
                    f"attachment {connection.id}: retention method, fastener specification, "
                    "quantity, locking method, and torque are not present in the canonical part closure"
                )
            else:
                unresolved.add(
                    f"attachment {connection.id}: bearing/shaft retention, clearance, and "
                    "mechanical travel limits are not present in the canonical part closure"
                )
            continue

        domain = connection.domain or "unspecified"
        if connection.kind in {"power", "signal"} and domain in {
            "electrical",
            "signal",
        }:
            spatial: list[tuple[str, TransformSpec]] = []
            nonspatial: list[str] = []
            provenance: dict[str, Mapping[str, Any]] = {}
            for endpoint, key in zip(connection.endpoints, endpoint_keys, strict=True):
                component = component_by_id[endpoint.component]
                part = resolved.parts[component.part]
                connector = part.connector_map[endpoint.port]
                provenance[key] = _provenance_dict(connector.provenance)
                pose = resolved.physical.connector_poses.get(key)
                if pose is not None:
                    spatial.append((key, pose))
                    continue
                if part.physical_role == "part" or connector.spatial:
                    raise BuildInstructionError(
                        f"connection {connection.id!r} endpoint {key!r} is a physical-part "
                        "connector but lacks a resolved pose"
                    )
                nonspatial.append(key)
            poses = [pose for _key, pose in spatial]
            wiring.append(
                WiringInstruction(
                    connection.id,
                    connection.kind,
                    domain,
                    endpoint_keys,
                    {key: _pose_dict(pose) for key, pose in spatial},
                    provenance,
                    tuple(nonspatial),
                    _minimum_spanning_length(poses),
                )
            )
            unresolved.add(
                f"connection {connection.id}: conductor type/gauge, insulation, protection, "
                "connector hardware, physical route, strain relief, and routed length are not "
                "present in the canonical part closure"
            )
            if len(endpoint_keys) > 2:
                unresolved.add(
                    f"connection {connection.id}: multi-endpoint net connectivity is known, "
                    "but physical branch topology and harness routing are not specified"
                )
        else:
            model_connections.append(
                ModelConnection(connection.id, connection.kind, domain, endpoint_keys)
            )

    for component in resolved.specification.components:
        if component.condition != "verified":
            unresolved.add(
                f"component {component.id}: condition is {component.condition!r}; inspect, "
                "identify, and qualify the exact physical instance before construction"
            )
        part = resolved.parts[component.part]
        if part.provenance.kind == "estimated":
            unresolved.add(
                f"part {part.id}: part geometry/provenance is estimated and must be "
                "replaced or independently verified before construction"
            )
        for body in part.bodies:
            for solid in body.solids:
                if solid.provenance.kind == "estimated":
                    unresolved.add(
                        f"part {part.id} solid {body.id}/{solid.id}: geometry is estimated"
                    )
        for connector in part.connectors:
            if connector.provenance.kind == "estimated":
                unresolved.add(
                    f"part {part.id} connector {connector.id}: pose is estimated"
                )

    steps: list[AssemblyStep] = []
    steps.append(
        AssemblyStep(
            1,
            "Verify exact closure",
            "Match every component part, PMDL model version/content hash, and physical "
            "instance against this build record. Stop on any substitution or damage.",
            f"Recorded closure equals {resolved.assembly_sha256} and PMDL closure equals "
            f"{resolved.system.pmdl_sha256}.",
        )
    )
    for item in mechanical:
        steps.append(
            AssemblyStep(
                len(steps) + 1,
                f"Assemble {item.connection_id}",
                f"Bring connector `{item.child_connector}` to the canonical frame of "
                f"`{item.parent_connector}` using the declared {item.joint_kind} joint. "
                "Do not choose retention hardware until its unresolved release gate is closed.",
                "Connector origins and orientations coincide, joint motion matches its "
                "declared coordinate bindings, and no undeclared constraint was introduced.",
                item.connection_id,
            )
        )
    for item in wiring:
        steps.append(
            AssemblyStep(
                len(steps) + 1,
                f"Implement net {item.connection_id}",
                f"Connect exactly these canonical endpoints: {', '.join(item.endpoints)}. "
                "The resolved geometry supplies connector locations only; complete and review "
                "the conductor/routing/protection release gate before energizing.",
                "Continuity, isolation, polarity/direction, strain relief, and protection are "
                "measured against an approved wiring record.",
                item.connection_id,
            )
        )
    steps.append(
        AssemblyStep(
            len(steps) + 1,
            "Release gate",
            "Close every unresolved item with measured, reviewed evidence; regenerate and "
            "hash the canonical part closure if physical geometry or connectors change.",
            "No unresolved item remains and independent low-energy checks pass before power-up.",
        )
    )

    plan = BuildPlan(
        resolved.specification.id,
        resolved.assembly_sha256,
        resolved.system.pmdl_sha256,
        _controller_identities(resolved),
        _bom(resolved),
        tuple(placements),
        tuple(mechanical),
        tuple(wiring),
        tuple(model_connections),
        tuple(steps),
        (
            "Simulation and visualization are engineering evidence, not authorization to build or energize hardware.",
            "Use independent over-current protection, emergency stop, mechanical travel limits, and guarded low-energy commissioning.",
            "Any physical change to a part body, connector, model, parameter, or topology requires a new resolved closure hash.",
        ),
        tuple(sorted(unresolved)),
    )
    if output is not None:
        plan.write(output)
    return plan


generate_build_plan = generate_build_instructions


__all__ = [
    "AssemblyStep",
    "BOMItem",
    "BuildInstructionError",
    "BuildPlan",
    "MechanicalInstruction",
    "ModelConnection",
    "PlacementInstruction",
    "WiringInstruction",
    "generate_build_instructions",
    "generate_build_plan",
]
