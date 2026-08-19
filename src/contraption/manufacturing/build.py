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
import hashlib
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
    static_part_id: str
    static_part_version: str
    static_part_sha256: str
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
class PurchaseBOMItem:
    procurement_record_id: str
    procurement_sha256: str
    manufacturer: str | None
    units_required: int
    identifiers: tuple[Mapping[str, Any], ...]
    provides: tuple[Mapping[str, Any], ...]
    documents: tuple[Mapping[str, Any], ...]
    offers: tuple[Mapping[str, Any], ...]

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
    endpoint_fabrication: Mapping[str, Mapping[str, Any] | None]
    implementation: Mapping[str, Any] | None

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
    endpoint_fabrication: Mapping[str, Mapping[str, Any] | None]
    implementation: Mapping[str, Any] | None
    routed_length_m: float | None = None
    conductor_specification: Mapping[str, Any] | None = None
    termination: Mapping[str, Any] | None = None
    protection: Mapping[str, Any] | None = None
    route: Mapping[str, Any] | None = None

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
    procurement_sha256: str
    controllers: tuple[Mapping[str, str], ...]
    bill_of_materials: tuple[BOMItem, ...]
    purchase_bill_of_materials: tuple[PurchaseBOMItem, ...]
    placements: tuple[PlacementInstruction, ...]
    mechanical: tuple[MechanicalInstruction, ...]
    wiring: tuple[WiringInstruction, ...]
    model_connections: tuple[ModelConnection, ...]
    steps: tuple[AssemblyStep, ...]
    safety_notes: tuple[str, ...]
    unresolved: tuple[str, ...]
    schema: str = "contraption.build-plan/v4"

    def __post_init__(self) -> None:
        pattern = re.compile(r"sha256:[0-9a-f]{64}\Z")
        if pattern.fullmatch(self.assembly_sha256) is None:
            raise BuildInstructionError("build plan has an invalid assembly_sha256")
        if pattern.fullmatch(self.pmdl_sha256) is None:
            raise BuildInstructionError("build plan has an invalid pmdl_sha256")
        if pattern.fullmatch(self.procurement_sha256) is None:
            raise BuildInstructionError("build plan has an invalid procurement_sha256")
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
            "procurement_sha256": self.procurement_sha256,
            "controllers": [dict(item) for item in self.controllers],
            "build_ready": self.build_ready,
            "bill_of_materials": [item.to_dict() for item in self.bill_of_materials],
            "purchase_bill_of_materials": [
                item.to_dict() for item in self.purchase_bill_of_materials
            ],
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
            f"Procurement closure: `{self.procurement_sha256}`  ",
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
            "| Part instance | Static part | Model | Qty | Components | Provenance |",
            "|---|---|---|---:|---|---|",
        ]
        for item in self.bill_of_materials:
            reference = (
                f" ({item.provenance_reference})" if item.provenance_reference else ""
            )
            lines.append(
                f"| {item.part_id} | {item.static_part_id}@{item.static_part_version} | "
                f"{item.model_id}@{item.model_version} | "
                f"{item.quantity} | {', '.join(item.component_ids)} | "
                f"{item.provenance_kind}: {item.provenance_source}{reference} |"
            )
        lines.extend(["", "## Purchase bill of materials", ""])
        if not self.purchase_bill_of_materials:
            lines.append("- No exact procurement records were selected.")
        for item in self.purchase_bill_of_materials:
            identifiers = ", ".join(
                f"{value['scheme']}={value['value']}" for value in item.identifiers
            )
            lines.append(
                f"- `{item.procurement_record_id}` × {item.units_required}: "
                f"{item.manufacturer or 'manufacturer unspecified'}; {identifiers}; "
                f"record `{item.procurement_sha256}`."
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
                f"({item.behavior_binding}{coordinate}{aliases}); fabrication "
                f"{'specified' if item.implementation is not None else 'missing'}."
            )
        lines.extend(["", "## Wiring/net schedule", ""])
        for item in self.wiring:
            virtual = (
                f" Nonspatial model boundary: {', '.join(item.nonspatial_endpoints)}."
                if item.nonspatial_endpoints
                else ""
            )
            routed = (
                "unresolved"
                if item.routed_length_m is None
                else f"{item.routed_length_m:.6g} m"
            )
            lines.append(
                f"- `{item.connection_id}` ({item.kind}/{item.domain}): "
                f"{', '.join(item.endpoints)}. Geometric lower bound "
                f"{item.straight_line_lower_bound_m:.6g} m; routed length {routed}."
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
        try:
            static = assembly.instantiations[part_id].static
        except KeyError as exc:
            raise BuildInstructionError(
                f"resolved part {part_id!r} has no canonical model instantiation"
            ) from exc
        result.append(
            BOMItem(
                part_id=part.id,
                static_part_id=static.id,
                static_part_version=static.version,
                static_part_sha256=static.sha256,
                model_id=part.model.id,
                model_version=part.model.version,
                quantity=len(groups[part_id]),
                component_ids=tuple(sorted(groups[part_id])),
                provenance_kind=part.provenance.kind,
                provenance_source=part.provenance.source,
                provenance_reference=part.provenance.reference,
            )
        )
    return tuple(result)


def _procurement_projection(
    assembly: ResolvedAssembly,
    physical_bom: Sequence[BOMItem],
) -> tuple[str, tuple[PurchaseBOMItem, ...], tuple[str, ...]]:
    """Select only unambiguous records and hash every relevant candidate.

    A record may provide several static parts, so selected purchase quantities
    are the maximum number of record units needed to cover each provision.
    Ambiguous candidates remain release gates rather than being ranked by an
    undocumented preference.
    """

    registry = assembly.instantiations.procurement
    required: dict[str, int] = {}
    procurement_required: set[str] = set()
    for item in physical_bom:
        part = assembly.parts[item.part_id]
        if part.physical_role != "part":
            continue
        required[item.static_part_id] = (
            required.get(item.static_part_id, 0) + item.quantity
        )
        if part.provenance.kind in {"catalog", "vendor"}:
            procurement_required.add(item.static_part_id)

    relevant: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    unresolved: set[str] = set()
    for static_part_id in sorted(required):
        candidates = registry.for_part(static_part_id)
        for record in candidates:
            relevant[record.id] = record
        if not candidates:
            if static_part_id in procurement_required:
                unresolved.add(
                    f"static part {static_part_id}: catalog/vendor provenance requires an "
                    "evidence-backed procurement record for this exact version and hash"
                )
            continue
        if len(candidates) > 1:
            if static_part_id in procurement_required:
                unresolved.add(
                    f"static part {static_part_id}: procurement is ambiguous across records "
                    + ", ".join(record.id for record in candidates)
                )
            continue
        selected[candidates[0].id] = candidates[0]

    purchase_bom: list[PurchaseBOMItem] = []
    for record_id in sorted(selected):
        record = selected[record_id]
        units = 1
        covered = False
        for provision in record.provides:
            quantity = required.get(provision.part)
            if quantity is None:
                continue
            covered = True
            units = max(units, math.ceil(quantity / provision.quantity))
        if not covered:  # pragma: no cover - selected through registry.for_part.
            raise BuildInstructionError(
                f"selected procurement record {record.id!r} covers no BOM part"
            )
        if record.lifecycle.status in {
            "not_recommended_for_new_design",
            "last_time_buy",
            "obsolete",
        }:
            unresolved.add(
                f"procurement record {record.id}: lifecycle status is "
                f"{record.lifecycle.status!r}"
            )
        if record.offers and all(
            offer.availability in {"out_of_stock", "discontinued"}
            for offer in record.offers
        ):
            unresolved.add(
                f"procurement record {record.id}: every evidenced offer is unavailable"
            )
        purchase_bom.append(
            PurchaseBOMItem(
                procurement_record_id=record.id,
                procurement_sha256=record.sha256,
                manufacturer=record.manufacturer,
                units_required=units,
                identifiers=tuple(item.to_dict() for item in record.identifiers),
                provides=tuple(item.to_dict() for item in record.provides),
                documents=tuple(item.to_dict() for item in record.documents),
                offers=tuple(item.to_dict() for item in record.offers),
            )
        )

    payload = [relevant[key].to_dict() for key in sorted(relevant)]
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest, tuple(purchase_bom), tuple(sorted(unresolved))


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
    physical_bom = _bom(resolved)
    procurement_sha256, purchase_bom, procurement_gates = _procurement_projection(
        resolved, physical_bom
    )
    unresolved: set[str] = set(procurement_gates)
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
        endpoint_fabrication: dict[str, Mapping[str, Any] | None] = {}
        physical_endpoint_present = False
        for endpoint, key in zip(connection.endpoints, endpoint_keys, strict=True):
            component = component_by_id[endpoint.component]
            part = resolved.parts[component.part]
            connector = part.connector_map[endpoint.port]
            endpoint_fabrication[key] = (
                None
                if connector.fabrication is None
                else connector.fabrication.to_dict()
            )
            if part.physical_role != "part":
                continue
            physical_endpoint_present = True
            if connector.fabrication is None:
                unresolved.add(
                    f"connection {connection.id} endpoint {key}: connector fabrication "
                    "record is missing"
                )
                continue
            missing_paths = tuple(
                dict.fromkeys(
                    (
                        *connector.fabrication.connector_missing_fields(),
                        *connector.fabrication.missing,
                    )
                )
            )
            for path in missing_paths:
                unresolved.add(
                    f"connection {connection.id} endpoint {key}: connector fabrication "
                    f"field {path!r} is missing"
                )
        implementation_payload = (
            None
            if connection.implementation is None
            else connection.implementation.to_dict()
        )
        if physical_endpoint_present and connection.kind in {
            "attachment",
            "power",
            "signal",
        }:
            if connection.implementation is None:
                unresolved.add(
                    f"connection {connection.id}: selected fabrication implementation "
                    "is missing"
                )
            else:
                for path in connection.implementation.implementation_missing_fields(
                    endpoint_count=len(connection.endpoints)
                ):
                    unresolved.add(
                        f"connection {connection.id}: fabrication implementation field "
                        f"{path!r} is missing"
                    )
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
                    endpoint_fabrication,
                    implementation_payload,
                )
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
                    endpoint_fabrication,
                    implementation_payload,
                    None
                    if connection.implementation is None
                    or connection.implementation.route is None
                    else connection.implementation.route.routed_length_m,
                    None
                    if connection.implementation is None
                    or connection.implementation.conductor is None
                    else connection.implementation.conductor.to_dict(),
                    None
                    if connection.implementation is None
                    or connection.implementation.termination is None
                    else connection.implementation.termination.to_dict(),
                    None
                    if connection.implementation is None
                    or connection.implementation.protection is None
                    else connection.implementation.protection.to_dict(),
                    None
                    if connection.implementation is None
                    or connection.implementation.route is None
                    else connection.implementation.route.to_dict(),
                )
            )
        else:
            model_connections.append(
                ModelConnection(connection.id, connection.kind, domain, endpoint_keys)
            )

    physical_instances = tuple(
        (component, "component", component.condition)
        for component in resolved.specification.components
    ) + tuple(
        (
            world_object,
            "world object",
            resolved.instantiations[
                world_object.part
            ].model_instance.condition,
        )
        for world_object in resolved.physical.world_objects
    )
    for component, instance_kind, condition in physical_instances:
        if condition not in {"inspected", "calibrated"}:
            unresolved.add(
                f"{instance_kind} {component.id}: condition is {condition!r}; inspect, "
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
        implementation_instruction = (
            " Follow the typed fabrication implementation recorded in this step."
            if item.implementation is not None
            else " Do not choose retention hardware until its release gate is closed."
        )
        steps.append(
            AssemblyStep(
                len(steps) + 1,
                f"Assemble {item.connection_id}",
                f"Bring connector `{item.child_connector}` to the canonical frame of "
                f"`{item.parent_connector}` using the declared {item.joint_kind} joint."
                + implementation_instruction,
                "Connector origins and orientations coincide, joint motion matches its "
                "declared coordinate bindings, and no undeclared constraint was introduced.",
                item.connection_id,
            )
        )
    for item in wiring:
        implementation_instruction = (
            " Follow the typed conductor, termination, protection, and route records."
            if item.implementation is not None
            else " Complete and review the fabrication release gate before energizing."
        )
        steps.append(
            AssemblyStep(
                len(steps) + 1,
                f"Implement net {item.connection_id}",
                f"Connect exactly these canonical endpoints: {', '.join(item.endpoints)}. "
                "The resolved geometry supplies connector locations."
                + implementation_instruction,
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
        contraption_id=resolved.specification.id,
        assembly_sha256=resolved.assembly_sha256,
        pmdl_sha256=resolved.system.pmdl_sha256,
        procurement_sha256=procurement_sha256,
        controllers=_controller_identities(resolved),
        bill_of_materials=physical_bom,
        purchase_bill_of_materials=purchase_bom,
        placements=tuple(placements),
        mechanical=tuple(mechanical),
        wiring=tuple(wiring),
        model_connections=tuple(model_connections),
        steps=tuple(steps),
        safety_notes=(
            "Simulation and visualization are engineering evidence, not authorization to build or energize hardware.",
            "Use independent over-current protection, emergency stop, mechanical travel limits, and guarded low-energy commissioning.",
            "Any physical change to a part body, connector, model, parameter, or topology requires a new resolved closure hash.",
        ),
        unresolved=tuple(sorted(unresolved)),
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
    "PurchaseBOMItem",
    "WiringInstruction",
    "generate_build_instructions",
    "generate_build_plan",
]
