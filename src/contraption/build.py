"""Deterministic, code-only build-instruction generation.

The generator translates component, connection, geometry, and purchasing
metadata into an auditable bill of materials and ordered assembly checklist.
It never calls a language model and it does not invent missing dimensions,
wire ratings, torque values, or fasteners: gaps are emitted as explicit
``unresolved`` items that must be closed before a real build.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class BuildInstructionError(ValueError):
    """Raised for malformed data that cannot produce deterministic steps."""


def _mapping(value: Any, label: str = "value") -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return result
    if is_dataclass(value):
        result = asdict(value)
        if isinstance(result, Mapping):
            return result
    raise BuildInstructionError(f"{label} must be an object or expose to_dict()")


def _objects(value: Any, label: str) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        result: list[Mapping[str, Any]] = []
        for key in sorted(value, key=str):
            item = dict(_mapping(value[key], f"{label}.{key}"))
            item.setdefault("id", str(key))
            result.append(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_mapping(item, label) for item in value]
    raise BuildInstructionError(f"{label} must be an array or object")


def _text(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean_number(value: Any) -> float | None:
    direct = _number(value)
    if direct is not None:
        return direct
    if isinstance(value, Mapping):
        for key in ("value", "mean", "nominal"):
            result = _number(value.get(key))
            if result is not None:
                return result
    return None


def _model_label(component: Mapping[str, Any]) -> str:
    model = component.get("model", component.get("model_id", component.get("category")))
    if isinstance(model, Mapping):
        for key in ("id", "name", "model", "category", "taxonomy_path"):
            if key in model:
                value = model[key]
                return "/".join(map(str, value)) if isinstance(value, Sequence) and not isinstance(value, str) else str(value)
        return json.dumps(model, sort_keys=True, separators=(",", ":"))
    return _text(model, "unspecified component")


def _metadata(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("metadata", {})
    return value if isinstance(value, Mapping) else {}


def _component_id(component: Mapping[str, Any], index: int = 0) -> str:
    value = component.get("id", component.get("name"))
    return _text(value, f"component_{index}")


def _endpoint(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        component, separator, port = value.partition(".")
        return component, port if separator else "unspecified"
    if isinstance(value, Mapping):
        return (
            _text(value.get("component", value.get("component_id")), "unspecified"),
            _text(value.get("port", value.get("port_name")), "unspecified"),
        )
    return "unspecified", "unspecified"


def _endpoints(connection: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    raw = connection.get("endpoints")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(_endpoint(value) for value in raw)
    for first, second in (("from", "to"), ("source", "target"), ("a", "b")):
        if first in connection or second in connection:
            return (_endpoint(connection.get(first)), _endpoint(connection.get(second)))
    return ()


def _position(component: Mapping[str, Any]) -> tuple[float, float, float] | None:
    candidates: list[Any] = []
    geometry = component.get("geometry")
    metadata = _metadata(component)
    parameters = component.get("parameters", {})
    if isinstance(geometry, Mapping):
        candidates.extend((geometry.get("position"), geometry.get("translation")))
        geometry_metadata = geometry.get("metadata", {})
        if isinstance(geometry_metadata, Mapping):
            candidates.extend(
                (
                    geometry_metadata.get("translation_m"),
                    geometry_metadata.get("position"),
                )
            )
    if isinstance(metadata, Mapping):
        candidates.extend((metadata.get("position"), metadata.get("placement")))
    if isinstance(parameters, Mapping):
        candidates.append(parameters.get("position"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            candidate = [candidate.get(axis) for axis in ("x", "y", "z")]
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)) and len(candidate) == 3:
            numbers = tuple(_mean_number(value) for value in candidate)
            if all(value is not None for value in numbers):
                return numbers[0], numbers[1], numbers[2]  # type: ignore[return-value]
    return None


def _round_wire_length(value: float) -> float:
    # 20% routing allowance plus a 10 cm service loop, rounded up to 5 cm.
    return math.ceil((1.2 * value + 0.1) / 0.05 - 1e-12) * 0.05


@dataclass(frozen=True)
class BOMItem:
    item: str
    quantity: float
    unit: str
    component_ids: tuple[str, ...] = ()
    manufacturer: str = ""
    part_number: str = ""
    supplier: str = ""
    purchase_url: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WiringInstruction:
    connection_id: str
    source: str
    destination: str
    signal: str
    wire_specification: str
    color: str
    length_m: float | None
    protection: str
    verification: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FastenerRequirement:
    connection_id: str
    fastener: str
    quantity: int | None
    material: str
    torque_nm: float | None
    locking_method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssemblyStep:
    number: int
    title: str
    instruction: str
    verification: str
    dependencies: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BuildPlan:
    """Complete deterministic result of build-plan generation."""

    contraption_id: str
    bill_of_materials: tuple[BOMItem, ...]
    steps: tuple[AssemblyStep, ...]
    wiring: tuple[WiringInstruction, ...]
    fasteners: tuple[FastenerRequirement, ...]
    safety_notes: tuple[str, ...]
    unresolved: tuple[str, ...]
    assumptions: tuple[str, ...]
    schema: str = "contraption.build-plan/v1"

    @property
    def bom(self) -> tuple[BOMItem, ...]:
        return self.bill_of_materials

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contraption_id": self.contraption_id,
            "bill_of_materials": [item.to_dict() for item in self.bill_of_materials],
            "steps": [item.to_dict() for item in self.steps],
            "wiring": [item.to_dict() for item in self.wiring],
            "fasteners": [item.to_dict() for item in self.fasteners],
            "safety_notes": list(self.safety_notes),
            "unresolved": list(self.unresolved),
            "assumptions": list(self.assumptions),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent) + "\n"

    def to_markdown(self) -> str:
        def cell(value: Any) -> str:
            return _text(value, "—").replace("|", "\\|").replace("\n", " ") or "—"

        lines = [f"# Build plan: {self.contraption_id}", "", f"Schema: `{self.schema}`", ""]
        lines.extend(
            [
                "## Safety gate",
                "",
                *[f"- {note}" for note in self.safety_notes],
                "",
                "## Bill of materials",
                "",
                "| Item | Qty | Unit | Part number | Components | Notes |",
                "|---|---:|---|---|---|---|",
            ]
        )
        for item in self.bill_of_materials:
            lines.append(
                f"| {cell(item.item)} | {item.quantity:g} | {cell(item.unit)} | "
                f"{cell(item.part_number)} | {cell(', '.join(item.component_ids))} | {cell(item.notes)} |"
            )
        lines.extend(["", "## Fasteners", ""])
        if self.fasteners:
            lines.extend(
                [
                    "| Connection | Fastener | Qty | Material | Torque (N·m) | Locking |",
                    "|---|---|---:|---|---:|---|",
                ]
            )
            for item in self.fasteners:
                lines.append(
                    f"| {cell(item.connection_id)} | {cell(item.fastener)} | "
                    f"{cell(item.quantity)} | {cell(item.material)} | {cell(item.torque_nm)} | "
                    f"{cell(item.locking_method)} |"
                )
        else:
            lines.append("No fasteners were declared.")
        lines.extend(["", "## Wiring schedule", ""])
        if self.wiring:
            lines.extend(
                [
                    "| ID | From | To | Signal | Wire | Color | Length (m) | Protection |",
                    "|---|---|---|---|---|---|---:|---|",
                ]
            )
            for item in self.wiring:
                lines.append(
                    f"| {cell(item.connection_id)} | {cell(item.source)} | "
                    f"{cell(item.destination)} | {cell(item.signal)} | "
                    f"{cell(item.wire_specification)} | {cell(item.color)} | "
                    f"{cell(item.length_m)} | {cell(item.protection)} |"
                )
        else:
            lines.append("No electrical or signal wiring was declared.")
        lines.extend(["", "## Assembly procedure", ""])
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
        lines.extend(["", "## Declared assumptions", ""])
        lines.extend([f"- {item}" for item in self.assumptions] or ["- None."])
        return "\n".join(lines).rstrip() + "\n"

    def write(self, destination: str | Path) -> Mapping[str, Path]:
        path = Path(destination)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".json":
                path.write_text(self.to_json(), encoding="utf-8")
            else:
                path.write_text(self.to_markdown(), encoding="utf-8")
            return {path.name: path}
        path.mkdir(parents=True, exist_ok=True)
        markdown = path / "BUILD_INSTRUCTIONS.md"
        machine = path / "build-plan.json"
        markdown.write_text(self.to_markdown(), encoding="utf-8")
        machine.write_text(self.to_json(), encoding="utf-8")
        return {markdown.name: markdown, machine.name: machine}


def _connection_kind(connection: Mapping[str, Any]) -> str:
    kind = _text(connection.get("kind", connection.get("domain")), "unknown").lower()
    metadata = _metadata(connection)
    domain = _text(metadata.get("domain"), "").lower()
    if kind in {"power", "electrical", "wire"} or domain == "electrical":
        return "power"
    if kind in {"signal", "measurement", "command", "data"}:
        return "signal"
    if kind in {"attachment", "mechanical", "fastener", "mount"} or domain == "mechanical":
        return "attachment"
    return kind


def _component_bom(components: Sequence[Mapping[str, Any]]) -> tuple[BOMItem, ...]:
    groups: dict[tuple[str, str, str, str, str], list[str]] = {}
    notes: dict[tuple[str, str, str, str, str], set[str]] = {}
    for index, component in enumerate(components):
        identifier = _component_id(component, index)
        metadata = _metadata(component)
        purchasing = component.get("purchasing", metadata.get("purchasing", {}))
        purchasing = purchasing if isinstance(purchasing, Mapping) else {}
        label = _text(
            purchasing.get("name", metadata.get("display_name")), _model_label(component)
        )
        manufacturer = _text(purchasing.get("manufacturer", metadata.get("manufacturer")))
        part_number = _text(
            purchasing.get("part_number", metadata.get("part_number", metadata.get("sku")))
        )
        supplier = _text(purchasing.get("supplier", metadata.get("supplier")))
        url = _text(purchasing.get("url", purchasing.get("purchase_url", metadata.get("url"))))
        key = label, manufacturer, part_number, supplier, url
        groups.setdefault(key, []).append(identifier)
        condition = _text(component.get("condition", metadata.get("condition")))
        if condition:
            notes.setdefault(key, set()).add(f"condition: {condition}")
    result = []
    for key in sorted(groups):
        label, manufacturer, part_number, supplier, url = key
        ids = tuple(sorted(groups[key]))
        result.append(
            BOMItem(
                label,
                float(len(ids)),
                "each",
                ids,
                manufacturer,
                part_number,
                supplier,
                url,
                "; ".join(sorted(notes.get(key, set()))),
            )
        )
    return tuple(result)


def _fastener(connection: Mapping[str, Any], connection_id: str) -> FastenerRequirement | None:
    metadata = _metadata(connection)
    raw = connection.get(
        "fastener", metadata.get("fastener", metadata.get("fasteners"))
    )
    if isinstance(raw, str):
        raw = {"type": raw}
    if not isinstance(raw, Mapping):
        flattened = {
            "type": metadata.get("fastener_type", metadata.get("thread")),
            "size": metadata.get("fastener_size"),
            "quantity": metadata.get("fastener_quantity"),
            "material": metadata.get("fastener_material"),
            "torque_nm": metadata.get("torque_nm"),
            "locking_method": metadata.get("locking_method"),
        }
        raw = flattened if any(value is not None for value in flattened.values()) else None
    if raw is None:
        return None
    fastener_type = _text(raw.get("type", raw.get("name")), "unspecified fastener")
    size = _text(raw.get("size", raw.get("thread")))
    quantity_value = raw.get("quantity")
    quantity = quantity_value if isinstance(quantity_value, int) and not isinstance(quantity_value, bool) and quantity_value > 0 else None
    torque = _number(raw.get("torque_nm", raw.get("torque")))
    return FastenerRequirement(
        connection_id,
        " ".join(value for value in (fastener_type, size) if value),
        quantity,
        _text(raw.get("material"), "unspecified"),
        torque,
        _text(raw.get("locking_method", raw.get("locking")), "unspecified"),
    )


def generate_build_instructions(
    specification: Mapping[str, Any] | Any,
    output: str | Path | None = None,
) -> BuildPlan:
    """Generate an auditable build plan from a spec or ``to_dict`` object."""

    spec = _mapping(specification, "contraption specification")
    components = _objects(spec.get("components"), "components")
    connections = _objects(spec.get("connections"), "connections")
    if not components:
        raise BuildInstructionError("a build plan requires at least one component")
    identifiers = [_component_id(component, index) for index, component in enumerate(components)]
    if len(set(identifiers)) != len(identifiers):
        raise BuildInstructionError("component identifiers must be unique")
    component_by_id = dict(zip(identifiers, components))
    contraption_id = _text(spec.get("id", spec.get("name")), "contraption")
    unresolved: set[str] = set()
    assumptions: set[str] = {
        "Estimated wire lengths use straight-line placement distance, 20% routing allowance, and a 0.10 m service loop.",
        "Assembly is de-energized until the electrical verification step passes.",
    }

    wiring: list[WiringInstruction] = []
    fasteners: list[FastenerRequirement] = []
    mechanical_connections: list[tuple[str, tuple[tuple[str, str], ...], Mapping[str, Any]]] = []
    connection_ids: set[str] = set()
    for index, connection in enumerate(connections):
        connection_id = _text(connection.get("id", connection.get("name")), f"connection_{index}")
        if connection_id in connection_ids:
            raise BuildInstructionError(f"duplicate connection id {connection_id!r}")
        connection_ids.add(connection_id)
        endpoints = _endpoints(connection)
        if len(endpoints) < 2:
            raise BuildInstructionError(
                f"connection {connection_id!r} requires at least two endpoints; "
                "no build instructions were generated"
            )
        unknown_components = sorted(
            {component for component, _port in endpoints if component not in component_by_id}
        )
        if unknown_components:
            raise BuildInstructionError(
                f"connection {connection_id!r} references component(s) absent from "
                f"the BOM: {unknown_components}"
            )
        kind = _connection_kind(connection)
        metadata = _metadata(connection)
        if kind not in {"power", "signal", "attachment"}:
            raise BuildInstructionError(
                f"connection {connection_id!r} has unsupported build kind {kind!r}"
            )
        raw_legs = metadata.get("legs", [])
        raw_legs = raw_legs if isinstance(raw_legs, Sequence) and not isinstance(raw_legs, (str, bytes)) else []
        if len(endpoints) > 2:
            assumptions.add(
                f"Hyperedge {connection_id} is expanded from its first endpoint to each later endpoint; shared metadata applies per leg unless metadata.legs overrides it."
            )
        for endpoint_index, destination_endpoint in enumerate(endpoints[1:], start=1):
            leg_id = connection_id if len(endpoints) == 2 else f"{connection_id}:{endpoint_index}"
            leg_metadata = dict(metadata)
            if endpoint_index - 1 < len(raw_legs) and isinstance(raw_legs[endpoint_index - 1], Mapping):
                leg_metadata.update(raw_legs[endpoint_index - 1])
            leg_connection = dict(connection)
            leg_connection["metadata"] = leg_metadata
            leg_endpoints = (endpoints[0], destination_endpoint)
            if kind in {"power", "signal"}:
                source = f"{leg_endpoints[0][0]}.{leg_endpoints[0][1]}"
                destination = f"{leg_endpoints[1][0]}.{leg_endpoints[1][1]}"
                length = _mean_number(
                    leg_metadata.get("wire_length_m", connection.get("length_m"))
                )
                if length is None:
                    first = _position(component_by_id.get(leg_endpoints[0][0], {}))
                    second = _position(component_by_id.get(leg_endpoints[1][0], {}))
                    if first is not None and second is not None:
                        length = _round_wire_length(
                            math.sqrt(
                                sum((left - right) ** 2 for left, right in zip(first, second))
                            )
                        )
                    else:
                        unresolved.add(
                            f"Connection {leg_id}: declare wire_length_m or both component positions."
                        )
                gauge = _text(
                    leg_metadata.get(
                        "wire_specification",
                        leg_metadata.get("wire_gauge", leg_metadata.get("awg")),
                    ),
                    "unspecified gauge",
                )
                if gauge != "unspecified gauge" and gauge.isdigit():
                    gauge = f"{gauge} AWG"
                if gauge == "unspecified gauge":
                    unresolved.add(
                        f"Connection {leg_id}: select wire gauge from worst-case current, voltage drop, and insulation rating."
                    )
                wiring.append(
                    WiringInstruction(
                        leg_id,
                        source,
                        destination,
                        _text(leg_metadata.get("signal", connection.get("signal")), kind),
                        gauge,
                        _text(leg_metadata.get("color"), "unassigned"),
                        length,
                        _text(
                            leg_metadata.get("protection", leg_metadata.get("fuse")),
                            "verify source protection",
                        ),
                        _text(
                            leg_metadata.get("verification"),
                            "Continuity/polarity check; verify no short to chassis before energizing.",
                        ),
                    )
                )
            else:
                mechanical_connections.append((leg_id, leg_endpoints, leg_connection))
                requirement = _fastener(leg_connection, leg_id)
                if requirement is None:
                    unresolved.add(
                        f"Connection {leg_id}: declare the attachment method/fastener, quantity, and rated torque."
                    )
                else:
                    fasteners.append(requirement)
                    if requirement.quantity is None:
                        unresolved.add(f"Connection {leg_id}: declare fastener quantity.")
                    if requirement.torque_nm is None and "adhesive" not in requirement.fastener.lower():
                        unresolved.add(f"Connection {leg_id}: obtain manufacturer torque specification.")

    bom = list(_component_bom(components))
    fastener_groups: dict[tuple[str, str], int] = {}
    fastener_connections: dict[tuple[str, str], list[str]] = {}
    for item in fasteners:
        if item.quantity is not None:
            key = item.fastener, item.material
            fastener_groups[key] = fastener_groups.get(key, 0) + item.quantity
            fastener_connections.setdefault(key, []).append(item.connection_id)
    for key in sorted(fastener_groups):
        name, material = key
        bom.append(
            BOMItem(
                name,
                float(fastener_groups[key]),
                "each",
                tuple(sorted(fastener_connections[key])),
                notes=f"material: {material}",
            )
        )
    for item in wiring:
        if item.length_m is not None:
            bom.append(
                BOMItem(
                    f"{item.wire_specification} wire ({item.color})",
                    item.length_m,
                    "m",
                    (item.connection_id,),
                    notes="Cut only after confirming routed length.",
                )
            )
    bom.sort(key=lambda item: (item.item.lower(), item.part_number, item.component_ids))
    wiring.sort(key=lambda item: (0 if item.signal == "power" else 1, item.connection_id))
    fasteners.sort(key=lambda item: item.connection_id)
    mechanical_connections.sort(key=lambda item: item[0])

    safety: set[str] = {
        "This generated plan is not a structural, electrical, fire, or regulatory certification; validate ratings and local requirements before construction.",
        "Wear eye protection during fabrication and keep the workpiece clamped; do not hold parts by hand while drilling or cutting.",
    }
    if wiring:
        safety.add("Disconnect and physically isolate every power source while wiring; fuse each source close to its positive terminal.")
        safety.add("Perform continuity, polarity, insulation, and chassis-short checks with current-limited power before full-power testing.")
    if mechanical_connections:
        safety.add("Guard pinch/crush points and support moving assemblies before loosening an attachment.")
    for component in components:
        label = f"{_model_label(component)} {_text(component.get('category'))}".lower()
        metadata = _metadata(component)
        parameters = component.get("parameters", {})
        parameters = parameters if isinstance(parameters, Mapping) else {}
        voltage = next(
            (
                _mean_number(mapping.get(key))
                for mapping in (parameters, metadata)
                for key in ("voltage", "voltage_v", "rated_voltage_v", "maximum_voltage_v")
                if _mean_number(mapping.get(key)) is not None
            ),
            None,
        )
        if voltage is not None and voltage >= 30.0:
            safety.add("The declared electrical system reaches 30 V or more; use touch-safe enclosures, rated connectors, and qualified review.")
        if any(token in label for token in ("battery", "lipo", "lithium")):
            safety.add("Charge and store each rechargeable battery with the manufacturer-approved charger in a non-combustible location; never charge unattended.")
        if any(token in label for token in ("motor", "wheel", "gear", "actuator")):
            safety.add("Keep hair, clothing, fingers, and cables clear of rotating or actuated parts; test first with the mechanism lifted and current-limited.")
        fabrication = _text(metadata.get("fabrication_method"), "").lower()
        if "print" in fabrication:
            safety.add("Operate 3D printers on a stable non-combustible surface with ventilation appropriate to the material; do not leave a printer unattended unless its safety system is rated for it.")
        raw_notes = metadata.get("safety_notes", [])
        if isinstance(raw_notes, str):
            raw_notes = [raw_notes]
        if isinstance(raw_notes, Sequence):
            safety.update(str(note) for note in raw_notes if str(note).strip())

    steps: list[AssemblyStep] = []

    def add_step(title: str, instruction: str, verification: str) -> None:
        number = len(steps) + 1
        dependencies = (number - 1,) if number > 1 else ()
        steps.append(AssemblyStep(number, title, instruction, verification, dependencies))

    add_step(
        "Resolve the safety gate",
        "Read every safety note and close every unresolved item below. Confirm that component labels, ratings, dimensions, condition, and purchased part numbers match the BOM.",
        "A builder records sign-off for each unresolved item and quarantines damaged or unverified safety-critical parts.",
    )
    fabricated: list[str] = []
    for index, component in enumerate(components):
        metadata = _metadata(component)
        method = _text(metadata.get("fabrication_method"))
        if method:
            fabricated.append(f"{_component_id(component, index)} using {method}")
    if fabricated:
        add_step(
            "Fabricate custom parts",
            "Fabricate " + "; ".join(sorted(fabricated)) + ". Preserve declared geometry dimensions, port access, and coordinate frames.",
            "Measure critical dimensions and record material/process settings against the component version.",
        )
    for connection_id, endpoints, connection in mechanical_connections:
        requirement = _fastener(connection, connection_id)
        attach = (
            f"using {requirement.quantity or 'the validated quantity of'} {requirement.fastener}"
            if requirement is not None
            else "using the resolved attachment method"
        )
        torque = (
            f" Tighten to {requirement.torque_nm:g} N·m"
            if requirement is not None and requirement.torque_nm is not None
            else " Apply the validated manufacturer torque"
        )
        add_step(
            f"Mechanical connection {connection_id}",
            f"Align {endpoints[0][0]}.{endpoints[0][1]} with {endpoints[1][0]}.{endpoints[1][1]} {attach}.{torque}; apply the declared locking method.",
            "Confirm full seating, free intended motion, no interference through the full range, and witness-mark tightened fasteners.",
        )
    if not mechanical_connections:
        add_step(
            "Place and secure components",
            "Place components according to their declared geometry and coordinate frames, using the attachment method resolved in the safety gate.",
            "All parts are restrained against expected inertial loads and no ports are obstructed.",
        )
    if wiring:
        add_step(
            "Route and terminate wiring",
            "With all sources isolated, follow the wiring schedule in order. Cut after routing, provide strain relief/service loops, keep power and sensor wiring separated where practical, and label both ends with the connection ID.",
            "Each listed continuity/polarity verification passes and conductors cannot chafe, enter motion envelopes, or carry mechanical load.",
        )
    controls = spec.get("controls", [])
    if controls:
        add_step(
            "Verify control and emergency behavior",
            "Load the versioned controller only after electrical checks. Keep the mechanism unloaded or lifted, apply current-limited power, confirm sensor sign conventions, then test stop/disable behavior before commanded motion.",
            "Every control maps to the declared target; loss of command or sensor validity reaches the defined safe state.",
        )
    add_step(
        "Commission incrementally",
        "Inspect against the specification, energize one protected domain at a time under current limiting, then increase load and travel in small steps while logging temperature, current, vibration, and unexpected motion.",
        "The assembled identifiers and versions match the build record, protective devices operate, and measured behavior remains inside declared validity ranges.",
    )

    plan = BuildPlan(
        contraption_id,
        tuple(bom),
        tuple(steps),
        tuple(wiring),
        tuple(fasteners),
        tuple(sorted(safety)),
        tuple(sorted(unresolved)),
        tuple(sorted(assumptions)),
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
    "FastenerRequirement",
    "WiringInstruction",
    "generate_build_instructions",
    "generate_build_plan",
]
