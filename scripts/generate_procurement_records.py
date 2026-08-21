#!/usr/bin/env python3
"""Generate the adjacent evidence-backed procurement catalog.

The generator is intentionally closed over tracked component inputs,
explicit kit contents/quantities, current static-part provenance, and one
preserved legacy KEMET identity snapshot.  It never performs network access or
creates offers, prices, availability claims, or non-unknown lifecycle status.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from contraption.catalog.instantiations import StaticPartSpec
from contraption.catalog.procurement import (
    ProcurementEvidenceSpec,
    ProcurementIdentifierSpec,
    ProcurementLifecycleSpec,
    ProcurementRecord,
    ProcurementRegistry,
)
from contraption.part_import.procurement_extraction import (
    bind_record,
    extract_component_procurement_file,
    static_part_provision,
    write_procurement_records,
)


PRIOR_INPUT_ROOT = Path("outputs/part-import-2026-08-18/component_inputs")
SCANNER_INPUT_ROOT = Path("outputs/scanner-part-import/component_inputs")
LEGACY_EVIDENCE = Path(
    "model_catalog/procurement/evidence/legacy_static_part_identities.json"
)
UNBOUND_LOCATIONS: Mapping[str, Path] = {
    "murata.ncp18xf101j03rb": Path(
        "thermoelectric/thermistors/instantiations/murata_ncp18_100r"
    ),
    "murata.ncp18xf151j03rb": Path(
        "thermoelectric/thermistors/instantiations/murata_ncp18_150r"
    ),
    "murata.ncp18xm221j03rb": Path(
        "thermoelectric/thermistors/instantiations/murata_ncp18_220r"
    ),
    "murata.ncp18xm331j03rb": Path(
        "thermoelectric/thermistors/instantiations/murata_ncp18_330r"
    ),
    "murata.ncp18xq471j03rb": Path(
        "thermoelectric/thermistors/instantiations/murata_ncp18_470r"
    ),
    "murata.ncp18xq681j03rb": Path(
        "thermoelectric/thermistors/instantiations/murata_ncp18_680r"
    ),
    "murata.ncp18xq102j03rb": Path(
        "thermoelectric/thermistors/instantiations/murata_ncp18_1k"
    ),
    "murata.ncp18xw152j03rb": Path(
        "thermoelectric/thermistors/instantiations/murata_ncp18_1k5"
    ),
    "murata.ncp18xw222j03rb": Path(
        "thermoelectric/thermistors/instantiations/murata_ncp18_2k2"
    ),
    "murata.ncp18xw332j03rb": Path(
        "thermoelectric/thermistors/instantiations/murata_ncp18_3k3"
    ),
    "product.six-rechargeable-aa-nimh-cells-and-matched-charger": Path(
        "electrochemical/batteries/nimh_battery_packs/instantiations/"
        "scanner_nimh_battery"
    ),
}


@dataclass(frozen=True, slots=True)
class Binding:
    part_id: str
    quantity: int
    input_path: tuple[str | int, ...]
    expected_value: Any
    static_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProductPlan:
    product: str
    bindings: tuple[Binding, ...]


SCANNER_PLANS: Mapping[str, tuple[ProductPlan, ...]] = {
    "romi_drive.json": (
        ProductPlan(
            "Romi Chassis Kit, item 3500",
            (
                Binding(
                    "scanner.romi_chassis",
                    1,
                    ("included", 4),
                    "163 x 149 x 70 mm chassis",
                    ("Pololu", "3500"),
                ),
                Binding(
                    "scanner.wheel",
                    2,
                    ("included", 1),
                    "two 70 x 8 mm wheels",
                    ("Pololu", "wheel"),
                ),
                Binding(
                    "scanner.gearmotor",
                    2,
                    ("included", 0),
                    "two 120:1 HP mini plastic brushed DC gearmotors",
                    ("Pololu", "1520"),
                ),
            ),
        ),
    ),
    "romi_arm.json": (
        ProductPlan(
            "Robot Arm Kit for Romi, item 3550",
            (
                Binding(
                    "scanner.arm_linkage",
                    1,
                    ("product",),
                    "Robot Arm Kit for Romi, item 3550",
                    ("arm", "kit"),
                ),
                Binding(
                    "scanner.position_servo",
                    2,
                    ("features", "lift_and_tilt_servos"),
                    2,
                    ("Pololu", "3550"),
                ),
            ),
        ),
    ),
    "romi_control.json": (
        ProductPlan(
            "Romi 32U4 Control Board, item 3544",
            (
                Binding(
                    "scanner.control_board",
                    1,
                    ("product",),
                    "Romi 32U4 Control Board, item 3544",
                    ("Pololu", "3544"),
                ),
            ),
        ),
    ),
    "romi_encoders.json": (
        ProductPlan(
            "Romi Encoder Pair Kit, item 3542",
            (
                Binding(
                    "scanner.encoder_pair",
                    1,
                    ("product",),
                    "Romi Encoder Pair Kit, item 3542",
                    ("Pololu", "3542"),
                ),
            ),
        ),
    ),
    "power.json": (
        ProductPlan(
            "Pololu 6V 2.5A step-up/step-down regulator S13V25F6, item 4981",
            (
                Binding(
                    "scanner.servo_regulator",
                    1,
                    ("products", 1),
                    "Pololu 6V 2.5A step-up/step-down regulator S13V25F6, item 4981",
                    ("Pololu", "4981"),
                ),
            ),
        ),
    ),
    "camera_compute.json": (
        ProductPlan(
            "Raspberry Pi Zero 2 W",
            (
                Binding(
                    "scanner.compute",
                    1,
                    ("products", 0),
                    "Raspberry Pi Zero 2 W",
                    ("Raspberry Pi Zero 2 W",),
                ),
            ),
        ),
        ProductPlan(
            "Camera Module 3 Wide",
            (
                Binding(
                    "scanner.camera",
                    1,
                    ("products", 1),
                    "Camera Module 3 Wide",
                    ("Raspberry Pi", "Camera Module 3 Wide"),
                ),
            ),
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class StaticEntry:
    spec: StaticPartSpec
    path: Path
    source: bytes


def _digest(source: bytes) -> str:
    return "sha256:" + hashlib.sha256(source).hexdigest()


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing JSON evidence file: {path}")
    source = path.read_bytes()
    try:
        value = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON evidence in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return value, source


def _source_name(repository: Path, path: Path) -> str:
    return path.resolve().relative_to(repository.resolve()).as_posix()


def _load_static_parts(repository: Path) -> dict[str, StaticEntry]:
    catalog = repository / "model_catalog"
    result: dict[str, StaticEntry] = {}
    for path in sorted(catalog.rglob("static.part")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe static part: {path}")
        source = path.read_bytes()
        static = StaticPartSpec.from_json(source.decode("utf-8"))
        if static.id in result:
            raise ValueError(f"duplicate static part id {static.id!r}")
        result[static.id] = StaticEntry(static, path, source)
    if not result:
        raise ValueError("model_catalog contains no static parts")
    return result


def _product(record: ProcurementRecord) -> str:
    values = [
        item.value for item in record.identifiers if item.scheme == "product_name"
    ]
    if len(values) != 1:
        raise ValueError(
            f"procurement record {record.id!r} must have one product_name for migration"
        )
    return values[0]


def _lookup(data: Mapping[str, Any], path: tuple[str | int, ...]) -> Any:
    value: Any = data
    for segment in path:
        try:
            value = value[segment]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"missing input evidence at {_locator(path)}") from exc
    return value


def _locator(path: tuple[str | int, ...]) -> str:
    result = "$"
    for segment in path:
        result += f"[{segment}]" if isinstance(segment, int) else f".{segment}"
    return result


def _static_text(entry: StaticEntry) -> str:
    provenance = entry.spec.provenance
    return " ".join(
        (
            entry.spec.id.replace("_", " ").replace("-", " "),
            entry.spec.name,
            provenance.source,
            "" if provenance.reference is None else provenance.reference,
        )
    ).casefold()


def _dedupe_evidence(
    evidence: list[ProcurementEvidenceSpec],
) -> tuple[ProcurementEvidenceSpec, ...]:
    result: list[ProcurementEvidenceSpec] = []
    seen: set[tuple[str, str, str | None]] = set()
    for item in evidence:
        key = (item.source, item.sha256, item.locator)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)


def _bind_planned_record(
    record: ProcurementRecord,
    *,
    bindings: tuple[Binding, ...],
    input_data: Mapping[str, Any],
    input_path: Path,
    repository: Path,
    statics: Mapping[str, StaticEntry],
) -> ProcurementRecord:
    input_source = input_path.read_bytes()
    input_name = _source_name(repository, input_path)
    evidence = list(record.evidence)
    provisions = []
    for binding in bindings:
        actual = _lookup(input_data, binding.input_path)
        if actual != binding.expected_value:
            raise ValueError(
                f"{input_name} {_locator(binding.input_path)} changed: "
                f"expected {binding.expected_value!r}, got {actual!r}"
            )
        try:
            entry = statics[binding.part_id]
        except KeyError as exc:
            raise ValueError(f"missing static part {binding.part_id!r}") from exc
        static_text = _static_text(entry)
        missing_markers = [
            marker
            for marker in binding.static_markers
            if marker.casefold() not in static_text
        ]
        if missing_markers:
            raise ValueError(
                f"static part {binding.part_id!r} no longer carries expected "
                f"provenance marker(s): {', '.join(missing_markers)}"
            )
        provisions.append(
            static_part_provision(entry.spec, quantity=binding.quantity)
        )
        evidence.append(
            ProcurementEvidenceSpec(
                input_name,
                _digest(input_source),
                _locator(binding.input_path),
            )
        )
        evidence.append(
            ProcurementEvidenceSpec(
                _source_name(repository, entry.path),
                _digest(entry.source),
                "$.provenance",
            )
        )
    return replace(
        bind_record(record, provisions),
        evidence=_dedupe_evidence(evidence),
    )


def _prior_records(
    repository: Path, statics: Mapping[str, StaticEntry]
) -> tuple[ProcurementRecord, ...]:
    root = repository / PRIOR_INPUT_ROOT
    paths = tuple(sorted(root.glob("*.json")))
    if len(paths) != 20:
        raise ValueError(f"expected 20 prior component inputs, found {len(paths)}")
    records: list[ProcurementRecord] = []
    family_counts = {"Yageo": 0, "Murata": 0}
    for path in paths:
        data, _source = _strict_json(path)
        manufacturer = data.get("manufacturer")
        if manufacturer not in family_counts:
            raise ValueError(f"unexpected prior input manufacturer in {path}: {manufacturer!r}")
        family_counts[manufacturer] += 1
        extracted = extract_component_procurement_file(
            path, source_name=_source_name(repository, path)
        )
        if len(extracted) != 1:
            raise ValueError(f"expected one procurement identity from {path}")
        record = extracted[0]
        if manufacturer == "Murata":
            records.append(record)
            continue
        product = data.get("product")
        urls = set(data.get("source_urls", ()))
        candidates = []
        for entry in statics.values():
            provenance = entry.spec.provenance
            reference = provenance.reference
            identity_text = " ".join(
                (provenance.source, "" if reference is None else reference)
            )
            if (
                isinstance(product, str)
                and product.casefold() in identity_text.casefold()
            ) or (reference is not None and reference in urls):
                candidates.append(entry)
        if len(candidates) != 1:
            raise ValueError(
                f"Yageo product {product!r} must match exactly one static part; "
                f"found {[entry.spec.id for entry in candidates]}"
            )
        entry = candidates[0]
        records.append(
            replace(
                bind_record(record, (static_part_provision(entry.spec),)),
                evidence=_dedupe_evidence(
                    list(record.evidence)
                    + [
                        ProcurementEvidenceSpec(
                            _source_name(repository, entry.path),
                            _digest(entry.source),
                            "$.provenance",
                        )
                    ]
                ),
            )
        )
    if family_counts != {"Yageo": 10, "Murata": 10}:
        raise ValueError(f"unexpected prior input family counts: {family_counts}")
    return tuple(records)


def _scanner_records(
    repository: Path, statics: Mapping[str, StaticEntry]
) -> tuple[ProcurementRecord, ...]:
    root = repository / SCANNER_INPUT_ROOT
    paths = tuple(sorted(root.glob("*.json")))
    if {path.name for path in paths} != set(SCANNER_PLANS):
        raise ValueError(
            "scanner component input set changed; update and review the explicit plan"
        )
    records: list[ProcurementRecord] = []
    for path in paths:
        data, _source = _strict_json(path)
        plans = {item.product: item for item in SCANNER_PLANS[path.name]}
        seen: set[str] = set()
        for record in extract_component_procurement_file(
            path, source_name=_source_name(repository, path)
        ):
            product = _product(record)
            plan = plans.get(product)
            if plan is None:
                records.append(record)
                continue
            seen.add(product)
            records.append(
                _bind_planned_record(
                    record,
                    bindings=plan.bindings,
                    input_data=data,
                    input_path=path,
                    repository=repository,
                    statics=statics,
                )
            )
        missing = sorted(set(plans) - seen)
        if missing:
            raise ValueError(
                f"planned scanner product(s) were not extracted from {path}: {missing}"
            )
    if len(records) != 8:
        raise ValueError(f"expected 8 scanner procurement records, found {len(records)}")
    return tuple(records)


def _legacy_kemet_record(
    repository: Path, statics: Mapping[str, StaticEntry]
) -> ProcurementRecord:
    evidence_path = repository / LEGACY_EVIDENCE
    data, _snapshot_source = _strict_json(evidence_path)
    if data.get("format") != "legacy-static-procurement-evidence-1":
        raise ValueError("unsupported legacy procurement evidence format")
    source = data.get("source")
    records = data.get("records")
    if not isinstance(source, Mapping) or not isinstance(records, list) or len(records) != 1:
        raise ValueError("malformed legacy KEMET evidence snapshot")
    identity = records[0]
    if not isinstance(identity, Mapping):
        raise ValueError("legacy KEMET identity must be an object")
    expected = {
        "static_part_id": "C1210C476K8RAC",
        "manufacturer": "KEMET",
        "manufacturer_part_number": "C1210C476K8RAC",
    }
    for name, value in expected.items():
        if identity.get(name) != value:
            raise ValueError(f"legacy KEMET {name} changed from {value!r}")
    entry = statics[expected["static_part_id"]]
    if "c1210c476k8rac" not in _static_text(entry):
        raise ValueError("current KEMET static part lost its ordering-code provenance")
    commit = source.get("git_commit")
    path = source.get("path")
    source_sha256 = source.get("sha256")
    if not all(isinstance(value, str) and value for value in (commit, path, source_sha256)):
        raise ValueError("legacy KEMET git evidence identity is incomplete")
    git_source = f"git:{commit}:{path}"
    return ProcurementRecord(
        format="procurement-record-1",
        id="kemet.c1210c476k8rac",
        version="1.0.0",
        manufacturer="KEMET",
        identifiers=(
            ProcurementIdentifierSpec(
                "manufacturer_part_number", "C1210C476K8RAC", "KEMET"
            ),
        ),
        documents=(),
        offers=(),
        lifecycle=ProcurementLifecycleSpec("unknown"),
        provides=(static_part_provision(entry.spec),),
        evidence=(
            ProcurementEvidenceSpec(
                git_source,
                source_sha256,
                "$.purchasing.manufacturer",
            ),
            ProcurementEvidenceSpec(
                git_source,
                source_sha256,
                "$.purchasing.manufacturer_part_number",
            ),
            ProcurementEvidenceSpec(
                _source_name(repository, entry.path),
                _digest(entry.source),
                "$.provenance",
            ),
        ),
    )


def expected_registry(repository: Path) -> ProcurementRegistry:
    repository = repository.resolve()
    statics = _load_static_parts(repository)
    records = list(_prior_records(repository, statics))
    records.extend(_scanner_records(repository, statics))
    records.append(_legacy_kemet_record(repository, statics))
    registry = ProcurementRegistry(records)
    registry.validate_provisions(
        {
            part_id: (entry.spec.version, entry.spec.sha256)
            for part_id, entry in statics.items()
        }
    )
    return registry


def _record_paths(
    repository: Path,
    registry: ProcurementRegistry,
    statics: Mapping[str, StaticEntry],
) -> dict[str, Path]:
    catalog = repository / "model_catalog"
    result: dict[str, Path] = {}
    for record in registry.values():
        if record.provides:
            directory = statics[record.provides[0].part].path.parent
        else:
            try:
                directory = catalog / UNBOUND_LOCATIONS[record.id]
            except KeyError as exc:
                raise ValueError(
                    f"unbound procurement record {record.id!r} has no reviewed location"
                ) from exc
        path = directory / f"{record.id}.procurement"
        relative = _source_name(repository, path)
        if relative in result:
            raise ValueError(f"duplicate procurement output path: {relative}")
        result[relative] = path
    unused = sorted(set(UNBOUND_LOCATIONS) - set(registry))
    if unused:
        raise ValueError(
            "unbound procurement locations reference missing records: "
            + ", ".join(unused)
        )
    return result


def generate(repository: Path, *, check: bool) -> dict[str, Any]:
    repository = repository.resolve()
    registry = expected_registry(repository)
    catalog = repository / "model_catalog"
    statics = _load_static_parts(repository)
    record_paths = _record_paths(repository, registry, statics)
    expected = {
        relative: registry[path.stem].to_json()
        for relative, path in record_paths.items()
    }
    changed: list[str] = []
    actual = {
        _source_name(repository, path): path
        for path in sorted(catalog.rglob("*.procurement"))
    }
    extra = sorted(set(actual) - set(expected))
    if extra:
        raise ValueError(
            "unexpected procurement record files: " + ", ".join(extra)
        )
    for relative, rendered in sorted(expected.items()):
        path = record_paths[relative]
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            changed.append(relative)
    if not check:
        write_procurement_records(
            catalog,
            registry.values(),
            overwrite=True,
            unbound_locations=UNBOUND_LOCATIONS,
        )

    bound = [record for record in registry.values() if record.provides]
    provision_count = sum(len(record.provides) for record in registry.values())
    quantity = sum(
        provision.quantity
        for record in registry.values()
        for provision in record.provides
    )
    return {
        "format": "procurement-generation-report-1",
        "check": check,
        "changed": changed,
        "registry_sha256": registry.sha256,
        "record_count": len(registry),
        "bound_record_count": len(bound),
        "unbound_record_count": len(registry) - len(bound),
        "provision_count": provision_count,
        "provided_quantity": quantity,
        "bindings": {
            record.id: [item.to_dict() for item in record.provides]
            for record in registry.values()
            if record.provides
        },
        "locations": {
            record_id: _source_name(repository, path)
            for record_id, path in sorted(
                (path.stem, path) for path in record_paths.values()
            )
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--repository", type=Path)
    arguments = parser.parse_args(argv)
    repository = (
        Path(__file__).resolve().parents[1]
        if arguments.repository is None
        else arguments.repository
    )
    try:
        report = generate(repository, check=arguments.check)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 1 if arguments.check and report["changed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
