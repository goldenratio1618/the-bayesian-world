#!/usr/bin/env python3
"""Deterministically migrate catalog parts to ``static-part-2``.

The migration never synthesizes fabrication dimensions or hardware.  Physical
connectors receive a typed ``status: missing`` record listing the fields that
would be required for their domain; nonphysical boundary/software connectors
receive ``fabrication: null``.  A narrowly-scoped deterministic extractor also
preserves explicit component-input construction facts with exact source hashes
and JSON locators.  Legacy purchasing and identity fields are removed only
after the caller has had an opportunity to extract procurement records from the
v1 sources (the migration report includes their exact source hashes and values).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from contraption.catalog.instantiations import ModelInstantiationSpec, StaticPartSpec


RESERVED_IDENTITY = frozenset(
    {
        "datasheet_url",
        "datasheet_urls",
        "manufacturer",
        "manufacturer_item_number",
        "manufacturer_part_number",
        "mpn",
        "part_number",
        "product",
        "purchase_url",
        "purchase_urls",
        "source_urls",
        "supplier",
        "supplier_sku",
    }
)

_ROMI_ARM_INPUT = Path(
    "outputs/scanner-part-import/component_inputs/romi_arm.json"
)
_ROMI_FEEDBACK_TEXT = "separate potentiometer feedback wire on each servo"
_ROMI_FEEDBACK_LOCATOR = "$.features.feedback"
_ROMI_SERVO_ID = "scanner.position_servo"
_ROMI_FEEDBACK_CONNECTOR = "position_measurement"


def _digest(source: bytes) -> str:
    return "sha256:" + hashlib.sha256(source).hexdigest()


def _fabrication_kind(domain: str, interface: str) -> tuple[str, tuple[str, ...]]:
    if domain in {"mechanical", "rigid_mechanical"}:
        if interface == "rotational-shaft":
            return "rotary_support", ("retention", "bearing", "travel")
        return "fixed_mount", ("retention",)
    if domain in {"electrical", "signal"}:
        return "electrical_termination", ("conductor", "termination")
    if domain == "optical":
        return (
            "optical_alignment",
            ("standards", "alignment_tolerance_m", "alignment_tolerance_rad"),
        )
    return "other", ("standards",)


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    source = path.read_bytes()
    try:
        value = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value, source


def _romi_feedback_fabrication(
    connector: Mapping[str, Any], *, repository: Path
) -> dict[str, Any] | None:
    """Extract the one explicit servo-wire fact without filling its unknowns."""

    if connector.get("id") != _ROMI_FEEDBACK_CONNECTOR:
        return None
    expected_connector = {
        "model_port": "position_measurement",
        "domain": "signal",
        "interface": "logic-signal",
    }
    for name, expected in expected_connector.items():
        if connector.get(name) != expected:
            raise ValueError(
                f"{_ROMI_SERVO_ID}.{_ROMI_FEEDBACK_CONNECTOR}: expected "
                f"{name}={expected!r}, got {connector.get(name)!r}"
            )
    source_path = repository / _ROMI_ARM_INPUT
    source_data, source = _strict_json(source_path)
    features = source_data.get("features")
    feedback = features.get("feedback") if isinstance(features, Mapping) else None
    if feedback != _ROMI_FEEDBACK_TEXT:
        raise ValueError(
            f"{source_path}: {_ROMI_FEEDBACK_LOCATOR} must equal "
            f"{_ROMI_FEEDBACK_TEXT!r}; refusing to infer conductor details"
        )
    return {
        "kind": "electrical_termination",
        "status": "partial",
        "missing": [
            "conductor.standard",
            "conductor.material",
            "conductor.cross_section_m2",
            "conductor.insulation_standard",
            "conductor.voltage_rating_v",
            "conductor.temperature_rating_k",
            "termination",
        ],
        "conductor": {"conductor_count": 1},
        "evidence": [
            {
                "kind": "component_input",
                "source": _ROMI_ARM_INPUT.as_posix(),
                "locator": _ROMI_FEEDBACK_LOCATOR,
                "sha256": _digest(source),
            }
        ],
    }


def _identity_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    purchasing = data.get("purchasing", {})
    if isinstance(purchasing, Mapping):
        result.update(
            (name, purchasing[name])
            for name in RESERVED_IDENTITY
            if name in purchasing
        )
    metadata = data.get("metadata", {})
    if isinstance(metadata, Mapping):
        for name in RESERVED_IDENTITY:
            if name in metadata and name not in result:
                result[name] = metadata[name]
    return result


def _migrate_static(
    path: Path, *, repository: Path | None = None
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    repository = repository or Path(__file__).resolve().parents[1]
    data, source = _strict_json(path)
    if data.get("format") not in {"static-part-1", "static-part-2"}:
        raise ValueError(f"{path}: unsupported static part format {data.get('format')!r}")
    legacy_identity = _identity_payload(data)
    data["format"] = "static-part-2"
    data.pop("purchasing", None)
    metadata = data.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must be an object")
    for name in RESERVED_IDENTITY:
        metadata.pop(name, None)
    physical_role = data.get("physical_role")
    connectors = data.get("connectors")
    if not isinstance(connectors, list):
        raise ValueError(f"{path}: connectors must be an array")
    feedback_fact_applied = False
    for index, connector in enumerate(connectors):
        if not isinstance(connector, dict):
            raise ValueError(f"{path}: connectors[{index}] must be an object")
        if data.get("id") == _ROMI_SERVO_ID:
            extracted = _romi_feedback_fabrication(
                connector, repository=repository
            )
            if extracted is not None:
                connector["fabrication"] = extracted
                feedback_fact_applied = True
                continue
        if "fabrication" in connector:
            continue
        if physical_role != "part":
            connector["fabrication"] = None
            continue
        kind, missing = _fabrication_kind(
            str(connector.get("domain")), str(connector.get("interface"))
        )
        connector["fabrication"] = {
            "kind": kind,
            "status": "missing",
            "missing": list(missing),
        }
    if data.get("id") == _ROMI_SERVO_ID and not feedback_fact_applied:
        raise ValueError(
            f"{path}: {_ROMI_SERVO_ID} lacks the expected "
            f"{_ROMI_FEEDBACK_CONNECTOR!r} connector"
        )
    static = StaticPartSpec.from_dict(data)
    rendered = json.dumps(
        static.to_dict(), indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    report = {
        "path": path.as_posix(),
        "source_sha256": _digest(source),
        "static_part_id": static.id,
        "static_part_version": static.version,
        "static_part_sha256": static.sha256,
        "legacy_identity": legacy_identity,
    }
    return rendered, report, static.to_dict()


def _migrate_model(path: Path) -> tuple[str, dict[str, Any]]:
    data, source = _strict_json(path)
    metadata = data.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must be an object")
    removed = {
        name: metadata.pop(name)
        for name in sorted(RESERVED_IDENTITY)
        if name in metadata
    }
    model = ModelInstantiationSpec.from_dict(data)
    rendered = (
        source.decode("utf-8")
        if not removed
        else json.dumps(
            model.to_dict(), indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )
    return rendered, {
        "path": path.as_posix(),
        "source_sha256": _digest(source),
        "removed_identity": removed,
    }


def _json_object_bytes(source: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected a JSON object")
    return value


def restore_no_identity_models_from_git_head(
    roots: tuple[Path, ...], *, repository: Path
) -> tuple[Path, ...]:
    """Restore byte-only migration churn without checkout/reset.

    A file is restored only when HEAD had no removable top-level identity and
    the current and HEAD model instances parse to exactly the same typed value.
    Any semantic difference fails closed instead of overwriting user work.
    """

    repository = repository.resolve()
    paths = tuple(
        sorted(
            {
                path.resolve()
                for root in roots
                if root.exists()
                for path in root.resolve().rglob("*.model")
            }
        )
    )
    restored: list[Path] = []
    for path in paths:
        try:
            relative = path.relative_to(repository).as_posix()
        except ValueError as exc:
            raise ValueError(f"model path escapes repository: {path}") from exc
        result = subprocess.run(
            ["git", "-C", str(repository), "show", f"HEAD:{relative}"],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            continue
        head_source = result.stdout
        head_data = _json_object_bytes(head_source, f"HEAD:{relative}")
        head_metadata = head_data.get("metadata", {})
        if not isinstance(head_metadata, Mapping):
            raise ValueError(f"HEAD:{relative}: metadata must be an object")
        if set(head_metadata) & RESERVED_IDENTITY:
            continue
        current_source = path.read_bytes()
        current_data = _json_object_bytes(current_source, str(path))
        head_model = ModelInstantiationSpec.from_dict(head_data)
        current_model = ModelInstantiationSpec.from_dict(current_data)
        if current_model != head_model:
            raise ValueError(
                f"refusing to restore semantically changed no-identity model: {path}"
            )
        if current_source != head_source:
            path.write_bytes(head_source)
            restored.append(path)
    return tuple(restored)


def migrate(
    roots: tuple[Path, ...], *, check: bool, repository: Path | None = None
) -> dict[str, Any]:
    repository = repository or Path(__file__).resolve().parents[1]
    static_paths = tuple(
        sorted(
            {
                path.resolve()
                for root in roots
                if root.exists()
                for path in root.resolve().rglob("static.part")
            }
        )
    )
    if not static_paths:
        raise ValueError("no static.part files found under migration roots")
    writes: dict[Path, str] = {}
    static_reports: list[dict[str, Any]] = []
    model_reports: list[dict[str, Any]] = []
    for path in static_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe static part path: {path}")
        rendered, report, _canonical = _migrate_static(
            path, repository=repository
        )
        writes[path] = rendered
        static_reports.append(report)
        for model_path in sorted(path.parent.glob("*.model")):
            if model_path.is_symlink() or not model_path.is_file():
                raise ValueError(f"unsafe model instance path: {model_path}")
            model_rendered, model_report = _migrate_model(model_path)
            writes[model_path] = model_rendered
            model_reports.append(model_report)
    changed = [
        path
        for path, rendered in writes.items()
        if path.read_text(encoding="utf-8") != rendered
    ]
    if not check:
        for path in changed:
            path.write_text(writes[path], encoding="utf-8", newline="\n")
    return {
        "schema": "static-part-2-migration-report/v1",
        "check": check,
        "changed": [path.as_posix() for path in changed],
        "static_parts": static_reports,
        "model_instances": model_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, dest="roots")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--restore-no-identity-models-from-head", action="store_true")
    arguments = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    roots = tuple(arguments.roots or (
        repository / "model_catalog",
        repository / "assembled_contraptions" / "examples" / "test_systems",
    ))
    try:
        restored = (
            restore_no_identity_models_from_git_head(
                roots, repository=repository
            )
            if arguments.restore_no_identity_models_from_head
            else ()
        )
        report = migrate(roots, check=arguments.check)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report["restored_no_identity_models"] = [
        path.as_posix() for path in restored
    ]
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.write_text(rendered, encoding="utf-8", newline="\n")
    sys.stdout.write(rendered)
    return 1 if arguments.check and report["changed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
