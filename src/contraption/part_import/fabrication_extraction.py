"""Deterministic, host-owned fabrication extraction for part imports.

The modeling agent may choose connector ids and poses, but it is not an
authority for construction facts.  A non-missing ``connector.fabrication``
record is retained only when the component input contains the strict
``connector_fabrication`` array defined here.  Everything else is normalized
to a typed missing record (or ``null`` for non-part physical roles).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

from ..catalog.instantiations import StaticPartSpec
from ..fabrication import ConnectorFabricationSpec, FabricationSpecError
from ..strict_json import loads_strict_json


class FabricationExtractionError(ValueError):
    """A deterministic connector-fabrication source or overlay is invalid."""


HOST_FABRICATION_CONTEXT_FORMAT = "host-fabrication-context-1"
HOST_FABRICATION_CONTEXT_FILENAME = ".host-fabrication-context.json"
HOST_FABRICATION_RECEIPT_FORMAT = "host-fabrication-receipt-1"
HOST_FABRICATION_RECEIPT_FILENAME = ".host-fabrication-receipt.json"


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise FabricationExtractionError(f"{context} must be an object with string keys")
    return value


def _array(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FabricationExtractionError(f"{context} must be an array")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise FabricationExtractionError(f"{context} must be a nonempty trimmed string")
    return value


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _digest(payload)


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = loads_strict_json(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise FabricationExtractionError(f"{context} is not strict UTF-8 JSON: {exc}") from exc
    return _object(value, context)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _expected_fabrication(domain: str, interface: str) -> tuple[str, tuple[str, ...]]:
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


def extract_connector_fabrication(
    source: bytes, *, source_name: str
) -> tuple[dict[str, Any], ...]:
    """Extract only explicitly typed component-input fabrication records.

    The optional component-input field has this exact shape::

        "connector_fabrication": [
          {"part": "part.id", "connector": "connector_id", "fabrication": {...}}
        ]

    Source records may not author their own evidence.  The host adds an exact
    component-input hash and JSON locator after validation.
    """

    try:
        value = loads_strict_json(source)
    except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise FabricationExtractionError(f"component input is not strict UTF-8 JSON: {exc}") from exc
    data = _object(value, "component input")
    raw_records = data.get("connector_fabrication")
    if raw_records is None:
        return ()
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    source_digest = _digest(source)
    for index, raw in enumerate(_array(raw_records, "component_input.connector_fabrication")):
        item = _object(raw, f"component_input.connector_fabrication[{index}]")
        if set(item) != {"part", "connector", "fabrication"}:
            raise FabricationExtractionError(
                f"component_input.connector_fabrication[{index}] must contain only "
                "part, connector, and fabrication"
            )
        part = _text(item["part"], f"connector_fabrication[{index}].part")
        connector = _text(item["connector"], f"connector_fabrication[{index}].connector")
        key = (part, connector)
        if key in seen:
            raise FabricationExtractionError(
                f"duplicate connector fabrication selector {part}.{connector}"
            )
        seen.add(key)
        fabrication = dict(
            _object(item["fabrication"], f"connector_fabrication[{index}].fabrication")
        )
        if "evidence" in fabrication:
            raise FabricationExtractionError(
                "component connector_fabrication records cannot author evidence; "
                "the host binds the exact input locator and hash"
            )
        status = fabrication.get("status")
        if status in {"partial", "specified"}:
            fabrication["evidence"] = [
                {
                    "kind": "component_input",
                    "source": _text(source_name, "source_name"),
                    "locator": f"$.connector_fabrication[{index}].fabrication",
                    "sha256": source_digest,
                }
            ]
        try:
            parsed = ConnectorFabricationSpec.from_dict(fabrication)
        except (FabricationSpecError, KeyError, TypeError) as exc:
            raise FabricationExtractionError(
                f"connector_fabrication[{index}] is invalid: {exc}"
            ) from exc
        records.append(
            {"part": part, "connector": connector, "fabrication": parsed.to_dict()}
        )
    return tuple(records)


def write_host_fabrication_context(
    run_root: str | Path,
    *,
    component_input: str | Path,
    source_name: str,
) -> Path:
    """Write a self-verifying context beside the protected input snapshot."""

    root = Path(run_root).resolve()
    component = Path(component_input).resolve()
    if component.is_symlink() or not component.is_file():
        raise FabricationExtractionError(f"component input is not a regular file: {component}")
    if root != component and root not in component.parents:
        raise FabricationExtractionError("component input is outside the modeling run root")
    payload = component.read_bytes()
    records = extract_connector_fabrication(payload, source_name=source_name)
    relative = component.relative_to(root).as_posix()
    context = {
        "format": HOST_FABRICATION_CONTEXT_FORMAT,
        "source": {
            "name": _text(source_name, "source_name"),
            "path": relative,
            "sha256": _digest(payload),
        },
        "records": list(records),
    }
    path = root / HOST_FABRICATION_CONTEXT_FILENAME
    _write_json_atomic(path, context)
    return path


def _load_context(path: Path) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_file():
        raise FabricationExtractionError(
            f"host fabrication context is not a regular file: {path}"
        )
    context = _read_json(path, "host fabrication context")
    if set(context) != {"format", "source", "records"} or context.get("format") != HOST_FABRICATION_CONTEXT_FORMAT:
        raise FabricationExtractionError("host fabrication context has an invalid schema")
    source = _object(context["source"], "host fabrication context source")
    if set(source) != {"name", "path", "sha256"}:
        raise FabricationExtractionError("host fabrication context source has invalid fields")
    raw_relative = _text(source["path"], "host fabrication context source.path")
    if "\\" in raw_relative or "\x00" in raw_relative:
        raise FabricationExtractionError("host fabrication context source.path is unsafe")
    relative = PurePosixPath(raw_relative)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise FabricationExtractionError("host fabrication context source.path is unsafe")
    root = path.parent.resolve()
    component = (root / Path(*relative.parts)).resolve()
    if component.is_symlink() or not component.is_file() or root not in component.parents:
        raise FabricationExtractionError("host fabrication component snapshot is unsafe")
    payload = component.read_bytes()
    if source.get("sha256") != _digest(payload):
        raise FabricationExtractionError("host fabrication component snapshot hash changed")
    expected = extract_connector_fabrication(
        payload, source_name=_text(source["name"], "host fabrication context source.name")
    )
    raw_records = _array(context["records"], "host fabrication context records")
    if list(expected) != list(raw_records):
        raise FabricationExtractionError("host fabrication context records changed")
    return expected


def proposal_fabrication_context_path(candidate_root: str | Path) -> Path:
    candidate = Path(candidate_root).resolve()
    return candidate.parent.parent / HOST_FABRICATION_CONTEXT_FILENAME


def proposal_fabrication_receipt_path(candidate_root: str | Path) -> Path:
    candidate = Path(candidate_root).resolve()
    return candidate.parent.parent / HOST_FABRICATION_RECEIPT_FILENAME


def _physical_static_entries(root: Path) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path in sorted(root.rglob("static.part")):
        if path.is_symlink() or not path.is_file():
            raise FabricationExtractionError(
                f"fabrication receipt static part is unsafe: {path}"
            )
        try:
            payload = path.read_bytes()
            static = StaticPartSpec.from_json(payload.decode("utf-8"))
        except Exception as exc:
            raise FabricationExtractionError(
                f"fabrication receipt static part is invalid: {path}: {exc}"
            ) from exc
        if static.physical_role != "part" or not static.connectors:
            continue
        relative = path.relative_to(root).as_posix()
        normalized = relative.casefold()
        if normalized in seen_paths:
            raise FabricationExtractionError(
                f"case-colliding fabrication receipt static path: {relative}"
            )
        seen_paths.add(normalized)
        static_data = static.to_dict()
        fabrication = [
            {
                "connector": connector["id"],
                "fabrication": connector.get("fabrication"),
            }
            for connector in static_data["connectors"]
        ]
        entries.append(
            {
                "path": relative,
                "file_sha256": _digest(payload),
                "part": static.id,
                "version": static.version,
                "static_sha256": static.sha256,
                "fabrication": fabrication,
                "fabrication_sha256": _canonical_digest(fabrication),
            }
        )
    return tuple(entries)


def build_proposal_fabrication_receipt(
    candidate_root: str | Path, context_path: str | Path
) -> dict[str, Any] | None:
    """Bind normalized connector fabrication to protected input context."""

    root = Path(candidate_root).resolve()
    context = Path(context_path).resolve()
    expected_context = proposal_fabrication_context_path(root)
    if context != expected_context:
        raise FabricationExtractionError(
            "host fabrication context is not at the protected proposal path"
        )
    _load_context(context)
    entries = _physical_static_entries(root)
    if not entries:
        return None
    return {
        "format": HOST_FABRICATION_RECEIPT_FORMAT,
        "context": {
            "path": context.name,
            "sha256": _digest(context.read_bytes()),
        },
        "statics": list(entries),
    }


def validate_fabrication_receipt(
    candidate_root: str | Path,
    *,
    context_path: str | Path,
    receipt: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Validate an in-memory protected receipt during materialization."""

    root = Path(candidate_root).resolve()
    context = Path(context_path).resolve()
    entries = _physical_static_entries(root)
    if not entries:
        raise FabricationExtractionError(
            "host fabrication receipt exists without physical connector parts"
        )
    if context.is_symlink() or not context.is_file():
        raise FabricationExtractionError(
            "importer-produced physical connector parts require protected fabrication context"
        )
    receipt = _object(receipt, "host fabrication receipt")
    if set(receipt) != {"format", "context", "statics"}:
        raise FabricationExtractionError(
            "host fabrication receipt has invalid fields"
        )
    if receipt.get("format") != HOST_FABRICATION_RECEIPT_FORMAT:
        raise FabricationExtractionError(
            "unsupported host fabrication receipt format"
        )
    receipt_context = _object(
        receipt.get("context"), "host fabrication receipt context"
    )
    if set(receipt_context) != {"path", "sha256"}:
        raise FabricationExtractionError(
            "host fabrication receipt context has invalid fields"
        )
    if receipt_context.get("path") != context.name:
        raise FabricationExtractionError(
            "host fabrication receipt context path changed"
        )
    if receipt_context.get("sha256") != _digest(context.read_bytes()):
        raise FabricationExtractionError(
            "host fabrication receipt context hash changed"
        )
    _load_context(context)
    raw_statics = _array(receipt.get("statics"), "host fabrication receipt statics")
    if list(raw_statics) != list(entries):
        raise FabricationExtractionError(
            "host fabrication receipt static bytes or fabrication payload changed"
        )
    return tuple(root / Path(*PurePosixPath(item["path"]).parts) for item in entries)


def verify_fabrication_receipt(
    candidate_root: str | Path,
    *,
    context_path: str | Path,
    receipt_path: str | Path,
) -> tuple[Path, ...]:
    """Verify one protected receipt against exact candidate or snapshot bytes."""

    root = Path(candidate_root).resolve()
    context = Path(context_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    entries = _physical_static_entries(root)
    if not entries:
        if receipt_file.exists():
            raise FabricationExtractionError(
                "host fabrication receipt exists without physical connector parts"
            )
        return ()
    if context.is_symlink() or not context.is_file():
        raise FabricationExtractionError(
            "importer-produced physical connector parts require protected fabrication context"
        )
    if receipt_file.is_symlink() or not receipt_file.is_file():
        raise FabricationExtractionError(
            "importer-produced physical connector parts require a protected fabrication receipt"
        )
    if receipt_file.parent != context.parent:
        raise FabricationExtractionError(
            "host fabrication receipt and context must share a protected directory"
        )
    receipt = _read_json(receipt_file, "host fabrication receipt")
    return validate_fabrication_receipt(
        root, context_path=context, receipt=receipt
    )


def verify_proposal_fabrication_receipt(
    candidate_root: str | Path,
) -> tuple[Path, ...]:
    """Verify the protected receipt when this is an importer-produced proposal."""

    root = Path(candidate_root).resolve()
    context = proposal_fabrication_context_path(root)
    receipt = proposal_fabrication_receipt_path(root)
    entries = _physical_static_entries(root)
    if not context.exists() and not receipt.exists():
        # Legacy/manual staged bundles do not claim host fabrication authority.
        return ()
    if not entries:
        if receipt.exists():
            raise FabricationExtractionError(
                "host fabrication receipt exists without physical connector parts"
            )
        if context.exists():
            _load_context(context)
        return ()
    return verify_fabrication_receipt(
        root, context_path=context, receipt_path=receipt
    )


def materialize_proposal_fabrication(
    candidate_root: str | Path, context_path: str | Path
) -> tuple[Path, ...]:
    """Replace agent-authored fabrication with deterministic source records."""

    root = Path(candidate_root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise FabricationExtractionError(f"candidate root is not a regular directory: {root}")
    context = Path(context_path).resolve()
    records = _load_context(context)
    by_key = {
        (str(item["part"]), str(item["connector"])): item["fabrication"]
        for item in records
    }
    matched: set[tuple[str, str]] = set()
    written: list[Path] = []
    for static_path in sorted(root.rglob("static.part")):
        if static_path.is_symlink() or not static_path.is_file():
            raise FabricationExtractionError(f"static part is not a regular file: {static_path}")
        data = dict(_read_json(static_path, str(static_path)))
        if data.get("format") != "static-part-2":
            raise FabricationExtractionError(
                f"{static_path}: deterministic fabrication requires static-part-2"
            )
        part_id = _text(data.get("id"), f"{static_path}.id")
        role = _text(data.get("physical_role"), f"{static_path}.physical_role")
        connectors = _array(data.get("connectors"), f"{static_path}.connectors")
        for index, raw_connector in enumerate(connectors):
            connector = _object(raw_connector, f"{static_path}.connectors[{index}]")
            if not isinstance(raw_connector, dict):
                raise FabricationExtractionError(
                    f"{static_path}.connectors[{index}] must be a mutable object"
                )
            connector_id = _text(connector.get("id"), f"{static_path}.connectors[{index}].id")
            if role != "part":
                raw_connector["fabrication"] = None
                continue
            kind, missing = _expected_fabrication(
                _text(connector.get("domain"), f"{static_path}.connectors[{index}].domain"),
                _text(connector.get("interface"), f"{static_path}.connectors[{index}].interface"),
            )
            key = (part_id, connector_id)
            selected = by_key.get(key)
            if selected is None:
                raw_connector["fabrication"] = {
                    "kind": kind,
                    "status": "missing",
                    "missing": list(missing),
                }
                continue
            if key in matched:
                raise FabricationExtractionError(
                    f"fabrication selector {part_id}.{connector_id} matched more than once"
                )
            parsed = ConnectorFabricationSpec.from_dict(
                _object(selected, f"fabrication selector {part_id}.{connector_id}")
            )
            if parsed.kind != kind:
                raise FabricationExtractionError(
                    f"fabrication selector {part_id}.{connector_id} kind {parsed.kind!r} "
                    f"does not match connector kind {kind!r}"
                )
            raw_connector["fabrication"] = parsed.to_dict()
            matched.add(key)
        try:
            static = StaticPartSpec.from_dict(data)
        except Exception as exc:
            raise FabricationExtractionError(
                f"{static_path}: normalized static part is invalid: {exc}"
            ) from exc
        static_path.write_text(
            json.dumps(static.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        written.append(static_path)
    unmatched = sorted(set(by_key) - matched)
    if unmatched:
        rendered = ", ".join(f"{part}.{connector}" for part, connector in unmatched)
        raise FabricationExtractionError(
            "component input names fabrication selectors absent from the candidate: " + rendered
        )
    return tuple(written)


__all__ = [
    "FabricationExtractionError",
    "HOST_FABRICATION_CONTEXT_FILENAME",
    "HOST_FABRICATION_CONTEXT_FORMAT",
    "HOST_FABRICATION_RECEIPT_FILENAME",
    "HOST_FABRICATION_RECEIPT_FORMAT",
    "build_proposal_fabrication_receipt",
    "extract_connector_fabrication",
    "materialize_proposal_fabrication",
    "proposal_fabrication_context_path",
    "proposal_fabrication_receipt_path",
    "validate_fabrication_receipt",
    "verify_fabrication_receipt",
    "verify_proposal_fabrication_receipt",
    "write_host_fabrication_context",
]
