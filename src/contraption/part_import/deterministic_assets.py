"""Host-owned deterministic asset ingestion bundled with Luna proposals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Iterable, Mapping

from ..shape import ShapeArtifact, ShapeUncertainty, import_shape
from ..shape.backends import (
    GeometryBackendError,
    automatic_tessellator,
    backend_identity,
    linked_source_relative_paths,
    linked_source_paths,
    missing_backend_message,
)
from ..shape.ingestion import ShapeImportError, Tessellator
from ..strict_json import loads_strict_json


class DeterministicAssetError(ValueError):
    """Raised when a declared host-side ingestion plan is unsafe or incomplete."""


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
HOST_SHAPE_RECEIPT_FORMAT = "host-shape-receipt-1"
HOST_SHAPE_RECEIPT_FILENAME = ".host-shape-receipt.json"


@dataclass(frozen=True, slots=True)
class DeterministicPlan:
    shapes: tuple[dict[str, Any], ...] = ()
    optical_sensors: tuple[dict[str, Any], ...] = ()
    documents: tuple[dict[str, Any], ...] = ()


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DeterministicAssetError(f"{context} must be an object with string keys")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], required: set[str], context: str) -> None:
    unknown, missing = sorted(set(value) - allowed), sorted(required - set(value))
    if unknown:
        raise DeterministicAssetError(f"{context} has unknown keys {unknown}")
    if missing:
        raise DeterministicAssetError(f"{context} is missing keys {missing}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DeterministicAssetError(f"{context} must be a nonempty trimmed string")
    return value


def _identifier(value: Any, context: str) -> str:
    result = _text(value, context)
    if _IDENTIFIER.fullmatch(result) is None:
        raise DeterministicAssetError(f"{context} must be an identifier")
    return result


def _relative(value: Any, context: str) -> PurePosixPath:
    raw = _text(value, context)
    if "\\" in raw or "\x00" in raw:
        raise DeterministicAssetError(f"{context} must be a POSIX relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise DeterministicAssetError(f"{context} must be a safe relative path")
    return path


def _resolve_below(root: Path, relative: PurePosixPath, context: str) -> Path:
    target = (root / Path(*relative.parts)).resolve()
    if target != root and root not in target.parents:
        raise DeterministicAssetError(f"{context} escapes {root}")
    return target


def _source_closure(source: Path) -> tuple[Path, ...]:
    values: list[Path] = [source]
    if source.suffix.lower() == ".obj":
        for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.partition("#")[0].strip()
            if not line.lower().startswith("mtllib "):
                continue
            for item in line.split()[1:]:
                linked = (source.parent / item).resolve()
                source_root = source.parent.resolve()
                if linked != source_root and source_root not in linked.parents:
                    raise DeterministicAssetError(f"{source}:{line_number}: mtllib must remain in the source directory")
                if not linked.is_file():
                    raise FileNotFoundError(linked)
                values.append(linked)
    try:
        values.extend(linked_source_paths(source))
    except GeometryBackendError as exc:
        raise DeterministicAssetError(
            f"cannot determine deterministic source closure for {source}: {exc}"
        ) from exc
    for candidate in (source.with_suffix(source.suffix + ".optical.json"), source.with_suffix(".optical.json")):
        if candidate.is_file():
            values.append(candidate.resolve())
            break
    return tuple(dict.fromkeys(values))


def load_plan(
    component_information: str | Path,
    *,
    source_directory: str | Path | None = None,
) -> DeterministicPlan:
    component = Path(component_information).resolve()
    source_root = (
        component.parent
        if source_directory is None
        else Path(source_directory).resolve()
    )
    if not source_root.is_dir():
        raise DeterministicAssetError(
            f"component source directory is missing or not a directory: {source_root}"
        )
    value = _object(loads_strict_json(component.read_text(encoding="utf-8")), "component information")
    raw_plan = value.get("deterministic_ingestion")
    if raw_plan is None:
        return DeterministicPlan()
    plan = _object(raw_plan, "deterministic_ingestion")
    _keys(
        plan,
        {"format", "shapes", "optical_sensors", "documents"},
        {"format"},
        "deterministic_ingestion",
    )
    if plan["format"] != "deterministic-part-ingestion-1":
        raise DeterministicAssetError("deterministic_ingestion.format must be deterministic-part-ingestion-1")
    raw_shapes = plan.get("shapes", [])
    raw_sensors = plan.get("optical_sensors", [])
    raw_documents = plan.get("documents", [])
    if (
        not isinstance(raw_shapes, list)
        or not isinstance(raw_sensors, list)
        or not isinstance(raw_documents, list)
        or not (raw_shapes or raw_sensors or raw_documents)
    ):
        raise DeterministicAssetError(
            "deterministic_ingestion requires a nonempty shapes, optical_sensors, or documents array"
        )
    shapes: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_shapes):
        item = _object(raw, f"deterministic_ingestion.shapes[{index}]")
        names = {
            "source",
            "catalog_directory",
            "body",
            "solid",
            "artifact_id",
            "version",
            "metres_per_source_unit",
            "density_kg_m3",
            "surface_uncertainty",
            "provenance",
            "archive_member",
        }
        required = {"source", "catalog_directory", "body", "solid", "artifact_id", "metres_per_source_unit"}
        _keys(item, names, required, f"deterministic_ingestion.shapes[{index}]")
        source_relative = _relative(item["source"], f"shapes[{index}].source")
        source = _resolve_below(source_root, source_relative, f"shapes[{index}].source")
        if not source.is_file() or source.is_symlink():
            raise DeterministicAssetError(f"shape source is missing or not regular: {source}")
        archive_member = item.get("archive_member")
        if source.suffix.casefold() == ".zip":
            if archive_member is None:
                raise DeterministicAssetError(
                    f"shapes[{index}].archive_member is required for ZIP sources"
                )
            archive_member = _relative(
                archive_member, f"shapes[{index}].archive_member"
            ).as_posix()
            if PurePosixPath(archive_member).suffix.casefold() not in {
                ".obj",
                ".stl",
                ".ply",
                ".step",
                ".stp",
                ".iges",
                ".igs",
                ".brep",
                ".fcstd",
                ".gltf",
                ".glb",
                ".wrl",
                ".vrml",
                ".ctmesh",
            }:
                raise DeterministicAssetError(
                    f"shapes[{index}].archive_member has an unsupported geometry extension"
                )
        elif archive_member is not None:
            raise DeterministicAssetError(
                f"shapes[{index}].archive_member is only valid for ZIP sources"
            )
        scale = item["metres_per_source_unit"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) <= 0:
            raise DeterministicAssetError(f"shapes[{index}].metres_per_source_unit must be positive")
        density = item.get("density_kg_m3")
        if density is not None and (isinstance(density, bool) or not isinstance(density, (int, float)) or not math.isfinite(float(density)) or float(density) <= 0):
            raise DeterministicAssetError(f"shapes[{index}].density_kg_m3 must be positive")
        normalized_shape = {
                "source": str(source),
                "catalog_directory": _relative(item["catalog_directory"], f"shapes[{index}].catalog_directory").as_posix(),
                "body": _identifier(item["body"], f"shapes[{index}].body"),
                "solid": _identifier(item["solid"], f"shapes[{index}].solid"),
                "artifact_id": _identifier(item["artifact_id"], f"shapes[{index}].artifact_id"),
                "version": _text(item.get("version", "1.0.0"), f"shapes[{index}].version"),
                "metres_per_source_unit": float(scale),
                "density_kg_m3": None if density is None else float(density),
                "surface_uncertainty": (
                    None
                    if item.get("surface_uncertainty") is None
                    else ShapeUncertainty.from_dict(
                        _object(item["surface_uncertainty"], f"shapes[{index}].surface_uncertainty")
                    ).to_dict()
                ),
                "provenance": dict(_object(item.get("provenance", {}), f"shapes[{index}].provenance")),
            }
        if archive_member is not None:
            normalized_shape["archive_member"] = archive_member
        shapes.append(normalized_shape)
    keys = [(item["catalog_directory"].casefold(), item["body"].casefold(), item["solid"].casefold()) for item in shapes]
    if len(keys) != len(set(keys)):
        raise DeterministicAssetError("deterministic shape targets must be unique")
    sensors: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_sensors):
        item = _object(raw, f"deterministic_ingestion.optical_sensors[{index}]")
        names = {"source", "catalog_directory", "body", "pose_connector", "artifact_port"}
        _keys(item, names, names, f"deterministic_ingestion.optical_sensors[{index}]")
        source_relative = _relative(item["source"], f"optical_sensors[{index}].source")
        source = _resolve_below(source_root, source_relative, f"optical_sensors[{index}].source")
        if not source.is_file() or source.is_symlink():
            raise DeterministicAssetError(f"optical sensor source is missing or not regular: {source}")
        from ..optics import OpticalSchemaError, OpticalSensor
        try:
            descriptor = OpticalSensor.load(source)
        except (OSError, OpticalSchemaError) as exc:
            raise DeterministicAssetError(f"invalid optical sensor source {source}: {exc}") from exc
        pose_connector = _identifier(item["pose_connector"], f"optical_sensors[{index}].pose_connector")
        if descriptor.mount_connector != pose_connector:
            raise DeterministicAssetError(
                f"optical sensor {descriptor.id!r} mount_connector does not match target pose_connector"
            )
        sensors.append(
            {
                "source": str(source),
                "catalog_directory": _relative(item["catalog_directory"], f"optical_sensors[{index}].catalog_directory").as_posix(),
                "body": _identifier(item["body"], f"optical_sensors[{index}].body"),
                "pose_connector": pose_connector,
                "artifact_port": _identifier(item["artifact_port"], f"optical_sensors[{index}].artifact_port"),
                "sensor_id": descriptor.id,
            }
        )
    sensor_targets = [item["catalog_directory"].casefold() for item in sensors]
    if len(sensor_targets) != len(set(sensor_targets)):
        raise DeterministicAssetError("each instantiation may have at most one deterministic optical sensor descriptor")
    documents: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_documents):
        item = _object(raw, f"deterministic_ingestion.documents[{index}]")
        _keys(
            item,
            {"source", "source_format"},
            {"source"},
            f"deterministic_ingestion.documents[{index}]",
        )
        source_relative = _relative(item["source"], f"documents[{index}].source")
        source = _resolve_below(source_root, source_relative, f"documents[{index}].source")
        if not source.is_file() or source.is_symlink():
            raise DeterministicAssetError(
                f"document source is missing or not regular: {source}"
            )
        source_format = item.get("source_format", "auto")
        if source_format not in {"auto", "pdf", "dxf", "kicad", "librepcb"}:
            raise DeterministicAssetError(
                f"documents[{index}].source_format must be auto, pdf, dxf, kicad, or librepcb"
            )
        documents.append({"source": str(source), "source_format": source_format})
    document_sources = [item["source"].casefold() for item in documents]
    if len(document_sources) != len(set(document_sources)):
        raise DeterministicAssetError("deterministic document sources must be unique")
    return DeterministicPlan(tuple(shapes), tuple(sensors), tuple(documents))


def input_paths(
    component_information: str | Path,
    *,
    source_directory: str | Path | None = None,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    plan = load_plan(
        component_information, source_directory=source_directory
    )
    for item in plan.shapes:
        paths.extend(_source_closure(Path(item["source"])))
    paths.extend(Path(item["source"]) for item in plan.optical_sensors)
    paths.extend(Path(item["source"]) for item in plan.documents)
    return tuple(dict.fromkeys(paths))


def _tree_hashes(root: Path) -> dict[str, str]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DeterministicAssetError(
                f"deterministic staged trees may not contain symlinks: {path}"
            )
        if path.is_file():
            files.append(path)
    if not files:
        raise DeterministicAssetError(f"deterministic staged tree is empty: {root}")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


def _stage_shape_source(
    source: Path,
    input_root: Path,
    archive_member: str | None,
) -> tuple[Path, dict[str, Any] | None]:
    input_root.mkdir()
    if archive_member is not None:
        raw_target = input_root / source.name
        _snapshot_regular_file(source, raw_target, "shape archive")
        from .archive_ingestion import (
            DeterministicArchiveError,
            extract_shape_archive,
        )

        try:
            extracted = extract_shape_archive(
                raw_target,
                input_root / "extracted",
                member=archive_member,
            )
        except DeterministicArchiveError as exc:
            raise DeterministicAssetError(
                f"deterministic shape archive extraction failed for {source}: {exc}"
            ) from exc
        return extracted.selected_path, {
            "format": "zip",
            "source_name": source.name,
            "bytes": extracted.archive_bytes,
            "sha256": extracted.archive_sha256,
            "member": extracted.selected_member,
        }

    # Capture the primary bytes exactly once before parsing any active-content
    # or linked-resource declarations. Every preflight, backend call, evidence
    # copy, and hash below operates on this private sibling tree rather than on
    # the mutable catalog download path.
    source_parent = source.parent.resolve()
    staged_source = input_root / source.name
    _snapshot_regular_file(source, staged_source, "shape source")
    relatives: list[PurePosixPath] = []
    if staged_source.suffix.casefold() == ".obj":
        try:
            lines = staged_source.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise DeterministicAssetError("OBJ source must be strict UTF-8") from exc
        for line_number, raw in enumerate(lines, 1):
            line = raw.partition("#")[0].strip()
            if not line.casefold().startswith("mtllib "):
                continue
            for value in line.split()[1:]:
                try:
                    relative = _relative(
                        value, f"OBJ mtllib at line {line_number}"
                    )
                except DeterministicAssetError as exc:
                    raise DeterministicAssetError(
                        f"unsafe OBJ mtllib at line {line_number}: {value!r}"
                    ) from exc
                relatives.append(relative)
    try:
        relatives.extend(linked_source_relative_paths(staged_source))
    except GeometryBackendError as exc:
        raise DeterministicAssetError(
            f"cannot determine private shape source closure for {source}: {exc}"
        ) from exc
    for candidate in (
        source.with_suffix(source.suffix + ".optical.json"),
        source.with_suffix(".optical.json"),
    ):
        if candidate.is_file() and not candidate.is_symlink():
            relatives.append(
                PurePosixPath(candidate.resolve().relative_to(source_parent).as_posix())
            )
            break
    for relative in dict.fromkeys(relatives):
        original = _resolve_below(
            source_parent,
            relative,
            "shape linked source",
        )
        target = input_root / Path(*relative.parts)
        _snapshot_regular_file(original, target, "shape linked source")
    try:
        private_closure = _source_closure(staged_source)
    except (OSError, UnicodeDecodeError, GeometryBackendError) as exc:
        raise DeterministicAssetError(
            f"private shape source closure is invalid for {source}: {exc}"
        ) from exc
    expected = {staged_source.resolve()}
    expected.update((input_root / Path(*item.parts)).resolve() for item in relatives)
    if set(path.resolve() for path in private_closure) != expected:
        raise DeterministicAssetError(
            "private shape source closure changed after snapshot"
        )
    return staged_source, None


def _snapshot_regular_file(source: Path, target: Path, context: str) -> None:
    resolved = source.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise DeterministicAssetError(f"{context} is not a regular file: {resolved}")
    size = resolved.stat().st_size
    if size <= 0:
        raise DeterministicAssetError(f"{context} must not be empty: {resolved}")
    payload = resolved.read_bytes()
    if len(payload) != size:
        raise DeterministicAssetError(f"{context} changed while being snapshotted")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)


def _native_backend_identity(source: Path) -> dict[str, str]:
    suffix = source.suffix.casefold().lstrip(".")
    return {"id": f"contraption-native-{suffix}", "version": "1"}


def stage_plan(
    component_information: str | Path,
    destination: str | Path,
    *,
    tessellator: Tessellator | None = None,
    source_directory: str | Path | None = None,
) -> Path | None:
    plan = load_plan(
        component_information, source_directory=source_directory
    )
    if not plan.shapes and not plan.optical_sensors and not plan.documents:
        return None
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    staged: list[dict[str, Any]] = []
    for index, item in enumerate(plan.shapes):
        entry = dict(item)
        source = Path(item["source"])
        source_root = root / f"shape-{index:03d}"
        source_root.mkdir()
        input_root = source_root / "input"
        staged_source, archive_evidence = _stage_shape_source(
            source,
            input_root,
            item.get("archive_member"),
        )
        suffix = staged_source.suffix.casefold()
        external_suffixes = {
            ".step",
            ".stp",
            ".iges",
            ".igs",
            ".brep",
            ".fcstd",
            ".gltf",
            ".glb",
            ".wrl",
            ".vrml",
        }
        selected_backend: Tessellator | None = None
        if suffix in external_suffixes:
            selected_backend = tessellator or automatic_tessellator(staged_source)
            if selected_backend is None:
                raise DeterministicAssetError(
                    missing_backend_message(staged_source)
                    + "; conversion is required before Luna dispatch"
                )
            identity = backend_identity(selected_backend)
        elif suffix == ".ply":
            selected_backend = automatic_tessellator(staged_source)
            if selected_backend is None:
                raise DeterministicAssetError("native PLY backend is unavailable")
            identity = backend_identity(selected_backend)
        else:
            identity = _native_backend_identity(staged_source)
        prepared_root = source_root / "prepared"
        try:
            result = import_shape(
                staged_source,
                prepared_root,
                artifact_id=item["artifact_id"],
                version=item["version"],
                metres_per_source_unit=item["metres_per_source_unit"],
                density_kg_m3=item.get("density_kg_m3"),
                tessellator=selected_backend,
                surface_uncertainty=(
                    None
                    if item.get("surface_uncertainty") is None
                    else ShapeUncertainty.from_dict(item["surface_uncertainty"])
                ),
                provenance={
                    "kind": "deterministic-preflight",
                    "backend": identity,
                    "archive": archive_evidence,
                    "declared": dict(item.get("provenance", {})),
                },
            )
        except (OSError, ShapeImportError, GeometryBackendError) as exc:
            raise DeterministicAssetError(
                f"deterministic shape conversion failed before Luna for {source}: {exc}"
            ) from exc
        ShapeArtifact.load(result.manifest_path)
        entry["source"] = staged_source.relative_to(root).as_posix()
        entry["input_root"] = input_root.relative_to(root).as_posix()
        entry["input_sha256"] = _tree_hashes(input_root)
        entry["prepared_root"] = prepared_root.relative_to(root).as_posix()
        entry["prepared_sha256"] = _tree_hashes(prepared_root)
        entry["backend"] = identity
        if archive_evidence is not None:
            entry["archive"] = archive_evidence
        staged.append(entry)
    staged_sensors: list[dict[str, Any]] = []
    for index, item in enumerate(plan.optical_sensors):
        entry = dict(item)
        source = Path(item["source"])
        source_root = root / f"optical-sensor-{index:03d}"
        source_root.mkdir()
        (source_root / source.name).write_bytes(source.read_bytes())
        entry["source"] = (Path(f"optical-sensor-{index:03d}") / source.name).as_posix()
        entry["input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        staged_sensors.append(entry)
    staged_documents: list[dict[str, Any]] = []
    from .document_ingestion import (
        DeterministicDocumentError,
        canonical_extraction_bytes,
        extract_supported_document,
    )
    for index, item in enumerate(plan.documents):
        source = Path(item["source"])
        try:
            extraction = extract_supported_document(
                source, source_format=item["source_format"]
            )
        except DeterministicDocumentError as exc:
            raise DeterministicAssetError(
                f"deterministic document extraction failed for {source}: {exc}"
            ) from exc
        document_root = root / f"document-{index:03d}"
        document_root.mkdir()
        extraction_path = document_root / "extraction.json"
        extraction_payload = canonical_extraction_bytes(extraction)
        extraction_path.write_bytes(extraction_payload)
        source_payload = source.read_bytes()
        source_digest = hashlib.sha256(source_payload).hexdigest()
        if extraction.get("source", {}).get("sha256") != "sha256:" + source_digest:
            raise DeterministicAssetError(
                f"document source changed during deterministic extraction: {source}"
            )
        staged_documents.append(
            {
                "source_name": source.name,
                "source_format": item["source_format"],
                "input_bytes": len(source_payload),
                "input_sha256": source_digest,
                "extraction": (Path(f"document-{index:03d}") / "extraction.json").as_posix(),
                "extraction_format": extraction["format"],
                "extraction_sha256": hashlib.sha256(extraction_payload).hexdigest(),
            }
        )
    plan_path = root / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "format": "deterministic-part-ingestion-staged-1",
                "shapes": staged,
                "optical_sensors": staged_sensors,
                "documents": staged_documents,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_path


def _verified_hash_tree(
    plan_file: Path,
    root_value: Any,
    hashes_value: Any,
    context: str,
) -> Path:
    root = _resolve_below(
        plan_file.parent,
        _relative(root_value, f"{context}.root"),
        f"{context} root",
    )
    if not root.is_dir() or root.is_symlink():
        raise DeterministicAssetError(
            f"{context} root is missing, linked, or not a directory: {root}"
        )
    raw_hashes = _object(hashes_value, f"{context}.sha256")
    if not raw_hashes:
        raise DeterministicAssetError(f"{context}.sha256 must not be empty")
    expected: dict[str, str] = {}
    for raw_name, raw_digest in raw_hashes.items():
        name = _relative(raw_name, f"{context}.sha256 key").as_posix()
        digest = _text(raw_digest, f"{context}.sha256[{name!r}]")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise DeterministicAssetError(f"{context} has an invalid SHA-256 digest")
        if name.casefold() in {item.casefold() for item in expected}:
            raise DeterministicAssetError(
                f"{context} has duplicate case-insensitive paths"
            )
        expected[name] = digest
    actual: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DeterministicAssetError(f"{context} contains a symlink: {path}")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = path
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise DeterministicAssetError(
            f"{context} file closure changed (missing={missing}, extra={extra})"
        )
    for name, path in actual.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise DeterministicAssetError(f"{context} hash mismatch: {path}")
    return root


def _verify_staged_shapes(
    value: Mapping[str, Any],
    plan_file: Path,
) -> tuple[tuple[Mapping[str, Any], Path, ShapeArtifact], ...]:
    raw_shapes = value.get("shapes", [])
    if not isinstance(raw_shapes, list):
        raise DeterministicAssetError("staged shapes must be an array")
    verified: list[tuple[Mapping[str, Any], Path, ShapeArtifact]] = []
    base_fields = {
        "source",
        "catalog_directory",
        "body",
        "solid",
        "artifact_id",
        "version",
        "metres_per_source_unit",
        "density_kg_m3",
        "surface_uncertainty",
        "provenance",
        "input_root",
        "input_sha256",
        "prepared_root",
        "prepared_sha256",
        "backend",
    }
    for index, raw in enumerate(raw_shapes):
        item = _object(raw, f"staged shapes[{index}]")
        _keys(
            item,
            base_fields | {"archive_member", "archive"},
            base_fields,
            f"staged shapes[{index}]",
        )
        input_root = _verified_hash_tree(
            plan_file,
            item["input_root"],
            item["input_sha256"],
            f"staged shapes[{index}] input",
        )
        source = _resolve_below(
            plan_file.parent,
            _relative(item["source"], f"staged shapes[{index}].source"),
            "staged shape source",
        )
        if not source.is_file() or source.is_symlink() or input_root not in source.parents:
            raise DeterministicAssetError(
                f"staged shapes[{index}] source is outside its verified input tree"
            )
        prepared_root = _verified_hash_tree(
            plan_file,
            item["prepared_root"],
            item["prepared_sha256"],
            f"staged shapes[{index}] prepared",
        )
        backend = _object(item["backend"], f"staged shapes[{index}].backend")
        _keys(
            backend,
            {"id", "version", "executable_sha256"},
            {"id", "version"},
            f"staged shapes[{index}].backend",
        )
        _text(backend["id"], f"staged shapes[{index}].backend.id")
        version = _text(backend["version"], f"staged shapes[{index}].backend.version")
        if version in {"unversioned", "system", "unknown"}:
            raise DeterministicAssetError(
                f"staged shapes[{index}] backend version is not reproducible"
            )
        if "executable_sha256" in backend and re.fullmatch(
            r"[0-9a-f]{64}",
            _text(
                backend["executable_sha256"],
                f"staged shapes[{index}].backend.executable_sha256",
            ),
        ) is None:
            raise DeterministicAssetError(
                f"staged shapes[{index}] backend executable digest is invalid"
            )
        if "archive" in item:
            archive = _object(item["archive"], f"staged shapes[{index}].archive")
            _keys(
                archive,
                {"format", "source_name", "bytes", "sha256", "member"},
                {"format", "source_name", "bytes", "sha256", "member"},
                f"staged shapes[{index}].archive",
            )
            member = _relative(
                archive["member"], f"staged shapes[{index}].archive.member"
            )
            if (
                archive["format"] != "zip"
                or member.as_posix() != item.get("archive_member")
            ):
                raise DeterministicAssetError(
                    f"staged shapes[{index}] archive evidence changed"
                )
            raw_archive = _resolve_below(
                input_root,
                _relative(
                    archive["source_name"],
                    f"staged shapes[{index}].archive.source_name",
                ),
                f"staged shapes[{index}] archive source",
            )
            archive_digest = _text(
                archive["sha256"], f"staged shapes[{index}].archive.sha256"
            )
            if (
                not raw_archive.is_file()
                or raw_archive.is_symlink()
                or isinstance(archive["bytes"], bool)
                or not isinstance(archive["bytes"], int)
                or archive["bytes"] <= 0
                or re.fullmatch(r"[0-9a-f]{64}", archive_digest) is None
                or raw_archive.stat().st_size != archive["bytes"]
                or hashlib.sha256(raw_archive.read_bytes()).hexdigest()
                != archive_digest
            ):
                raise DeterministicAssetError(
                    f"staged shapes[{index}] archive source evidence changed"
                )
            selected_archive_source = _resolve_below(
                input_root / "extracted",
                member,
                f"staged shapes[{index}] selected archive member",
            )
            if source != selected_archive_source:
                raise DeterministicAssetError(
                    f"staged shapes[{index}] selected archive member changed"
                )
        elif "archive_member" in item:
            raise DeterministicAssetError(
                f"staged shapes[{index}] lacks declared archive evidence"
            )
        manifest = prepared_root / "shape.artifact.json"
        try:
            artifact = ShapeArtifact.load(manifest)
        except (OSError, ValueError) as exc:
            raise DeterministicAssetError(
                f"staged shapes[{index}] prepared artifact is invalid: {exc}"
            ) from exc
        if artifact.id != item["artifact_id"] or artifact.version != item["version"]:
            raise DeterministicAssetError(
                f"staged shapes[{index}] prepared artifact identity changed"
            )
        expected_provenance = {
            "kind": "deterministic-preflight",
            "backend": dict(backend),
            "archive": item.get("archive"),
            "declared": dict(
                _object(
                    item["provenance"],
                    f"staged shapes[{index}].provenance",
                )
            ),
        }
        if dict(artifact.provenance) != expected_provenance:
            raise DeterministicAssetError(
                f"staged shapes[{index}] prepared artifact provenance changed"
            )
        verified.append((item, prepared_root, artifact))
    return tuple(verified)


def modeling_context_paths(plan_path: str | Path) -> tuple[Path, ...]:
    """Return verified extracted document JSON paths safe to expose to Luna.

    Raw document paths are deliberately absent.  Callers can include these
    normalized JSON records in an inert input block after staging.
    """

    plan_file = Path(plan_path).resolve()
    value = _object(
        loads_strict_json(plan_file.read_text(encoding="utf-8")),
        "staged deterministic plan",
    )
    _keys(
        value,
        {"format", "shapes", "optical_sensors", "documents"},
        {"format"},
        "staged deterministic plan",
    )
    if value["format"] != "deterministic-part-ingestion-staged-1":
        raise DeterministicAssetError("unsupported staged deterministic ingestion format")
    _verify_staged_shapes(value, plan_file)
    paths: list[Path] = []
    for index, raw in enumerate(value.get("documents", [])):
        item = _object(raw, f"staged documents[{index}]")
        _keys(
            item,
            {
                "source_name",
                "source_format",
                "input_bytes",
                "input_sha256",
                "extraction",
                "extraction_format",
                "extraction_sha256",
            },
            {
                "source_name",
                "source_format",
                "input_bytes",
                "input_sha256",
                "extraction",
                "extraction_format",
                "extraction_sha256",
            },
            f"staged documents[{index}]",
        )
        _text(item["source_name"], f"staged documents[{index}].source_name")
        if item["source_format"] not in {"auto", "pdf", "dxf", "kicad", "librepcb"}:
            raise DeterministicAssetError(
                f"staged documents[{index}].source_format is unsupported"
            )
        if (
            isinstance(item["input_bytes"], bool)
            or not isinstance(item["input_bytes"], int)
            or item["input_bytes"] <= 0
        ):
            raise DeterministicAssetError(
                f"staged documents[{index}].input_bytes must be a positive integer"
            )
        extraction = _resolve_below(
            plan_file.parent,
            _relative(item["extraction"], f"staged documents[{index}].extraction"),
            "staged document extraction",
        )
        if not extraction.is_file() or extraction.is_symlink():
            raise DeterministicAssetError(
                f"staged document extraction is missing or not regular: {extraction}"
            )
        expected = _text(
            item["extraction_sha256"],
            f"staged documents[{index}].extraction_sha256",
        )
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise DeterministicAssetError("invalid staged document extraction digest")
        payload = extraction.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise DeterministicAssetError(
                f"staged deterministic document extraction hash mismatch: {extraction}"
            )
        extracted = _object(
            loads_strict_json(payload.decode("utf-8")),
            f"staged documents[{index}] extraction",
        )
        if extracted.get("format") != item["extraction_format"]:
            raise DeterministicAssetError(
                f"staged documents[{index}] extraction format changed"
            )
        source = _object(
            extracted.get("source"),
            f"staged documents[{index}] extraction source",
        )
        expected_input = _text(
            item["input_sha256"], f"staged documents[{index}].input_sha256"
        )
        if not re.fullmatch(r"[0-9a-f]{64}", expected_input):
            raise DeterministicAssetError("invalid staged document input digest")
        if source.get("sha256") != "sha256:" + expected_input:
            raise DeterministicAssetError(
                f"staged documents[{index}] source evidence hash changed"
            )
        if source.get("bytes") != item["input_bytes"]:
            raise DeterministicAssetError(
                f"staged documents[{index}] source evidence length changed"
            )
        paths.append(extraction)
    return tuple(paths)


def bundle_staged_plan(candidate_root: str | Path, plan_path: str | Path) -> tuple[Path, ...]:
    candidate = Path(candidate_root).resolve()
    plan_file = Path(plan_path).resolve()
    value = _object(loads_strict_json(plan_file.read_text(encoding="utf-8")), "staged deterministic plan")
    _keys(
        value,
        {"format", "shapes", "optical_sensors", "documents"},
        {"format"},
        "staged deterministic plan",
    )
    if value["format"] != "deterministic-part-ingestion-staged-1":
        raise DeterministicAssetError("unsupported staged deterministic ingestion format")
    if not value.get("shapes") and not value.get("optical_sensors") and not value.get("documents"):
        raise DeterministicAssetError("staged deterministic ingestion has no assets")
    modeling_context_paths(plan_file)
    verified_shapes = _verify_staged_shapes(value, plan_file)
    written: list[Path] = []
    for index, (item, prepared_root, _prepared_artifact) in enumerate(verified_shapes):
        catalog_directory = _resolve_below(candidate, _relative(item["catalog_directory"], f"staged shapes[{index}].catalog_directory"), "catalog directory")
        static_path = catalog_directory / "static.part"
        if not static_path.is_file():
            raise DeterministicAssetError(f"Luna proposal is missing target static.part: {static_path}")
        shape_root = catalog_directory / "shape" / item["solid"]
        if shape_root.exists():
            raise DeterministicAssetError(
                f"Luna proposal may not supply host-owned shape artifacts: {shape_root}"
            )
        shutil.copytree(prepared_root, shape_root, copy_function=shutil.copyfile)
        if _tree_hashes(shape_root) != dict(item["prepared_sha256"]):
            raise DeterministicAssetError(
                f"copied host-owned shape artifact changed during transfer: {shape_root}"
            )
        manifest_path = shape_root / "shape.artifact.json"
        try:
            artifact = ShapeArtifact.load(manifest_path)
        except (OSError, ValueError) as exc:
            raise DeterministicAssetError(
                f"copied host-owned shape artifact is invalid: {exc}"
            ) from exc
        static = _object(loads_strict_json(static_path.read_text(encoding="utf-8")), "static.part")
        body = next((value for value in static.get("bodies", []) if value.get("id") == item["body"]), None)
        if body is None:
            raise DeterministicAssetError(f"target body {item['body']!r} is missing from {static_path}")
        solid = next((value for value in body.get("solids", []) if value.get("id") == item["solid"]), None)
        if solid is None:
            raise DeterministicAssetError(f"target solid {item['solid']!r} is missing from {static_path}")
        surface = artifact.surface_for("analysis")
        low, high = surface.bounds_m[:3], surface.bounds_m[3:]
        relative_manifest = manifest_path.relative_to(catalog_directory).as_posix()
        solid["geometry"] = {
            "kind": "shape", "dimensions_m": [high[axis] - low[axis] for axis in range(3)],
            "shape_uri": relative_manifest, "shape_sha256": "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "surface_id": surface.id,
        }
        static_path.write_text(json.dumps(static, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
        written.extend(path for path in shape_root.rglob("*") if path.is_file())
        written.append(static_path)
    for index, item in enumerate(value.get("optical_sensors", [])):
        source = _resolve_below(
            plan_file.parent,
            _relative(item["source"], f"staged optical_sensors[{index}].source"),
            "staged optical sensor source",
        )
        expected = _text(item["input_sha256"], f"staged optical_sensors[{index}].input_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise DeterministicAssetError(f"staged deterministic optical sensor hash mismatch: {source}")
        from ..optics import OpticalSchemaError, OpticalSensor
        try:
            descriptor = OpticalSensor.load(source)
        except (OSError, OpticalSchemaError) as exc:
            raise DeterministicAssetError(f"invalid staged optical sensor {source}: {exc}") from exc
        if descriptor.id != item["sensor_id"] or descriptor.mount_connector != item["pose_connector"]:
            raise DeterministicAssetError("staged optical sensor identity or mount changed")
        catalog_directory = _resolve_below(
            candidate,
            _relative(item["catalog_directory"], f"staged optical_sensors[{index}].catalog_directory"),
            "optical sensor catalog directory",
        )
        static_path = catalog_directory / "static.part"
        if not static_path.is_file():
            raise DeterministicAssetError(f"Luna proposal is missing target static.part: {static_path}")
        destination = catalog_directory / "sensor.optical.json"
        if destination.exists():
            raise DeterministicAssetError(f"Luna proposal may not supply host-owned optical sensor descriptor: {destination}")
        payload = source.read_bytes()
        destination.write_bytes(payload)
        static = _object(loads_strict_json(static_path.read_text(encoding="utf-8")), "static.part")
        static["optical_sensors"] = [
            {
                "id": descriptor.id,
                "body": item["body"],
                "pose_connector": item["pose_connector"],
                "artifact_port": item["artifact_port"],
                "descriptor_uri": "sensor.optical.json",
                "descriptor_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        ]
        static_path.write_text(json.dumps(static, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
        written.extend((destination, static_path))
    return tuple(sorted(set(written)))


def proposal_shape_receipt_path(candidate_root: str | Path) -> Path:
    """Return the protected receipt beside the modeling run, not its candidate."""

    candidate = Path(candidate_root).resolve()
    return candidate.parent.parent / HOST_SHAPE_RECEIPT_FILENAME


def build_proposal_shape_receipt(
    candidate_root: str | Path,
    plan_path: str | Path,
    host_artifacts: Iterable[str | Path],
) -> dict[str, Any] | None:
    """Bind the exact host-created shape trees without consulting their manifests."""

    candidate = Path(candidate_root).resolve()
    receipt_path = proposal_shape_receipt_path(candidate)
    receipt_root = receipt_path.parent.resolve()
    plan = Path(plan_path).resolve()
    if plan.is_symlink() or not plan.is_file():
        raise DeterministicAssetError(f"deterministic plan is not regular: {plan}")
    try:
        plan_relative = plan.relative_to(receipt_root).as_posix()
    except ValueError as exc:
        raise DeterministicAssetError(
            "deterministic plan must remain below the protected receipt root"
        ) from exc
    _relative(plan_relative, "shape receipt plan path")

    supplied: set[Path] = set()
    for raw in host_artifacts:
        path = Path(raw)
        if not path.is_absolute():
            path = candidate / path
        resolved = path.resolve()
        if resolved != candidate and candidate not in resolved.parents:
            raise DeterministicAssetError(
                f"host shape artifact escapes candidate root: {raw}"
            )
        if resolved.is_symlink() or not resolved.is_file():
            raise DeterministicAssetError(
                f"host shape artifact is not a regular file: {resolved}"
            )
        supplied.add(resolved)

    manifests = sorted(
        path for path in supplied if path.name == "shape.artifact.json"
    )
    if not manifests:
        return None
    roots = tuple(sorted({manifest.parent.resolve() for manifest in manifests}))
    for index, root in enumerate(roots):
        if root == candidate or candidate not in root.parents:
            raise DeterministicAssetError(
                f"host shape root escapes candidate: {root}"
            )
        if root.is_symlink() or not root.is_dir():
            raise DeterministicAssetError(f"host shape root is unsafe: {root}")
        for other in roots[index + 1 :]:
            if root in other.parents or other in root.parents:
                raise DeterministicAssetError("host shape roots may not overlap")

    artifacts: list[dict[str, Any]] = []
    for root in roots:
        actual: list[Path] = []
        for path in root.rglob("*"):
            if path.is_symlink():
                raise DeterministicAssetError(
                    f"host shape tree contains a symlink: {path}"
                )
            if path.is_file():
                actual.append(path.resolve())
        if root / "shape.artifact.json" not in actual:
            raise DeterministicAssetError(
                f"host shape root lacks its fixed manifest: {root}"
            )
        if any(path not in supplied for path in actual):
            raise DeterministicAssetError(
                f"host shape receipt input omits a created tree file: {root}"
            )
        for path in sorted(actual):
            payload = path.read_bytes()
            artifacts.append(
                {
                    "path": path.relative_to(candidate).as_posix(),
                    "bytes": len(payload),
                    "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                }
            )
    artifact_paths = [item["path"].casefold() for item in artifacts]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise DeterministicAssetError(
            "host shape receipt paths must be unique case-insensitively"
        )
    plan_payload = plan.read_bytes()
    return {
        "format": HOST_SHAPE_RECEIPT_FORMAT,
        "plan": {
            "path": plan_relative,
            "bytes": len(plan_payload),
            "sha256": "sha256:" + hashlib.sha256(plan_payload).hexdigest(),
        },
        "roots": [root.relative_to(candidate).as_posix() for root in roots],
        "artifacts": artifacts,
    }


def verify_proposal_shape_receipt(
    candidate_root: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return only paths authorized by a protected exact host shape receipt."""

    candidate = Path(candidate_root).resolve()
    receipt = (
        proposal_shape_receipt_path(candidate)
        if receipt_path is None
        else Path(receipt_path).resolve()
    )
    manifests = tuple(sorted(candidate.rglob("shape.artifact.json")))
    if not receipt.exists():
        if manifests:
            raise DeterministicAssetError(
                "shape artifacts require a protected host shape receipt"
            )
        return ()
    if receipt.is_symlink() or not receipt.is_file():
        raise DeterministicAssetError(
            f"host shape receipt is not a regular file: {receipt}"
        )
    if receipt == candidate or candidate in receipt.parents:
        raise DeterministicAssetError("host shape receipt may not be inside the candidate")
    value = _object(
        loads_strict_json(receipt.read_text(encoding="utf-8")),
        "host shape receipt",
    )
    _keys(
        value,
        {"format", "plan", "roots", "artifacts"},
        {"format", "plan", "roots", "artifacts"},
        "host shape receipt",
    )
    if value["format"] != HOST_SHAPE_RECEIPT_FORMAT:
        raise DeterministicAssetError("unsupported host shape receipt format")

    raw_plan = _object(value["plan"], "host shape receipt plan")
    _keys(
        raw_plan,
        {"path", "bytes", "sha256"},
        {"path", "bytes", "sha256"},
        "host shape receipt plan",
    )
    plan = _resolve_below(
        receipt.parent.resolve(),
        _relative(raw_plan["path"], "host shape receipt plan.path"),
        "host shape receipt plan",
    )
    plan_bytes = raw_plan["bytes"]
    plan_sha = _text(raw_plan["sha256"], "host shape receipt plan.sha256")
    if (
        plan.is_symlink()
        or not plan.is_file()
        or isinstance(plan_bytes, bool)
        or not isinstance(plan_bytes, int)
        or plan_bytes <= 0
        or re.fullmatch(r"sha256:[0-9a-f]{64}", plan_sha) is None
    ):
        raise DeterministicAssetError("host shape receipt plan evidence is invalid")
    plan_payload = plan.read_bytes()
    if (
        len(plan_payload) != plan_bytes
        or "sha256:" + hashlib.sha256(plan_payload).hexdigest() != plan_sha
    ):
        raise DeterministicAssetError("host shape receipt plan hash changed")

    raw_roots = value["roots"]
    if not isinstance(raw_roots, list) or not raw_roots:
        raise DeterministicAssetError("host shape receipt roots must be nonempty")
    roots: list[Path] = []
    root_names: set[str] = set()
    for index, raw_root in enumerate(raw_roots):
        relative = _relative(raw_root, f"host shape receipt roots[{index}]")
        name = relative.as_posix()
        if name.casefold() in root_names:
            raise DeterministicAssetError(
                "host shape receipt roots must be unique case-insensitively"
            )
        root_names.add(name.casefold())
        root = _resolve_below(candidate, relative, "host shape receipt root")
        if root == candidate or root.is_symlink() or not root.is_dir():
            raise DeterministicAssetError(f"host shape receipt root is unsafe: {root}")
        roots.append(root)
    if [path.relative_to(candidate).as_posix() for path in roots] != sorted(
        path.relative_to(candidate).as_posix() for path in roots
    ):
        raise DeterministicAssetError("host shape receipt roots must be sorted")
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if root in other.parents or other in root.parents:
                raise DeterministicAssetError("host shape receipt roots may not overlap")

    raw_artifacts = value["artifacts"]
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise DeterministicAssetError("host shape receipt artifacts must be nonempty")
    expected: dict[str, tuple[Path, int, str]] = {}
    for index, raw_artifact in enumerate(raw_artifacts):
        artifact = _object(raw_artifact, f"host shape receipt artifacts[{index}]")
        _keys(
            artifact,
            {"path", "bytes", "sha256"},
            {"path", "bytes", "sha256"},
            f"host shape receipt artifacts[{index}]",
        )
        relative = _relative(
            artifact["path"], f"host shape receipt artifacts[{index}].path"
        )
        name = relative.as_posix()
        if name.casefold() in {item.casefold() for item in expected}:
            raise DeterministicAssetError(
                "host shape receipt artifact paths must be unique case-insensitively"
            )
        path = _resolve_below(candidate, relative, "host shape receipt artifact")
        if not any(root == path.parent or root in path.parents for root in roots):
            raise DeterministicAssetError(
                f"host shape receipt artifact is outside its roots: {path}"
            )
        byte_length = artifact["bytes"]
        digest = _text(
            artifact["sha256"], f"host shape receipt artifacts[{index}].sha256"
        )
        if (
            isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or byte_length < 0
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        ):
            raise DeterministicAssetError(
                f"host shape receipt artifact evidence is invalid: {name}"
            )
        expected[name] = (path, byte_length, digest)
    if list(expected) != sorted(expected):
        raise DeterministicAssetError("host shape receipt artifacts must be sorted")

    actual: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise DeterministicAssetError(
                    f"host shape receipt tree contains a symlink: {path}"
                )
            if path.is_file():
                name = path.relative_to(candidate).as_posix()
                actual[name] = path.resolve()
    if set(actual) != set(expected):
        raise DeterministicAssetError(
            "host shape receipt artifact set changed "
            f"(missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))})"
        )
    expected_manifests = {root / "shape.artifact.json" for root in roots}
    if set(path.resolve() for path in manifests) != expected_manifests:
        raise DeterministicAssetError("host shape receipt manifest set changed")
    for name, (path, byte_length, digest) in expected.items():
        if path.is_symlink() or not path.is_file():
            raise DeterministicAssetError(
                f"host shape receipt artifact is not regular: {path}"
            )
        payload = path.read_bytes()
        if (
            len(payload) != byte_length
            or "sha256:" + hashlib.sha256(payload).hexdigest() != digest
        ):
            raise DeterministicAssetError(
                f"host shape receipt artifact hash changed: {name}"
            )
    from ..shape.artifacts import ShapeArtifact

    for manifest in sorted(expected_manifests):
        ShapeArtifact.load(manifest)
    return tuple(expected[name][0] for name in sorted(expected))


__all__ = [
    "DeterministicAssetError",
    "DeterministicPlan",
    "HOST_SHAPE_RECEIPT_FILENAME",
    "HOST_SHAPE_RECEIPT_FORMAT",
    "build_proposal_shape_receipt",
    "bundle_staged_plan",
    "input_paths",
    "load_plan",
    "modeling_context_paths",
    "proposal_shape_receipt_path",
    "stage_plan",
    "verify_proposal_shape_receipt",
]
