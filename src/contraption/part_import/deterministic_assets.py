"""Host-owned deterministic asset ingestion bundled with Luna proposals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from ..shape import ShapeUncertainty, import_shape
from ..strict_json import loads_strict_json


class DeterministicAssetError(ValueError):
    """Raised when a declared host-side ingestion plan is unsafe or incomplete."""


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class DeterministicPlan:
    shapes: tuple[dict[str, Any], ...] = ()
    optical_sensors: tuple[dict[str, Any], ...] = ()


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
                if linked.parent != source.parent.resolve():
                    raise DeterministicAssetError(f"{source}:{line_number}: mtllib must remain in the source directory")
                if not linked.is_file():
                    raise FileNotFoundError(linked)
                values.append(linked)
    for candidate in (source.with_suffix(source.suffix + ".optical.json"), source.with_suffix(".optical.json")):
        if candidate.is_file():
            values.append(candidate.resolve())
            break
    return tuple(dict.fromkeys(values))


def load_plan(component_information: str | Path) -> DeterministicPlan:
    component = Path(component_information).resolve()
    value = _object(loads_strict_json(component.read_text(encoding="utf-8")), "component information")
    raw_plan = value.get("deterministic_ingestion")
    if raw_plan is None:
        return DeterministicPlan()
    plan = _object(raw_plan, "deterministic_ingestion")
    _keys(plan, {"format", "shapes", "optical_sensors"}, {"format"}, "deterministic_ingestion")
    if plan["format"] != "deterministic-part-ingestion-1":
        raise DeterministicAssetError("deterministic_ingestion.format must be deterministic-part-ingestion-1")
    raw_shapes = plan.get("shapes", [])
    raw_sensors = plan.get("optical_sensors", [])
    if not isinstance(raw_shapes, list) or not isinstance(raw_sensors, list) or not (raw_shapes or raw_sensors):
        raise DeterministicAssetError("deterministic_ingestion requires a nonempty shapes or optical_sensors array")
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
        }
        required = {"source", "catalog_directory", "body", "solid", "artifact_id", "metres_per_source_unit"}
        _keys(item, names, required, f"deterministic_ingestion.shapes[{index}]")
        source_relative = _relative(item["source"], f"shapes[{index}].source")
        source = _resolve_below(component.parent, source_relative, f"shapes[{index}].source")
        if not source.is_file() or source.is_symlink():
            raise DeterministicAssetError(f"shape source is missing or not regular: {source}")
        scale = item["metres_per_source_unit"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) <= 0:
            raise DeterministicAssetError(f"shapes[{index}].metres_per_source_unit must be positive")
        density = item.get("density_kg_m3")
        if density is not None and (isinstance(density, bool) or not isinstance(density, (int, float)) or not math.isfinite(float(density)) or float(density) <= 0):
            raise DeterministicAssetError(f"shapes[{index}].density_kg_m3 must be positive")
        shapes.append(
            {
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
        )
    keys = [(item["catalog_directory"].casefold(), item["body"].casefold(), item["solid"].casefold()) for item in shapes]
    if len(keys) != len(set(keys)):
        raise DeterministicAssetError("deterministic shape targets must be unique")
    sensors: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_sensors):
        item = _object(raw, f"deterministic_ingestion.optical_sensors[{index}]")
        names = {"source", "catalog_directory", "body", "pose_connector", "artifact_port"}
        _keys(item, names, names, f"deterministic_ingestion.optical_sensors[{index}]")
        source_relative = _relative(item["source"], f"optical_sensors[{index}].source")
        source = _resolve_below(component.parent, source_relative, f"optical_sensors[{index}].source")
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
    return DeterministicPlan(tuple(shapes), tuple(sensors))


def input_paths(component_information: str | Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    plan = load_plan(component_information)
    for item in plan.shapes:
        paths.extend(_source_closure(Path(item["source"])))
    paths.extend(Path(item["source"]) for item in plan.optical_sensors)
    return tuple(dict.fromkeys(paths))


def stage_plan(component_information: str | Path, destination: str | Path) -> Path | None:
    plan = load_plan(component_information)
    if not plan.shapes and not plan.optical_sensors:
        return None
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    staged: list[dict[str, Any]] = []
    for index, item in enumerate(plan.shapes):
        entry = dict(item)
        source = Path(item["source"])
        closure = _source_closure(source)
        source_root = root / f"shape-{index:03d}"
        source_root.mkdir()
        for path in closure:
            target = source_root / path.name
            target.write_bytes(path.read_bytes())
        entry["source"] = (Path(f"shape-{index:03d}") / source.name).as_posix()
        entry["input_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in closure}
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
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps({"format": "deterministic-part-ingestion-staged-1", "shapes": staged, "optical_sensors": staged_sensors}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path


def bundle_staged_plan(candidate_root: str | Path, plan_path: str | Path) -> tuple[Path, ...]:
    candidate = Path(candidate_root).resolve()
    plan_file = Path(plan_path).resolve()
    value = _object(loads_strict_json(plan_file.read_text(encoding="utf-8")), "staged deterministic plan")
    _keys(value, {"format", "shapes", "optical_sensors"}, {"format"}, "staged deterministic plan")
    if value["format"] != "deterministic-part-ingestion-staged-1":
        raise DeterministicAssetError("unsupported staged deterministic ingestion format")
    if not value.get("shapes") and not value.get("optical_sensors"):
        raise DeterministicAssetError("staged deterministic ingestion has no assets")
    written: list[Path] = []
    for index, item in enumerate(value.get("shapes", [])):
        source = _resolve_below(plan_file.parent, _relative(item["source"], f"staged shapes[{index}].source"), "staged source")
        expected_hashes = _object(item["input_sha256"], f"staged shapes[{index}].input_sha256")
        for name, digest in expected_hashes.items():
            path = source.parent / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise DeterministicAssetError(f"staged deterministic source hash mismatch: {path}")
        catalog_directory = _resolve_below(candidate, _relative(item["catalog_directory"], f"staged shapes[{index}].catalog_directory"), "catalog directory")
        static_path = catalog_directory / "static.part"
        if not static_path.is_file():
            raise DeterministicAssetError(f"Luna proposal is missing target static.part: {static_path}")
        shape_root = catalog_directory / "shape" / item["solid"]
        result = import_shape(
            source, shape_root, artifact_id=item["artifact_id"], version=item["version"],
            metres_per_source_unit=item["metres_per_source_unit"], density_kg_m3=item.get("density_kg_m3"),
            surface_uncertainty=(
                None
                if item.get("surface_uncertainty") is None
                else ShapeUncertainty.from_dict(item["surface_uncertainty"])
            ),
            provenance={"kind": "deterministic-luna-bundle", **dict(item.get("provenance", {}))},
        )
        static = _object(loads_strict_json(static_path.read_text(encoding="utf-8")), "static.part")
        body = next((value for value in static.get("bodies", []) if value.get("id") == item["body"]), None)
        if body is None:
            raise DeterministicAssetError(f"target body {item['body']!r} is missing from {static_path}")
        solid = next((value for value in body.get("solids", []) if value.get("id") == item["solid"]), None)
        if solid is None:
            raise DeterministicAssetError(f"target solid {item['solid']!r} is missing from {static_path}")
        surface = result.artifact.surface_for("analysis")
        low, high = surface.bounds_m[:3], surface.bounds_m[3:]
        relative_manifest = result.manifest_path.relative_to(catalog_directory).as_posix()
        solid["geometry"] = {
            "kind": "shape", "dimensions_m": [high[axis] - low[axis] for axis in range(3)],
            "shape_uri": relative_manifest, "shape_sha256": "sha256:" + hashlib.sha256(result.manifest_path.read_bytes()).hexdigest(),
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


__all__ = ["DeterministicAssetError", "DeterministicPlan", "bundle_staged_plan", "input_paths", "load_plan", "stage_plan"]
