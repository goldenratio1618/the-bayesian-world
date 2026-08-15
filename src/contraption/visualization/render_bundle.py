"""Strict, source-independent assets for the offline assembly viewer.

The physical scene owns bodies, poses, connectors, and source provenance.  A
render bundle owns only exact display representations derived from canonical
shape and optical-observation artifacts.  It is deliberately complete: when a
bundle is present every physical solid has exactly one surface binding, and
every payload is content-addressed.  The viewer never follows source-file URIs
or invents geometry for a missing binding.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
import struct
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from ..shape.artifacts import OpticalMaterial, ShapeArtifact
    from ..shape.mesh import TriangleMesh


RENDER_BUNDLE_SCHEMA = "contraption.render-bundle/v1"
TRIANGLE_SURFACE_SCHEMA = "contraption.triangle-surface/v1"

_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")
_LAYER_MODES = {"rgb", "depth", "segmentation", "uncertainty", "reconstruction"}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_INLINE_RASTER_BYTES = 64 * 1024 * 1024


class RenderBundleError(ValueError):
    """Raised when a render asset is incomplete, stale, or ambiguous."""


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RenderBundleError(f"{label} must be an object with string keys")
    return value


def _list(value: Any, label: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise RenderBundleError(f"{label} must be an array")
    if nonempty and not value:
        raise RenderBundleError(f"{label} must not be empty")
    return value


def _keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    label: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise RenderBundleError(f"{label} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise RenderBundleError(
            f"{label} contains unsupported fields that would be ignored: {', '.join(unknown)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenderBundleError(f"{label} must be a non-empty string")
    return value


def _identifier(value: Any, label: str) -> str:
    result = _text(value, label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise RenderBundleError(f"{label} must be a canonical identifier")
    return result


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise RenderBundleError(
            f"{label} must use canonical form 'sha256:' followed by 64 lowercase hex digits"
        )
    return value


def _number(
    value: Any,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderBundleError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RenderBundleError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise RenderBundleError(f"{label} must be greater than zero")
    if nonnegative and result < 0.0:
        raise RenderBundleError(f"{label} must be non-negative")
    return result


def _integer(value: Any, label: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RenderBundleError(f"{label} must be an integer")
    if nonnegative and value < 0:
        raise RenderBundleError(f"{label} must be non-negative")
    return value


def _vector(value: Any, label: str, length: int) -> list[float]:
    items = _list(value, label)
    if len(items) != length:
        raise RenderBundleError(f"{label} must contain exactly {length} numbers")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(items)]


def _pose(value: Any, label: str) -> dict[str, Any]:
    pose = _object(value, label)
    _keys(
        pose,
        required={"translation_m", "rotation_quaternion_wxyz"},
        label=label,
    )
    quaternion = _vector(
        pose["rotation_quaternion_wxyz"],
        f"{label}.rotation_quaternion_wxyz",
        4,
    )
    norm = math.sqrt(sum(item * item for item in quaternion))
    if abs(norm - 1.0) > 1e-9:
        raise RenderBundleError(
            f"{label}.rotation_quaternion_wxyz must be normalized (norm={norm:.12g})"
        )
    first_nonzero = next((item for item in quaternion if abs(item) > 1e-15), 0.0)
    if first_nonzero < 0.0:
        raise RenderBundleError(
            f"{label}.rotation_quaternion_wxyz must use the canonical quaternion sign"
        )
    return {
        "translation_m": _vector(pose["translation_m"], f"{label}.translation_m", 3),
        "rotation_quaternion_wxyz": quaternion,
    }


def content_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical digest for a mapping whose ``sha256`` is omitted."""

    payload = dict(value)
    payload.pop("sha256", None)
    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RenderBundleError(f"content is not canonical JSON: {exc}") from exc
    return "sha256:" + hashlib.sha256(serialized).hexdigest()


def _material(value: Any, label: str) -> dict[str, Any]:
    material = _object(value, label)
    _keys(
        material,
        required={"id", "base_color_linear_rgba", "optical_material_sha256"},
        label=label,
    )
    color = _vector(
        material["base_color_linear_rgba"], f"{label}.base_color_linear_rgba", 4
    )
    if any(item < 0.0 or item > 1.0 for item in color):
        raise RenderBundleError(f"{label}.base_color_linear_rgba values must be in [0, 1]")
    optical_hash = material["optical_material_sha256"]
    if optical_hash is not None:
        optical_hash = _hash(optical_hash, f"{label}.optical_material_sha256")
    return {
        "id": _text(material["id"], f"{label}.id"),
        "base_color_linear_rgba": color,
        "optical_material_sha256": optical_hash,
    }


def _surface(value: Any, label: str) -> dict[str, Any]:
    surface = _object(value, label)
    _keys(
        surface,
        required={
            "schema",
            "sha256",
            "shape_manifest_sha256",
            "shape_artifact_sha256",
            "shape_id",
            "surface_id",
            "source_surface_sha256",
            "vertices_m",
            "triangles",
            "vertex_normals",
            "vertex_rgba_linear",
            "materials",
            "triangle_materials",
            "vertex_uncertainty_m",
        },
        label=label,
    )
    if surface["schema"] != TRIANGLE_SURFACE_SCHEMA:
        raise RenderBundleError(
            f"{label}.schema must be {TRIANGLE_SURFACE_SCHEMA!r}"
        )
    vertices = [
        _vector(vertex, f"{label}.vertices_m[{index}]", 3)
        for index, vertex in enumerate(_list(surface["vertices_m"], f"{label}.vertices_m", nonempty=True))
    ]
    if len(vertices) < 3:
        raise RenderBundleError(f"{label}.vertices_m must contain at least three vertices")
    triangles: list[list[int]] = []
    for index, raw_triangle in enumerate(
        _list(surface["triangles"], f"{label}.triangles", nonempty=True)
    ):
        triangle_label = f"{label}.triangles[{index}]"
        raw_indices = _list(raw_triangle, triangle_label)
        if len(raw_indices) != 3:
            raise RenderBundleError(f"{triangle_label} must contain exactly three indices")
        indices = [
            _integer(item, f"{triangle_label}[{offset}]", nonnegative=True)
            for offset, item in enumerate(raw_indices)
        ]
        if len(set(indices)) != 3 or max(indices) >= len(vertices):
            raise RenderBundleError(
                f"{triangle_label} must contain three distinct in-range vertex indices"
            )
        first, second, third = (vertices[item] for item in indices)
        edge_a = [second[axis] - first[axis] for axis in range(3)]
        edge_b = [third[axis] - first[axis] for axis in range(3)]
        cross = [
            edge_a[1] * edge_b[2] - edge_a[2] * edge_b[1],
            edge_a[2] * edge_b[0] - edge_a[0] * edge_b[2],
            edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0],
        ]
        if sum(item * item for item in cross) <= 1e-30:
            raise RenderBundleError(f"{triangle_label} is geometrically degenerate")
        triangles.append(indices)

    normals: list[list[float]] | None = None
    if surface["vertex_normals"] is not None:
        raw_normals = _list(surface["vertex_normals"], f"{label}.vertex_normals")
        if len(raw_normals) != len(vertices):
            raise RenderBundleError(
                f"{label}.vertex_normals must match vertices_m in length"
            )
        normals = []
        for index, raw_normal in enumerate(raw_normals):
            normal_label = f"{label}.vertex_normals[{index}]"
            normal = _vector(raw_normal, normal_label, 3)
            norm = math.sqrt(sum(item * item for item in normal))
            if abs(norm - 1.0) > 1e-6:
                raise RenderBundleError(f"{normal_label} must be normalized")
            normals.append(normal)

    materials = [
        _material(item, f"{label}.materials[{index}]")
        for index, item in enumerate(_list(surface["materials"], f"{label}.materials", nonempty=True))
    ]
    material_ids = [item["id"] for item in materials]
    if len(set(material_ids)) != len(material_ids):
        raise RenderBundleError(f"{label}.materials contains duplicate ids")
    triangle_materials = _list(
        surface["triangle_materials"], f"{label}.triangle_materials"
    )
    if len(triangle_materials) != len(triangles):
        raise RenderBundleError(
            f"{label}.triangle_materials must match triangles in length"
        )
    normalized_triangle_materials = [
        _integer(item, f"{label}.triangle_materials[{index}]", nonnegative=True)
        for index, item in enumerate(triangle_materials)
    ]
    if any(item >= len(materials) for item in normalized_triangle_materials):
        raise RenderBundleError(f"{label}.triangle_materials references an unknown material")

    uncertainty: list[float] | None = None
    if surface["vertex_uncertainty_m"] is not None:
        raw_uncertainty = _list(
            surface["vertex_uncertainty_m"], f"{label}.vertex_uncertainty_m"
        )
        if len(raw_uncertainty) != len(vertices):
            raise RenderBundleError(
                f"{label}.vertex_uncertainty_m must match vertices_m in length"
            )
        uncertainty = [
            _number(item, f"{label}.vertex_uncertainty_m[{index}]", nonnegative=True)
            for index, item in enumerate(raw_uncertainty)
        ]

    normalized = {
        "schema": TRIANGLE_SURFACE_SCHEMA,
        "sha256": _hash(surface["sha256"], f"{label}.sha256"),
        "shape_manifest_sha256": _hash(
            surface["shape_manifest_sha256"], f"{label}.shape_manifest_sha256"
        ),
        "shape_artifact_sha256": _hash(
            surface["shape_artifact_sha256"], f"{label}.shape_artifact_sha256"
        ),
        "shape_id": _text(surface["shape_id"], f"{label}.shape_id"),
        "surface_id": _text(surface["surface_id"], f"{label}.surface_id"),
        "source_surface_sha256": _hash(
            surface["source_surface_sha256"], f"{label}.source_surface_sha256"
        ),
        "vertices_m": vertices,
        "triangles": triangles,
        "vertex_normals": normals,
        "vertex_rgba_linear": None,
        "materials": materials,
        "triangle_materials": normalized_triangle_materials,
        "vertex_uncertainty_m": uncertainty,
    }
    if surface["vertex_rgba_linear"] is not None:
        raw_colors = _list(surface["vertex_rgba_linear"], f"{label}.vertex_rgba_linear")
        if len(raw_colors) != len(vertices):
            raise RenderBundleError(
                f"{label}.vertex_rgba_linear must match vertices_m in length"
            )
        colors = [
            _vector(item, f"{label}.vertex_rgba_linear[{index}]", 4)
            for index, item in enumerate(raw_colors)
        ]
        if any(channel < 0.0 or channel > 1.0 for color in colors for channel in color):
            raise RenderBundleError(f"{label}.vertex_rgba_linear values must be in [0, 1]")
        normalized["vertex_rgba_linear"] = colors
    expected = content_sha256(normalized)
    if normalized["sha256"] != expected:
        raise RenderBundleError(
            f"{label}.sha256 is stale or incorrect; expected {expected}"
        )
    return normalized


def _surface_dimensions(surface: Mapping[str, Any]) -> list[float]:
    vertices = surface["vertices_m"]
    return [
        max(vertex[axis] for vertex in vertices) - min(vertex[axis] for vertex in vertices)
        for axis in range(3)
    ]


def _prefixed_digest(value: str, label: str) -> str:
    return _hash(f"sha256:{value}", label)


def _render_material(material: "OpticalMaterial") -> dict[str, Any]:
    payload = material.to_dict()
    return {
        "id": material.id,
        "base_color_linear_rgba": list(material.base_color_linear_rgba),
        "optical_material_sha256": content_sha256(payload),
    }


def _verified_ctmesh(
    artifact: "ShapeArtifact",
    surface_id: str,
) -> tuple[Any, "TriangleMesh"]:
    from ..shape.artifacts import ShapeArtifactError
    from ..shape.mesh import MeshError, TriangleMesh
    surface = next((item for item in artifact.surfaces if item.id == surface_id), None)
    if surface is None:
        raise RenderBundleError(
            f"shape {artifact.id!r} does not contain declared surface {surface_id!r}"
        )
    if "render" not in surface.purposes:
        raise RenderBundleError(
            f"shape {artifact.id!r} surface {surface_id!r} is not authored for rendering"
        )
    if surface.kind != "ctmesh":
        raise RenderBundleError(
            f"shape {artifact.id!r} surface {surface_id!r} is not canonical CTMESH"
        )
    try:
        path = artifact.resolve(surface.content)
        payload = path.read_bytes()
    except (OSError, ShapeArtifactError) as exc:
        raise RenderBundleError(
            f"cannot resolve render surface {artifact.id}/{surface.id}: {exc}"
        ) from exc
    if len(payload) != surface.content.byte_length:
        raise RenderBundleError(
            f"render surface content length mismatch for {artifact.id}/{surface.id}"
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != surface.content.sha256:
        raise RenderBundleError(
            f"render surface content hash mismatch for {artifact.id}/{surface.id}"
        )
    try:
        mesh = TriangleMesh.from_bytes(payload)
    except MeshError as exc:
        raise RenderBundleError(
            f"invalid CTMESH render surface {artifact.id}/{surface.id}: {exc}"
        ) from exc
    if len(mesh.vertices_m) != surface.vertex_count or len(mesh.triangles) != surface.triangle_count:
        raise RenderBundleError(
            f"render surface counts do not match manifest for {artifact.id}/{surface.id}"
        )
    low, high = mesh.bounds_m
    actual_bounds = [float(item) for item in (*low, *high)]
    for index, (declared, actual) in enumerate(zip(surface.bounds_m, actual_bounds)):
        if abs(float(declared) - actual) > max(1e-9, abs(float(declared)) * 1e-6):
            raise RenderBundleError(
                f"render surface bound {index} does not match manifest for "
                f"{artifact.id}/{surface.id}"
            )
    return surface, mesh


def _surface_from_artifact(
    artifact: "ShapeArtifact", surface_id: str, shape_manifest_sha256: str
) -> dict[str, Any]:
    from ..shape.artifacts import OpticalMaterial
    surface, mesh = _verified_ctmesh(artifact, surface_id)
    materials_by_id = {item.id: item for item in artifact.optical_materials}
    if surface.material_ids:
        materials = [
            _render_material(materials_by_id[material_id])
            for material_id in surface.material_ids
        ]
    else:
        materials = [
            _render_material(
                OpticalMaterial(
                    id="default",
                    provenance={
                        "kind": "derived",
                        "source": "viewer neutral material for an unassigned canonical surface",
                    },
                )
            )
        ]
    if mesh.face_material is None:
        triangle_materials = [0] * len(mesh.triangles)
    else:
        triangle_materials = mesh.face_material.astype(int).tolist()
        if triangle_materials and max(triangle_materials) >= len(materials):
            raise RenderBundleError(
                f"CTMESH face_material references an absent optical material for "
                f"{artifact.id}/{surface.id}"
            )
    value = {
        "schema": TRIANGLE_SURFACE_SCHEMA,
        "sha256": "sha256:" + "0" * 64,
        "shape_manifest_sha256": _hash(
            shape_manifest_sha256, "shape_manifest_sha256"
        ),
        "shape_artifact_sha256": _prefixed_digest(
            artifact.artifact_sha256, "shape_artifact_sha256"
        ),
        "shape_id": artifact.id,
        "surface_id": surface.id,
        "source_surface_sha256": _prefixed_digest(
            surface.content.sha256, "source_surface_sha256"
        ),
        "vertices_m": mesh.vertices_m.tolist(),
        "triangles": mesh.triangles.astype(int).tolist(),
        "vertex_normals": None
        if mesh.vertex_normals is None
        else mesh.vertex_normals.tolist(),
        "vertex_rgba_linear": None
        if mesh.vertex_rgba_linear is None
        else mesh.vertex_rgba_linear.tolist(),
        "materials": materials,
        "triangle_materials": triangle_materials,
        "vertex_uncertainty_m": None,
    }
    value["sha256"] = content_sha256(value)
    return value


def materialize_render_bundle(
    *,
    assembly_sha256: str,
    scene: Mapping[str, Any],
    solid_shapes: Mapping[str, str | Path | "ShapeArtifact"],
    sensors: Sequence[Mapping[str, Any]] = (),
    observations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Load verified canonical shape surfaces into a portable browser bundle.

    ``solid_shapes`` keys use ``component/body/solid``. Manifest paths are
    loaded without admitting source representations; only the selected
    canonical CTMESH render payload is resolved, length-checked, hash-checked,
    and decoded. Every physical solid must be supplied exactly once.
    """

    from ..shape.artifacts import ShapeArtifact, ShapeArtifactError

    if not isinstance(solid_shapes, Mapping) or any(
        not isinstance(key, str) for key in solid_shapes
    ):
        raise RenderBundleError("solid_shapes must be an object with string keys")
    solid_geometry = {
        f"{component['id']}/{body['id']}/{solid['id']}": solid["geometry"]
        for component in scene["components"]
        for body in component["bodies"]
        for solid in body["solids"]
    }
    declared_solids = set(solid_geometry)
    missing = sorted(declared_solids - set(solid_shapes))
    extra = sorted(set(solid_shapes) - declared_solids)
    if missing or extra:
        raise RenderBundleError(
            f"solid_shapes must exactly match the physical scene; missing={missing}, extra={extra}"
        )
    surfaces: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, str]] = []
    for solid_key in sorted(declared_solids):
        geometry = solid_geometry[solid_key]
        if geometry["kind"] != "shape":
            raise RenderBundleError(
                f"solid {solid_key!r} is {geometry['kind']!r}; canonical shape "
                "materialization requires kind='shape' for every bound solid"
            )
        source = solid_shapes[solid_key]
        try:
            if isinstance(source, ShapeArtifact):
                artifact = source
                manifest_path = getattr(source, "_manifest_path", None)
                if manifest_path is None:
                    raise RenderBundleError(
                        f"loaded shape artifact for solid {solid_key!r} has no manifest path; "
                        "pass its manifest path so the static.part file hash can be verified"
                    )
                manifest_path = Path(manifest_path)
            else:
                manifest_path = Path(source).resolve()
                artifact = ShapeArtifact.load(manifest_path, verify_content=False)
        except (OSError, ShapeArtifactError, TypeError) as exc:
            raise RenderBundleError(
                f"cannot load shape artifact for solid {solid_key!r}: {exc}"
            ) from exc
        expected_manifest_sha256 = geometry["shape_sha256"]
        actual_manifest_sha256 = "sha256:" + hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise RenderBundleError(
                f"shape manifest file digest mismatch for solid {solid_key!r}: "
                f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
            )
        surface = _surface_from_artifact(
            artifact, geometry["surface_id"], actual_manifest_sha256
        )
        surfaces.setdefault(surface["sha256"], surface)
        component, body, solid = solid_key.split("/", 2)
        bindings.append(
            {
                "component": component,
                "body": body,
                "solid": solid,
                "surface_sha256": surface["sha256"],
            }
        )
    bundle = {
        "schema": RENDER_BUNDLE_SCHEMA,
        "sha256": "sha256:" + "0" * 64,
        "assembly_sha256": assembly_sha256,
        "surfaces": surfaces,
        "solid_bindings": bindings,
        "sensors": [dict(item) for item in sensors],
        "observations": [dict(item) for item in observations],
    }
    bundle["sha256"] = content_sha256(bundle)
    return normalize_render_bundle(
        bundle,
        assembly_sha256=assembly_sha256,
        scene=scene,
    )


def shape_artifacts_from_registry(
    *, scene: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Path]:
    """Resolve exact solid shape manifests from catalog instantiation directories.

    ``ResolvedAssembly`` intentionally contains no source directories. Callers that
    have the ``PartInstantiationRegistry`` may use this helper to resolve each
    ``shape_uri`` below its owning ``static.part`` directory. The physical scene's
    artifact and surface hashes remain authoritative and are checked later by
    :func:`materialize_render_bundle`.
    """

    if not isinstance(registry, Mapping):
        raise RenderBundleError("registry must be a PartInstantiation mapping")
    result: dict[str, Path] = {}
    for component in scene["components"]:
        shaped = [
            (body, solid)
            for body in component["bodies"]
            for solid in body["solids"]
            if solid["geometry"]["kind"] == "shape"
        ]
        if not shaped:
            continue
        try:
            instantiation = registry[component["part"]]
        except KeyError as exc:
            raise RenderBundleError(
                f"catalog registry has no instantiation {component['part']!r} used by "
                f"component {component['id']!r}"
            ) from exc
        directory_value = getattr(instantiation, "directory", None)
        if directory_value is None:
            raise RenderBundleError(
                f"catalog instantiation {component['part']!r} has no source directory"
            )
        directory = Path(directory_value).resolve()
        if not directory.is_dir():
            raise RenderBundleError(
                f"catalog instantiation {component['part']!r} has no valid directory"
            )
        for body, solid in shaped:
            uri = solid["geometry"]["shape_uri"]
            relative = Path(*PurePosixPath(uri).parts)
            manifest = (directory / relative).resolve()
            if manifest != directory and directory not in manifest.parents:
                raise RenderBundleError(
                    f"shape_uri for {component['id']}/{body['id']}/{solid['id']} escapes "
                    "its static.part directory"
                )
            if not manifest.is_file():
                raise RenderBundleError(
                    f"shape manifest does not exist for "
                    f"{component['id']}/{body['id']}/{solid['id']}: {manifest}"
                )
            result[f"{component['id']}/{body['id']}/{solid['id']}"] = manifest
    return result


def _projection(value: Any, label: str) -> dict[str, Any]:
    projection = _object(value, label)
    _keys(
        projection,
        required={
            "kind",
            "resolution_px",
            "focal_length_px",
            "principal_point_px",
            "clipping_m",
        },
        label=label,
    )
    if projection["kind"] != "pinhole":
        raise RenderBundleError(f"{label}.kind must be 'pinhole' for spatial POV rendering")
    resolution_raw = _list(projection["resolution_px"], f"{label}.resolution_px")
    if len(resolution_raw) != 2:
        raise RenderBundleError(f"{label}.resolution_px must contain width and height")
    resolution = [
        _integer(item, f"{label}.resolution_px[{index}]", nonnegative=True)
        for index, item in enumerate(resolution_raw)
    ]
    if any(item <= 0 for item in resolution):
        raise RenderBundleError(f"{label}.resolution_px values must be positive")
    focal = _vector(projection["focal_length_px"], f"{label}.focal_length_px", 2)
    if any(item <= 0.0 for item in focal):
        raise RenderBundleError(f"{label}.focal_length_px values must be positive")
    principal = _vector(
        projection["principal_point_px"], f"{label}.principal_point_px", 2
    )
    if not (0.0 <= principal[0] <= resolution[0] and 0.0 <= principal[1] <= resolution[1]):
        raise RenderBundleError(f"{label}.principal_point_px must lie inside the sensor")
    clipping = _vector(projection["clipping_m"], f"{label}.clipping_m", 2)
    if clipping[0] <= 0.0 or clipping[1] <= clipping[0]:
        raise RenderBundleError(f"{label}.clipping_m must be positive and increasing")
    return {
        "kind": "pinhole",
        "resolution_px": resolution,
        "focal_length_px": focal,
        "principal_point_px": principal,
        "clipping_m": clipping,
    }


def _sensor(value: Any, label: str, optical_connectors: set[str]) -> dict[str, Any]:
    sensor = _object(value, label)
    _keys(
        sensor,
        required={"id", "display_name", "connector", "projection", "descriptor_sha256"},
        label=label,
    )
    connector = _text(sensor["connector"], f"{label}.connector")
    if connector not in optical_connectors:
        raise RenderBundleError(
            f"{label}.connector {connector!r} is not a declared spatial optical connector"
        )
    return {
        "id": _identifier(sensor["id"], f"{label}.id"),
        "display_name": _text(sensor["display_name"], f"{label}.display_name"),
        "connector": connector,
        "projection": _projection(sensor["projection"], f"{label}.projection"),
        "descriptor_sha256": _hash(
            sensor["descriptor_sha256"], f"{label}.descriptor_sha256"
        ),
    }


def sensor_view_from_descriptor(
    sensor: Any, *, component_id: str | None = None
) -> dict[str, Any]:
    """Project a strict ``OpticalSensor`` into the viewer's POV declaration."""

    if getattr(sensor, "format", None) != "optical-sensor-1":
        raise RenderBundleError("sensor descriptor must use optical-sensor-1")
    local_connector = getattr(sensor, "mount_connector", None)
    connector = local_connector
    if connector is None:
        raise RenderBundleError(
            f"optical sensor {getattr(sensor, 'id', '<unknown>')!r} has no mount_connector"
        )
    if getattr(sensor, "projection", None) != "pinhole":
        raise RenderBundleError("viewer POV currently requires a pinhole optical sensor")
    viewer_id = str(sensor.id)
    if component_id is not None:
        component = _identifier(component_id, "sensor component_id")
        if "." in str(local_connector):
            raise RenderBundleError(
                "catalog optical sensor mount_connector must be local to its static.part"
            )
        connector = f"{component}.{local_connector}"
        viewer_id = f"{component}.{sensor.id}"
    return {
        "id": viewer_id,
        "display_name": str(sensor.display_name or sensor.id),
        "connector": str(connector),
        "projection": {
            "kind": "pinhole",
            "resolution_px": list(sensor.resolution_px),
            "focal_length_px": list(sensor.focal_length_px),
            "principal_point_px": list(sensor.principal_point_px),
            "clipping_m": [float(sensor.near_clip_m), float(sensor.far_clip_m)],
        },
        "descriptor_sha256": _prefixed_digest(
            str(sensor.artifact_sha256), "sensor.artifact_sha256"
        ),
    }


def optical_sensors_from_registry(
    *,
    scene: Mapping[str, Any],
    registry: Mapping[str, Any],
    component_models: Mapping[str, Any],
) -> list[tuple[str, Any]]:
    """Load only sensors admitted by exact static-part/model bindings.

    The returned tuples contain the component id and the immutable source
    ``OpticalSensor``. Instance qualification is a viewer projection and never
    changes the descriptor or its canonical digest. An unbound descriptor file
    beside a static part is deliberately ignored.
    """

    from ..catalog import validate_optical_sensors
    from ..physics.physical import PhysicalSpecError

    if not isinstance(registry, Mapping):
        raise RenderBundleError("registry must be a PartInstantiation mapping")
    if not isinstance(component_models, Mapping):
        raise RenderBundleError("component_models must map component ids to verified PMDL models")
    result: list[tuple[str, Any]] = []
    for component in scene["components"]:
        try:
            instantiation = registry[component["part"]]
        except KeyError as exc:
            raise RenderBundleError(
                f"catalog registry has no instantiation {component['part']!r} used by "
                f"component {component['id']!r}"
            ) from exc
        if getattr(instantiation, "id", None) != component["part"]:
            raise RenderBundleError(
                f"catalog registry key {component['part']!r} resolves to a different "
                "model instantiation"
            )
        static = getattr(instantiation, "static", None)
        model_instance = getattr(instantiation, "model_instance", None)
        if static is None or model_instance is None:
            raise RenderBundleError(
                f"catalog instantiation {component['part']!r} lacks parsed static/model data"
            )
        bindings = tuple(getattr(static, "optical_sensors", ()))
        if not bindings:
            continue
        directory_value = getattr(instantiation, "directory", None)
        if directory_value is None:
            raise RenderBundleError(
                f"catalog instantiation {component['part']!r} has no source directory"
            )
        try:
            model = component_models[component["id"]]
        except KeyError as exc:
            raise RenderBundleError(
                f"component {component['id']!r} has a static optical sensor binding "
                "but no verified PMDL model"
            ) from exc
        expected_model = model_instance.model
        try:
            model_digest = "sha256:" + hashlib.sha256(
                model.to_json().encode("utf-8")
            ).hexdigest()
            model_id = model.id
            model_version = model.version
        except (AttributeError, TypeError, ValueError) as exc:
            raise RenderBundleError(
                f"component {component['id']!r} does not have a canonical PMDL model"
            ) from exc
        if (
            model_id != expected_model.id
            or model_version != expected_model.version
            or model_digest != expected_model.sha256
            or component.get("model") != model_id
        ):
            raise RenderBundleError(
                f"component {component['id']!r} optical sensor PMDL identity/hash "
                "does not match its resolved instantiation"
            )
        directory = Path(directory_value).resolve()
        try:
            sensors = validate_optical_sensors(static, directory, model)
        except (OSError, PhysicalSpecError) as exc:
            raise RenderBundleError(
                f"invalid optical sensor closure for component {component['id']!r}: {exc}"
            ) from exc

        scene_bodies = {item["id"] for item in component["bodies"]}
        scene_connectors = {
            item["id"]: item
            for item in component["connectors"]
            if item["body"] is not None
        }
        for binding, sensor in zip(bindings, sensors, strict=True):
            if binding.body not in scene_bodies:
                raise RenderBundleError(
                    f"sensor {binding.id!r} body {binding.body!r} is absent from "
                    f"component {component['id']!r}"
                )
            connector = scene_connectors.get(binding.pose_connector)
            if (
                connector is None
                or connector["body"] != binding.body
                or connector["domain"] != "optical"
            ):
                raise RenderBundleError(
                    f"sensor {binding.id!r} pose connector {binding.pose_connector!r} "
                    f"is not an optical connector on body {binding.body!r}"
                )
            result.append((component["id"], sensor))
    return result


def _decode_png(value: Any, label: str, width: int, height: int) -> tuple[str, bytes]:
    encoded = _text(value, label)
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RenderBundleError(f"{label} must be canonical base64") from exc
    if len(decoded) > _MAX_INLINE_RASTER_BYTES:
        raise RenderBundleError(
            f"{label} exceeds the {_MAX_INLINE_RASTER_BYTES}-byte offline-viewer limit"
        )
    if len(decoded) < 24 or not decoded.startswith(_PNG_SIGNATURE) or decoded[12:16] != b"IHDR":
        raise RenderBundleError(f"{label} must contain a PNG with an IHDR chunk")
    png_width = int.from_bytes(decoded[16:20], "big")
    png_height = int.from_bytes(decoded[20:24], "big")
    if [png_width, png_height] != [width, height]:
        raise RenderBundleError(
            f"{label} PNG dimensions {png_width}x{png_height} do not match "
            f"declared {width}x{height}"
        )
    canonical = base64.b64encode(decoded).decode("ascii")
    if canonical != encoded:
        raise RenderBundleError(f"{label} must use canonical padded base64")
    return canonical, decoded


def _layer(
    value: Any,
    label: str,
    *,
    mode: str,
    surfaces: Mapping[str, Mapping[str, Any]],
    source_observation_sha256: str,
    sensor: Mapping[str, Any],
) -> dict[str, Any]:
    layer = _object(value, label)
    kind = _text(layer.get("kind"), f"{label}.kind")
    if kind == "raster":
        _keys(
            layer,
            required={
                "kind",
                "sha256",
                "source_observation_sha256",
                "source_output_sha256",
                "source_output_media_type",
                "source_output_dtype",
                "source_output_shape",
                "display_transform",
                "display_range",
                "media_type",
                "width_px",
                "height_px",
                "data_base64",
            },
            label=label,
        )
        if layer["media_type"] != "image/png":
            raise RenderBundleError(f"{label}.media_type must be 'image/png'")
        if layer["source_output_media_type"] != "application/vnd.numpy.npy":
            raise RenderBundleError(
                f"{label}.source_output_media_type must be 'application/vnd.numpy.npy'"
            )
        width = _integer(layer["width_px"], f"{label}.width_px", nonnegative=True)
        height = _integer(layer["height_px"], f"{label}.height_px", nonnegative=True)
        if width <= 0 or height <= 0:
            raise RenderBundleError(f"{label} dimensions must be positive")
        encoded, decoded = _decode_png(
            layer["data_base64"], f"{label}.data_base64", width, height
        )
        digest = _hash(layer["sha256"], f"{label}.sha256")
        expected = "sha256:" + hashlib.sha256(decoded).hexdigest()
        if digest != expected:
            raise RenderBundleError(f"{label}.sha256 is stale or incorrect; expected {expected}")
        source_digest = _hash(
            layer["source_observation_sha256"], f"{label}.source_observation_sha256"
        )
        if source_digest != source_observation_sha256:
            raise RenderBundleError(
                f"{label}.source_observation_sha256 differs from its observation artifact"
            )
        expected_dtype = "int32" if mode == "segmentation" else "float32"
        if layer["source_output_dtype"] != expected_dtype:
            raise RenderBundleError(
                f"{label}.source_output_dtype must be {expected_dtype!r}"
            )
        source_shape = _list(layer["source_output_shape"], f"{label}.source_output_shape")
        expected_shape = [height, width, 3] if mode == "rgb" else [height, width]
        if source_shape != expected_shape:
            raise RenderBundleError(
                f"{label}.source_output_shape must equal {expected_shape}"
            )
        if [width, height] != sensor["projection"]["resolution_px"]:
            raise RenderBundleError(
                f"{label} dimensions must equal the bound sensor resolution"
            )
        transforms = {
            "rgb": "linear-rgb-clamped-to-srgb8",
            "depth": "depth-near-white-far-black",
            "segmentation": "stable-integer-label-colors",
            "uncertainty": "uncertainty-log-blue-yellow-infinite-magenta",
        }
        if mode not in transforms or layer["display_transform"] != transforms[mode]:
            raise RenderBundleError(f"{label}.display_transform is not canonical for {mode}")
        display_range = layer["display_range"]
        if mode in {"rgb", "segmentation"}:
            if display_range is not None:
                raise RenderBundleError(f"{label}.display_range must be null for {mode}")
        else:
            display_range = _vector(display_range, f"{label}.display_range", 2)
            if display_range[0] < 0 or display_range[1] < display_range[0]:
                raise RenderBundleError(f"{label}.display_range must be non-negative and ordered")
            if mode == "depth" and display_range != sensor["projection"]["clipping_m"]:
                raise RenderBundleError(
                    f"{label}.display_range must equal the sensor clipping range"
                )
        return {
            "kind": "raster",
            "sha256": digest,
            "source_observation_sha256": source_digest,
            "source_output_sha256": _hash(
                layer["source_output_sha256"], f"{label}.source_output_sha256"
            ),
            "source_output_media_type": "application/vnd.numpy.npy",
            "source_output_dtype": expected_dtype,
            "source_output_shape": source_shape,
            "display_transform": transforms[mode],
            "display_range": display_range,
            "media_type": "image/png",
            "width_px": width,
            "height_px": height,
            "data_base64": encoded,
        }
    if kind == "surface":
        if mode != "reconstruction":
            raise RenderBundleError(f"{label} surface layers are valid only for reconstruction")
        _keys(
            layer,
            required={
                "kind",
                "source_observation_sha256",
                "surface_sha256",
                "world_pose",
            },
            label=label,
        )
        surface_sha256 = _hash(layer["surface_sha256"], f"{label}.surface_sha256")
        if surface_sha256 not in surfaces:
            raise RenderBundleError(f"{label}.surface_sha256 references an absent surface")
        return {
            "kind": "surface",
            "source_observation_sha256": _hash(
                layer["source_observation_sha256"],
                f"{label}.source_observation_sha256",
            ),
            "surface_sha256": surface_sha256,
            "world_pose": _pose(layer["world_pose"], f"{label}.world_pose"),
        }
    raise RenderBundleError(f"{label}.kind must be 'raster' or 'surface'")


def _matrix(value: Any, label: str) -> list[float]:
    matrix = _vector(value, label, 16)
    if any(abs(matrix[12 + index] - expected) > 1e-9 for index, expected in enumerate((0, 0, 0, 1))):
        raise RenderBundleError(f"{label} must have homogeneous final row [0, 0, 0, 1]")
    rows = (matrix[0:3], matrix[4:7], matrix[8:11])
    for index, row in enumerate(rows):
        if abs(sum(item * item for item in row) - 1.0) > 1e-9:
            raise RenderBundleError(f"{label} rotation row {index} is not unit length")
    for first in range(3):
        for second in range(first):
            if abs(sum(rows[first][axis] * rows[second][axis] for axis in range(3))) > 1e-9:
                raise RenderBundleError(f"{label} rotation rows are not orthogonal")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if abs(determinant - 1.0) > 1e-9:
        raise RenderBundleError(
            f"{label} rotation must be right-handed with determinant +1"
        )
    return matrix


def _pose_matrix(value: Mapping[str, Any]) -> list[float]:
    tx, ty, tz = value["translation_m"]
    w, x, y, z = value["rotation_quaternion_wxyz"]
    return [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), tx,
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), ty,
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), tz,
        0.0, 0.0, 0.0, 1.0,
    ]


def normalize_render_bundle(
    value: Any,
    *,
    assembly_sha256: str,
    scene: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize a complete render bundle for one scene."""

    bundle = _object(value, "render_bundle")
    _keys(
        bundle,
        required={
            "schema",
            "sha256",
            "assembly_sha256",
            "surfaces",
            "solid_bindings",
            "sensors",
            "observations",
        },
        label="render_bundle",
    )
    if bundle["schema"] != RENDER_BUNDLE_SCHEMA:
        raise RenderBundleError(
            f"render_bundle.schema must be {RENDER_BUNDLE_SCHEMA!r}"
        )
    bound_assembly = _hash(bundle["assembly_sha256"], "render_bundle.assembly_sha256")
    if bound_assembly != assembly_sha256:
        raise RenderBundleError(
            f"render bundle assembly hash mismatch ({bound_assembly} != {assembly_sha256})"
        )

    raw_surfaces = _object(bundle["surfaces"], "render_bundle.surfaces")
    surfaces: dict[str, dict[str, Any]] = {}
    for key in sorted(raw_surfaces):
        digest = _hash(key, f"render_bundle.surfaces key {key!r}")
        surface = _surface(raw_surfaces[key], f"render_bundle.surfaces[{key!r}]")
        if surface["sha256"] != digest:
            raise RenderBundleError(
                f"render_bundle.surfaces key {digest} does not match payload hash "
                f"{surface['sha256']}"
            )
        surfaces[digest] = surface

    solid_geometry: dict[str, Mapping[str, Any]] = {}
    optical_connectors: set[str] = set()
    for component in scene["components"]:
        for body in component["bodies"]:
            for solid in body["solids"]:
                solid_geometry[f"{component['id']}/{body['id']}/{solid['id']}"] = solid[
                    "geometry"
                ]
        for connector in component["connectors"]:
            if connector["body"] is not None and connector["domain"] == "optical":
                optical_connectors.add(f"{component['id']}.{connector['id']}")

    bindings: list[dict[str, str]] = []
    bound_solids: set[str] = set()
    used_surfaces: set[str] = set()
    for index, raw_binding in enumerate(
        _list(bundle["solid_bindings"], "render_bundle.solid_bindings", nonempty=True)
    ):
        label = f"render_bundle.solid_bindings[{index}]"
        binding = _object(raw_binding, label)
        _keys(
            binding,
            required={"component", "body", "solid", "surface_sha256"},
            label=label,
        )
        component = _identifier(binding["component"], f"{label}.component")
        body = _identifier(binding["body"], f"{label}.body")
        solid = _identifier(binding["solid"], f"{label}.solid")
        key = f"{component}/{body}/{solid}"
        if key not in solid_geometry:
            raise RenderBundleError(f"{label} references unknown solid {key!r}")
        if key in bound_solids:
            raise RenderBundleError(f"render bundle repeats solid binding {key!r}")
        surface_sha256 = _hash(binding["surface_sha256"], f"{label}.surface_sha256")
        if surface_sha256 not in surfaces:
            raise RenderBundleError(f"{label}.surface_sha256 references an absent surface")
        bound_surface = surfaces[surface_sha256]
        geometry = solid_geometry[key]
        if geometry["kind"] != "shape":
            raise RenderBundleError(
                f"{label} supplies canonical surface data for non-shape geometry {key!r}"
            )
        if bound_surface["shape_manifest_sha256"] != geometry["shape_sha256"]:
            raise RenderBundleError(
                f"{label} surface is not bound to the solid's exact shape manifest"
            )
        if bound_surface["surface_id"] != geometry["surface_id"]:
            raise RenderBundleError(
                f"{label} surface_id differs from the solid's authored surface_id"
            )
        declared_dimensions = solid_geometry[key]["dimensions_m"]
        actual_dimensions = _surface_dimensions(surfaces[surface_sha256])
        for axis, (declared, actual) in enumerate(zip(declared_dimensions, actual_dimensions)):
            tolerance = max(1e-9, abs(float(declared)) * 1e-6)
            if abs(float(declared) - actual) > tolerance:
                raise RenderBundleError(
                    f"{label} surface extent axis {axis} is {actual:.12g} m, but solid "
                    f"declares {float(declared):.12g} m"
                )
        bound_solids.add(key)
        used_surfaces.add(surface_sha256)
        bindings.append(
            {
                "component": component,
                "body": body,
                "solid": solid,
                "surface_sha256": surface_sha256,
            }
        )
    missing_solids = sorted(set(solid_geometry) - bound_solids)
    if missing_solids:
        raise RenderBundleError(
            "render bundle must bind every physical solid; missing " + ", ".join(missing_solids)
        )

    sensors = [
        _sensor(item, f"render_bundle.sensors[{index}]", optical_connectors)
        for index, item in enumerate(_list(bundle["sensors"], "render_bundle.sensors"))
    ]
    sensor_ids = [item["id"] for item in sensors]
    if len(set(sensor_ids)) != len(sensor_ids):
        raise RenderBundleError("render_bundle.sensors contains duplicate ids")
    sensor_set = set(sensor_ids)
    sensor_by_id = {item["id"]: item for item in sensors}

    frame_count = (
        len(scene["body_pose_frames"]["frames"])
        if "body_pose_frames" in scene
        else 1
    )
    observations: list[dict[str, Any]] = []
    observation_keys: set[tuple[int, str]] = set()
    for index, raw_observation in enumerate(
        _list(bundle["observations"], "render_bundle.observations")
    ):
        label = f"render_bundle.observations[{index}]"
        observation = _object(raw_observation, label)
        _keys(
            observation,
            required={
                "id",
                "artifact_sha256",
                "frame_index",
                "sensor",
                "sensor_descriptor_sha256",
                "optical_scene_sha256",
                "assembly_id",
                "assembly_sha256",
                "assembly_frame",
                "mount_connector",
                "mount_transform_sha256",
                "pose_world_from_sensor_row_major",
                "layers",
            },
            label=label,
        )
        observation_id = _identifier(observation["id"], f"{label}.id")
        artifact_sha256 = _hash(
            observation["artifact_sha256"], f"{label}.artifact_sha256"
        )
        frame_index = _integer(
            observation["frame_index"], f"{label}.frame_index", nonnegative=True
        )
        if frame_index >= frame_count:
            raise RenderBundleError(f"{label}.frame_index is outside the scene frame range")
        sensor = _identifier(observation["sensor"], f"{label}.sensor")
        if sensor not in sensor_set:
            raise RenderBundleError(f"{label}.sensor references unknown sensor {sensor!r}")
        sensor_record = sensor_by_id[sensor]
        descriptor_sha256 = _hash(
            observation["sensor_descriptor_sha256"],
            f"{label}.sensor_descriptor_sha256",
        )
        if descriptor_sha256 != sensor_record["descriptor_sha256"]:
            raise RenderBundleError(
                f"{label}.sensor_descriptor_sha256 differs from its sensor declaration"
            )
        observation_assembly = _hash(
            observation["assembly_sha256"], f"{label}.assembly_sha256"
        )
        if observation_assembly != bound_assembly:
            raise RenderBundleError(f"{label} is bound to another physical assembly")
        if observation["assembly_id"] != scene["contraption_id"]:
            raise RenderBundleError(f"{label}.assembly_id differs from the rendered assembly")
        if observation["assembly_frame"] != "world":
            raise RenderBundleError(f"{label}.assembly_frame must be 'world'")
        mount_connector = _text(
            observation["mount_connector"], f"{label}.mount_connector"
        )
        if mount_connector != sensor_record["connector"]:
            raise RenderBundleError(
                f"{label}.mount_connector differs from its sensor declaration"
            )
        matrix = _matrix(
            observation["pose_world_from_sensor_row_major"],
            f"{label}.pose_world_from_sensor_row_major",
        )
        mount_transform_sha256 = _hash(
            observation["mount_transform_sha256"],
            f"{label}.mount_transform_sha256",
        )
        matrix_digest = "sha256:" + hashlib.sha256(
            struct.pack("<16d", *matrix)
        ).hexdigest()
        if mount_transform_sha256 != matrix_digest:
            raise RenderBundleError(
                f"{label}.mount_transform_sha256 does not hash its exact pose matrix"
            )
        connector_poses = (
            scene["body_pose_frames"]["frames"][frame_index]["connector_poses"]
            if "body_pose_frames" in scene
            else scene["connector_poses"]
        )
        physical_matrix = _pose_matrix(connector_poses[mount_connector])
        if any(abs(actual - expected) > 1e-9 for actual, expected in zip(matrix, physical_matrix)):
            raise RenderBundleError(
                f"{label} pose differs from the exact physical connector pose at its frame"
            )
        key = (frame_index, sensor)
        if key in observation_keys:
            raise RenderBundleError(
                f"render bundle repeats observation for frame {frame_index}, sensor {sensor!r}"
            )
        observation_keys.add(key)
        raw_layers = _object(observation["layers"], f"{label}.layers")
        if not raw_layers:
            raise RenderBundleError(f"{label}.layers must not be empty")
        unknown_modes = sorted(set(raw_layers) - _LAYER_MODES)
        if unknown_modes:
            raise RenderBundleError(
                f"{label}.layers contains unsupported modes: {', '.join(unknown_modes)}"
            )
        layers = {
            mode: _layer(
                raw_layers[mode],
                f"{label}.layers.{mode}",
                mode=mode,
                surfaces=surfaces,
                source_observation_sha256=artifact_sha256,
                sensor=sensor_record,
            )
            for mode in sorted(raw_layers)
        }
        for layer in layers.values():
            if layer["kind"] == "surface":
                used_surfaces.add(layer["surface_sha256"])
        observations.append({
            "id": observation_id,
            "artifact_sha256": artifact_sha256,
            "frame_index": frame_index,
            "sensor": sensor,
            "sensor_descriptor_sha256": descriptor_sha256,
            "optical_scene_sha256": _hash(
                observation["optical_scene_sha256"], f"{label}.optical_scene_sha256"
            ),
            "assembly_id": scene["contraption_id"],
            "assembly_sha256": observation_assembly,
            "assembly_frame": "world",
            "mount_connector": mount_connector,
            "mount_transform_sha256": mount_transform_sha256,
            "pose_world_from_sensor_row_major": matrix,
            "layers": layers,
        })

    unused_surfaces = sorted(set(surfaces) - used_surfaces)
    if unused_surfaces:
        raise RenderBundleError(
            "render bundle contains unreferenced surfaces: " + ", ".join(unused_surfaces)
        )

    normalized = {
        "schema": RENDER_BUNDLE_SCHEMA,
        "sha256": _hash(bundle["sha256"], "render_bundle.sha256"),
        "assembly_sha256": bound_assembly,
        "surfaces": surfaces,
        "solid_bindings": bindings,
        "sensors": sensors,
        "observations": observations,
    }
    expected = content_sha256(normalized)
    if normalized["sha256"] != expected:
        raise RenderBundleError(
            f"render_bundle.sha256 is stale or incorrect; expected {expected}"
        )
    return normalized


__all__ = [
    "RENDER_BUNDLE_SCHEMA",
    "TRIANGLE_SURFACE_SCHEMA",
    "RenderBundleError",
    "content_sha256",
    "materialize_render_bundle",
    "normalize_render_bundle",
    "optical_sensors_from_registry",
    "sensor_view_from_descriptor",
    "shape_artifacts_from_registry",
]
