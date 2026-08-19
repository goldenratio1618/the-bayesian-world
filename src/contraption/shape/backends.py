"""Bounded deterministic mesh readers and optional CAD/scene adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import tempfile
from typing import Any, Callable
from urllib.parse import unquote, urlsplit
import zipfile

import numpy as np

from ..strict_json import loads_strict_json
from .artifacts import OpticalMaterial, ShapeUncertainty
from .mesh import MeshError, TriangleMesh


class GeometryBackendError(ValueError):
    """Raised when deterministic geometry conversion must fail closed."""


@dataclass(frozen=True, slots=True)
class GeometryLimits:
    max_source_bytes: int = 256 * 1024 * 1024
    max_linked_files: int = 128
    max_linked_bytes: int = 512 * 1024 * 1024
    max_vertices: int = 5_000_000
    max_triangles: int = 10_000_000
    max_ply_header_bytes: int = 64 * 1024
    max_properties: int = 64
    max_external_output_bytes: int = 768 * 1024 * 1024
    external_timeout_seconds: int = 180

    def __post_init__(self) -> None:
        for name in (
            "max_source_bytes",
            "max_linked_files",
            "max_linked_bytes",
            "max_vertices",
            "max_triangles",
            "max_ply_header_bytes",
            "max_properties",
            "max_external_output_bytes",
            "external_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GeometryBackendError(f"{name} must be a positive integer")


def _read_bounded(path: Path, limit: int, context: str) -> bytes:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise GeometryBackendError(f"{context} is not a regular file: {resolved}")
    size = resolved.stat().st_size
    if size <= 0 or size > limit:
        raise GeometryBackendError(
            f"{context} size must be between 1 and {limit} bytes: {resolved}"
        )
    payload = resolved.read_bytes()
    if len(payload) != size:
        raise GeometryBackendError(f"{context} changed while being read: {resolved}")
    return payload


_PLY_TYPES: dict[str, tuple[str, type, int | None]] = {
    "char": ("b", int, 8),
    "int8": ("b", int, 8),
    "uchar": ("B", int, 8),
    "uint8": ("B", int, 8),
    "short": ("h", int, 16),
    "int16": ("h", int, 16),
    "ushort": ("H", int, 16),
    "uint16": ("H", int, 16),
    "int": ("i", int, 32),
    "int32": ("i", int, 32),
    "uint": ("I", int, 32),
    "uint32": ("I", int, 32),
    "float": ("f", float, None),
    "float32": ("f", float, None),
    "double": ("d", float, None),
    "float64": ("d", float, None),
}


@dataclass(frozen=True, slots=True)
class _PlyProperty:
    name: str
    scalar_type: str | None = None
    count_type: str | None = None
    item_type: str | None = None

    @property
    def is_list(self) -> bool:
        return self.count_type is not None


@dataclass(frozen=True, slots=True)
class _PlyElement:
    name: str
    count: int
    properties: tuple[_PlyProperty, ...]


def _ply_header(payload: bytes, limits: GeometryLimits) -> tuple[str, tuple[_PlyElement, ...], int]:
    endings = (b"end_header\n", b"end_header\r\n")
    candidates = [(payload.find(marker), marker) for marker in endings]
    candidates = [(index, marker) for index, marker in candidates if index >= 0]
    if not candidates:
        raise GeometryBackendError("PLY has no end_header marker")
    marker_index, marker = min(candidates, key=lambda item: item[0])
    body_start = marker_index + len(marker)
    if body_start > limits.max_ply_header_bytes:
        raise GeometryBackendError("PLY header exceeds the safety limit")
    try:
        lines = payload[:body_start].decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise GeometryBackendError("PLY header must be ASCII") from exc
    if not lines or lines[0] != "ply":
        raise GeometryBackendError("PLY source does not begin with the ply magic line")
    if len(lines) < 3 or not lines[1].startswith("format "):
        raise GeometryBackendError("PLY format declaration is missing")
    parts = lines[1].split()
    if len(parts) != 3 or parts[2] != "1.0" or parts[1] not in {
        "ascii",
        "binary_little_endian",
        "binary_big_endian",
    }:
        raise GeometryBackendError("PLY format must be ASCII or binary version 1.0")
    elements: list[tuple[str, int, list[_PlyProperty]]] = []
    for line_number, line in enumerate(lines[2:], 3):
        if not line or line.startswith("comment ") or line.startswith("obj_info "):
            continue
        fields = line.split()
        if fields[0] == "end_header":
            break
        if fields[0] == "element":
            if len(fields) != 3 or not fields[2].isdigit():
                raise GeometryBackendError(f"PLY header line {line_number} has invalid element")
            count = int(fields[2])
            if count <= 0:
                raise GeometryBackendError("PLY elements must have positive counts")
            elements.append((fields[1], count, []))
            continue
        if fields[0] == "property":
            if not elements:
                raise GeometryBackendError("PLY property appears before its element")
            if len(fields) == 3 and fields[1] in _PLY_TYPES:
                prop = _PlyProperty(fields[2], scalar_type=fields[1])
            elif (
                len(fields) == 5
                and fields[1] == "list"
                and fields[2] in _PLY_TYPES
                and fields[3] in _PLY_TYPES
            ):
                prop = _PlyProperty(fields[4], count_type=fields[2], item_type=fields[3])
            else:
                raise GeometryBackendError(f"PLY header line {line_number} has invalid property")
            if any(existing.name == prop.name for existing in elements[-1][2]):
                raise GeometryBackendError(
                    f"PLY element {elements[-1][0]!r} has duplicate property {prop.name!r}"
                )
            elements[-1][2].append(prop)
            if len(elements[-1][2]) > limits.max_properties:
                raise GeometryBackendError("PLY element exceeds the property-count limit")
            continue
        raise GeometryBackendError(f"unsupported PLY header directive {fields[0]!r}")
    result = tuple(_PlyElement(name, count, tuple(props)) for name, count, props in elements)
    if tuple(item.name for item in result) != ("vertex", "face"):
        raise GeometryBackendError("PLY must contain exactly vertex then face elements")
    if result[0].count > limits.max_vertices or result[1].count > limits.max_triangles:
        raise GeometryBackendError("PLY exceeds the vertex or triangle safety limit")
    return parts[1], result, body_start


def _ascii_scalar(token: str, kind: str, context: str) -> int | float:
    converter = _PLY_TYPES[kind][1]
    try:
        value = converter(token)
    except ValueError as exc:
        raise GeometryBackendError(f"{context} contains an invalid {kind} value") from exc
    if isinstance(value, float) and not math.isfinite(value):
        raise GeometryBackendError(f"{context} contains NaN or infinity")
    return value


def _linear_channel(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _ply_values(
    payload: bytes,
    mode: str,
    elements: tuple[_PlyElement, ...],
    body_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[list[dict[str, Any]]] = [[], []]
    if mode == "ascii":
        try:
            lines = payload[body_start:].decode("ascii", errors="strict").splitlines()
        except UnicodeDecodeError as exc:
            raise GeometryBackendError("ASCII PLY body contains non-ASCII bytes") from exc
        cursor = 0
        for element_index, element in enumerate(elements):
            for row in range(element.count):
                if cursor >= len(lines):
                    raise GeometryBackendError("ASCII PLY body ended early")
                tokens = lines[cursor].split()
                cursor += 1
                offset = 0
                record: dict[str, Any] = {}
                for prop in element.properties:
                    if prop.is_list:
                        if offset >= len(tokens):
                            raise GeometryBackendError("ASCII PLY list count is missing")
                        count = int(_ascii_scalar(tokens[offset], prop.count_type or "", "PLY list count"))
                        offset += 1
                        if count != 3 or offset + count > len(tokens):
                            raise GeometryBackendError("ASCII PLY list length is invalid")
                        record[prop.name] = [
                            _ascii_scalar(token, prop.item_type or "", "PLY list")
                            for token in tokens[offset : offset + count]
                        ]
                        offset += count
                    else:
                        if offset >= len(tokens):
                            raise GeometryBackendError("ASCII PLY scalar value is missing")
                        record[prop.name] = _ascii_scalar(
                            tokens[offset], prop.scalar_type or "", "PLY scalar"
                        )
                        offset += 1
                if offset != len(tokens):
                    raise GeometryBackendError("ASCII PLY row contains extra values")
                records[element_index].append(record)
        if any(line.strip() for line in lines[cursor:]):
            raise GeometryBackendError("ASCII PLY has trailing data")
        return records[0], records[1]

    endian = "<" if mode == "binary_little_endian" else ">"
    offset = body_start
    for element_index, element in enumerate(elements):
        for _row in range(element.count):
            record = {}
            for prop in element.properties:
                if prop.is_list:
                    count_struct = struct.Struct(endian + _PLY_TYPES[prop.count_type or ""][0])
                    if offset + count_struct.size > len(payload):
                        raise GeometryBackendError("binary PLY list count escapes the source")
                    count = int(count_struct.unpack_from(payload, offset)[0])
                    offset += count_struct.size
                    if count != 3:
                        raise GeometryBackendError(
                            "binary PLY faces must contain exactly three indices"
                        )
                    item_struct = struct.Struct(endian + _PLY_TYPES[prop.item_type or ""][0])
                    byte_count = count * item_struct.size
                    if offset + byte_count > len(payload):
                        raise GeometryBackendError("binary PLY list escapes the source")
                    record[prop.name] = [
                        item_struct.unpack_from(payload, offset + index * item_struct.size)[0]
                        for index in range(count)
                    ]
                    offset += byte_count
                else:
                    scalar_struct = struct.Struct(endian + _PLY_TYPES[prop.scalar_type or ""][0])
                    if offset + scalar_struct.size > len(payload):
                        raise GeometryBackendError("binary PLY scalar escapes the source")
                    value = scalar_struct.unpack_from(payload, offset)[0]
                    offset += scalar_struct.size
                    if isinstance(value, float) and not math.isfinite(value):
                        raise GeometryBackendError("binary PLY contains NaN or infinity")
                    record[prop.name] = value
            records[element_index].append(record)
    if offset != len(payload):
        raise GeometryBackendError("binary PLY has trailing data")
    return records[0], records[1]


def native_ply_tessellator(
    path: Path,
    scale: float,
    *,
    limits: GeometryLimits = GeometryLimits(),
):
    """Decode a bounded triangle-only PLY without optional software."""

    payload = _read_bounded(path, limits.max_source_bytes, "PLY source")
    mode, elements, body_start = _ply_header(payload, limits)
    vertex_props = {item.name: item for item in elements[0].properties}
    if any(item.is_list for item in elements[0].properties):
        raise GeometryBackendError("PLY vertex properties must be scalar")
    supported_vertex_properties = {
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "red",
        "green",
        "blue",
        "alpha",
    }
    unsupported_vertex_properties = sorted(
        set(vertex_props) - supported_vertex_properties
    )
    if unsupported_vertex_properties:
        raise GeometryBackendError(
            "PLY vertex properties cannot be preserved by canonical CTMESH: "
            + ", ".join(unsupported_vertex_properties)
        )
    if not {"x", "y", "z"} <= set(vertex_props):
        raise GeometryBackendError("PLY vertices require x, y, and z properties")
    texture_names = {"u", "v", "s", "t", "texture_u", "texture_v"}
    if texture_names & set(vertex_props):
        raise GeometryBackendError(
            "PLY texture coordinates cannot be represented by canonical CTMESH; "
            "provide a strict optical sidecar or a UV-capable future schema"
        )
    normal_names = {"nx", "ny", "nz"} & set(vertex_props)
    if normal_names and normal_names != {"nx", "ny", "nz"}:
        raise GeometryBackendError("PLY normals must provide nx, ny, and nz together")
    color_names = {"red", "green", "blue", "alpha"} & set(vertex_props)
    if color_names and not {"red", "green", "blue"} <= color_names:
        raise GeometryBackendError("PLY colors must provide red, green, and blue together")
    face_props = elements[1].properties
    if len(face_props) != 1 or not face_props[0].is_list or face_props[0].name not in {
        "vertex_indices",
        "vertex_index",
    }:
        raise GeometryBackendError(
            "PLY faces must contain exactly one vertex_indices list property"
        )
    integer_types = {
        "char", "int8", "uchar", "uint8", "short", "int16", "ushort",
        "uint16", "int", "int32", "uint", "uint32",
    }
    if (
        face_props[0].count_type not in integer_types
        or face_props[0].item_type not in integer_types
    ):
        raise GeometryBackendError(
            "PLY face list counts and vertex indices must use integer types"
        )
    vertices, faces = _ply_values(payload, mode, elements, body_start)
    coordinates = [
        [float(item["x"]) * scale, float(item["y"]) * scale, float(item["z"]) * scale]
        for item in vertices
    ]
    triangles: list[list[int]] = []
    key = face_props[0].name
    for index, face in enumerate(faces):
        values = face[key]
        if len(values) != 3:
            raise GeometryBackendError(
                f"PLY face {index} is not a triangle; ambiguous polygon triangulation is forbidden"
            )
        triangle = [int(value) for value in values]
        if any(value < 0 or value >= len(coordinates) for value in triangle):
            raise GeometryBackendError(f"PLY face {index} has an out-of-range vertex")
        triangles.append(triangle)
    normals = None
    if normal_names:
        normals = [[float(item["nx"]), float(item["ny"]), float(item["nz"])] for item in vertices]
    colors = None
    if color_names:
        allowed_color_types = {
            "uchar",
            "uint8",
            "ushort",
            "uint16",
            "uint",
            "uint32",
            "float",
            "float32",
            "double",
            "float64",
        }
        if any(
            vertex_props[name].scalar_type not in allowed_color_types
            for name in color_names
        ):
            raise GeometryBackendError(
                "PLY colors must use unsigned integer or floating-point properties"
            )
        colors = []
        for item in vertices:
            converted: list[float] = []
            for name in ("red", "green", "blue"):
                prop = vertex_props[name]
                raw = float(item[name])
                bits = _PLY_TYPES[prop.scalar_type or ""][2]
                normalized = raw if bits is None else raw / float((1 << bits) - 1)
                if not 0.0 <= normalized <= 1.0:
                    raise GeometryBackendError("PLY color channels must be in their declared range")
                converted.append(_linear_channel(normalized))
            alpha = 1.0
            if "alpha" in item:
                prop = vertex_props["alpha"]
                raw = float(item["alpha"])
                bits = _PLY_TYPES[prop.scalar_type or ""][2]
                alpha = raw if bits is None else raw / float((1 << bits) - 1)
                if not 0.0 <= alpha <= 1.0:
                    raise GeometryBackendError("PLY alpha must be in its declared range")
            colors.append([*converted, alpha])
    try:
        mesh = TriangleMesh(coordinates, triangles, normals, colors)
    except MeshError as exc:
        raise GeometryBackendError(f"PLY mesh is invalid: {exc}") from exc
    from .ingestion import TessellatedShape

    return TessellatedShape(mesh)


native_ply_tessellator.backend_id = "contraption-native-ply"  # type: ignore[attr-defined]
native_ply_tessellator.backend_version = "1"  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class NativePlyTessellator:
    limits: GeometryLimits = GeometryLimits()
    backend_id: str = "contraption-native-ply"
    backend_version: str = "1"

    def __call__(self, path: Path, scale: float):
        return native_ply_tessellator(path, scale, limits=self.limits)


def _safe_relative_link(uri: str, context: str) -> PurePosixPath | None:
    if uri.startswith("data:"):
        return None
    parsed = urlsplit(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise GeometryBackendError(f"{context} must be a local relative URI")
    decoded = unquote(parsed.path)
    if "\\" in decoded or "\x00" in decoded:
        raise GeometryBackendError(f"{context} must be a POSIX relative URI")
    relative = PurePosixPath(decoded)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise GeometryBackendError(f"{context} escapes the source directory")
    return relative


def _safe_link(root: Path, uri: str, context: str) -> Path | None:
    relative = _safe_relative_link(uri, context)
    if relative is None:
        return None
    target = (root / Path(*relative.parts)).resolve()
    if target != root and root not in target.parents:
        raise GeometryBackendError(f"{context} escapes the source directory")
    if not target.is_file() or target.is_symlink():
        raise GeometryBackendError(f"{context} is missing or not regular: {target}")
    return target


def _gltf_document(path: Path, limits: GeometryLimits) -> dict[str, Any]:
    payload = _read_bounded(path, limits.max_source_bytes, "glTF source")
    if path.suffix.casefold() == ".glb":
        return _glb_json(payload)
    try:
        value = loads_strict_json(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeometryBackendError(f"glTF JSON is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise GeometryBackendError("glTF JSON root must be an object")
    return value


def linked_source_relative_paths(
    source: str | Path,
    *,
    limits: GeometryLimits = GeometryLimits(),
) -> tuple[PurePosixPath, ...]:
    """Return syntax-validated glTF links before their private snapshot exists."""

    path = Path(source).resolve()
    if path.suffix.casefold() not in {".gltf", ".glb"}:
        return ()
    value = _gltf_document(path, limits)
    links: list[PurePosixPath] = []
    for collection_name in ("buffers", "images"):
        collection = value.get(collection_name, [])
        if not isinstance(collection, list):
            raise GeometryBackendError(f"glTF {collection_name} must be an array")
        for index, item in enumerate(collection):
            if not isinstance(item, dict):
                raise GeometryBackendError(
                    f"glTF {collection_name}[{index}] must be an object"
                )
            uri = item.get("uri")
            if uri is None:
                continue
            if not isinstance(uri, str) or not uri:
                raise GeometryBackendError(
                    f"glTF {collection_name}[{index}].uri is invalid"
                )
            linked = _safe_relative_link(
                uri, f"glTF {collection_name}[{index}].uri"
            )
            if linked is not None:
                links.append(linked)
    unique = tuple(dict.fromkeys(links))
    if len(unique) > limits.max_linked_files:
        raise GeometryBackendError("glTF exceeds the linked-file count limit")
    return unique


def _glb_json(payload: bytes) -> dict[str, Any]:
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise GeometryBackendError("GLB source has invalid magic or length")
    version, total = struct.unpack_from("<II", payload, 4)
    if version != 2 or total != len(payload):
        raise GeometryBackendError("GLB must be version 2 with an exact declared length")
    offset = 12
    chunks = 0
    json_payload: bytes | None = None
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise GeometryBackendError("GLB chunk header escapes the source")
        length, kind = struct.unpack_from("<II", payload, offset)
        offset += 8
        if length % 4 or offset + length > len(payload):
            raise GeometryBackendError("GLB chunk has invalid alignment or length")
        chunk = payload[offset : offset + length]
        offset += length
        chunks += 1
        if chunks > 16:
            raise GeometryBackendError("GLB exceeds the chunk-count limit")
        if kind == 0x4E4F534A:
            if json_payload is not None or chunks != 1:
                raise GeometryBackendError("GLB must contain one leading JSON chunk")
            json_payload = chunk.rstrip(b" \t\r\n\x00")
    if offset != len(payload) or json_payload is None:
        raise GeometryBackendError("GLB JSON chunk is missing")
    try:
        value = loads_strict_json(json_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeometryBackendError(f"GLB JSON chunk is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise GeometryBackendError("GLB JSON root must be an object")
    return value


def linked_source_paths(
    source: str | Path,
    *,
    limits: GeometryLimits = GeometryLimits(),
) -> tuple[Path, ...]:
    """Return bounded local glTF resources that must join source evidence."""

    path = Path(source).resolve()
    suffix = path.suffix.casefold()
    if suffix not in {".gltf", ".glb"}:
        return ()
    links: list[Path] = []
    for relative in linked_source_relative_paths(path, limits=limits):
        target = (path.parent / Path(*relative.parts)).resolve()
        if target != path.parent and path.parent not in target.parents:
            raise GeometryBackendError("glTF linked source escapes the source directory")
        if not target.is_file() or target.is_symlink():
            raise GeometryBackendError(
                f"glTF linked source is missing or not regular: {target}"
            )
        links.append(target)
    unique = tuple(dict.fromkeys(links))
    if len(unique) > limits.max_linked_files:
        raise GeometryBackendError("glTF exceeds the linked-file count limit")
    total = sum(item.stat().st_size for item in unique)
    if total > limits.max_linked_bytes:
        raise GeometryBackendError("glTF exceeds the linked-byte safety limit")
    return unique


def _material_from_factor(identifier: str, factor: Any, roughness: Any, metallic: Any) -> OpticalMaterial:
    values = np.asarray(factor, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise GeometryBackendError("scene material base-color factor is invalid")
    return OpticalMaterial(
        identifier,
        base_color_linear_rgba=tuple(float(item) for item in values),
        roughness=float(roughness),
        metallic=float(metallic),
        uncertainty=ShapeUncertainty(
            "uniform", {"reason": "source scene provides nominal material values only"}
        ),
        provenance={"kind": "source", "format": "scene-material"},
    )


@dataclass(frozen=True, slots=True)
class TrimeshTessellator:
    limits: GeometryLimits = GeometryLimits()
    backend_id: str = "trimesh-scene"

    @property
    def backend_version(self) -> str:
        try:
            module = importlib.import_module("trimesh")
        except ImportError:
            return "unavailable"
        return str(getattr(module, "__version__", "unknown"))

    def __call__(self, path: Path, scale: float):
        payload = _read_bounded(path, self.limits.max_source_bytes, "scene source")
        if path.suffix.casefold() in {".wrl", ".vrml"}:
            try:
                text = payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise GeometryBackendError("VRML must be UTF-8 text") from exc
            if re.search(r"\b(?:Script|Inline|EXTERNPROTO|ImageTexture|MovieTexture)\b", text):
                raise GeometryBackendError(
                    "VRML executable, external, or textured nodes are not accepted"
                )
        # Validate and bind the complete local glTF closure before importing or
        # calling the optional backend. In particular, no remote or escaping
        # URI may reach trimesh's loader.
        linked_sources = linked_source_paths(path, limits=self.limits)
        try:
            trimesh = importlib.import_module("trimesh")
        except ImportError as exc:
            raise GeometryBackendError(
                "glTF/GLB/WRL ingestion requires the optional 'geometry' dependency "
                "(trimesh>=4.8,<5)"
            ) from exc
        try:
            loaded = trimesh.load(path, force="scene", process=False)
        except Exception as exc:
            raise GeometryBackendError(f"trimesh rejected {path.name}: {exc}") from exc
        scene = loaded if hasattr(loaded, "graph") and hasattr(loaded, "geometry") else trimesh.Scene(loaded)
        nodes = sorted(scene.graph.nodes_geometry)
        if not nodes:
            raise GeometryBackendError("scene contains no triangle geometry")
        all_vertices: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        vertex_colors: list[np.ndarray | None] = []
        face_material: list[np.ndarray | None] = []
        materials: list[OpticalMaterial] = []
        material_by_key: dict[tuple[Any, ...], int] = {}
        vertex_offset = 0
        face_total = 0
        for node in nodes:
            try:
                transform, geometry_name = scene.graph.get(node)
                geometry = scene.geometry[geometry_name].copy()
                geometry.apply_transform(transform)
            except Exception as exc:
                raise GeometryBackendError(f"cannot resolve scene node {node!r}: {exc}") from exc
            vertices = np.asarray(geometry.vertices, dtype=np.float64) * scale
            faces = np.asarray(geometry.faces)
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise GeometryBackendError("scene backend returned non-triangle faces")
            if len(vertices) + vertex_offset > self.limits.max_vertices:
                raise GeometryBackendError("scene exceeds the vertex safety limit")
            face_total += len(faces)
            if face_total > self.limits.max_triangles:
                raise GeometryBackendError("scene exceeds the triangle safety limit")
            all_vertices.append(vertices)
            all_faces.append(np.asarray(faces, dtype=np.uint32) + vertex_offset)
            vertex_offset += len(vertices)
            visual = getattr(geometry, "visual", None)
            kind = None if visual is None else getattr(visual, "kind", None)
            colors: np.ndarray | None = None
            regions: np.ndarray | None = None
            if kind == "vertex":
                raw = np.asarray(visual.vertex_colors, dtype=float)
                if raw.shape != (len(vertices), 4):
                    raise GeometryBackendError("scene vertex colors have an invalid shape")
                normalized = raw / 255.0
                colors = np.column_stack(
                    (
                        np.vectorize(_linear_channel)(normalized[:, 0]),
                        np.vectorize(_linear_channel)(normalized[:, 1]),
                        np.vectorize(_linear_channel)(normalized[:, 2]),
                        normalized[:, 3],
                    )
                )
            elif kind == "face":
                raw = np.asarray(visual.face_colors, dtype=np.uint8)
                if raw.shape != (len(faces), 4):
                    raise GeometryBackendError("scene face colors have an invalid shape")
                indices: list[int] = []
                for rgba in raw:
                    key = ("face", *[int(item) for item in rgba])
                    if key not in material_by_key:
                        normalized = np.asarray(rgba, dtype=float) / 255.0
                        identifier = "face-" + "".join(f"{int(item):02x}" for item in rgba)
                        material_by_key[key] = len(materials)
                        materials.append(
                            _material_from_factor(
                                identifier,
                                [
                                    _linear_channel(normalized[0]),
                                    _linear_channel(normalized[1]),
                                    _linear_channel(normalized[2]),
                                    normalized[3],
                                ],
                                0.5,
                                0.0,
                            )
                        )
                    indices.append(material_by_key[key])
                regions = np.asarray(indices, dtype=np.uint32)
            elif kind == "texture":
                uv = getattr(visual, "uv", None)
                if uv is not None and np.asarray(uv).size:
                    raise GeometryBackendError(
                        "scene UV coordinates cannot be represented by canonical CTMESH"
                    )
                material = visual.material
                texture_fields = (
                    "image",
                    "baseColorTexture",
                    "metallicRoughnessTexture",
                    "normalTexture",
                    "occlusionTexture",
                    "emissiveTexture",
                )
                if any(getattr(material, name, None) is not None for name in texture_fields):
                    raise GeometryBackendError(
                        "scene UV textures cannot be represented by canonical CTMESH"
                    )
                factor = getattr(material, "baseColorFactor", None)
                if factor is None:
                    factor = np.asarray(getattr(material, "diffuse", [128, 128, 128, 255]), dtype=float) / 255.0
                    factor = [
                        _linear_channel(float(factor[0])),
                        _linear_channel(float(factor[1])),
                        _linear_channel(float(factor[2])),
                        float(factor[3]),
                    ]
                roughness = getattr(material, "roughnessFactor", 0.5)
                metallic = getattr(material, "metallicFactor", 0.0)
                key = (
                    "material",
                    json.dumps(np.asarray(factor).tolist()),
                    float(roughness),
                    float(metallic),
                )
                if key not in material_by_key:
                    material_by_key[key] = len(materials)
                    materials.append(
                        _material_from_factor(
                            f"material-{len(materials)}",
                            factor,
                            roughness,
                            metallic,
                        )
                    )
                regions = np.full(len(faces), material_by_key[key], dtype=np.uint32)
            elif kind not in {None, "none"}:
                raise GeometryBackendError(f"unsupported scene visual kind {kind!r}")
            vertex_colors.append(colors)
            face_material.append(regions)
        if any(item is not None for item in vertex_colors) and any(
            item is None for item in vertex_colors
        ):
            raise GeometryBackendError("scene mixes colored and uncolored vertex geometry")
        if any(item is not None for item in face_material) and any(
            item is None for item in face_material
        ):
            raise GeometryBackendError("scene mixes materialized and unmaterialized geometry")
        try:
            mesh = TriangleMesh(
                np.concatenate(all_vertices),
                np.concatenate(all_faces),
                vertex_rgba_linear=(
                    np.concatenate([item for item in vertex_colors if item is not None])
                    if vertex_colors and vertex_colors[0] is not None
                    else None
                ),
                face_material=(
                    np.concatenate([item for item in face_material if item is not None])
                    if face_material and face_material[0] is not None
                    else None
                ),
            ).with_computed_normals()
        except (MeshError, ValueError) as exc:
            raise GeometryBackendError(f"scene mesh is invalid: {exc}") from exc
        from .ingestion import TessellatedShape

        return TessellatedShape(
            mesh,
            tuple(materials),
            linked_sources,
        )


@dataclass(frozen=True, slots=True)
class FreeCADTessellator:
    executable: str
    limits: GeometryLimits = GeometryLimits()
    backend_id: str = "freecad-command"
    backend_version: str | None = None
    executable_sha256: str | None = None

    def __post_init__(self) -> None:
        executable = Path(self.executable).resolve()
        if not executable.is_file():
            raise GeometryBackendError(
                f"FreeCAD executable is not a regular file: {executable}"
            )
        digest = hashlib.sha256()
        total = 0
        with executable.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.limits.max_source_bytes:
                    raise GeometryBackendError(
                        "FreeCAD executable exceeds the backend hash safety limit"
                    )
                digest.update(chunk)
        try:
            completed = subprocess.run(
                [str(executable), "--version"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GeometryBackendError(
                f"cannot identify FreeCAD converter version: {exc}"
            ) from exc
        version_lines = (completed.stdout or completed.stderr).decode(
            "utf-8", errors="replace"
        ).strip().splitlines()
        if (
            completed.returncode != 0
            or not version_lines
            or not version_lines[0]
            or len(version_lines[0]) > 256
        ):
            raise GeometryBackendError(
                "FreeCAD converter did not provide a bounded successful version"
            )
        object.__setattr__(self, "executable", str(executable))
        object.__setattr__(self, "backend_version", version_lines[0])
        object.__setattr__(self, "executable_sha256", digest.hexdigest())

    def __call__(self, path: Path, scale: float):
        _read_bounded(path, self.limits.max_source_bytes, "CAD source")
        if path.suffix.casefold() == ".fcstd":
            _preflight_fcstd(path, self.limits)
        worker = Path(__file__).with_name("_freecad_worker.py").resolve()
        if not worker.is_file():
            raise GeometryBackendError("bundled FreeCAD worker is missing")
        with tempfile.TemporaryDirectory(prefix="contraption-freecad-") as temporary:
            root = Path(temporary)
            output = root / "mesh.json"
            home = root / "home"
            home.mkdir()
            environment = {
                "HOME": str(home),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": os.environ.get("PATH", ""),
                "PYTHONNOUSERSITE": "1",
            }
            command = [
                self.executable,
                "--console",
                "--safe-mode",
                str(worker),
                str(path.resolve()),
                str(output),
                repr(float(scale)),
                repr(max(1.0e-5 / float(scale), 1.0e-6)),
                str(self.limits.max_vertices),
                str(self.limits.max_triangles),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.limits.external_timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise GeometryBackendError(f"FreeCAD conversion failed: {exc}") from exc
            if len(completed.stdout) + len(completed.stderr) > 2 * 1024 * 1024:
                raise GeometryBackendError("FreeCAD emitted excessive process output")
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace")[-1000:].strip()
                raise GeometryBackendError(
                    f"FreeCAD rejected {path.name} (exit {completed.returncode}): {detail}"
                )
            if not output.is_file() or output.stat().st_size > self.limits.max_external_output_bytes:
                raise GeometryBackendError("FreeCAD did not emit a bounded mesh record")
            try:
                value = loads_strict_json(output.read_bytes())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GeometryBackendError(f"FreeCAD mesh record is invalid: {exc}") from exc
        if not isinstance(value, dict) or set(value) != {
            "format",
            "vertices_m",
            "triangles",
        } or value["format"] != "contraption-freecad-mesh-1":
            raise GeometryBackendError("FreeCAD mesh record has an invalid schema")
        if len(value["vertices_m"]) > self.limits.max_vertices or len(value["triangles"]) > self.limits.max_triangles:
            raise GeometryBackendError("FreeCAD output exceeds geometry limits")
        try:
            mesh = TriangleMesh(value["vertices_m"], value["triangles"]).with_computed_normals()
        except (MeshError, TypeError, ValueError) as exc:
            raise GeometryBackendError(f"FreeCAD output mesh is invalid: {exc}") from exc
        from .ingestion import TessellatedShape

        return TessellatedShape(mesh)


def _preflight_fcstd(path: Path, limits: GeometryLimits) -> None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > 4_096:
                raise GeometryBackendError("FCStd entry count exceeds the safety limit")
            total = 0
            document: bytes | None = None
            seen: set[str] = set()
            for info in infos:
                name = info.filename
                if "\\" in name or "\x00" in name:
                    raise GeometryBackendError("FCStd contains an unsafe member path")
                raw_name = name[:-1] if info.is_dir() and name.endswith("/") else name
                pieces = raw_name.split("/")
                member = PurePosixPath(raw_name)
                if (
                    not raw_name
                    or member.is_absolute()
                    or any(part in {"", ".", ".."} for part in pieces)
                ):
                    raise GeometryBackendError("FCStd contains an unsafe member path")
                folded = member.as_posix().casefold()
                if folded in seen:
                    raise GeometryBackendError("FCStd contains duplicate member paths")
                seen.add(folded)
                if info.flag_bits & 0x1:
                    raise GeometryBackendError("encrypted FCStd members are not accepted")
                if info.compress_type not in {
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                }:
                    raise GeometryBackendError(
                        "FCStd contains an unsupported compression method"
                    )
                mode = info.external_attr >> 16
                kind = stat.S_IFMT(mode)
                if kind == stat.S_IFLNK or kind not in {
                    0,
                    stat.S_IFREG,
                    stat.S_IFDIR,
                }:
                    raise GeometryBackendError(
                        "FCStd links and special files are forbidden"
                    )
                if (
                    info.file_size > 0
                    and (
                        info.compress_size <= 0
                        or info.file_size > info.compress_size * 200
                    )
                ):
                    raise GeometryBackendError(
                        "FCStd member exceeds the compression-ratio limit"
                    )
                total += info.file_size
                if total > limits.max_linked_bytes:
                    raise GeometryBackendError("FCStd expands beyond the safety limit")
                if member.suffix.casefold() in {".py", ".fcmacro", ".so", ".dll"}:
                    raise GeometryBackendError("FCStd embedded executable content is forbidden")
                if member.as_posix() == "Document.xml":
                    if info.file_size > 64 * 1024 * 1024:
                        raise GeometryBackendError("FCStd Document.xml exceeds the safety limit")
                    document = archive.read(info)
            bad_member = archive.testzip()
            if bad_member is not None:
                raise GeometryBackendError(
                    f"FCStd CRC validation failed for {bad_member!r}"
                )
            if document is None:
                raise GeometryBackendError("FCStd Document.xml is missing")
            lowered = document.lower()
            if b"app::featurepython" in lowered or b"partdesign::featurepython" in lowered:
                raise GeometryBackendError("FCStd Python proxy objects are forbidden")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise GeometryBackendError(f"FCStd container is invalid: {exc}") from exc


def automatic_tessellator(
    source: str | Path,
    *,
    limits: GeometryLimits = GeometryLimits(),
) -> Callable[[Path, float], Any] | None:
    suffix = Path(source).suffix.casefold()
    if suffix == ".ply":
        return NativePlyTessellator(limits)
    if suffix in {".gltf", ".glb", ".wrl", ".vrml"}:
        try:
            importlib.import_module("trimesh")
        except ImportError:
            return None
        return TrimeshTessellator(limits)
    if suffix in {".step", ".stp", ".iges", ".igs", ".brep", ".fcstd"}:
        executable = shutil.which("FreeCADCmd") or shutil.which("freecadcmd")
        if executable is None:
            return None
        return FreeCADTessellator(executable, limits)
    return None


def missing_backend_message(source: str | Path) -> str:
    suffix = Path(source).suffix.casefold()
    if suffix in {".step", ".stp", ".iges", ".igs", ".brep", ".fcstd"}:
        return (
            f"{suffix} requires the FreeCAD command-line CAD kernel "
            "(FreeCADCmd/freecadcmd) or an explicit deterministic Tessellator"
        )
    if suffix in {".gltf", ".glb", ".wrl", ".vrml"}:
        return (
            f"{suffix} requires the optional 'geometry' dependency "
            "(trimesh>=4.8,<5) or an explicit deterministic Tessellator"
        )
    return f"{suffix} requires an explicit deterministic Tessellator"


def backend_identity(tessellator: Any) -> dict[str, str]:
    result = {
        "id": str(
            getattr(
                tessellator,
                "backend_id",
                getattr(tessellator, "__qualname__", type(tessellator).__qualname__),
            )
        ),
        "version": str(getattr(tessellator, "backend_version", "unversioned")),
    }
    digest = getattr(tessellator, "executable_sha256", None)
    if digest is not None:
        if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            raise GeometryBackendError("backend executable digest is invalid")
        result["executable_sha256"] = str(digest)
    if result["version"] in {"", "None", "unversioned", "system", "unknown"}:
        raise GeometryBackendError("deterministic backend must have an exact version")
    return result


__all__ = [
    "FreeCADTessellator",
    "GeometryBackendError",
    "GeometryLimits",
    "NativePlyTessellator",
    "TrimeshTessellator",
    "automatic_tessellator",
    "backend_identity",
    "linked_source_relative_paths",
    "linked_source_paths",
    "missing_backend_message",
    "native_ply_tessellator",
]
