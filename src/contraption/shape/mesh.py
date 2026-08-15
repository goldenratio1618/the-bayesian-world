"""Deterministic canonical triangle meshes and a compact binary encoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np

from ..strict_json import loads_strict_json


_MAGIC = b"CTMESH1\n"
_HEADER_LENGTH = struct.Struct("<I")


class MeshError(ValueError):
    """Raised when mesh topology or serialization is invalid."""


def _array(value: Any, width: int, dtype: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 2 or result.shape[1] != width:
        raise MeshError(f"{label} must have shape [N, {width}]")
    if not np.all(np.isfinite(result)):
        raise MeshError(f"{label} contains NaN or infinity")
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    """Canonical right-handed metric triangle surface.

    Vertices are float64 in memory for analysis. ``.ctmesh`` stores
    little-endian float32 vertices/normals and uint32 topology for transfer.
    """

    vertices_m: np.ndarray
    triangles: np.ndarray
    vertex_normals: np.ndarray | None = None
    vertex_rgba_linear: np.ndarray | None = None
    face_material: np.ndarray | None = None

    def __post_init__(self) -> None:
        vertices = _array(self.vertices_m, 3, np.float64, "vertices_m")
        triangles = np.asarray(self.triangles)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise MeshError("triangles must have shape [M, 3]")
        if triangles.dtype.kind not in {"i", "u"}:
            raise MeshError("triangles must contain integer indices")
        triangles = np.ascontiguousarray(triangles, dtype=np.uint32)
        if len(vertices) < 3 or len(triangles) < 1:
            raise MeshError("a triangle mesh requires at least three vertices and one face")
        if int(triangles.max(initial=0)) >= len(vertices):
            raise MeshError("triangle index exceeds the vertex array")
        if np.any(
            (triangles[:, 0] == triangles[:, 1])
            | (triangles[:, 1] == triangles[:, 2])
            | (triangles[:, 0] == triangles[:, 2])
        ):
            raise MeshError("triangles may not repeat a vertex index")
        faces = vertices[triangles]
        twice_area = np.cross(
            faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]
        )
        if np.any(np.einsum("ij,ij->i", twice_area, twice_area) <= 0.0):
            raise MeshError("mesh contains zero-area or collinear triangles")
        normals = self.vertex_normals
        if normals is not None:
            normals = _array(normals, 3, np.float64, "vertex_normals")
            if len(normals) != len(vertices):
                raise MeshError("vertex_normals count must equal vertices_m count")
            lengths = np.linalg.norm(normals, axis=1)
            if np.any(lengths <= 1e-15):
                raise MeshError("vertex normals must be nonzero")
            normals = normals / lengths[:, None]
        colors = self.vertex_rgba_linear
        if colors is not None:
            colors = _array(colors, 4, np.float64, "vertex_rgba_linear")
            if len(colors) != len(vertices) or np.any((colors < 0) | (colors > 1)):
                raise MeshError("vertex RGBA values must be one per vertex in [0, 1]")
        materials = self.face_material
        if materials is not None:
            materials = np.asarray(materials)
            if materials.ndim != 1 or len(materials) != len(triangles):
                raise MeshError("face_material must contain one integer per triangle")
            if materials.dtype.kind not in {"i", "u"} or np.any(materials < 0):
                raise MeshError("face_material values must be nonnegative integers")
            materials = np.ascontiguousarray(materials, dtype=np.uint32)
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "vertex_normals", normals)
        object.__setattr__(self, "vertex_rgba_linear", colors)
        object.__setattr__(self, "face_material", materials)

    @property
    def bounds_m(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices_m.min(axis=0), self.vertices_m.max(axis=0)

    @property
    def dimensions_m(self) -> tuple[float, float, float]:
        low, high = self.bounds_m
        return tuple(float(value) for value in high - low)

    @property
    def watertight(self) -> bool:
        """Whether every undirected edge has exactly two incident faces."""

        _edges, counts, _directions = self._edge_topology()
        return bool(np.all(counts == 2))

    def _edge_topology(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = np.concatenate(
            (
                self.triangles[:, [0, 1]],
                self.triangles[:, [1, 2]],
                self.triangles[:, [2, 0]],
            ),
            axis=0,
        )
        directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
        undirected = np.sort(edges, axis=1)
        unique, inverse, counts = np.unique(
            undirected, axis=0, return_inverse=True, return_counts=True
        )
        direction_sums = np.zeros(len(unique), dtype=np.int64)
        np.add.at(direction_sums, inverse, directions)
        return unique, counts, direction_sums

    @property
    def consistently_oriented(self) -> bool:
        """Whether paired surface edges are traversed in opposite directions."""

        _edges, counts, direction_sums = self._edge_topology()
        return bool(np.all((counts != 2) | (direction_sums == 0)))

    @property
    def manifold(self) -> bool:
        """Whether each vertex has a single disk or half-disk neighbourhood."""

        triangles = np.asarray(self.triangles, dtype=np.int64)
        if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
            return False
        _edges, edge_counts, _directions = self._edge_topology()
        if np.any(edge_counts > 2):
            return False
        incident: list[list[tuple[int, int]]] = [[] for _ in self.vertices_m]
        for a, b, c in triangles:
            incident[a].append((int(b), int(c)))
            incident[b].append((int(c), int(a)))
            incident[c].append((int(a), int(b)))
        for links in incident:
            if not links:
                return False
            adjacency: dict[int, set[int]] = {}
            for first, second in links:
                adjacency.setdefault(first, set()).add(second)
                adjacency.setdefault(second, set()).add(first)
            if any(len(neighbours) not in {1, 2} for neighbours in adjacency.values()):
                return False
            degree_one = sum(len(neighbours) == 1 for neighbours in adjacency.values())
            if degree_one not in {0, 2}:
                return False
            pending = [next(iter(adjacency))]
            visited: set[int] = set()
            while pending:
                vertex = pending.pop()
                if vertex in visited:
                    continue
                visited.add(vertex)
                pending.extend(adjacency[vertex] - visited)
            if len(visited) != len(adjacency):
                return False
        return True

    @property
    def closed_oriented_manifold(self) -> bool:
        return self.watertight and self.manifold and self.consistently_oriented

    def with_computed_normals(self) -> "TriangleMesh":
        faces = self.vertices_m[self.triangles]
        face_normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        lengths = np.linalg.norm(face_normals, axis=1)
        if np.any(lengths <= 1e-18):
            raise MeshError("mesh contains zero-area triangles")
        normals = np.zeros_like(self.vertices_m)
        for corner in range(3):
            np.add.at(normals, self.triangles[:, corner], face_normals)
        normal_lengths = np.linalg.norm(normals, axis=1)
        if np.any(normal_lengths <= 1e-18):
            raise MeshError("mesh contains vertices without a defined normal")
        return TriangleMesh(
            self.vertices_m,
            self.triangles,
            normals / normal_lengths[:, None],
            self.vertex_rgba_linear,
            self.face_material,
        )

    def to_inline_dict(self) -> dict[str, Any]:
        mesh = self if self.vertex_normals is not None else self.with_computed_normals()
        result: dict[str, Any] = {
            "schema": "contraption.triangle-mesh/v1",
            "vertices_m": mesh.vertices_m.tolist(),
            "triangles": mesh.triangles.astype(np.int64).tolist(),
            "vertex_normals": mesh.vertex_normals.tolist(),
        }
        if mesh.vertex_rgba_linear is not None:
            result["vertex_rgba_linear"] = mesh.vertex_rgba_linear.tolist()
        if mesh.face_material is not None:
            result["face_material"] = mesh.face_material.astype(np.int64).tolist()
        return result

    @classmethod
    def from_inline_dict(cls, value: dict[str, Any]) -> "TriangleMesh":
        allowed = {
            "schema",
            "vertices_m",
            "triangles",
            "vertex_normals",
            "vertex_rgba_linear",
            "face_material",
        }
        if set(value) - allowed or value.get("schema") != "contraption.triangle-mesh/v1":
            raise MeshError("invalid inline canonical triangle mesh")
        return cls(
            value["vertices_m"],
            value["triangles"],
            value.get("vertex_normals"),
            value.get("vertex_rgba_linear"),
            value.get("face_material"),
        )

    def to_bytes(self) -> bytes:
        mesh = self if self.vertex_normals is not None else self.with_computed_normals()
        arrays: list[tuple[str, np.ndarray]] = [
            ("vertices_m", np.asarray(mesh.vertices_m, dtype="<f4")),
            ("triangles", np.asarray(mesh.triangles, dtype="<u4")),
            ("vertex_normals", np.asarray(mesh.vertex_normals, dtype="<f4")),
        ]
        if mesh.vertex_rgba_linear is not None:
            arrays.append(("vertex_rgba_linear", np.asarray(mesh.vertex_rgba_linear, dtype="<f4")))
        if mesh.face_material is not None:
            arrays.append(("face_material", np.asarray(mesh.face_material, dtype="<u4")))
        offset = 0
        descriptors: list[dict[str, Any]] = []
        payloads: list[bytes] = []
        for name, array in arrays:
            raw = array.tobytes(order="C")
            descriptors.append(
                {
                    "name": name,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "offset": offset,
                    "bytes": len(raw),
                }
            )
            payloads.append(raw)
            offset += len(raw)
        header = json.dumps(
            {"schema": "contraption.ctmesh/v1", "arrays": descriptors},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return _MAGIC + _HEADER_LENGTH.pack(len(header)) + header + b"".join(payloads)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "TriangleMesh":
        if not payload.startswith(_MAGIC) or len(payload) < len(_MAGIC) + 4:
            raise MeshError("not a canonical CTMESH1 payload")
        start = len(_MAGIC)
        header_length = _HEADER_LENGTH.unpack(payload[start : start + 4])[0]
        header_start = start + 4
        body_start = header_start + header_length
        if body_start > len(payload):
            raise MeshError("CTMESH header escapes payload")
        try:
            header = loads_strict_json(payload[header_start:body_start])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MeshError(f"invalid CTMESH header: {exc}") from exc
        if set(header) != {"schema", "arrays"} or header["schema"] != "contraption.ctmesh/v1":
            raise MeshError("unsupported CTMESH schema")
        descriptors = header["arrays"]
        if not isinstance(descriptors, list):
            raise MeshError("CTMESH arrays must be a list")
        names = [
            item.get("name") if isinstance(item, dict) else None
            for item in descriptors
        ]
        required_order = ["vertices_m", "triangles", "vertex_normals"]
        optional_order = [
            name
            for name in ("vertex_rgba_linear", "face_material")
            if name in names
        ]
        if names != required_order + optional_order:
            raise MeshError("CTMESH arrays are not in canonical order")
        expected_dtypes = {
            "vertices_m": "<f4",
            "triangles": "<u4",
            "vertex_normals": "<f4",
            "vertex_rgba_linear": "<f4",
            "face_material": "<u4",
        }
        expected_widths = {
            "vertices_m": 3,
            "triangles": 3,
            "vertex_normals": 3,
            "vertex_rgba_linear": 4,
        }
        arrays: dict[str, np.ndarray] = {}
        expected_offset = 0
        for item in descriptors:
            if set(item) != {"name", "dtype", "shape", "offset", "bytes"}:
                raise MeshError("invalid CTMESH array descriptor")
            name = item["name"]
            if item["dtype"] != expected_dtypes[name]:
                raise MeshError(f"CTMESH {name} has noncanonical dtype")
            shape = item["shape"]
            if (
                not isinstance(shape, list)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in shape
                )
            ):
                raise MeshError(f"CTMESH {name} has invalid shape")
            valid_shape = (
                len(shape) == 1
                if name == "face_material"
                else len(shape) == 2 and shape[1] == expected_widths[name]
            )
            if not valid_shape:
                raise MeshError(f"CTMESH {name} has invalid shape")
            dtype = np.dtype(item["dtype"])
            byte_length = math.prod(shape) * dtype.itemsize
            if (
                isinstance(item["offset"], bool)
                or not isinstance(item["offset"], int)
                or isinstance(item["bytes"], bool)
                or not isinstance(item["bytes"], int)
                or item["offset"] != expected_offset
                or item["bytes"] != byte_length
            ):
                raise MeshError(f"CTMESH {name} is not canonically packed")
            offset = body_start + expected_offset
            end = offset + byte_length
            if end > len(payload):
                raise MeshError("CTMESH array escapes payload")
            array = np.frombuffer(payload[offset:end], dtype=dtype)
            try:
                arrays[name] = array.reshape(tuple(shape)).copy()
            except ValueError as exc:
                raise MeshError("CTMESH array shape/length mismatch") from exc
            expected_offset += byte_length
        if body_start + expected_offset != len(payload):
            raise MeshError("CTMESH payload has trailing or unreferenced bytes")
        if arrays["vertex_normals"].shape != arrays["vertices_m"].shape:
            raise MeshError("CTMESH normal count differs from vertex count")
        if (
            "vertex_rgba_linear" in arrays
            and len(arrays["vertex_rgba_linear"]) != len(arrays["vertices_m"])
        ):
            raise MeshError("CTMESH color count differs from vertex count")
        if (
            "face_material" in arrays
            and len(arrays["face_material"]) != len(arrays["triangles"])
        ):
            raise MeshError("CTMESH face-material count differs from triangle count")
        normal_lengths = np.linalg.norm(
            arrays["vertex_normals"].astype(np.float64), axis=1
        )
        if np.any(np.abs(normal_lengths - 1.0) > 1e-5):
            raise MeshError("CTMESH vertex normals must be unit length")
        result = cls(
            arrays["vertices_m"],
            arrays["triangles"],
            arrays["vertex_normals"],
            arrays.get("vertex_rgba_linear"),
            arrays.get("face_material"),
        )
        return result

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_bytes())
        return target

    @classmethod
    def read(cls, path: str | Path) -> "TriangleMesh":
        return cls.from_bytes(Path(path).read_bytes())

    def to_glb_bytes(self) -> bytes:
        """Return a minimal deterministic glTF 2.0 binary runtime surface."""

        mesh = self if self.vertex_normals is not None else self.with_computed_normals()
        positions = np.asarray(mesh.vertices_m, dtype="<f4").tobytes()
        normals = np.asarray(mesh.vertex_normals, dtype="<f4").tobytes()
        indices = np.asarray(mesh.triangles.reshape(-1), dtype="<u4").tobytes()
        chunks: list[bytes] = []
        views: list[dict[str, int]] = []
        offset = 0
        for raw, target in ((positions, 34962), (normals, 34962), (indices, 34963)):
            padding = (-offset) % 4
            if padding:
                chunks.append(b"\0" * padding)
                offset += padding
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
            chunks.append(raw)
            offset += len(raw)
        binary = b"".join(chunks)
        low, high = mesh.bounds_m
        document = {
            "asset": {"version": "2.0", "generator": "contraption.shape CTMESH1"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {"POSITION": 0, "NORMAL": 1},
                            "indices": 2,
                            "mode": 4,
                            "material": 0,
                        }
                    ]
                }
            ],
            "materials": [
                {
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [0.65, 0.68, 0.72, 1.0],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.65,
                    }
                }
            ],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": views,
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": len(mesh.vertices_m),
                    "type": "VEC3",
                    "min": low.astype(float).tolist(),
                    "max": high.astype(float).tolist(),
                },
                {"bufferView": 1, "componentType": 5126, "count": len(mesh.vertices_m), "type": "VEC3"},
                {"bufferView": 2, "componentType": 5125, "count": int(mesh.triangles.size), "type": "SCALAR"},
            ],
        }
        json_chunk = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        json_chunk += b" " * ((-len(json_chunk)) % 4)
        binary += b"\0" * ((-len(binary)) % 4)
        total = 12 + 8 + len(json_chunk) + 8 + len(binary)
        return (
            struct.pack("<4sII", b"glTF", 2, total)
            + struct.pack("<I4s", len(json_chunk), b"JSON")
            + json_chunk
            + struct.pack("<I4s", len(binary), b"BIN\0")
            + binary
        )


def box_mesh(dimensions_m: Iterable[float]) -> TriangleMesh:
    x, y, z = (float(value) / 2.0 for value in dimensions_m)
    vertices = np.asarray(
        [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.uint32,
    )
    return TriangleMesh(vertices, triangles).with_computed_normals()


def cylinder_mesh(diameter_m: float, length_m: float, segments: int = 32) -> TriangleMesh:
    if segments < 8:
        raise MeshError("a cylinder requires at least eight segments")
    radius, half = float(diameter_m) / 2.0, float(length_m) / 2.0
    vertices = [[0.0, 0.0, -half], [0.0, 0.0, half]]
    for z in (-half, half):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append([radius * math.cos(angle), radius * math.sin(angle), z])
    triangles: list[list[int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        bottom, bottom_next = 2 + index, 2 + nxt
        top, top_next = 2 + segments + index, 2 + segments + nxt
        triangles.extend(
            ([0, bottom_next, bottom], [1, top, top_next], [bottom, bottom_next, top_next], [bottom, top_next, top])
        )
    return TriangleMesh(vertices, triangles).with_computed_normals()


def sphere_mesh(diameter_m: float, rings: int = 16, segments: int = 32) -> TriangleMesh:
    if rings < 4 or segments < 8:
        raise MeshError("a sphere requires at least four rings and eight segments")
    radius = float(diameter_m) / 2.0
    if not math.isfinite(radius) or radius <= 0.0:
        raise MeshError("sphere diameter must be finite and positive")
    vertices: list[list[float]] = [[0.0, 0.0, radius], [0.0, 0.0, -radius]]
    for ring in range(1, rings):
        polar = math.pi * ring / rings
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append([radius * math.sin(polar) * math.cos(angle), radius * math.sin(polar) * math.sin(angle), radius * math.cos(polar)])
    triangles: list[list[int]] = []
    first_ring, last_ring = 2, 2 + (rings - 2) * segments
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.extend(([0, first_ring + index, first_ring + nxt], [1, last_ring + nxt, last_ring + index]))
    for ring in range(rings - 2):
        first, second = 2 + ring * segments, 2 + (ring + 1) * segments
        for index in range(segments):
            nxt = (index + 1) % segments
            triangles.extend(([first + index, second + index, second + nxt], [first + index, second + nxt, first + nxt]))
    return TriangleMesh(vertices, triangles).with_computed_normals()


def combine_meshes(items: Iterable[tuple[TriangleMesh, np.ndarray | None]]) -> TriangleMesh:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    face_materials: list[np.ndarray] = []
    has_materials = False
    offset = 0
    for mesh, transform in items:
        current = mesh.vertices_m
        if transform is not None:
            matrix = np.asarray(transform, dtype=float)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise MeshError("mesh transform must be a finite 4x4 matrix")
            homogeneous = np.concatenate((current, np.ones((len(current), 1))), axis=1)
            current = (homogeneous @ matrix.T)[:, :3]
        vertices.append(current)
        faces.append(mesh.triangles.astype(np.uint64) + offset)
        if mesh.face_material is not None:
            has_materials = True
            face_materials.append(mesh.face_material)
        else:
            face_materials.append(np.zeros(len(mesh.triangles), dtype=np.uint32))
        offset += len(current)
    if not vertices:
        raise MeshError("cannot combine an empty mesh collection")
    materials = np.concatenate(face_materials) if has_materials else None
    return TriangleMesh(np.concatenate(vertices), np.concatenate(faces), face_material=materials).with_computed_normals()
