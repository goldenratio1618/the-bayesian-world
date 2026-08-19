"""Deterministic, agent-independent shape and optical-property ingestion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
from typing import Any, Callable, Iterable

import numpy as np

from .artifacts import (
    ContentReference,
    DerivedMassProperties,
    OpticalMaterial,
    OpticalMaterialLibrary,
    PhysicalField,
    ShapeArtifact,
    ShapeArtifactError,
    ShapeUncertainty,
    SourceRepresentation,
    SurfaceRepresentation,
)
from .mechanics import MassPropertyError, mass_properties
from .mesh import MeshError, TriangleMesh
from .backends import (
    GeometryBackendError,
    automatic_tessellator,
    missing_backend_message,
    native_ply_tessellator,
)


class ShapeImportError(ShapeArtifactError):
    """Raised when deterministic source conversion cannot be completed."""


@dataclass(frozen=True, slots=True)
class ImportResult:
    manifest_path: Path
    artifact: ShapeArtifact
    imported_sources: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class TessellatedShape:
    """Complete deterministic output of an external CAD/GLB tessellator."""

    mesh: TriangleMesh
    optical_materials: tuple[OpticalMaterial, ...] = ()
    linked_sources: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, TriangleMesh):
            raise TypeError("tessellated shape mesh must be TriangleMesh")
        materials = tuple(self.optical_materials)
        if any(not isinstance(item, OpticalMaterial) for item in materials):
            raise TypeError("tessellated optical materials must be OpticalMaterial records")
        sources = tuple(Path(item).resolve() for item in self.linked_sources)
        if any(not item.is_file() for item in sources):
            raise FileNotFoundError(next(item for item in sources if not item.is_file()))
        object.__setattr__(self, "optical_materials", materials)
        object.__setattr__(self, "linked_sources", sources)


Tessellator = Callable[[Path, float], TessellatedShape]


_MEDIA_TYPES = {
    ".obj": "model/obj", ".mtl": "model/mtl", ".stl": "model/stl", ".ply": "model/ply",
    ".step": "model/step", ".stp": "model/step", ".iges": "model/iges", ".igs": "model/iges",
    ".brep": "model/brep", ".fcstd": "application/vnd.freecad",
    ".gltf": "model/gltf+json", ".glb": "model/gltf-binary", ".ctmesh": "application/vnd.contraption.ctmesh",
    ".wrl": "model/vrml", ".vrml": "model/vrml",
    ".json": "application/json",
}
_SOURCE_FORMATS = {
    ".obj": "obj", ".stl": "stl", ".ply": "ply", ".step": "step", ".stp": "step",
    ".iges": "iges", ".igs": "iges", ".brep": "brep",
    ".fcstd": "fcstd", ".gltf": "gltf", ".glb": "glb", ".wrl": "wrl", ".vrml": "wrl",
    ".ctmesh": "ctmesh",
}


def _slug(value: str) -> str:
    result = "-".join("".join(character.lower() if character.isalnum() else " " for character in value).split())
    return result or "shape"


def _copy_source(source: Path, root: Path) -> Path:
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    destination = root / "source" / f"{digest[:16]}-{source.name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise ShapeImportError(f"content-address collision for {source}")
    destination.write_bytes(payload)
    return destination


def _linear_channel(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _parse_mtl(path: Path) -> tuple[OpticalMaterial, ...]:
    records: list[tuple[str, dict[str, Any]]] = []
    current: dict[str, Any] | None = None
    name = ""
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        items = line.split()
        command, values = items[0].lower(), items[1:]
        if command == "newmtl":
            if not values:
                raise ShapeImportError(f"{path}:{line_number}: newmtl requires a name")
            if current is not None:
                records.append((name, current))
            name, current = "_".join(values), {}
            continue
        if current is None:
            continue
        try:
            if command in {"kd", "ke", "tf"} and len(values) >= 3:
                current[command] = tuple(float(item) for item in values[:3])
            elif command in {"d", "tr", "ni", "ns", "pr", "pm", "ps", "pc", "pcr"} and values:
                current[command] = float(values[0])
            elif command == "illum" and values:
                current[command] = int(values[0])
            elif command.startswith("map_") or command in {
                "bump",
                "decal",
                "disp",
                "refl",
            }:
                raise ShapeImportError(
                    f"{path}:{line_number}: texture/UV material maps cannot be "
                    "represented by canonical CTMESH"
                )
        except ShapeImportError:
            raise
        except ValueError as exc:
            raise ShapeImportError(f"{path}:{line_number}: invalid MTL numeric value") from exc
    if current is not None:
        records.append((name, current))
    materials: list[OpticalMaterial] = []
    for material_name, values in records:
        diffuse = values.get("kd", (0.5, 0.5, 0.5))
        alpha = values.get("d", 1.0 - values.get("tr", 0.0))
        transmission = min(1.0, max(0.0, 1.0 - alpha))
        illum = values.get("illum", 2)
        refractive = values.get("ni", 1.5)
        roughness = values.get("pr")
        if roughness is None:
            shininess = min(1000.0, max(0.0, values.get("ns", 10.0)))
            roughness = math.sqrt(2.0 / (shininess + 2.0))
        metallic = min(1.0, max(0.0, values.get("pm", 0.0)))
        model = "dielectric" if transmission > 0.0 or illum in {4, 6, 7, 9} else ("conductor" if metallic > 0.5 else "principled")
        emission = values.get("ke", (0.0, 0.0, 0.0))
        if any(value > 0.0 for value in emission):
            model = "emissive" if max(diffuse) <= 0.0 else model
        materials.append(
            OpticalMaterial(
                id=_slug(material_name), model=model,
                base_color_linear_rgba=(*(_linear_channel(value) for value in diffuse), min(1.0, max(0.0, alpha))),
                roughness=min(1.0, max(0.0, roughness)), metallic=metallic, transmission=transmission,
                refractive_index=max(1.0, refractive), emission_linear_rgb=tuple(max(0.0, value) for value in emission),
                uncertainty=ShapeUncertainty(
                    "uniform",
                    {"reason": "MTL contains nominal values but no metrology uncertainty"},
                ),
                provenance={"kind": "source", "format": "mtl", "file": path.name, "material_name": material_name},
            )
        )
    return tuple(materials)


def _obj_mesh(path: Path, scale: float) -> tuple[TriangleMesh, tuple[OpticalMaterial, ...], tuple[Path, ...]]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    face_material: list[int] = []
    material_names: list[str] = []
    current_material = -1
    libraries: list[Path] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        items = line.split()
        command, values = items[0].lower(), items[1:]
        try:
            if command == "v" and len(values) >= 3:
                vertices.append([float(item) * scale for item in values[:3]])
            elif command == "mtllib":
                for item in values:
                    library = (path.parent / item).resolve()
                    if path.parent.resolve() != library.parent and path.parent.resolve() not in library.parents:
                        raise ShapeImportError(f"{path}:{line_number}: MTL reference escapes source directory")
                    libraries.append(library)
            elif command == "usemtl":
                name = "_".join(values)
                if name not in material_names:
                    material_names.append(name)
                current_material = material_names.index(name)
            elif command == "f" and len(values) >= 3:
                indices: list[int] = []
                for item in values:
                    raw_index = int(item.split("/", 1)[0])
                    index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                    if not 0 <= index < len(vertices):
                        raise ShapeImportError(f"{path}:{line_number}: vertex index is out of range")
                    indices.append(index)
                for offset in range(1, len(indices) - 1):
                    triangles.append([indices[0], indices[offset], indices[offset + 1]])
                    face_material.append(max(0, current_material))
        except ValueError as exc:
            raise ShapeImportError(f"{path}:{line_number}: invalid OBJ value") from exc
    if not vertices or not triangles:
        raise ShapeImportError(f"{path}: OBJ contains no triangle surface")
    imported_materials: list[OpticalMaterial] = []
    for library in libraries:
        if not library.is_file():
            raise ShapeImportError(f"OBJ material library is missing: {library}")
        imported_materials.extend(_parse_mtl(library))
    by_source_name = {_slug(item.id): item for item in imported_materials}
    ordered: list[OpticalMaterial] = []
    for name in material_names:
        key = _slug(name)
        ordered.append(by_source_name.get(key, OpticalMaterial(key, provenance={"kind": "inferred", "source": "OBJ material name"}, uncertainty=ShapeUncertainty("uniform", {"reason": "no material properties"}))))
    if not ordered:
        ordered = list(imported_materials)
    material_indices = np.asarray(face_material, dtype=np.uint32) if ordered else None
    mesh = TriangleMesh(vertices, triangles, face_material=material_indices).with_computed_normals()
    return mesh, tuple(ordered), tuple(dict.fromkeys(libraries))


def _stl_mesh(path: Path, scale: float) -> TriangleMesh:
    payload = path.read_bytes()
    triangles_xyz: list[list[tuple[float, float, float]]] = []
    if len(payload) >= 84:
        count = struct.unpack_from("<I", payload, 80)[0]
        if len(payload) == 84 + count * 50:
            for index in range(count):
                offset = 84 + index * 50 + 12
                points = struct.unpack_from("<9f", payload, offset)
                triangles_xyz.append([tuple(points[item:item + 3]) for item in (0, 3, 6)])
    if not triangles_xyz:
        current: list[tuple[float, float, float]] = []
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ShapeImportError(f"{path}: invalid binary STL length") from exc
        for line_number, raw in enumerate(text.splitlines(), 1):
            items = raw.strip().split()
            if items and items[0].lower() == "vertex":
                if len(items) != 4:
                    raise ShapeImportError(f"{path}:{line_number}: malformed STL vertex")
                try:
                    current.append(tuple(float(value) for value in items[1:]))
                except ValueError as exc:
                    raise ShapeImportError(f"{path}:{line_number}: invalid STL coordinate") from exc
                if len(current) == 3:
                    triangles_xyz.append(current)
                    current = []
    if not triangles_xyz:
        raise ShapeImportError(f"{path}: STL contains no triangles")
    vertices: list[list[float]] = []
    indices: dict[tuple[float, float, float], int] = {}
    faces: list[list[int]] = []
    for triangle in triangles_xyz:
        face: list[int] = []
        for point in triangle:
            scaled = tuple(float(value) * scale for value in point)
            if scaled not in indices:
                indices[scaled] = len(vertices)
                vertices.append(list(scaled))
            face.append(indices[scaled])
        faces.append(face)
    return TriangleMesh(vertices, faces).with_computed_normals()


def _load_mesh(path: Path, scale: float, tessellator: Tessellator | None) -> tuple[TriangleMesh, tuple[OpticalMaterial, ...], tuple[Path, ...]]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _obj_mesh(path, scale)
    if suffix == ".stl":
        return _stl_mesh(path, scale), (), ()
    if suffix == ".ctmesh":
        mesh = TriangleMesh.read(path)
        if scale != 1.0:
            mesh = TriangleMesh(mesh.vertices_m * scale, mesh.triangles, mesh.vertex_normals, mesh.vertex_rgba_linear, mesh.face_material)
        return mesh, (), ()
    if suffix == ".ply":
        result = native_ply_tessellator(path, scale)
        return result.mesh, result.optical_materials, result.linked_sources
    if suffix in {".step", ".stp", ".iges", ".igs", ".brep", ".fcstd", ".gltf", ".glb", ".wrl", ".vrml"}:
        selected = tessellator or automatic_tessellator(path)
        if selected is None:
            raise ShapeImportError(
                missing_backend_message(path)
                + "; source evidence was not approximated or replaced by a bounding box"
            )
        result = selected(path, scale)
        if not isinstance(result, TessellatedShape):
            raise ShapeImportError(
                "tessellator must return TessellatedShape with canonical mesh and available optical materials"
            )
        return result.mesh, result.optical_materials, result.linked_sources
    raise ShapeImportError(f"unsupported source geometry extension {suffix!r}")


def _sidecar_materials(source: Path) -> tuple[OpticalMaterial, ...]:
    candidates = (source.with_suffix(source.suffix + ".optical.json"), source.with_suffix(".optical.json"))
    for candidate in candidates:
        if candidate.is_file():
            return OpticalMaterialLibrary.load(candidate).materials
    return ()


def import_shape(
    source: str | Path,
    output_directory: str | Path,
    *,
    artifact_id: str | None = None,
    version: str = "1.0.0",
    metres_per_source_unit: float = 1.0,
    density_kg_m3: float | None = None,
    optical_materials: Iterable[OpticalMaterial] = (),
    tessellator: Tessellator | None = None,
    surface_uncertainty: ShapeUncertainty | None = None,
    provenance: dict[str, Any] | None = None,
) -> ImportResult:
    """Import a source deterministically and emit source-independent surfaces.

    This function is deliberately free of Luna/LLM calls.  Available MTL and
    ``*.optical.json`` properties are parsed by strict deterministic code.
    """

    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    scale = float(metres_per_source_unit)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ShapeImportError("metres_per_source_unit must be finite and positive")
    suffix = source_path.suffix.lower()
    if suffix not in _SOURCE_FORMATS:
        raise ShapeImportError(f"unsupported source extension {suffix!r}")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        mesh, discovered, linked_sources = _load_mesh(source_path, scale, tessellator)
    except (OSError, UnicodeError, MeshError, GeometryBackendError) as exc:
        raise ShapeImportError(f"cannot import {source_path}: {exc}") from exc
    explicit = tuple(optical_materials)
    sidecar = _sidecar_materials(source_path)
    materials = explicit or sidecar or discovered
    if not materials:
        materials = (
            OpticalMaterial(
                "uncharacterized", model="principled",
                uncertainty=ShapeUncertainty("uniform", {"base_color_linear": [0.0, 1.0], "roughness": [0.0, 1.0]}),
                provenance={"kind": "inferred", "reason": "source contains no optical properties"},
            ),
        )
    if mesh.face_material is not None and int(mesh.face_material.max(initial=0)) >= len(materials):
        raise ShapeImportError("mesh material-region indices exceed imported optical material count")

    imported: list[Path] = []
    source_records: list[SourceRepresentation] = []
    for index, linked in enumerate((source_path, *linked_sources)):
        copied = _copy_source(linked, output)
        imported.append(copied)
        linked_suffix = linked.suffix.lower()
        linked_format = _SOURCE_FORMATS.get(linked_suffix, "material_library")
        source_records.append(
            SourceRepresentation(
                id="primary" if index == 0 else f"linked-{index}", format=linked_format,
                content=ContentReference.from_path(copied, relative_to=output, media_type=_MEDIA_TYPES.get(linked_suffix, "application/octet-stream")),
                metres_per_source_unit=scale if index == 0 else 1.0,
                provenance={"kind": "imported", "original_name": linked.name},
            )
        )
    sidecar_candidates = (source_path.with_suffix(source_path.suffix + ".optical.json"), source_path.with_suffix(".optical.json"))
    for candidate in sidecar_candidates:
        if candidate.is_file():
            copied = _copy_source(candidate, output)
            imported.append(copied)
            source_records.append(SourceRepresentation(f"optical-sidecar-{len(source_records)}", "material_library", ContentReference.from_path(copied, relative_to=output, media_type="application/json"), 1.0, provenance={"kind": "imported", "original_name": candidate.name}))
            break

    canonical_path = output / "canonical.ctmesh"
    canonical_path.write_bytes(mesh.to_bytes())
    runtime_path = output / "runtime.glb"
    runtime_path.write_bytes(mesh.to_glb_bytes())
    low, high = mesh.bounds_m
    if surface_uncertainty is None:
        maximum_extent = float(np.max(high - low))
        half_width = max(1e-7, maximum_extent * 0.005)
        surface_uncertainty = ShapeUncertainty(
            "uniform",
            {
                "lower_m": -half_width,
                "upper_m": half_width,
                "basis": "conservative default because source geometry declares no metrology uncertainty",
            },
        )
    elif not isinstance(surface_uncertainty, ShapeUncertainty):
        raise TypeError("surface_uncertainty must be ShapeUncertainty")
    canonical = SurfaceRepresentation(
        "canonical", "ctmesh", ContentReference.from_path(canonical_path, relative_to=output, media_type="application/vnd.contraption.ctmesh"),
        ("analysis", "ray_trace", "render", "collision"), len(mesh.vertices_m), len(mesh.triangles),
        tuple(float(value) for value in np.concatenate((low, high))), mesh.watertight, mesh.manifold,
        tuple(item.id for item in materials), surface_uncertainty,
    )
    fields: tuple[PhysicalField, ...] = ()
    derived: DerivedMassProperties | None = None
    if density_kg_m3 is not None:
        density = float(density_kg_m3)
        fields = (PhysicalField("mass-density", "mass_density", "kg/m^3", "constant", density),)
        if mesh.closed_oriented_manifold:
            try:
                properties = mass_properties(mesh, density)
            except MassPropertyError as exc:
                raise ShapeImportError(f"cannot derive mass properties: {exc}") from exc
            derived = DerivedMassProperties(
                canonical.id, fields[0].id, properties.mass_kg, properties.volume_m3,
                tuple(float(value) for value in properties.center_of_mass_m),
                tuple(float(value) for value in properties.inertia_kg_m2.reshape(-1)),
                surface_uncertainty,
            )
    artifact = ShapeArtifact(
        id=artifact_id or _slug(source_path.stem), version=version, sources=tuple(source_records), surfaces=(canonical,),
        optical_materials=materials, physical_fields=fields, derived_mass_properties=derived,
        caches=(ContentReference.from_path(runtime_path, relative_to=output, media_type="model/gltf-binary"),),
        provenance=provenance or {"kind": "deterministic-import", "source_name": source_path.name},
        metadata={"importer": "contraption.shape.import_shape", "source_format": _SOURCE_FORMATS[suffix]},
    )
    manifest = output / "shape.artifact.json"
    artifact.write(manifest)
    loaded = ShapeArtifact.load(manifest)
    return ImportResult(manifest, loaded, tuple(imported))
