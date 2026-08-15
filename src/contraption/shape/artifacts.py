"""Strict, source-independent physical shape artifacts.

``shape-artifact-1`` keeps immutable source evidence separate from canonical
runtime representations.  Simulation code consumes only the canonical surface,
volume, material, and physical-field records; it never interprets source files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..strict_json import loads_strict_json


class ShapeArtifactError(ValueError):
    """Raised when a shape manifest or one of its content references is invalid."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ShapeArtifactError(f"{context} must be an object with string keys")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], required: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ShapeArtifactError(f"{context} has unknown keys: {', '.join(unknown)}")
    if missing:
        raise ShapeArtifactError(f"{context} is missing keys: {', '.join(missing)}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ShapeArtifactError(f"{context} must be a nonempty trimmed string")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ShapeArtifactError(f"{context} must be a finite number")
    return float(value)


def _vector(value: Any, length: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ShapeArtifactError(f"{context} must contain exactly {length} numbers")
    return tuple(_number(item, f"{context}[{index}]") for index, item in enumerate(value))


def _json_value(value: Any, context: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ShapeArtifactError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ShapeArtifactError(f"{context} contains a non-string key")
        return {key: _json_value(value[key], f"{context}.{key}") for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{context}[{index}]") for index, item in enumerate(value)]
    raise ShapeArtifactError(f"{context} contains unsupported type {type(value).__name__}")


def _relative_uri(value: Any, context: str) -> str:
    text = _text(value, context)
    if "\\" in text or "\x00" in text:
        raise ShapeArtifactError(f"{context} must be a POSIX relative URI")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ShapeArtifactError(f"{context} must remain below the manifest directory")
    return text


@dataclass(frozen=True, slots=True)
class ContentReference:
    uri: str
    media_type: str
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        _relative_uri(self.uri, "content.uri")
        _text(self.media_type, "content.media_type")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ShapeArtifactError("content.sha256 must be a lowercase SHA-256 hex digest")
        if isinstance(self.byte_length, bool) or not isinstance(self.byte_length, int) or self.byte_length < 0:
            raise ShapeArtifactError("content.byte_length must be a nonnegative integer")

    @classmethod
    def from_path(cls, path: str | Path, *, relative_to: str | Path, media_type: str) -> "ContentReference":
        source, root = Path(path).resolve(), Path(relative_to).resolve()
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ShapeArtifactError(f"content {source} is outside artifact root {root}") from exc
        payload = source.read_bytes()
        return cls(relative, media_type, hashlib.sha256(payload).hexdigest(), len(payload))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContentReference":
        value = _mapping(value, "content")
        names = {"uri", "media_type", "sha256", "byte_length"}
        _keys(value, names, names, "content")
        return cls(
            _relative_uri(value["uri"], "content.uri"),
            _text(value["media_type"], "content.media_type"),
            _text(value["sha256"], "content.sha256"),
            value["byte_length"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"uri": self.uri, "media_type": self.media_type, "sha256": self.sha256, "byte_length": self.byte_length}


@dataclass(frozen=True, slots=True)
class ShapeUncertainty:
    distribution: str = "fixed"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    correlation_group: str | None = None

    def __post_init__(self) -> None:
        if self.distribution not in {"fixed", "normal", "lognormal", "uniform", "triangular", "empirical"}:
            raise ShapeArtifactError(f"unsupported uncertainty distribution {self.distribution!r}")
        object.__setattr__(self, "parameters", _json_value(_mapping(self.parameters, "uncertainty.parameters")))
        if self.correlation_group is not None:
            _text(self.correlation_group, "uncertainty.correlation_group")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShapeUncertainty":
        value = _mapping(value, "uncertainty")
        names = {"distribution", "parameters", "correlation_group"}
        _keys(value, names, set(), "uncertainty")
        return cls(
            _text(value.get("distribution", "fixed"), "uncertainty.distribution"),
            _mapping(value.get("parameters", {}), "uncertainty.parameters"),
            None if value.get("correlation_group") is None else _text(value["correlation_group"], "uncertainty.correlation_group"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"distribution": self.distribution, "parameters": _json_value(self.parameters)}
        if self.correlation_group is not None:
            result["correlation_group"] = self.correlation_group
        return result


@dataclass(frozen=True, slots=True)
class SpectralOpticalSample:
    wavelength_nm: float
    reflectance: float | None = None
    transmittance: float | None = None
    refractive_index: float | None = None
    extinction_coefficient: float | None = None
    emission_w_sr_m2_nm: float | None = None

    def __post_init__(self) -> None:
        if not 100.0 <= _number(self.wavelength_nm, "spectrum.wavelength_nm") <= 1_000_000.0:
            raise ShapeArtifactError("spectrum wavelength is outside the supported optical range")
        for name in ("reflectance", "transmittance"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= _number(value, f"spectrum.{name}") <= 1.0:
                raise ShapeArtifactError(f"spectrum.{name} must be in [0, 1]")
        for name in ("refractive_index", "extinction_coefficient", "emission_w_sr_m2_nm"):
            value = getattr(self, name)
            if value is not None and _number(value, f"spectrum.{name}") < 0.0:
                raise ShapeArtifactError(f"spectrum.{name} must be nonnegative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpectralOpticalSample":
        value = _mapping(value, "spectral sample")
        names = {"wavelength_nm", "reflectance", "transmittance", "refractive_index", "extinction_coefficient", "emission_w_sr_m2_nm"}
        _keys(value, names, {"wavelength_nm"}, "spectral sample")
        def optional(name: str) -> float | None:
            return None if value.get(name) is None else _number(value[name], f"spectral sample.{name}")
        return cls(
            _number(value["wavelength_nm"], "spectral sample.wavelength_nm"),
            optional("reflectance"), optional("transmittance"), optional("refractive_index"),
            optional("extinction_coefficient"), optional("emission_w_sr_m2_nm"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {"wavelength_nm": self.wavelength_nm}
        for name in ("reflectance", "transmittance", "refractive_index", "extinction_coefficient", "emission_w_sr_m2_nm"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class OpticalMaterial:
    id: str
    model: str = "principled"
    base_color_linear_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    roughness: float = 0.5
    metallic: float = 0.0
    transmission: float = 0.0
    refractive_index: float = 1.5
    extinction_coefficient: float = 0.0
    absorption_per_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scattering_per_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    phase_anisotropy: float = 0.0
    emission_linear_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    double_sided: bool = False
    spectrum: tuple[SpectralOpticalSample, ...] = ()
    uncertainty: ShapeUncertainty = ShapeUncertainty()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.id, "optical_material.id")
        if self.model not in {"lambertian", "principled", "dielectric", "conductor", "emissive", "measured"}:
            raise ShapeArtifactError(f"unsupported optical material model {self.model!r}")
        color = _vector(self.base_color_linear_rgba, 4, "optical_material.base_color_linear_rgba")
        if any(not 0.0 <= value <= 1.0 for value in color):
            raise ShapeArtifactError("base color channels must be in [0, 1]")
        for name in ("roughness", "metallic", "transmission"):
            if not 0.0 <= _number(getattr(self, name), f"optical_material.{name}") <= 1.0:
                raise ShapeArtifactError(f"optical_material.{name} must be in [0, 1]")
        if _number(self.refractive_index, "optical_material.refractive_index") < 1.0:
            raise ShapeArtifactError("refractive index must be at least 1")
        if _number(self.extinction_coefficient, "optical_material.extinction_coefficient") < 0.0:
            raise ShapeArtifactError("extinction coefficient must be nonnegative")
        for name in ("absorption_per_m", "scattering_per_m", "emission_linear_rgb"):
            vector = _vector(getattr(self, name), 3, f"optical_material.{name}")
            if any(value < 0.0 for value in vector):
                raise ShapeArtifactError(f"optical_material.{name} must be nonnegative")
        if not -1.0 <= _number(self.phase_anisotropy, "optical_material.phase_anisotropy") <= 1.0:
            raise ShapeArtifactError("phase anisotropy must be in [-1, 1]")
        if not isinstance(self.double_sided, bool):
            raise ShapeArtifactError("optical_material.double_sided must be boolean")
        wavelengths = [sample.wavelength_nm for sample in self.spectrum]
        if wavelengths != sorted(set(wavelengths)):
            raise ShapeArtifactError("spectral samples must have unique increasing wavelengths")
        object.__setattr__(self, "base_color_linear_rgba", color)
        object.__setattr__(self, "absorption_per_m", _vector(self.absorption_per_m, 3, "optical_material.absorption_per_m"))
        object.__setattr__(self, "scattering_per_m", _vector(self.scattering_per_m, 3, "optical_material.scattering_per_m"))
        object.__setattr__(self, "emission_linear_rgb", _vector(self.emission_linear_rgb, 3, "optical_material.emission_linear_rgb"))
        object.__setattr__(self, "provenance", _json_value(_mapping(self.provenance, "optical_material.provenance")))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OpticalMaterial":
        value = _mapping(value, "optical material")
        names = {
            "id", "model", "base_color_linear_rgba", "roughness", "metallic", "transmission",
            "refractive_index", "extinction_coefficient", "absorption_per_m", "scattering_per_m",
            "phase_anisotropy", "emission_linear_rgb", "double_sided", "spectrum", "uncertainty", "provenance",
        }
        _keys(value, names, {"id"}, "optical material")
        return cls(
            id=_text(value["id"], "optical material.id"),
            model=_text(value.get("model", "principled"), "optical material.model"),
            base_color_linear_rgba=_vector(value.get("base_color_linear_rgba", (0.5, 0.5, 0.5, 1.0)), 4, "optical material.base_color_linear_rgba"),
            roughness=_number(value.get("roughness", 0.5), "optical material.roughness"),
            metallic=_number(value.get("metallic", 0.0), "optical material.metallic"),
            transmission=_number(value.get("transmission", 0.0), "optical material.transmission"),
            refractive_index=_number(value.get("refractive_index", 1.5), "optical material.refractive_index"),
            extinction_coefficient=_number(value.get("extinction_coefficient", 0.0), "optical material.extinction_coefficient"),
            absorption_per_m=_vector(value.get("absorption_per_m", (0, 0, 0)), 3, "optical material.absorption_per_m"),
            scattering_per_m=_vector(value.get("scattering_per_m", (0, 0, 0)), 3, "optical material.scattering_per_m"),
            phase_anisotropy=_number(value.get("phase_anisotropy", 0.0), "optical material.phase_anisotropy"),
            emission_linear_rgb=_vector(value.get("emission_linear_rgb", (0, 0, 0)), 3, "optical material.emission_linear_rgb"),
            double_sided=value.get("double_sided", False),
            spectrum=tuple(SpectralOpticalSample.from_dict(item) for item in value.get("spectrum", [])),
            uncertainty=ShapeUncertainty.from_dict(value.get("uncertainty", {})),
            provenance=_mapping(value.get("provenance", {}), "optical material.provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "model": self.model, "base_color_linear_rgba": list(self.base_color_linear_rgba),
            "roughness": self.roughness, "metallic": self.metallic, "transmission": self.transmission,
            "refractive_index": self.refractive_index, "extinction_coefficient": self.extinction_coefficient,
            "absorption_per_m": list(self.absorption_per_m), "scattering_per_m": list(self.scattering_per_m),
            "phase_anisotropy": self.phase_anisotropy, "emission_linear_rgb": list(self.emission_linear_rgb),
            "double_sided": self.double_sided, "spectrum": [sample.to_dict() for sample in self.spectrum],
            "uncertainty": self.uncertainty.to_dict(), "provenance": _json_value(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class OpticalMaterialLibrary:
    """Standalone deterministic optical data imported alongside a source shape."""

    id: str
    version: str
    materials: tuple[OpticalMaterial, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    format: str = "optical-material-1"

    def __post_init__(self) -> None:
        if self.format != "optical-material-1":
            raise ShapeArtifactError(f"unsupported optical material format {self.format!r}")
        _text(self.id, "optical material library.id")
        _text(self.version, "optical material library.version")
        if not self.materials:
            raise ShapeArtifactError("optical material libraries require at least one material")
        identifiers = [item.id for item in self.materials]
        if len(identifiers) != len(set(identifiers)):
            raise ShapeArtifactError("optical material library IDs must be unique")
        object.__setattr__(self, "provenance", _json_value(_mapping(self.provenance, "optical material library.provenance")))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OpticalMaterialLibrary":
        value = _mapping(value, "optical material library")
        names = {"format", "id", "version", "materials", "provenance"}
        _keys(value, names, {"format", "id", "version", "materials"}, "optical material library")
        return cls(
            _text(value["id"], "optical material library.id"),
            _text(value["version"], "optical material library.version"),
            tuple(OpticalMaterial.from_dict(item) for item in value["materials"]),
            _mapping(value.get("provenance", {}), "optical material library.provenance"),
            _text(value["format"], "optical material library.format"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OpticalMaterialLibrary":
        source = Path(path)
        try:
            value = loads_strict_json(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ShapeArtifactError(f"cannot load optical material library {source}: {exc}") from exc
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "id": self.id, "version": self.version, "materials": [item.to_dict() for item in self.materials], "provenance": _json_value(self.provenance)}


@dataclass(frozen=True, slots=True)
class SourceRepresentation:
    id: str
    format: str
    content: ContentReference
    metres_per_source_unit: float
    transform_to_canonical_row_major: tuple[float, ...] = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    license: str | None = None

    def __post_init__(self) -> None:
        _text(self.id, "source.id")
        if self.format not in {"step", "brep", "fcstd", "cadquery", "openscad", "stl", "obj", "ply", "gltf", "glb", "point_cloud", "depth_frames", "scan_frames", "optical_sensor", "optical_observation", "ctmesh", "procedural", "material_library", "texture"}:
            raise ShapeArtifactError(f"unsupported source format {self.format!r}")
        if _number(self.metres_per_source_unit, "source.metres_per_source_unit") <= 0.0:
            raise ShapeArtifactError("source unit scale must be positive")
        object.__setattr__(self, "transform_to_canonical_row_major", _vector(self.transform_to_canonical_row_major, 16, "source.transform_to_canonical_row_major"))
        object.__setattr__(self, "provenance", _json_value(_mapping(self.provenance, "source.provenance")))
        if self.license is not None:
            _text(self.license, "source.license")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRepresentation":
        value = _mapping(value, "source")
        names = {"id", "format", "content", "metres_per_source_unit", "transform_to_canonical_row_major", "provenance", "license"}
        _keys(value, names, {"id", "format", "content", "metres_per_source_unit"}, "source")
        return cls(
            _text(value["id"], "source.id"), _text(value["format"], "source.format"),
            ContentReference.from_dict(value["content"]), _number(value["metres_per_source_unit"], "source.metres_per_source_unit"),
            _vector(value.get("transform_to_canonical_row_major", (1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)), 16, "source.transform_to_canonical_row_major"),
            _mapping(value.get("provenance", {}), "source.provenance"),
            None if value.get("license") is None else _text(value["license"], "source.license"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {"id": self.id, "format": self.format, "content": self.content.to_dict(), "metres_per_source_unit": self.metres_per_source_unit, "transform_to_canonical_row_major": list(self.transform_to_canonical_row_major), "provenance": _json_value(self.provenance)}
        if self.license is not None:
            result["license"] = self.license
        return result


@dataclass(frozen=True, slots=True)
class SurfaceRepresentation:
    id: str
    kind: str
    content: ContentReference
    purposes: tuple[str, ...]
    vertex_count: int
    triangle_count: int
    bounds_m: tuple[float, float, float, float, float, float]
    watertight: bool
    manifold: bool
    material_ids: tuple[str, ...] = ()
    uncertainty: ShapeUncertainty = ShapeUncertainty()

    def __post_init__(self) -> None:
        _text(self.id, "surface.id")
        if self.kind != "ctmesh":
            raise ShapeArtifactError("surface.kind must be 'ctmesh'")
        if not self.purposes or any(item not in {"analysis", "ray_trace", "render", "collision"} for item in self.purposes):
            raise ShapeArtifactError("surface purposes must use analysis/ray_trace/render/collision")
        if len(set(self.purposes)) != len(self.purposes):
            raise ShapeArtifactError("surface purposes must be unique")
        for name in ("vertex_count", "triangle_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ShapeArtifactError(f"surface.{name} must be a positive integer")
        bounds = _vector(self.bounds_m, 6, "surface.bounds_m")
        if any(bounds[index] > bounds[index + 3] for index in range(3)):
            raise ShapeArtifactError("surface bounds minima may not exceed maxima")
        if not isinstance(self.watertight, bool) or not isinstance(self.manifold, bool):
            raise ShapeArtifactError("surface watertight/manifold flags must be booleans")
        if len(set(self.material_ids)) != len(self.material_ids):
            raise ShapeArtifactError("surface material IDs must be unique")
        object.__setattr__(self, "bounds_m", bounds)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SurfaceRepresentation":
        value = _mapping(value, "surface")
        names = {"id", "kind", "content", "purposes", "vertex_count", "triangle_count", "bounds_m", "watertight", "manifold", "material_ids", "uncertainty"}
        _keys(value, names, {"id", "kind", "content", "purposes", "vertex_count", "triangle_count", "bounds_m", "watertight", "manifold"}, "surface")
        return cls(
            _text(value["id"], "surface.id"), _text(value["kind"], "surface.kind"), ContentReference.from_dict(value["content"]),
            tuple(_text(item, "surface.purposes[]") for item in value["purposes"]), value["vertex_count"], value["triangle_count"],
            _vector(value["bounds_m"], 6, "surface.bounds_m"), value["watertight"], value["manifold"],
            tuple(_text(item, "surface.material_ids[]") for item in value.get("material_ids", [])), ShapeUncertainty.from_dict(value.get("uncertainty", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "content": self.content.to_dict(), "purposes": list(self.purposes), "vertex_count": self.vertex_count, "triangle_count": self.triangle_count, "bounds_m": list(self.bounds_m), "watertight": self.watertight, "manifold": self.manifold, "material_ids": list(self.material_ids), "uncertainty": self.uncertainty.to_dict()}


@dataclass(frozen=True, slots=True)
class VolumeRepresentation:
    id: str
    kind: str
    content: ContentReference
    purposes: tuple[str, ...]
    voxel_size_m: float | None = None
    dimensions: tuple[int, int, int] | None = None
    mutable_topology: bool = False
    uncertainty: ShapeUncertainty = ShapeUncertainty()

    def __post_init__(self) -> None:
        _text(self.id, "volume.id")
        if self.kind not in {"tetrahedral_mesh", "sparse_tsdf", "sparse_sdf", "sparse_occupancy", "nanovdb"}:
            raise ShapeArtifactError(f"unsupported volume kind {self.kind!r}")
        if not self.purposes or any(item not in {"mechanics", "ray_trace", "reconstruction", "collision"} for item in self.purposes):
            raise ShapeArtifactError("volume purposes are invalid")
        if self.voxel_size_m is not None and _number(self.voxel_size_m, "volume.voxel_size_m") <= 0.0:
            raise ShapeArtifactError("volume.voxel_size_m must be positive")
        if self.dimensions is not None and (len(self.dimensions) != 3 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in self.dimensions)):
            raise ShapeArtifactError("volume.dimensions must contain three positive integers")
        if not isinstance(self.mutable_topology, bool):
            raise ShapeArtifactError("volume.mutable_topology must be boolean")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VolumeRepresentation":
        value = _mapping(value, "volume")
        names = {"id", "kind", "content", "purposes", "voxel_size_m", "dimensions", "mutable_topology", "uncertainty"}
        _keys(value, names, {"id", "kind", "content", "purposes"}, "volume")
        dimensions = value.get("dimensions")
        return cls(
            _text(value["id"], "volume.id"), _text(value["kind"], "volume.kind"), ContentReference.from_dict(value["content"]),
            tuple(_text(item, "volume.purposes[]") for item in value["purposes"]),
            None if value.get("voxel_size_m") is None else _number(value["voxel_size_m"], "volume.voxel_size_m"),
            None if dimensions is None else tuple(dimensions), value.get("mutable_topology", False), ShapeUncertainty.from_dict(value.get("uncertainty", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.id, "kind": self.kind, "content": self.content.to_dict(), "purposes": list(self.purposes), "mutable_topology": self.mutable_topology, "uncertainty": self.uncertainty.to_dict()}
        if self.voxel_size_m is not None:
            result["voxel_size_m"] = self.voxel_size_m
        if self.dimensions is not None:
            result["dimensions"] = list(self.dimensions)
        return result


@dataclass(frozen=True, slots=True)
class PhysicalField:
    id: str
    quantity: str
    unit: str
    representation: str
    constant_value: float | None = None
    material_values: Mapping[str, float] = field(default_factory=dict)
    content: ContentReference | None = None
    uncertainty: ShapeUncertainty = ShapeUncertainty()

    def __post_init__(self) -> None:
        _text(self.id, "physical_field.id")
        _text(self.quantity, "physical_field.quantity")
        _text(self.unit, "physical_field.unit")
        if self.representation not in {"constant", "per_material", "per_vertex", "per_cell", "voxel_grid"}:
            raise ShapeArtifactError(f"unsupported physical-field representation {self.representation!r}")
        values = {str(key): _number(value, f"physical_field.material_values.{key}") for key, value in _mapping(self.material_values, "physical_field.material_values").items()}
        object.__setattr__(self, "material_values", dict(sorted(values.items())))
        if self.representation == "constant" and self.constant_value is None:
            raise ShapeArtifactError("constant physical fields require constant_value")
        if self.representation == "per_material" and not values:
            raise ShapeArtifactError("per_material physical fields require material_values")
        if self.representation in {"per_vertex", "per_cell", "voxel_grid"} and self.content is None:
            raise ShapeArtifactError("array physical fields require content")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalField":
        value = _mapping(value, "physical field")
        names = {"id", "quantity", "unit", "representation", "constant_value", "material_values", "content", "uncertainty"}
        _keys(value, names, {"id", "quantity", "unit", "representation"}, "physical field")
        return cls(
            _text(value["id"], "physical field.id"), _text(value["quantity"], "physical field.quantity"), _text(value["unit"], "physical field.unit"),
            _text(value["representation"], "physical field.representation"),
            None if value.get("constant_value") is None else _number(value["constant_value"], "physical field.constant_value"),
            _mapping(value.get("material_values", {}), "physical field.material_values"),
            None if value.get("content") is None else ContentReference.from_dict(value["content"]), ShapeUncertainty.from_dict(value.get("uncertainty", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.id, "quantity": self.quantity, "unit": self.unit, "representation": self.representation, "material_values": dict(self.material_values), "uncertainty": self.uncertainty.to_dict()}
        if self.constant_value is not None:
            result["constant_value"] = self.constant_value
        if self.content is not None:
            result["content"] = self.content.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class DerivedMassProperties:
    source_surface: str
    density_field: str
    mass_kg: float
    volume_m3: float
    center_of_mass_m: tuple[float, float, float]
    inertia_kg_m2_row_major: tuple[float, ...]
    uncertainty: ShapeUncertainty = ShapeUncertainty()

    def __post_init__(self) -> None:
        _text(self.source_surface, "mass_properties.source_surface")
        _text(self.density_field, "mass_properties.density_field")
        if _number(self.mass_kg, "mass_properties.mass_kg") <= 0.0 or _number(self.volume_m3, "mass_properties.volume_m3") <= 0.0:
            raise ShapeArtifactError("mass and volume must be positive")
        object.__setattr__(self, "center_of_mass_m", _vector(self.center_of_mass_m, 3, "mass_properties.center_of_mass_m"))
        object.__setattr__(self, "inertia_kg_m2_row_major", _vector(self.inertia_kg_m2_row_major, 9, "mass_properties.inertia_kg_m2_row_major"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DerivedMassProperties":
        value = _mapping(value, "mass properties")
        names = {"source_surface", "density_field", "mass_kg", "volume_m3", "center_of_mass_m", "inertia_kg_m2_row_major", "uncertainty"}
        _keys(value, names, names - {"uncertainty"}, "mass properties")
        return cls(_text(value["source_surface"], "mass properties.source_surface"), _text(value["density_field"], "mass properties.density_field"), _number(value["mass_kg"], "mass properties.mass_kg"), _number(value["volume_m3"], "mass properties.volume_m3"), _vector(value["center_of_mass_m"], 3, "mass properties.center_of_mass_m"), _vector(value["inertia_kg_m2_row_major"], 9, "mass properties.inertia_kg_m2_row_major"), ShapeUncertainty.from_dict(value.get("uncertainty", {})))

    def to_dict(self) -> dict[str, Any]:
        return {"source_surface": self.source_surface, "density_field": self.density_field, "mass_kg": self.mass_kg, "volume_m3": self.volume_m3, "center_of_mass_m": list(self.center_of_mass_m), "inertia_kg_m2_row_major": list(self.inertia_kg_m2_row_major), "uncertainty": self.uncertainty.to_dict()}


@dataclass(frozen=True, slots=True)
class ShapeArtifact:
    id: str
    version: str
    sources: tuple[SourceRepresentation, ...]
    surfaces: tuple[SurfaceRepresentation, ...]
    volumes: tuple[VolumeRepresentation, ...] = ()
    optical_materials: tuple[OpticalMaterial, ...] = ()
    physical_fields: tuple[PhysicalField, ...] = ()
    derived_mass_properties: DerivedMassProperties | None = None
    caches: tuple[ContentReference, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = "shape-artifact-1"
    canonical_frame: str = "right_handed_z_up_x_forward_metres"
    _manifest_path: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.format != "shape-artifact-1":
            raise ShapeArtifactError(f"unsupported shape artifact format {self.format!r}")
        if self.canonical_frame != "right_handed_z_up_x_forward_metres":
            raise ShapeArtifactError("unsupported canonical coordinate frame")
        _text(self.id, "shape.id")
        _text(self.version, "shape.version")
        if not self.sources:
            raise ShapeArtifactError("shape artifacts require at least one source representation")
        if not self.surfaces and not self.volumes:
            raise ShapeArtifactError("shape artifacts require a canonical surface or volume")
        for label, items in (("source", self.sources), ("surface", self.surfaces), ("volume", self.volumes), ("optical material", self.optical_materials), ("physical field", self.physical_fields)):
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise ShapeArtifactError(f"{label} IDs must be unique")
        material_ids = {item.id for item in self.optical_materials}
        for surface in self.surfaces:
            missing = set(surface.material_ids) - material_ids
            if missing:
                raise ShapeArtifactError(f"surface {surface.id!r} references missing optical materials {sorted(missing)}")
        object.__setattr__(self, "provenance", _json_value(_mapping(self.provenance, "shape.provenance")))
        object.__setattr__(self, "metadata", _json_value(_mapping(self.metadata, "shape.metadata")))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, manifest_path: str | Path | None = None) -> "ShapeArtifact":
        value = _mapping(value, "shape artifact")
        names = {"format", "id", "version", "canonical_frame", "sources", "surfaces", "volumes", "optical_materials", "physical_fields", "derived_mass_properties", "caches", "provenance", "metadata"}
        _keys(value, names, {"format", "id", "version", "canonical_frame", "sources", "surfaces"}, "shape artifact")
        return cls(
            id=_text(value["id"], "shape.id"), version=_text(value["version"], "shape.version"),
            sources=tuple(SourceRepresentation.from_dict(item) for item in value["sources"]),
            surfaces=tuple(SurfaceRepresentation.from_dict(item) for item in value["surfaces"]),
            volumes=tuple(VolumeRepresentation.from_dict(item) for item in value.get("volumes", [])),
            optical_materials=tuple(OpticalMaterial.from_dict(item) for item in value.get("optical_materials", [])),
            physical_fields=tuple(PhysicalField.from_dict(item) for item in value.get("physical_fields", [])),
            derived_mass_properties=None if value.get("derived_mass_properties") is None else DerivedMassProperties.from_dict(value["derived_mass_properties"]),
            caches=tuple(ContentReference.from_dict(item) for item in value.get("caches", [])),
            provenance=_mapping(value.get("provenance", {}), "shape.provenance"), metadata=_mapping(value.get("metadata", {}), "shape.metadata"),
            format=_text(value["format"], "shape.format"), canonical_frame=_text(value["canonical_frame"], "shape.canonical_frame"),
            _manifest_path=None if manifest_path is None else Path(manifest_path).resolve(),
        )

    @classmethod
    def load(cls, path: str | Path, *, verify_content: bool = True) -> "ShapeArtifact":
        manifest = Path(path).resolve()
        try:
            value = loads_strict_json(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ShapeArtifactError(f"cannot load shape artifact {manifest}: {exc}") from exc
        artifact = cls.from_dict(value, manifest_path=manifest)
        if verify_content:
            artifact.verify()
        return artifact

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format, "id": self.id, "version": self.version, "canonical_frame": self.canonical_frame,
            "sources": [item.to_dict() for item in self.sources], "surfaces": [item.to_dict() for item in self.surfaces],
            "volumes": [item.to_dict() for item in self.volumes], "optical_materials": [item.to_dict() for item in self.optical_materials],
            "physical_fields": [item.to_dict() for item in self.physical_fields],
            "derived_mass_properties": None if self.derived_mass_properties is None else self.derived_mass_properties.to_dict(),
            "caches": [item.to_dict() for item in self.caches], "provenance": _json_value(self.provenance), "metadata": _json_value(self.metadata),
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return target

    def resolve(self, reference: ContentReference, *, base: str | Path | None = None) -> Path:
        root = Path(base).resolve() if base is not None else (self._manifest_path.parent if self._manifest_path is not None else None)
        if root is None:
            raise ShapeArtifactError("resolving content requires a loaded manifest or explicit base")
        target = (root / Path(*PurePosixPath(reference.uri).parts)).resolve()
        if target != root and root not in target.parents:
            raise ShapeArtifactError(f"content reference escapes artifact root: {reference.uri}")
        return target

    def verify(self, *, base: str | Path | None = None) -> None:
        references = [source.content for source in self.sources]
        references += [surface.content for surface in self.surfaces]
        references += [volume.content for volume in self.volumes]
        references += [field.content for field in self.physical_fields if field.content is not None]
        references += list(self.caches)
        for reference in references:
            path = self.resolve(reference, base=base)
            if not path.is_file():
                raise ShapeArtifactError(
                    f"content must be a regular file: {reference.uri}"
                )
            payload = path.read_bytes()
            if len(payload) != reference.byte_length:
                raise ShapeArtifactError(f"content length mismatch for {reference.uri}")
            if hashlib.sha256(payload).hexdigest() != reference.sha256:
                raise ShapeArtifactError(f"content hash mismatch for {reference.uri}")
        surface_meshes: dict[str, Any] = {}
        for surface in self.surfaces:
            if surface.content.media_type != "application/vnd.contraption.ctmesh":
                raise ShapeArtifactError(
                    f"surface {surface.id!r} must reference canonical CTMESH content"
                )
            try:
                import numpy as np
                from .mesh import TriangleMesh

                mesh = TriangleMesh.from_bytes(
                    self.resolve(surface.content, base=base).read_bytes()
                )
            except Exception as exc:
                raise ShapeArtifactError(
                    f"invalid canonical surface {surface.id!r}: {exc}"
                ) from exc
            surface_meshes[surface.id] = mesh
            if len(mesh.vertices_m) != surface.vertex_count:
                raise ShapeArtifactError(
                    f"surface {surface.id!r} vertex-count metadata mismatch"
                )
            if len(mesh.triangles) != surface.triangle_count:
                raise ShapeArtifactError(
                    f"surface {surface.id!r} triangle-count metadata mismatch"
                )
            low, high = mesh.bounds_m
            actual_bounds = np.concatenate((low, high))
            if not np.allclose(actual_bounds, surface.bounds_m, rtol=1e-6, atol=1e-9):
                raise ShapeArtifactError(
                    f"surface {surface.id!r} bounds metadata mismatch"
                )
            if mesh.watertight != surface.watertight:
                raise ShapeArtifactError(
                    f"surface {surface.id!r} watertight metadata mismatch"
                )
            if mesh.manifold != surface.manifold:
                raise ShapeArtifactError(
                    f"surface {surface.id!r} manifold metadata mismatch"
                )
            if mesh.face_material is not None:
                if not surface.material_ids or int(mesh.face_material.max()) >= len(surface.material_ids):
                    raise ShapeArtifactError(
                        f"surface {surface.id!r} face-material index exceeds material_ids"
                    )
            elif len(surface.material_ids) > 1:
                raise ShapeArtifactError(
                    f"surface {surface.id!r} requires face-material indices when multiple material_ids are declared"
                )
            if surface.uncertainty.distribution == "empirical":
                field_id = surface.uncertainty.parameters.get("field_id")
                field = next(
                    (item for item in self.physical_fields if item.id == field_id),
                    None,
                )
                if (
                    not isinstance(field_id, str)
                    or field is None
                    or field.representation != "per_vertex"
                    or field.quantity != "surface_position_standard_deviation"
                    or field.unit != "m"
                    or field.content is None
                    or field.content.media_type != "application/vnd.numpy.npy"
                ):
                    raise ShapeArtifactError(
                        f"surface {surface.id!r} empirical uncertainty requires a per-vertex surface_position_standard_deviation field in metres"
                    )
                try:
                    field_path = self.resolve(field.content, base=base)
                    with field_path.open("rb") as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0):
                            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise ShapeArtifactError(
                                f"surface {surface.id!r} uncertainty field uses unsupported NPY version {version}"
                            )
                        data_offset = stream.tell()
                    if (
                        dtype != np.dtype("<f4")
                        or fortran_order
                        or tuple(shape) != (surface.vertex_count,)
                        or data_offset + surface.vertex_count * dtype.itemsize
                        != field.content.byte_length
                    ):
                        raise ShapeArtifactError(
                            f"surface {surface.id!r} uncertainty NPY header/framing is not canonical"
                        )
                    with field_path.open("rb") as stream:
                        values = np.load(stream, allow_pickle=False)
                except Exception as exc:
                    if isinstance(exc, ShapeArtifactError):
                        raise
                    raise ShapeArtifactError(
                        f"surface {surface.id!r} uncertainty field is not valid NPY: {exc}"
                    ) from exc
                if (
                    values.dtype != np.dtype("<f4")
                    or values.shape != (surface.vertex_count,)
                    or not values.flags.c_contiguous
                    or not np.all(np.isfinite(values))
                    or np.any(values < 0.0)
                ):
                    raise ShapeArtifactError(
                        f"surface {surface.id!r} uncertainty field must be finite, nonnegative little-endian float32 with one value per vertex"
                    )
        if self.derived_mass_properties is not None:
            derived = self.derived_mass_properties
            mesh = surface_meshes.get(derived.source_surface)
            if mesh is None:
                raise ShapeArtifactError(
                    "derived mass properties reference a missing source surface"
                )
            density = next(
                (field for field in self.physical_fields if field.id == derived.density_field),
                None,
            )
            if (
                density is None
                or density.quantity != "mass_density"
                or density.unit != "kg/m^3"
                or density.representation != "constant"
                or density.constant_value is None
                or density.constant_value <= 0.0
            ):
                raise ShapeArtifactError(
                    "derived mass properties require a positive constant mass_density field in kg/m^3"
                )
            if not mesh.closed_oriented_manifold:
                raise ShapeArtifactError(
                    "derived mass properties require a closed, consistently oriented manifold surface"
                )
            try:
                from .mechanics import mass_properties

                computed = mass_properties(mesh, density.constant_value)
            except Exception as exc:
                raise ShapeArtifactError(f"cannot verify derived mass properties: {exc}") from exc
            if not math.isclose(computed.mass_kg, derived.mass_kg, rel_tol=1e-5, abs_tol=0.0):
                raise ShapeArtifactError("derived mass metadata does not match canonical geometry")
            if not math.isclose(computed.volume_m3, derived.volume_m3, rel_tol=1e-5, abs_tol=0.0):
                raise ShapeArtifactError("derived volume metadata does not match canonical geometry")
            scale = max(float(np.max(mesh.dimensions_m)), np.finfo(float).tiny)
            if not np.allclose(
                computed.center_of_mass_m,
                derived.center_of_mass_m,
                rtol=1e-5,
                atol=scale * 1e-7,
            ):
                raise ShapeArtifactError("derived center of mass metadata does not match canonical geometry")
            inertia_scale = max(
                float(np.max(np.abs(computed.inertia_kg_m2))),
                np.finfo(float).tiny,
            )
            if not np.allclose(
                computed.inertia_kg_m2.reshape(-1),
                derived.inertia_kg_m2_row_major,
                rtol=1e-5,
                atol=inertia_scale * 1e-7,
            ):
                raise ShapeArtifactError("derived inertia metadata does not match canonical geometry")
        # Some canonical/source records are manifests with their own strict,
        # relative content closure. Verify those closures transitively so a
        # shape artifact cannot remain valid after an observation sidecar or
        # sparse reconstruction block is modified.
        for source in self.sources:
            source_path = self.resolve(source.content, base=base)
            try:
                if source.format == "optical_sensor":
                    from contraption.optics.schemas import OpticalSensor

                    sensor = OpticalSensor.load(source_path)
                    expected = source.provenance.get("sensor_sha256")
                    if expected is not None and sensor.artifact_sha256 != expected:
                        raise ShapeArtifactError(
                            f"optical sensor digest mismatch for source {source.id!r}"
                        )
                elif source.format == "optical_observation":
                    from contraption.optics.schemas import ObservationArtifact

                    observation = ObservationArtifact.load(
                        source_path, verify_content=True
                    )
                    expected = source.provenance.get("observation_sha256")
                    if expected is not None and observation.artifact_sha256 != expected:
                        raise ShapeArtifactError(
                            f"optical observation digest mismatch for source {source.id!r}"
                        )
            except ShapeArtifactError:
                raise
            except Exception as exc:
                raise ShapeArtifactError(
                    f"invalid transitive optical source {source.id!r}: {exc}"
                ) from exc
        for volume in self.volumes:
            if volume.kind == "sparse_tsdf":
                if (
                    volume.content.media_type
                    != "application/vnd.contraption.reconstruction-state+json"
                ):
                    raise ShapeArtifactError(
                        f"sparse TSDF volume {volume.id!r} must reference a canonical reconstruction state"
                    )
                try:
                    from contraption.optics.schemas import ReconstructionState

                    ReconstructionState.load(
                        self.resolve(volume.content, base=base),
                        verify_content=True,
                    )
                except Exception as exc:
                    raise ShapeArtifactError(
                        f"invalid transitive sparse TSDF volume {volume.id!r}: {exc}"
                    ) from exc

    def surface_for(self, purpose: str, *, prefer_kind: str = "ctmesh") -> SurfaceRepresentation:
        candidates = [item for item in self.surfaces if purpose in item.purposes]
        if not candidates:
            raise ShapeArtifactError(f"shape {self.id!r} has no surface for purpose {purpose!r}")
        return next((item for item in candidates if item.kind == prefer_kind), candidates[0])

    @property
    def artifact_sha256(self) -> str:
        """Digest of the canonical manifest data, independent of JSON whitespace."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def load_surface(self, surface_or_purpose: str = "ray_trace") -> Any:
        """Resolve and load a canonical CTMESH surface by ID or purpose."""

        from .mesh import TriangleMesh

        surface = next((item for item in self.surfaces if item.id == surface_or_purpose), None)
        if surface is None:
            surface = self.surface_for(surface_or_purpose)
        if surface.kind != "ctmesh":
            raise ShapeArtifactError(
                f"surface {surface.id!r} is {surface.kind!r}; analysis requires a CTMESH surface"
            )
        return TriangleMesh.read(self.resolve(surface.content))
