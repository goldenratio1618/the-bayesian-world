"""Strict structured formats for optical simulation and reconstruction.

The formats in this module are deliberately file-oriented and backend-neutral.
Runtime renderers consume the same records whether a frame is generated on the
CPU, a CUDA device, or an eventual external controller/GPU service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
from typing import Any, Mapping, Sequence

from ..strict_json import loads_strict_json


class OpticalSchemaError(ValueError):
    """Raised when an optical structured artifact is malformed or unsafe."""


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise OpticalSchemaError(f"{context} must be an object with string keys")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], required: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise OpticalSchemaError(f"{context} has unknown keys: {', '.join(unknown)}")
    if missing:
        raise OpticalSchemaError(f"{context} is missing keys: {', '.join(missing)}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OpticalSchemaError(f"{context} must be a nonempty trimmed string")
    return value


def _number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise OpticalSchemaError(f"{context} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise OpticalSchemaError(f"{context} must be at least {minimum}")
    return result


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OpticalSchemaError(f"{context} must be an integer of at least {minimum}")
    return value


def _vector(value: Any, length: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise OpticalSchemaError(f"{context} must contain exactly {length} numbers")
    return tuple(_number(item, f"{context}[{index}]") for index, item in enumerate(value))


def _json_value(value: Any, context: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _number(value, context)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, f"{context}.{key}") for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{context}[{index}]") for index, item in enumerate(value)]
    raise OpticalSchemaError(f"{context} contains unsupported type {type(value).__name__}")


def _relative_uri(value: Any, context: str) -> str:
    text = _text(value, context)
    if "\\" in text or "\x00" in text:
        raise OpticalSchemaError(f"{context} must be a POSIX relative URI")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise OpticalSchemaError(f"{context} must stay below its artifact directory")
    return text


def _digest(value: Any, context: str) -> str:
    text = _text(value, context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise OpticalSchemaError(f"{context} must be a lowercase SHA-256 digest")
    return text


def _write_json(value: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return target


def _read_json(path: str | Path, context: str) -> tuple[Path, Mapping[str, Any]]:
    source = Path(path).resolve()
    try:
        value = loads_strict_json(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpticalSchemaError(f"cannot load {context} {source}: {exc}") from exc
    return source, _object(value, context)


def _rigid_transform(value: Any, context: str) -> tuple[float, ...]:
    matrix = _vector(value, 16, context)
    if any(
        abs(matrix[12 + index] - expected) > 1e-9
        for index, expected in enumerate((0, 0, 0, 1))
    ):
        raise OpticalSchemaError(
            f"{context} must have homogeneous final row [0, 0, 0, 1]"
        )
    rotation = (matrix[0:3], matrix[4:7], matrix[8:11])
    if any(
        abs(sum(item * item for item in row) - 1.0) > 1e-5
        for row in rotation
    ):
        raise OpticalSchemaError(f"{context} rotation rows must have unit length")
    if any(
        abs(sum(rotation[row][axis] * rotation[column][axis] for axis in range(3)))
        > 1e-5
        for row in range(3)
        for column in range(row)
    ):
        raise OpticalSchemaError(f"{context} rotation rows must be orthogonal")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-5:
        raise OpticalSchemaError(f"{context} rotation determinant must be +1")
    return matrix


@dataclass(frozen=True, slots=True)
class Pose:
    """Rigid transform from optical sensor coordinates to world coordinates.

    Sensor coordinates use +X right, +Y down, and +Z forward. The world frame
    is the project's right-handed, Z-up, metre frame.
    """

    transform_world_from_sensor_row_major: tuple[float, ...] = (
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    )

    def __post_init__(self) -> None:
        matrix = _rigid_transform(
            self.transform_world_from_sensor_row_major, "pose transform"
        )
        object.__setattr__(self, "transform_world_from_sensor_row_major", matrix)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Pose":
        value = _object(value, "pose")
        _keys(value, {"transform_world_from_sensor_row_major"}, {"transform_world_from_sensor_row_major"}, "pose")
        return cls(_vector(value["transform_world_from_sensor_row_major"], 16, "pose.transform"))

    def to_dict(self) -> dict[str, Any]:
        return {"transform_world_from_sensor_row_major": list(self.transform_world_from_sensor_row_major)}

    @property
    def artifact_sha256(self) -> str:
        """Digest of the exact little-endian float64 world-from-sensor matrix."""
        return hashlib.sha256(struct.pack("<16d", *self.transform_world_from_sensor_row_major)).hexdigest()


@dataclass(frozen=True, slots=True)
class SensorNoise:
    model: str = "gaussian_poisson"
    seed: int = 0
    read_noise_std_linear: float = 0.0
    shot_noise_scale: float = 0.0
    depth_noise_std_m: float = 0.0
    depth_quantization_m: float = 0.0
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.model not in {"none", "gaussian_poisson"}:
            raise OpticalSchemaError("sensor noise model must be 'none' or 'gaussian_poisson'")
        _integer(self.seed, "sensor.noise.seed")
        for name in ("read_noise_std_linear", "shot_noise_scale", "depth_noise_std_m", "depth_quantization_m"):
            _number(getattr(self, name), f"sensor.noise.{name}", minimum=0.0)
        probability = _number(self.dropout_probability, "sensor.noise.dropout_probability", minimum=0.0)
        if probability > 1.0:
            raise OpticalSchemaError("sensor.noise.dropout_probability must be at most 1")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SensorNoise":
        value = _object(value, "sensor noise")
        names = {"model", "seed", "read_noise_std_linear", "shot_noise_scale", "depth_noise_std_m", "depth_quantization_m", "dropout_probability"}
        _keys(value, names, set(), "sensor noise")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in ("model", "seed", "read_noise_std_linear", "shot_noise_scale", "depth_noise_std_m", "depth_quantization_m", "dropout_probability")}


@dataclass(frozen=True, slots=True)
class SpectralChannel:
    id: str
    center_wavelength_nm: float
    bandwidth_nm: float
    relative_response: float = 1.0

    def __post_init__(self) -> None:
        _text(self.id, "spectral channel.id")
        wavelength = _number(self.center_wavelength_nm, "spectral channel.center_wavelength_nm", minimum=100.0)
        if wavelength > 3000.0:
            raise OpticalSchemaError("spectral channel wavelength exceeds the supported optical range")
        _number(self.bandwidth_nm, "spectral channel.bandwidth_nm", minimum=1e-12)
        _number(self.relative_response, "spectral channel.relative_response", minimum=0.0)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpectralChannel":
        value = _object(value, "spectral channel")
        names = {"id", "center_wavelength_nm", "bandwidth_nm", "relative_response"}
        _keys(value, names, names - {"relative_response"}, "spectral channel")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "center_wavelength_nm": self.center_wavelength_nm, "bandwidth_nm": self.bandwidth_nm, "relative_response": self.relative_response}


@dataclass(frozen=True, slots=True)
class WirePayloadSpec:
    """A bounded, versioned transport contract; no controller is implemented here."""

    schema: str = "contraption.optical-frame/v1"
    encoding: str = "cbor-arrays"
    max_payload_bytes: int = 8 * 1024 * 1024
    max_frame_rate_hz: float = 30.0

    def __post_init__(self) -> None:
        if self.schema != "contraption.optical-frame/v1":
            raise OpticalSchemaError("unsupported optical wire schema")
        if self.encoding not in {"cbor-arrays", "raw-chunks"}:
            raise OpticalSchemaError("unsupported optical wire encoding")
        _integer(self.max_payload_bytes, "sensor.wire.max_payload_bytes", minimum=256)
        _number(self.max_frame_rate_hz, "sensor.wire.max_frame_rate_hz", minimum=1e-9)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WirePayloadSpec":
        value = _object(value, "wire payload")
        names = {"schema", "encoding", "max_payload_bytes", "max_frame_rate_hz"}
        _keys(value, names, set(), "wire payload")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "encoding": self.encoding, "max_payload_bytes": self.max_payload_bytes, "max_frame_rate_hz": self.max_frame_rate_hz}


_SENSOR_OUTPUTS = {"rgb_linear", "depth_m", "segmentation", "uncertainty"}


@dataclass(frozen=True, slots=True)
class OpticalSensor:
    id: str
    resolution_px: tuple[int, int]
    focal_length_px: tuple[float, float]
    principal_point_px: tuple[float, float]
    near_clip_m: float = 0.01
    far_clip_m: float = 100.0
    exposure_duration_s: float = 0.01
    readout_duration_s: float = 0.0
    processing_latency_s: float = 0.0
    outputs: tuple[str, ...] = ("rgb_linear", "depth_m", "segmentation", "uncertainty")
    spectral_channels: tuple[SpectralChannel, ...] = (
        SpectralChannel("red", 620.0, 100.0),
        SpectralChannel("green", 540.0, 100.0),
        SpectralChannel("blue", 460.0, 100.0),
    )
    noise: SensorNoise = SensorNoise()
    wire: WirePayloadSpec = WirePayloadSpec()
    display_name: str | None = None
    mount_connector: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = "optical-sensor-1"
    projection: str = "pinhole"

    def __post_init__(self) -> None:
        if self.format != "optical-sensor-1" or self.projection != "pinhole":
            raise OpticalSchemaError("unsupported optical sensor format or projection")
        _text(self.id, "sensor.id")
        if len(self.resolution_px) != 2:
            raise OpticalSchemaError("sensor.resolution_px must contain width and height")
        width, height = (_integer(item, "sensor.resolution_px[]", minimum=1) for item in self.resolution_px)
        if width * height > 16_777_216:
            raise OpticalSchemaError("sensor resolution exceeds the bounded 16-megapixel contract")
        focal = _vector(self.focal_length_px, 2, "sensor.focal_length_px")
        if min(focal) <= 0:
            raise OpticalSchemaError("sensor focal lengths must be positive")
        principal = _vector(self.principal_point_px, 2, "sensor.principal_point_px")
        near = _number(self.near_clip_m, "sensor.near_clip_m", minimum=0.0)
        far = _number(self.far_clip_m, "sensor.far_clip_m", minimum=0.0)
        if near <= 0 or far <= near:
            raise OpticalSchemaError("sensor clipping range must satisfy 0 < near < far")
        for name in ("exposure_duration_s", "readout_duration_s", "processing_latency_s"):
            _number(getattr(self, name), f"sensor.{name}", minimum=0.0)
        if not self.outputs or len(set(self.outputs)) != len(self.outputs) or set(self.outputs) - _SENSOR_OUTPUTS:
            raise OpticalSchemaError("sensor outputs must be unique supported optical products")
        if not self.spectral_channels or len({item.id for item in self.spectral_channels}) != len(self.spectral_channels):
            raise OpticalSchemaError("sensor spectral channel IDs must be nonempty and unique")
        if self.display_name is not None:
            _text(self.display_name, "sensor.display_name")
        if self.mount_connector is not None:
            _text(self.mount_connector, "sensor.mount_connector")
        object.__setattr__(self, "resolution_px", (width, height))
        object.__setattr__(self, "focal_length_px", focal)
        object.__setattr__(self, "principal_point_px", principal)
        object.__setattr__(self, "metadata", _json_value(_object(self.metadata, "sensor.metadata")))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OpticalSensor":
        value = _object(value, "optical sensor")
        names = {"format", "id", "projection", "resolution_px", "focal_length_px", "principal_point_px", "near_clip_m", "far_clip_m", "exposure_duration_s", "readout_duration_s", "processing_latency_s", "outputs", "spectral_channels", "noise", "wire", "display_name", "mount_connector", "metadata"}
        required = {"format", "id", "projection", "resolution_px", "focal_length_px", "principal_point_px"}
        _keys(value, names, required, "optical sensor")
        kwargs = dict(value)
        kwargs["resolution_px"] = tuple(value["resolution_px"])
        kwargs["focal_length_px"] = tuple(value["focal_length_px"])
        kwargs["principal_point_px"] = tuple(value["principal_point_px"])
        kwargs["outputs"] = tuple(value.get("outputs", ("rgb_linear", "depth_m", "segmentation", "uncertainty")))
        kwargs["spectral_channels"] = tuple(SpectralChannel.from_dict(item) for item in value.get("spectral_channels", [item.to_dict() for item in cls.__dataclass_fields__["spectral_channels"].default]))
        kwargs["noise"] = SensorNoise.from_dict(value.get("noise", {}))
        kwargs["wire"] = WirePayloadSpec.from_dict(value.get("wire", {}))
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "OpticalSensor":
        _path, value = _read_json(path, "optical sensor")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "format": self.format, "id": self.id, "projection": self.projection,
            "resolution_px": list(self.resolution_px), "focal_length_px": list(self.focal_length_px),
            "principal_point_px": list(self.principal_point_px), "near_clip_m": self.near_clip_m,
            "far_clip_m": self.far_clip_m, "exposure_duration_s": self.exposure_duration_s,
            "readout_duration_s": self.readout_duration_s, "processing_latency_s": self.processing_latency_s,
            "outputs": list(self.outputs), "spectral_channels": [item.to_dict() for item in self.spectral_channels],
            "noise": self.noise.to_dict(), "wire": self.wire.to_dict(), "metadata": _json_value(self.metadata),
        }
        if self.display_name is not None:
            result["display_name"] = self.display_name
        if self.mount_connector is not None:
            result["mount_connector"] = self.mount_connector
        return result

    def write(self, path: str | Path) -> Path:
        return _write_json(self.to_dict(), path)

    @property
    def artifact_sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SceneObject:
    id: str
    shape_artifact_uri: str
    shape_artifact_sha256: str
    segmentation_id: int
    transform_world_from_object_row_major: tuple[float, ...] = (
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    )
    surface_id: str | None = None
    surface_uncertainty_m: float = 0.0

    def __post_init__(self) -> None:
        _text(self.id, "scene object.id")
        _relative_uri(self.shape_artifact_uri, "scene object.shape_artifact_uri")
        _digest(self.shape_artifact_sha256, "scene object.shape_artifact_sha256")
        _integer(self.segmentation_id, "scene object.segmentation_id", minimum=1)
        transform = _rigid_transform(
            self.transform_world_from_object_row_major, "scene object transform"
        )
        object.__setattr__(self, "transform_world_from_object_row_major", transform)
        if self.surface_id is not None:
            _text(self.surface_id, "scene object.surface_id")
        _number(self.surface_uncertainty_m, "scene object.surface_uncertainty_m", minimum=0.0)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SceneObject":
        value = _object(value, "scene object")
        names = {"id", "shape_artifact_uri", "shape_artifact_sha256", "segmentation_id", "transform_world_from_object_row_major", "surface_id", "surface_uncertainty_m"}
        _keys(value, names, {"id", "shape_artifact_uri", "shape_artifact_sha256", "segmentation_id"}, "scene object")
        return cls(value["id"], value["shape_artifact_uri"], value["shape_artifact_sha256"], value["segmentation_id"], tuple(value.get("transform_world_from_object_row_major", Pose().transform_world_from_sensor_row_major)), value.get("surface_id"), value.get("surface_uncertainty_m", 0.0))

    def to_dict(self) -> dict[str, Any]:
        result = {"id": self.id, "shape_artifact_uri": self.shape_artifact_uri, "shape_artifact_sha256": self.shape_artifact_sha256, "segmentation_id": self.segmentation_id, "transform_world_from_object_row_major": list(self.transform_world_from_object_row_major), "surface_uncertainty_m": self.surface_uncertainty_m}
        if self.surface_id is not None:
            result["surface_id"] = self.surface_id
        return result


@dataclass(frozen=True, slots=True)
class OpticalLight:
    id: str
    kind: str
    color_linear_rgb: tuple[float, float, float]
    intensity: float
    position_m: tuple[float, float, float] | None = None
    direction_world: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        _text(self.id, "light.id")
        if self.kind not in {"point", "directional"}:
            raise OpticalSchemaError("light.kind must be point or directional")
        color = _vector(self.color_linear_rgb, 3, "light.color_linear_rgb")
        if any(item < 0 for item in color):
            raise OpticalSchemaError("light color must be nonnegative")
        _number(self.intensity, "light.intensity", minimum=0.0)
        if self.kind == "point" and self.position_m is None:
            raise OpticalSchemaError("point lights require position_m")
        if self.kind == "directional" and self.direction_world is None:
            raise OpticalSchemaError("directional lights require direction_world")
        if self.position_m is not None:
            object.__setattr__(self, "position_m", _vector(self.position_m, 3, "light.position_m"))
        if self.direction_world is not None:
            direction = _vector(self.direction_world, 3, "light.direction_world")
            norm = math.sqrt(sum(item * item for item in direction))
            if norm <= 1e-12:
                raise OpticalSchemaError("directional light direction must be nonzero")
            object.__setattr__(self, "direction_world", tuple(item / norm for item in direction))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OpticalLight":
        value = _object(value, "optical light")
        names = {"id", "kind", "color_linear_rgb", "intensity", "position_m", "direction_world"}
        _keys(value, names, {"id", "kind", "color_linear_rgb", "intensity"}, "optical light")
        return cls(value["id"], value["kind"], tuple(value["color_linear_rgb"]), value["intensity"], None if value.get("position_m") is None else tuple(value["position_m"]), None if value.get("direction_world") is None else tuple(value["direction_world"]))

    def to_dict(self) -> dict[str, Any]:
        result = {"id": self.id, "kind": self.kind, "color_linear_rgb": list(self.color_linear_rgb), "intensity": self.intensity}
        if self.position_m is not None:
            result["position_m"] = list(self.position_m)
        if self.direction_world is not None:
            result["direction_world"] = list(self.direction_world)
        return result


@dataclass(frozen=True, slots=True)
class OpticalScene:
    id: str
    objects: tuple[SceneObject, ...]
    lights: tuple[OpticalLight, ...] = ()
    environment_linear_rgb: tuple[float, float, float] = (0.02, 0.02, 0.02)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = "optical-scene-1"
    canonical_frame: str = "right_handed_z_up_x_forward_metres"
    _manifest_path: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.format != "optical-scene-1" or self.canonical_frame != "right_handed_z_up_x_forward_metres":
            raise OpticalSchemaError("unsupported optical scene format or coordinate frame")
        _text(self.id, "scene.id")
        if not self.objects or len({item.id for item in self.objects}) != len(self.objects):
            raise OpticalSchemaError("scene object IDs must be nonempty and unique")
        if len({item.segmentation_id for item in self.objects}) != len(self.objects):
            raise OpticalSchemaError("scene segmentation IDs must be unique")
        if len({item.id for item in self.lights}) != len(self.lights):
            raise OpticalSchemaError("scene light IDs must be unique")
        environment = _vector(self.environment_linear_rgb, 3, "scene.environment_linear_rgb")
        if any(item < 0 for item in environment):
            raise OpticalSchemaError("scene environment radiance must be nonnegative")
        object.__setattr__(self, "environment_linear_rgb", environment)
        object.__setattr__(self, "metadata", _json_value(_object(self.metadata, "scene.metadata")))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, manifest_path: str | Path | None = None) -> "OpticalScene":
        value = _object(value, "optical scene")
        names = {"format", "id", "canonical_frame", "objects", "lights", "environment_linear_rgb", "metadata"}
        _keys(value, names, {"format", "id", "canonical_frame", "objects"}, "optical scene")
        return cls(value["id"], tuple(SceneObject.from_dict(item) for item in value["objects"]), tuple(OpticalLight.from_dict(item) for item in value.get("lights", [])), tuple(value.get("environment_linear_rgb", (0.02, 0.02, 0.02))), _object(value.get("metadata", {}), "scene.metadata"), value["format"], value["canonical_frame"], None if manifest_path is None else Path(manifest_path).resolve())

    @classmethod
    def load(cls, path: str | Path, *, verify_shapes: bool = True) -> "OpticalScene":
        source, value = _read_json(path, "optical scene")
        result = cls.from_dict(value, manifest_path=source)
        if verify_shapes:
            result.verify()
        return result

    def resolve_shape(self, item: SceneObject) -> Path:
        if self._manifest_path is None:
            raise OpticalSchemaError("resolving scene shapes requires a loaded scene manifest")
        root = self._manifest_path.parent
        target = (root / Path(*PurePosixPath(item.shape_artifact_uri).parts)).resolve()
        if target != root and root not in target.parents:
            raise OpticalSchemaError("scene object shape reference escapes the scene directory")
        return target

    def verify(self) -> None:
        from contraption.shape import ShapeArtifact
        for item in self.objects:
            shape = ShapeArtifact.load(self.resolve_shape(item), verify_content=True)
            if shape.artifact_sha256 != item.shape_artifact_sha256:
                raise OpticalSchemaError(f"shape manifest digest mismatch for scene object {item.id!r}")
            if item.surface_id is not None and not any(surface.id == item.surface_id for surface in shape.surfaces):
                raise OpticalSchemaError(f"scene object {item.id!r} names an unknown surface")

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "id": self.id, "canonical_frame": self.canonical_frame, "objects": [item.to_dict() for item in self.objects], "lights": [item.to_dict() for item in self.lights], "environment_linear_rgb": list(self.environment_linear_rgb), "metadata": _json_value(self.metadata)}

    def write(self, path: str | Path) -> Path:
        return _write_json(self.to_dict(), path)

    @property
    def artifact_sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ContentReference:
    uri: str
    media_type: str
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        _relative_uri(self.uri, "content.uri")
        _text(self.media_type, "content.media_type")
        _digest(self.sha256, "content.sha256")
        _integer(self.byte_length, "content.byte_length")

    @classmethod
    def from_path(cls, path: str | Path, *, relative_to: str | Path, media_type: str) -> "ContentReference":
        source, root = Path(path).resolve(), Path(relative_to).resolve()
        try:
            uri = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise OpticalSchemaError(f"content {source} is outside {root}") from exc
        payload = source.read_bytes()
        return cls(uri, media_type, hashlib.sha256(payload).hexdigest(), len(payload))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContentReference":
        value = _object(value, "content reference")
        names = {"uri", "media_type", "sha256", "byte_length"}
        _keys(value, names, names, "content reference")
        return cls(value["uri"], value["media_type"], value["sha256"], value["byte_length"])

    def to_dict(self) -> dict[str, Any]:
        return {"uri": self.uri, "media_type": self.media_type, "sha256": self.sha256, "byte_length": self.byte_length}


@dataclass(frozen=True, slots=True)
class ObservationOutput:
    name: str
    dtype: str
    shape: tuple[int, ...]
    content: ContentReference

    def __post_init__(self) -> None:
        if self.name not in _SENSOR_OUTPUTS:
            raise OpticalSchemaError(f"unsupported observation output {self.name!r}")
        if self.dtype not in {"float32", "int32"}:
            raise OpticalSchemaError("observation output dtype must be float32 or int32")
        expected_dtype = "int32" if self.name == "segmentation" else "float32"
        if self.dtype != expected_dtype:
            raise OpticalSchemaError(f"observation output {self.name!r} must use {expected_dtype}")
        if not self.shape or any(_integer(item, "observation output.shape[]", minimum=1) < 1 for item in self.shape):
            raise OpticalSchemaError("observation output shape must be nonempty and positive")
        if math.prod(self.shape) > 50_331_648:
            raise OpticalSchemaError("observation output exceeds the bounded array contract")
        if self.content.media_type != "application/vnd.numpy.npy":
            raise OpticalSchemaError("observation outputs must use canonical NPY content")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationOutput":
        value = _object(value, "observation output")
        names = {"name", "dtype", "shape", "content"}
        _keys(value, names, names, "observation output")
        return cls(value["name"], value["dtype"], tuple(value["shape"]), ContentReference.from_dict(value["content"]))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape), "content": self.content.to_dict()}


@dataclass(frozen=True, slots=True)
class ObservationArtifact:
    id: str
    sensor_id: str
    sensor_sha256: str
    scene_sha256: str
    frame_index: int
    requested_at_s: float
    exposure_started_at_s: float
    exposure_duration_s: float
    ready_at_s: float
    pose: Pose
    seed: int
    outputs: tuple[ObservationOutput, ...]
    assembly_id: str | None = None
    assembly_sha256: str | None = None
    assembly_frame: str = "world"
    mount_connector: str | None = None
    mount_transform_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = "optical-observation-1"
    _manifest_path: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.format != "optical-observation-1":
            raise OpticalSchemaError("unsupported optical observation format")
        _text(self.id, "observation.id")
        _text(self.sensor_id, "observation.sensor_id")
        _digest(self.sensor_sha256, "observation.sensor_sha256")
        _digest(self.scene_sha256, "observation.scene_sha256")
        _integer(self.frame_index, "observation.frame_index")
        _integer(self.seed, "observation.seed")
        for name in ("requested_at_s", "exposure_started_at_s", "exposure_duration_s", "ready_at_s"):
            _number(getattr(self, name), f"observation.{name}", minimum=0.0)
        if self.exposure_started_at_s < self.requested_at_s or self.ready_at_s < self.exposure_started_at_s + self.exposure_duration_s:
            raise OpticalSchemaError("observation timestamps are not causally ordered")
        if not self.outputs or len({item.name for item in self.outputs}) != len(self.outputs):
            raise OpticalSchemaError("observation outputs must be nonempty and unique")
        assembly_binding = (self.assembly_id, self.assembly_sha256, self.mount_connector, self.mount_transform_sha256)
        if any(item is not None for item in assembly_binding) and not all(item is not None for item in assembly_binding):
            raise OpticalSchemaError("assembly-bound observations require assembly_id, assembly_sha256, mount_connector, and mount_transform_sha256 together")
        if self.assembly_id is not None:
            _text(self.assembly_id, "observation.assembly_id")
            _digest(self.assembly_sha256, "observation.assembly_sha256")
        _text(self.assembly_frame, "observation.assembly_frame")
        if self.mount_connector is not None:
            _text(self.mount_connector, "observation.mount_connector")
            if "." not in self.mount_connector:
                raise OpticalSchemaError("assembly-bound mount_connector must be a qualified '<component>.<connector>' ID")
            _digest(self.mount_transform_sha256, "observation.mount_transform_sha256")
        object.__setattr__(self, "metadata", _json_value(_object(self.metadata, "observation.metadata")))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, manifest_path: str | Path | None = None) -> "ObservationArtifact":
        value = _object(value, "optical observation")
        names = {"format", "id", "sensor_id", "sensor_sha256", "scene_sha256", "frame_index", "requested_at_s", "exposure_started_at_s", "exposure_duration_s", "ready_at_s", "pose", "seed", "outputs", "assembly_id", "assembly_sha256", "assembly_frame", "mount_connector", "mount_transform_sha256", "metadata"}
        optional = {"metadata", "assembly_id", "assembly_sha256", "assembly_frame", "mount_connector", "mount_transform_sha256"}
        _keys(value, names, names - optional, "optical observation")
        return cls(
            id=value["id"], sensor_id=value["sensor_id"], sensor_sha256=value["sensor_sha256"],
            scene_sha256=value["scene_sha256"], frame_index=value["frame_index"],
            requested_at_s=value["requested_at_s"], exposure_started_at_s=value["exposure_started_at_s"],
            exposure_duration_s=value["exposure_duration_s"], ready_at_s=value["ready_at_s"],
            pose=Pose.from_dict(value["pose"]), seed=value["seed"],
            outputs=tuple(ObservationOutput.from_dict(item) for item in value["outputs"]),
            assembly_id=value.get("assembly_id"), assembly_sha256=value.get("assembly_sha256"),
            assembly_frame=value.get("assembly_frame", "world"), mount_connector=value.get("mount_connector"),
            mount_transform_sha256=value.get("mount_transform_sha256"),
            metadata=_object(value.get("metadata", {}), "observation.metadata"), format=value["format"],
            _manifest_path=None if manifest_path is None else Path(manifest_path).resolve(),
        )

    @classmethod
    def load(cls, path: str | Path, *, verify_content: bool = True) -> "ObservationArtifact":
        source, value = _read_json(path, "optical observation")
        result = cls.from_dict(value, manifest_path=source)
        if verify_content:
            result.verify()
        return result

    @classmethod
    def from_arrays(
        cls,
        *,
        path: str | Path,
        arrays: Mapping[str, Any],
        id: str,
        sensor: OpticalSensor,
        scene_sha256: str,
        frame_index: int,
        requested_at_s: float,
        exposure_started_at_s: float,
        ready_at_s: float,
        pose: Pose,
        seed: int,
        assembly_id: str | None = None,
        assembly_sha256: str | None = None,
        assembly_frame: str = "world",
        mount_connector: str | None = None,
        mount_transform_sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ObservationArtifact":
        import numpy as np

        manifest = Path(path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        targets = [manifest.parent / f"{manifest.stem}.{name}.npy" for name in sensor.outputs]
        existing = [target for target in (manifest, *targets) if target.exists()]
        if existing:
            raise OpticalSchemaError(f"observation artifact would overwrite existing content: {existing[0]}")
        outputs: list[ObservationOutput] = []
        expected = set(sensor.outputs)
        if set(arrays) != expected:
            raise OpticalSchemaError(f"render products differ from sensor outputs: expected {sorted(expected)}")
        created: list[Path] = []
        try:
            for name, target in zip(sensor.outputs, targets, strict=True):
                value = np.asarray(arrays[name])
                dtype = np.dtype("<i4" if name == "segmentation" else "<f4")
                value = np.ascontiguousarray(value, dtype=dtype)
                width, height = sensor.resolution_px
                expected_shape = (height, width, 3) if name == "rgb_linear" else (height, width)
                if value.shape != expected_shape:
                    raise OpticalSchemaError(
                        f"observation output {name!r} has shape {value.shape}, expected {expected_shape}"
                    )
                with target.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                created.append(target)
                reference = ContentReference.from_path(target, relative_to=manifest.parent, media_type="application/vnd.numpy.npy")
                outputs.append(
                    ObservationOutput(
                        name,
                        "int32" if name == "segmentation" else "float32",
                        tuple(value.shape),
                        reference,
                    )
                )
            result = cls(
                id=id, sensor_id=sensor.id, sensor_sha256=sensor.artifact_sha256,
                scene_sha256=_digest(scene_sha256, "scene_sha256"), frame_index=frame_index,
                requested_at_s=requested_at_s, exposure_started_at_s=exposure_started_at_s,
                exposure_duration_s=sensor.exposure_duration_s, ready_at_s=ready_at_s,
                pose=pose, seed=seed, outputs=tuple(outputs), assembly_id=assembly_id,
                assembly_sha256=assembly_sha256, assembly_frame=assembly_frame,
                mount_connector=mount_connector, mount_transform_sha256=mount_transform_sha256,
                metadata=metadata or {}, _manifest_path=manifest.resolve(),
            )
            result.write(manifest)
            return result
        except Exception:
            for target in reversed(created):
                target.unlink(missing_ok=True)
            raise

    def _resolve(self, reference: ContentReference) -> Path:
        if self._manifest_path is None:
            raise OpticalSchemaError("resolving observation outputs requires a loaded/written manifest")
        root = self._manifest_path.parent
        target = (root / Path(*PurePosixPath(reference.uri).parts)).resolve()
        if target != root and root not in target.parents:
            raise OpticalSchemaError("observation content escapes its artifact directory")
        return target

    def verify(self) -> None:
        import numpy as np

        for output in self.outputs:
            path = self._resolve(output.content)
            if not path.is_file():
                raise OpticalSchemaError(
                    f"observation output must be a regular file: {output.name!r}"
                )
            payload = path.read_bytes()
            if len(payload) != output.content.byte_length or hashlib.sha256(payload).hexdigest() != output.content.sha256:
                raise OpticalSchemaError(f"observation output hash mismatch for {output.name!r}")
            try:
                with path.open("rb") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise OpticalSchemaError(
                            f"unsupported NPY version {version} for {output.name!r}"
                        )
                    data_offset = stream.tell()
            except OpticalSchemaError:
                raise
            except Exception as exc:
                raise OpticalSchemaError(
                    f"invalid NPY header for observation output {output.name!r}: {exc}"
                ) from exc
            expected_dtype = np.dtype("<i4" if output.dtype == "int32" else "<f4")
            if (
                dtype != expected_dtype
                or dtype.hasobject
                or dtype.fields is not None
                or dtype.subdtype is not None
            ):
                raise OpticalSchemaError(
                    f"observation array dtype mismatch for {output.name!r}"
                )
            if fortran_order:
                raise OpticalSchemaError(
                    f"observation array {output.name!r} must use C order"
                )
            if tuple(shape) != output.shape:
                raise OpticalSchemaError(
                    f"observation array shape mismatch for {output.name!r}"
                )
            expected_length = data_offset + math.prod(shape) * dtype.itemsize
            if expected_length != len(payload):
                raise OpticalSchemaError(
                    f"observation array framing mismatch for {output.name!r}"
                )

    def load_arrays(self) -> dict[str, Any]:
        import numpy as np

        self.verify()
        result: dict[str, Any] = {}
        for output in self.outputs:
            with self._resolve(output.content).open("rb") as stream:
                value = np.load(stream, allow_pickle=False)
            expected_dtype = np.dtype("<i4" if output.dtype == "int32" else "<f4")
            if value.dtype != expected_dtype or tuple(value.shape) != output.shape:
                raise OpticalSchemaError(f"observation array metadata mismatch for {output.name!r}")
            result[output.name] = value
        return result

    def to_dict(self) -> dict[str, Any]:
        result = {"format": self.format, "id": self.id, "sensor_id": self.sensor_id, "sensor_sha256": self.sensor_sha256, "scene_sha256": self.scene_sha256, "frame_index": self.frame_index, "requested_at_s": self.requested_at_s, "exposure_started_at_s": self.exposure_started_at_s, "exposure_duration_s": self.exposure_duration_s, "ready_at_s": self.ready_at_s, "pose": self.pose.to_dict(), "seed": self.seed, "outputs": [item.to_dict() for item in self.outputs], "assembly_frame": self.assembly_frame, "metadata": _json_value(self.metadata)}
        if self.assembly_id is not None:
            result["assembly_id"] = self.assembly_id
        if self.assembly_sha256 is not None:
            result["assembly_sha256"] = self.assembly_sha256
        if self.mount_connector is not None:
            result["mount_connector"] = self.mount_connector
        if self.mount_transform_sha256 is not None:
            result["mount_transform_sha256"] = self.mount_transform_sha256
        return result

    def write(self, path: str | Path) -> Path:
        target = _write_json(self.to_dict(), path)
        object.__setattr__(self, "_manifest_path", target.resolve())
        return target

    @property
    def artifact_sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ReconstructionBlockReference:
    index: tuple[int, int, int]
    content: ContentReference

    def __post_init__(self) -> None:
        if len(self.index) != 3 or any(isinstance(item, bool) or not isinstance(item, int) for item in self.index):
            raise OpticalSchemaError("reconstruction block index must contain three integers")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReconstructionBlockReference":
        value = _object(value, "reconstruction block")
        _keys(value, {"index", "content"}, {"index", "content"}, "reconstruction block")
        return cls(tuple(value["index"]), ContentReference.from_dict(value["content"]))

    def to_dict(self) -> dict[str, Any]:
        return {"index": list(self.index), "content": self.content.to_dict()}


@dataclass(frozen=True, slots=True)
class ReconstructionState:
    id: str
    voxel_size_m: float
    block_size: int
    origin_world_m: tuple[float, float, float]
    truncation_distance_m: float
    occupancy_prior_probability: float
    occupied_probability: float
    free_probability: float
    min_log_odds: float
    max_log_odds: float
    update_count: int
    blocks: tuple[ReconstructionBlockReference, ...]
    observation_sha256: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = "reconstruction-state-1"
    representation: str = "sparse_bayesian_tsdf_occupancy"
    canonical_frame: str = "right_handed_z_up_x_forward_metres"
    _manifest_path: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.format != "reconstruction-state-1" or self.representation != "sparse_bayesian_tsdf_occupancy":
            raise OpticalSchemaError("unsupported reconstruction state format")
        if self.canonical_frame != "right_handed_z_up_x_forward_metres":
            raise OpticalSchemaError("unsupported reconstruction coordinate frame")
        _text(self.id, "reconstruction.id")
        _number(self.voxel_size_m, "reconstruction.voxel_size_m", minimum=1e-12)
        block_size = _integer(self.block_size, "reconstruction.block_size", minimum=2)
        if block_size > 64:
            raise OpticalSchemaError("reconstruction.block_size may not exceed 64")
        object.__setattr__(self, "origin_world_m", _vector(self.origin_world_m, 3, "reconstruction.origin_world_m"))
        _number(self.truncation_distance_m, "reconstruction.truncation_distance_m", minimum=self.voxel_size_m)
        probability = _number(self.occupancy_prior_probability, "reconstruction.occupancy_prior_probability", minimum=1e-12)
        if probability >= 1.0:
            raise OpticalSchemaError("reconstruction occupancy prior must be below 1")
        occupied = _number(self.occupied_probability, "reconstruction.occupied_probability", minimum=1e-12)
        free = _number(self.free_probability, "reconstruction.free_probability", minimum=1e-12)
        if occupied >= 1.0 or free >= 1.0 or occupied <= probability or free >= probability:
            raise OpticalSchemaError("reconstruction sensor probabilities must satisfy free < prior < occupied < 1")
        minimum = _number(self.min_log_odds, "reconstruction.min_log_odds")
        maximum = _number(self.max_log_odds, "reconstruction.max_log_odds")
        if minimum >= maximum:
            raise OpticalSchemaError("reconstruction min_log_odds must be below max_log_odds")
        _integer(self.update_count, "reconstruction.update_count")
        if len({item.index for item in self.blocks}) != len(self.blocks):
            raise OpticalSchemaError("reconstruction block indices must be unique")
        for digest in self.observation_sha256:
            _digest(digest, "reconstruction.observation_sha256[]")
        object.__setattr__(self, "metadata", _json_value(_object(self.metadata, "reconstruction.metadata")))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, manifest_path: str | Path | None = None) -> "ReconstructionState":
        value = _object(value, "reconstruction state")
        names = {"format", "id", "representation", "canonical_frame", "voxel_size_m", "block_size", "origin_world_m", "truncation_distance_m", "occupancy_prior_probability", "occupied_probability", "free_probability", "min_log_odds", "max_log_odds", "update_count", "blocks", "observation_sha256", "metadata"}
        required = names - {"observation_sha256", "metadata"}
        _keys(value, names, required, "reconstruction state")
        return cls(
            id=value["id"], voxel_size_m=value["voxel_size_m"], block_size=value["block_size"],
            origin_world_m=tuple(value["origin_world_m"]), truncation_distance_m=value["truncation_distance_m"],
            occupancy_prior_probability=value["occupancy_prior_probability"], occupied_probability=value["occupied_probability"],
            free_probability=value["free_probability"], min_log_odds=value["min_log_odds"], max_log_odds=value["max_log_odds"],
            update_count=value["update_count"], blocks=tuple(ReconstructionBlockReference.from_dict(item) for item in value["blocks"]),
            observation_sha256=tuple(value.get("observation_sha256", [])), metadata=_object(value.get("metadata", {}), "reconstruction.metadata"),
            format=value["format"], representation=value["representation"], canonical_frame=value["canonical_frame"],
            _manifest_path=None if manifest_path is None else Path(manifest_path).resolve(),
        )

    @classmethod
    def load(cls, path: str | Path, *, verify_content: bool = True) -> "ReconstructionState":
        source, value = _read_json(path, "reconstruction state")
        result = cls.from_dict(value, manifest_path=source)
        if verify_content:
            result.verify()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "id": self.id, "representation": self.representation, "canonical_frame": self.canonical_frame, "voxel_size_m": self.voxel_size_m, "block_size": self.block_size, "origin_world_m": list(self.origin_world_m), "truncation_distance_m": self.truncation_distance_m, "occupancy_prior_probability": self.occupancy_prior_probability, "occupied_probability": self.occupied_probability, "free_probability": self.free_probability, "min_log_odds": self.min_log_odds, "max_log_odds": self.max_log_odds, "update_count": self.update_count, "blocks": [item.to_dict() for item in self.blocks], "observation_sha256": list(self.observation_sha256), "metadata": _json_value(self.metadata)}

    def write(self, path: str | Path) -> Path:
        target = _write_json(self.to_dict(), path)
        object.__setattr__(self, "_manifest_path", target.resolve())
        return target

    def resolve(self, reference: ContentReference) -> Path:
        if self._manifest_path is None:
            raise OpticalSchemaError("resolving reconstruction blocks requires a loaded/written manifest")
        root = self._manifest_path.parent
        target = (root / Path(*PurePosixPath(reference.uri).parts)).resolve()
        if target != root and root not in target.parents:
            raise OpticalSchemaError("reconstruction block escapes its artifact directory")
        return target

    def verify(self) -> None:
        for block in self.blocks:
            if block.content.media_type != "application/vnd.contraption.sparse-voxel-block":
                raise OpticalSchemaError(
                    f"reconstruction block at {block.index} has the wrong media type"
                )
            path = self.resolve(block.content)
            if not path.is_file():
                raise OpticalSchemaError(
                    f"reconstruction block at {block.index} must be a regular file"
                )
            payload = path.read_bytes()
            if len(payload) != block.content.byte_length or hashlib.sha256(payload).hexdigest() != block.content.sha256:
                raise OpticalSchemaError(f"reconstruction block hash mismatch at {block.index}")
            try:
                from .reconstruction import VoxelBlock

                decoded = VoxelBlock.from_bytes(payload)
            except Exception as exc:
                raise OpticalSchemaError(
                    f"invalid reconstruction block at {block.index}: {exc}"
                ) from exc
            if decoded.index != block.index:
                raise OpticalSchemaError(
                    f"reconstruction block index mismatch at {block.index}"
                )
            if decoded.block_size != self.block_size:
                raise OpticalSchemaError(
                    f"reconstruction block size mismatch at {block.index}"
                )
            import numpy as np

            if np.any(decoded.occupancy_log_odds < self.min_log_odds) or np.any(
                decoded.occupancy_log_odds > self.max_log_odds
            ):
                raise OpticalSchemaError(
                    f"reconstruction occupancy log odds escape manifest bounds at {block.index}"
                )

    @property
    def artifact_sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(payload).hexdigest()

    def as_shape_volume(self, *, manifest_path: str | Path | None = None, id: str = "reconstruction") -> Any:
        """Return a shape-artifact-1 ``VolumeRepresentation`` referencing this state."""
        from contraption.shape import ContentReference as ShapeContentReference
        from contraption.shape import ShapeUncertainty, VolumeRepresentation

        path = Path(manifest_path).resolve() if manifest_path is not None else self._manifest_path
        if path is None:
            raise OpticalSchemaError("shape volume export requires a written reconstruction manifest")
        root = path.parent
        content = ShapeContentReference.from_path(path, relative_to=root, media_type="application/vnd.contraption.reconstruction-state+json")
        if self.blocks:
            indices = [item.index for item in self.blocks]
            dimensions = tuple((max(item[axis] for item in indices) - min(item[axis] for item in indices) + 1) * self.block_size for axis in range(3))
        else:
            dimensions = None
        return VolumeRepresentation(id=id, kind="sparse_tsdf", content=content, purposes=("reconstruction", "ray_trace"), voxel_size_m=self.voxel_size_m, dimensions=dimensions, mutable_topology=True, uncertainty=ShapeUncertainty("normal", {"standard_deviation_m": self.voxel_size_m}))


__all__ = [
    "ContentReference", "ObservationArtifact", "ObservationOutput", "OpticalLight",
    "OpticalScene", "OpticalSchemaError", "OpticalSensor", "Pose",
    "ReconstructionBlockReference", "ReconstructionState", "SceneObject",
    "SensorNoise", "SpectralChannel", "WirePayloadSpec",
]
