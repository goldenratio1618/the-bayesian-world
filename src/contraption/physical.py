"""Canonical physical component packages and fail-closed assembly kinematics.

This module deliberately contains no visualization behavior and no dynamics.
It resolves the physical shape and connector graph that simulation, build, and
visualization tooling can share.  Component packages are immutable data.  A
resolved assembly may be reconfigured at different root poses and revolute
joint coordinates without changing the hash of its source assembly closure.

Connector frames use a right-handed coordinate system.  A revolute connector's
local +Z axis is its joint axis.  Quaternion fields use ``[w, x, y, z]``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .specs import FrozenDict, SpecError, StrictRecord
from .units import UnitError, parse_unit


class PhysicalSpecError(SpecError):
    """A physical package or assembly declaration is incomplete or invalid."""


class PhysicalAssemblyError(PhysicalSpecError):
    """Base class for physical assembly resolution failures."""


class AssemblyCycleError(PhysicalAssemblyError):
    """The simple tree assembler encountered a closed constraint loop."""


class AssemblyUnderconstrainedError(PhysicalAssemblyError):
    """One or more physical parts lack a unique pose or joint coordinate."""


class ConnectorCompatibilityError(PhysicalAssemblyError):
    """Connected physical interfaces or PMDL bindings are incompatible."""


class ConnectorCoincidenceError(PhysicalAssemblyError):
    """Connected connector frames do not satisfy their joint constraint."""


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROVENANCE_KINDS = frozenset(
    {
        "estimated",
        "catalog",
        "vendor",
        "cad",
        "scan",
        "measured",
        "manual",
        "derived",
        "boundary",
        "software",
    }
)
_PHYSICAL_ROLES = frozenset({"part", "boundary", "software"})
_SOLID_KINDS = frozenset({"box", "cylinder", "sphere", "mesh"})
_ATTACHMENT_KINDS = frozenset({"fixed", "revolute"})
_BEHAVIOR_BINDINGS = frozenset({"kinematic_only", "pmdl"})
_MECHANICAL_DOMAINS = frozenset({"mechanical", "rigid_mechanical"})


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PhysicalSpecError(f"{context} must be an object with string keys")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PhysicalSpecError(f"{context} must be an array")
    return value


def _keys(
    value: Mapping[str, Any],
    allowed: Iterable[str],
    context: str,
    required: Iterable[str] = (),
) -> None:
    allowed_set = set(allowed)
    required_set = set(required)
    unknown = sorted(set(value) - allowed_set)
    missing = sorted(required_set - set(value))
    if unknown:
        raise PhysicalSpecError(f"unknown {context} field(s): {', '.join(unknown)}")
    if missing:
        raise PhysicalSpecError(f"missing {context} field(s): {', '.join(missing)}")


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PhysicalSpecError(f"{context} must be a non-empty identifier")
    return value


def _symbol(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SYMBOL.fullmatch(value) is None:
        raise PhysicalSpecError(f"{context} must be a non-empty PMDL symbol")
    return value


def _state_reference(value: Any, context: str) -> str:
    reference = _text(value, context)
    try:
        component, state = reference.rsplit(".", 1)
    except ValueError as exc:
        raise PhysicalSpecError(
            f"{context} must be a namespaced PMDL state reference 'component.state'"
        ) from exc
    _identifier(component, f"{context} component")
    _symbol(state, f"{context} state")
    return reference


def split_state_reference(reference: str) -> tuple[str, str]:
    """Split a previously validated ``component.state`` reference."""

    value = _state_reference(reference, "state reference")
    component, state = value.rsplit(".", 1)
    return component, state


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhysicalSpecError(f"{context} must be a non-empty string")
    return value


def _optional_text(value: Any, context: str) -> str | None:
    return None if value is None else _text(value, context)


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalSpecError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PhysicalSpecError(f"{context} must be finite")
    return result


def _vector(value: Any, length: int, context: str) -> tuple[float, ...]:
    source = _sequence(value, context)
    if len(source) != length:
        raise PhysicalSpecError(f"{context} must contain exactly {length} values")
    return tuple(_number(item, f"{context}[{index}]") for index, item in enumerate(source))


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhysicalSpecError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _json_loads(source: str, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(source, object_pairs_hook=_strict_json_object)
    except PhysicalSpecError:
        raise
    except json.JSONDecodeError as exc:
        raise PhysicalSpecError(
            f"invalid {context} JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    return _mapping(value, context)


def _canonical_json_value(value: Any, context: str = "value") -> Any:
    """Return JSON-only data while rejecting nonfinite or opaque values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PhysicalSpecError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, StrictRecord):
        return _canonical_json_value(value.to_dict(), context)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PhysicalSpecError(f"{context} contains a non-string object key")
        return {
            key: _canonical_json_value(value[key], f"{context}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(item, f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    if hasattr(value, "to_dict"):
        return _canonical_json_value(value.to_dict(), context)
    raise PhysicalSpecError(f"{context} contains unsupported type {type(value).__name__}")


def _freeze_json(value: Any, context: str = "value") -> Any:
    """Recursively freeze JSON-compatible data for public resolved artifacts."""

    canonical = _canonical_json_value(value, context)
    if isinstance(canonical, dict):
        return FrozenDict(
            (key, _freeze_json(item, f"{context}.{key}"))
            for key, item in canonical.items()
        )
    if isinstance(canonical, list):
        return tuple(
            _freeze_json(item, f"{context}[{index}]")
            for index, item in enumerate(canonical)
        )
    return canonical


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    # Optimized q * [0,v] * q^-1 for a unit quaternion.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


@dataclass(frozen=True, slots=True)
class TransformSpec(StrictRecord):
    """Rigid transform with translation in metres and a unit WXYZ quaternion."""

    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        translation = _vector(self.translation_m, 3, "transform.translation_m")
        quaternion = _vector(
            self.rotation_quaternion_wxyz,
            4,
            "transform.rotation_quaternion_wxyz",
        )
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise PhysicalSpecError(
                "transform.rotation_quaternion_wxyz must be normalized; "
                f"norm={norm:.17g}"
            )
        # q and -q encode the same rotation.  Canonicalize the sign so hashes and
        # equality do not depend on that representational ambiguity.
        first_nonzero = next((value for value in quaternion if abs(value) > 1e-15), 0.0)
        if first_nonzero < 0:
            quaternion = tuple(-value for value in quaternion)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "rotation_quaternion_wxyz", quaternion)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransformSpec":
        data = _mapping(value, "transform")
        names = ("translation_m", "rotation_quaternion_wxyz")
        _keys(data, names, "transform", names)
        return cls(
            _vector(data["translation_m"], 3, "transform.translation_m"),
            _vector(
                data["rotation_quaternion_wxyz"],
                4,
                "transform.rotation_quaternion_wxyz",
            ),
        )

    @classmethod
    def identity(cls) -> "TransformSpec":
        return cls()

    @classmethod
    def rotation_about_z(cls, angle_rad: float) -> "TransformSpec":
        angle = _number(angle_rad, "angle_rad")
        half = 0.5 * angle
        return cls((0.0, 0.0, 0.0), (math.cos(half), 0.0, 0.0, math.sin(half)))

    @classmethod
    def from_roll_pitch_yaw(
        cls,
        translation_m: Sequence[float],
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
    ) -> "TransformSpec":
        """Build a transform using the right-handed extrinsic ZYX convention."""

        roll = _number(roll_rad, "roll_rad")
        pitch = _number(pitch_rad, "pitch_rad")
        yaw = _number(yaw_rad, "yaw_rad")
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        quaternion = (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
        return cls(_vector(translation_m, 3, "translation_m"), quaternion)  # type: ignore[arg-type]

    def roll_pitch_yaw(self) -> tuple[float, float, float]:
        """Return ZYX roll, pitch, yaw, rejecting the ambiguous gimbal-lock case."""

        w, x, y, z = self.rotation_quaternion_wxyz
        sin_pitch = 2.0 * (w * y - z * x)
        if abs(sin_pitch) >= 1.0 - 1e-12:
            raise PhysicalSpecError(
                "planar root state binding cannot use a pose at Euler gimbal lock"
            )
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return roll, pitch, yaw

    def with_planar_coordinates(
        self, *, x_m: float, y_m: float, yaw_rad: float
    ) -> "TransformSpec":
        """Replace absolute X/Y/yaw while preserving this pose's Z/roll/pitch."""

        roll, pitch, _yaw = self.roll_pitch_yaw()
        return TransformSpec.from_roll_pitch_yaw(
            (
                _number(x_m, "x_m"),
                _number(y_m, "y_m"),
                self.translation_m[2],
            ),
            roll,
            pitch,
            _number(yaw_rad, "yaw_rad"),
        )

    def compose(self, child: "TransformSpec") -> "TransformSpec":
        """Return ``self * child`` using parent-from-child transform order."""

        rotated = _rotate_vector(self.rotation_quaternion_wxyz, child.translation_m)
        translation = tuple(
            left + right for left, right in zip(self.translation_m, rotated, strict=True)
        )
        quaternion = _quaternion_multiply(
            self.rotation_quaternion_wxyz, child.rotation_quaternion_wxyz
        )
        norm = math.sqrt(sum(value * value for value in quaternion))
        quaternion = tuple(value / norm for value in quaternion)
        return TransformSpec(translation, quaternion)  # type: ignore[arg-type]

    def inverse(self) -> "TransformSpec":
        w, x, y, z = self.rotation_quaternion_wxyz
        inverse_rotation = (w, -x, -y, -z)
        inverse_translation = _rotate_vector(
            inverse_rotation,
            tuple(-value for value in self.translation_m),  # type: ignore[arg-type]
        )
        return TransformSpec(inverse_translation, inverse_rotation)

    def apply(self, point_m: Sequence[float]) -> tuple[float, float, float]:
        point = _vector(point_m, 3, "point_m")
        rotated = _rotate_vector(self.rotation_quaternion_wxyz, point)  # type: ignore[arg-type]
        return tuple(
            left + right for left, right in zip(self.translation_m, rotated, strict=True)
        )  # type: ignore[return-value]

    def angular_distance(self, other: "TransformSpec") -> float:
        left_w, left_x, left_y, left_z = self.rotation_quaternion_wxyz
        relative = _quaternion_multiply(
            (left_w, -left_x, -left_y, -left_z),
            other.rotation_quaternion_wxyz,
        )
        scalar = abs(relative[0])
        vector_norm = math.sqrt(sum(value * value for value in relative[1:]))
        # atan2 is stable near identity; acos(dot) magnifies round-off into a
        # false ~1e-8 rad constraint failure when the true residual is ~1e-16.
        return 2.0 * math.atan2(vector_norm, scalar)


@dataclass(frozen=True, slots=True)
class PlanarRootStateBindingSpec(StrictRecord):
    """Absolute planar root coordinates sourced from namespaced PMDL states."""

    kind: str
    x: str
    y: str
    yaw: str

    def __post_init__(self) -> None:
        if self.kind != "planar":
            raise PhysicalSpecError(
                f"root state_binding.kind must be 'planar', got {self.kind!r}"
            )
        references = (
            _state_reference(self.x, "root state_binding.x"),
            _state_reference(self.y, "root state_binding.y"),
            _state_reference(self.yaw, "root state_binding.yaw"),
        )
        if len(set(references)) != len(references):
            raise PhysicalSpecError("root planar state bindings x/y/yaw must be distinct")
        object.__setattr__(self, "x", references[0])
        object.__setattr__(self, "y", references[1])
        object.__setattr__(self, "yaw", references[2])

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanarRootStateBindingSpec":
        data = _mapping(value, "root state_binding")
        names = ("kind", "x", "y", "yaw")
        _keys(data, names, "root state_binding", names)
        return cls(
            _text(data["kind"], "root state_binding.kind"),
            _state_reference(data["x"], "root state_binding.x"),
            _state_reference(data["y"], "root state_binding.y"),
            _state_reference(data["yaw"], "root state_binding.yaw"),
        )


@dataclass(frozen=True, slots=True)
class CounterRotationKinematicsSpec(StrictRecord):
    """Keep a non-material connector frame fixed by undoing local joint spin."""

    kind: str
    state: str

    def __post_init__(self) -> None:
        if self.kind != "counter_rotation":
            raise PhysicalSpecError(
                "connector kinematics.kind must be 'counter_rotation', "
                f"got {self.kind!r}"
            )
        _symbol(self.state, "connector kinematics.state")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CounterRotationKinematicsSpec":
        data = _mapping(value, "connector kinematics")
        names = ("kind", "state")
        _keys(data, names, "connector kinematics", names)
        return cls(
            _text(data["kind"], "connector kinematics.kind"),
            _symbol(data["state"], "connector kinematics.state"),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceSpec(StrictRecord):
    kind: str
    source: str
    reference: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _PROVENANCE_KINDS:
            raise PhysicalSpecError(
                f"provenance.kind must be one of {sorted(_PROVENANCE_KINDS)}, got {self.kind!r}"
            )
        _text(self.source, "provenance.source")
        if self.reference is not None:
            _text(self.reference, "provenance.reference")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProvenanceSpec":
        data = _mapping(value, "provenance")
        _keys(data, ("kind", "source", "reference"), "provenance", ("kind", "source"))
        return cls(
            _text(data["kind"], "provenance.kind"),
            _text(data["source"], "provenance.source"),
            _optional_text(data.get("reference"), "provenance.reference"),
        )


@dataclass(frozen=True, slots=True)
class ModelReferenceSpec(StrictRecord):
    id: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        _identifier(self.id, "model.id")
        _text(self.version, "model.version")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise PhysicalSpecError(
                "model.sha256 must be 'sha256:' followed by 64 lowercase hexadecimal characters"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelReferenceSpec":
        data = _mapping(value, "model reference")
        names = ("id", "version", "sha256")
        _keys(data, names, "model reference", names)
        return cls(
            _identifier(data["id"], "model.id"),
            _text(data["version"], "model.version"),
            _text(data["sha256"], "model.sha256"),
        )


@dataclass(frozen=True, slots=True)
class GeometrySpec(StrictRecord):
    """Explicit solid geometry.

    ``dimensions_m`` is the axis-aligned XYZ extent in the solid's local
    frame.  Cylinders use Z as their axis and therefore require equal X/Y
    diameters.  Spheres require all three diameters to be equal.  Mesh extents
    are the declared metric extents of the referenced asset, not an inferred
    or viewer-selected scale.
    """

    kind: str
    dimensions_m: tuple[float, float, float]
    mesh_uri: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _SOLID_KINDS:
            raise PhysicalSpecError(
                f"geometry.kind must be one of {sorted(_SOLID_KINDS)}, got {self.kind!r}"
            )
        dimensions = _vector(self.dimensions_m, 3, "geometry.dimensions_m")
        if any(value <= 0 for value in dimensions):
            raise PhysicalSpecError("geometry.dimensions_m values must be positive")
        if self.kind == "cylinder" and not math.isclose(
            dimensions[0], dimensions[1], rel_tol=0.0, abs_tol=1e-12
        ):
            raise PhysicalSpecError(
                "cylinder geometry requires equal X/Y diameters in dimensions_m"
            )
        if self.kind == "sphere" and not (
            math.isclose(dimensions[0], dimensions[1], rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(
                dimensions[1], dimensions[2], rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise PhysicalSpecError(
                "sphere geometry requires three equal diameters in dimensions_m"
            )
        if self.kind == "mesh":
            _text(self.mesh_uri, "geometry.mesh_uri")
        elif self.mesh_uri is not None:
            raise PhysicalSpecError("geometry.mesh_uri is permitted only for mesh geometry")
        object.__setattr__(self, "dimensions_m", dimensions)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeometrySpec":
        data = _mapping(value, "geometry")
        _keys(
            data,
            ("kind", "dimensions_m", "mesh_uri"),
            "geometry",
            ("kind", "dimensions_m"),
        )
        return cls(
            _text(data["kind"], "geometry.kind"),
            _vector(data["dimensions_m"], 3, "geometry.dimensions_m"),  # type: ignore[arg-type]
            _optional_text(data.get("mesh_uri"), "geometry.mesh_uri"),
        )


@dataclass(frozen=True, slots=True)
class SolidSpec(StrictRecord):
    id: str
    geometry: GeometrySpec
    local_pose: TransformSpec
    provenance: ProvenanceSpec

    def __post_init__(self) -> None:
        _identifier(self.id, "solid.id")
        if not isinstance(self.geometry, GeometrySpec):
            raise PhysicalSpecError("solid.geometry must be a GeometrySpec")
        if not isinstance(self.local_pose, TransformSpec):
            raise PhysicalSpecError("solid.local_pose must be a TransformSpec")
        if not isinstance(self.provenance, ProvenanceSpec):
            raise PhysicalSpecError("solid.provenance must be a ProvenanceSpec")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SolidSpec":
        data = _mapping(value, "solid")
        names = ("id", "geometry", "local_pose", "provenance")
        _keys(data, names, "solid", names)
        return cls(
            _identifier(data["id"], "solid.id"),
            GeometrySpec.from_dict(_mapping(data["geometry"], "solid.geometry")),
            TransformSpec.from_dict(_mapping(data["local_pose"], "solid.local_pose")),
            ProvenanceSpec.from_dict(_mapping(data["provenance"], "solid.provenance")),
        )


@dataclass(frozen=True, slots=True)
class BodySpec(StrictRecord):
    id: str
    local_pose: TransformSpec
    solids: tuple[SolidSpec, ...]

    def __post_init__(self) -> None:
        _identifier(self.id, "body.id")
        if not isinstance(self.local_pose, TransformSpec):
            raise PhysicalSpecError("body.local_pose must be a TransformSpec")
        solids = tuple(self.solids)
        if not solids:
            raise PhysicalSpecError(f"physical body {self.id!r} must declare at least one solid")
        duplicates = _duplicates(solid.id for solid in solids)
        if duplicates:
            raise PhysicalSpecError(
                f"physical body {self.id!r} has duplicate solid id(s): {', '.join(duplicates)}"
            )
        if any(not isinstance(solid, SolidSpec) for solid in solids):
            raise PhysicalSpecError("body.solids must contain only SolidSpec values")
        object.__setattr__(self, "solids", solids)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BodySpec":
        data = _mapping(value, "body")
        names = ("id", "local_pose", "solids")
        _keys(data, names, "body", names)
        return cls(
            _identifier(data["id"], "body.id"),
            TransformSpec.from_dict(_mapping(data["local_pose"], "body.local_pose")),
            tuple(
                SolidSpec.from_dict(_mapping(item, f"body.solids[{index}]"))
                for index, item in enumerate(_sequence(data["solids"], "body.solids"))
            ),
        )


@dataclass(frozen=True, slots=True)
class PhysicalConnectorSpec(StrictRecord):
    id: str
    model_port: str | None
    body: str | None
    domain: str
    interface: str
    local_pose: TransformSpec | None
    provenance: ProvenanceSpec
    kinematics: CounterRotationKinematicsSpec | None = None
    joint_coordinate_state: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.id, "connector.id")
        if self.model_port is not None:
            _identifier(self.model_port, "connector.model_port")
        if self.body is not None:
            _identifier(self.body, "connector.body")
        _identifier(self.domain, "connector.domain")
        _text(self.interface, "connector.interface")
        if (self.body is None) != (self.local_pose is None):
            raise PhysicalSpecError(
                f"connector {self.id!r} must declare both body and local_pose, or explicitly null both"
            )
        if self.local_pose is not None and not isinstance(self.local_pose, TransformSpec):
            raise PhysicalSpecError("connector.local_pose must be a TransformSpec or null")
        if not isinstance(self.provenance, ProvenanceSpec):
            raise PhysicalSpecError("connector.provenance must be a ProvenanceSpec")
        if self.kinematics is not None:
            if not isinstance(self.kinematics, CounterRotationKinematicsSpec):
                raise PhysicalSpecError(
                    "connector.kinematics must be CounterRotationKinematicsSpec or null"
                )
            if not self.spatial:
                raise PhysicalSpecError(
                    f"connector {self.id!r} may not declare kinematics without a body pose"
                )
        if self.joint_coordinate_state is not None:
            object.__setattr__(
                self,
                "joint_coordinate_state",
                _symbol(
                    self.joint_coordinate_state,
                    "connector.joint_coordinate_state",
                ),
            )
            if self.interface != "rotational-shaft":
                raise PhysicalSpecError(
                    f"connector {self.id!r} may bind a joint coordinate state only "
                    "for interface 'rotational-shaft'"
                )

    @property
    def spatial(self) -> bool:
        return self.body is not None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalConnectorSpec":
        data = _mapping(value, "connector")
        names = (
            "id",
            "model_port",
            "body",
            "domain",
            "interface",
            "local_pose",
            "provenance",
            "kinematics",
            "joint_coordinate_state",
        )
        required = [
            "id",
            "model_port",
            "body",
            "domain",
            "interface",
            "local_pose",
            "provenance",
        ]
        if data.get("interface") == "rotational-shaft":
            required.append("joint_coordinate_state")
        _keys(
            data,
            names,
            "connector",
            required,
        )
        model_port = data["model_port"]
        if model_port is not None:
            model_port = _identifier(model_port, "connector.model_port")
        body = data["body"]
        if body is not None:
            body = _identifier(body, "connector.body")
        local_pose = data["local_pose"]
        return cls(
            _identifier(data["id"], "connector.id"),
            model_port,
            body,
            _identifier(data["domain"], "connector.domain"),
            _text(data["interface"], "connector.interface"),
            None
            if local_pose is None
            else TransformSpec.from_dict(_mapping(local_pose, "connector.local_pose")),
            ProvenanceSpec.from_dict(_mapping(data["provenance"], "connector.provenance")),
            None
            if data.get("kinematics") is None
            else CounterRotationKinematicsSpec.from_dict(
                _mapping(data["kinematics"], "connector.kinematics")
            ),
            None
            if data.get("joint_coordinate_state") is None
            else _symbol(
                data["joint_coordinate_state"],
                "connector.joint_coordinate_state",
            ),
        )


@dataclass(frozen=True, slots=True)
class SolidRadiusMeasureSpec(StrictRecord):
    """Radius measured as half one declared local solid extent."""

    kind: str
    body: str
    solid: str
    axis: str

    def __post_init__(self) -> None:
        if self.kind != "solid_radius":
            raise PhysicalSpecError(
                f"solid radius measure kind must be 'solid_radius', got {self.kind!r}"
            )
        _identifier(self.body, "parameter measure.body")
        _identifier(self.solid, "parameter measure.solid")
        if self.axis not in {"x", "y", "z"}:
            raise PhysicalSpecError("solid radius measure.axis must be x, y, or z")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SolidRadiusMeasureSpec":
        data = _mapping(value, "solid radius measure")
        names = ("kind", "body", "solid", "axis")
        _keys(data, names, "solid radius measure", names)
        return cls(
            _text(data["kind"], "parameter measure.kind"),
            _identifier(data["body"], "parameter measure.body"),
            _identifier(data["solid"], "parameter measure.solid"),
            _text(data["axis"], "parameter measure.axis"),
        )


@dataclass(frozen=True, slots=True)
class ConnectorDistanceMeasureSpec(StrictRecord):
    """Euclidean distance between two package-local connector origins."""

    kind: str
    first_connector: str
    second_connector: str

    def __post_init__(self) -> None:
        if self.kind != "connector_distance":
            raise PhysicalSpecError(
                "connector distance measure kind must be 'connector_distance', "
                f"got {self.kind!r}"
            )
        _identifier(self.first_connector, "parameter measure.first_connector")
        _identifier(self.second_connector, "parameter measure.second_connector")
        if self.first_connector == self.second_connector:
            raise PhysicalSpecError(
                "connector distance measure requires two distinct connectors"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConnectorDistanceMeasureSpec":
        data = _mapping(value, "connector distance measure")
        names = ("kind", "first_connector", "second_connector")
        _keys(data, names, "connector distance measure", names)
        return cls(
            _text(data["kind"], "parameter measure.kind"),
            _identifier(
                data["first_connector"], "parameter measure.first_connector"
            ),
            _identifier(
                data["second_connector"], "parameter measure.second_connector"
            ),
        )


PhysicalParameterMeasureSpec = SolidRadiusMeasureSpec | ConnectorDistanceMeasureSpec


def _parameter_measure(value: Mapping[str, Any]) -> PhysicalParameterMeasureSpec:
    data = _mapping(value, "physical parameter measure")
    kind = data.get("kind")
    if kind == "solid_radius":
        return SolidRadiusMeasureSpec.from_dict(data)
    if kind == "connector_distance":
        return ConnectorDistanceMeasureSpec.from_dict(data)
    raise PhysicalSpecError(
        "physical parameter measure.kind must be 'solid_radius' or "
        f"'connector_distance', got {kind!r}"
    )


@dataclass(frozen=True, slots=True)
class PhysicalParameterBindingSpec(StrictRecord):
    """Bind one PMDL parameter to a typed measurement of package geometry."""

    model_parameter: str
    unit: str
    absolute_tolerance: float
    measure: PhysicalParameterMeasureSpec

    def __post_init__(self) -> None:
        _symbol(self.model_parameter, "parameter binding.model_parameter")
        unit = _text(self.unit, "parameter binding.unit")
        try:
            parsed_unit = parse_unit(unit)
            metre = parse_unit("m")
        except UnitError as exc:
            raise PhysicalSpecError(
                f"parameter binding.unit is invalid: {exc}"
            ) from exc
        if not parsed_unit.compatible_with(metre):
            raise PhysicalSpecError(
                f"parameter binding.unit must measure length, got {unit!r}"
            )
        tolerance = _number(
            self.absolute_tolerance, "parameter binding.absolute_tolerance"
        )
        if tolerance < 0:
            raise PhysicalSpecError(
                "parameter binding.absolute_tolerance must be nonnegative"
            )
        if not isinstance(
            self.measure, (SolidRadiusMeasureSpec, ConnectorDistanceMeasureSpec)
        ):
            raise PhysicalSpecError(
                "parameter binding.measure must be a supported typed physical measure"
            )
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "absolute_tolerance", tolerance)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalParameterBindingSpec":
        data = _mapping(value, "physical parameter binding")
        names = ("model_parameter", "unit", "absolute_tolerance", "measure")
        _keys(data, names, "physical parameter binding", names)
        return cls(
            _symbol(data["model_parameter"], "parameter binding.model_parameter"),
            _text(data["unit"], "parameter binding.unit"),
            _number(
                data["absolute_tolerance"],
                "parameter binding.absolute_tolerance",
            ),
            _parameter_measure(
                _mapping(data["measure"], "physical parameter binding.measure")
            ),
        )


@dataclass(frozen=True, slots=True)
class ComponentPackageSpec(StrictRecord):
    format: str
    id: str
    version: str
    physical_role: str
    model: ModelReferenceSpec
    bodies: tuple[BodySpec, ...]
    connectors: tuple[PhysicalConnectorSpec, ...]
    parameter_bindings: tuple[PhysicalParameterBindingSpec, ...]
    provenance: ProvenanceSpec

    def __post_init__(self) -> None:
        if self.format != "component-package-1":
            raise PhysicalSpecError(
                f"unsupported component package format {self.format!r}; expected 'component-package-1'"
            )
        _identifier(self.id, "component_package.id")
        _text(self.version, "component_package.version")
        if self.physical_role not in _PHYSICAL_ROLES:
            raise PhysicalSpecError(
                "component_package.physical_role must be one of "
                f"{sorted(_PHYSICAL_ROLES)}, got {self.physical_role!r}"
            )
        if not isinstance(self.model, ModelReferenceSpec):
            raise PhysicalSpecError("component_package.model must be a ModelReferenceSpec")
        if not isinstance(self.provenance, ProvenanceSpec):
            raise PhysicalSpecError("component_package.provenance must be a ProvenanceSpec")
        bodies = tuple(self.bodies)
        connectors = tuple(self.connectors)
        parameter_bindings = tuple(self.parameter_bindings)
        body_duplicates = _duplicates(body.id for body in bodies)
        connector_duplicates = _duplicates(connector.id for connector in connectors)
        if body_duplicates:
            raise PhysicalSpecError(
                "component package has duplicate body id(s): " + ", ".join(body_duplicates)
            )
        if connector_duplicates:
            raise PhysicalSpecError(
                "component package has duplicate connector id(s): "
                + ", ".join(connector_duplicates)
            )
        if self.physical_role == "part" and not bodies:
            raise PhysicalSpecError(
                "physical_role 'part' requires at least one explicit body with solids; "
                "placeholder geometry is not permitted"
            )
        if self.physical_role != "part" and bodies:
            raise PhysicalSpecError(
                f"physical_role {self.physical_role!r} must declare bodies: []; "
                "nonphysical boundaries/software may not carry hidden geometry"
            )
        expected_nonphysical_provenance = {
            "boundary": "boundary",
            "software": "software",
        }.get(self.physical_role)
        if (
            expected_nonphysical_provenance is not None
            and self.provenance.kind != expected_nonphysical_provenance
        ):
            raise PhysicalSpecError(
                f"physical_role {self.physical_role!r} requires package provenance.kind "
                f"{expected_nonphysical_provenance!r}"
            )
        if self.physical_role == "part" and self.provenance.kind in {"boundary", "software"}:
            raise PhysicalSpecError(
                "physical_role 'part' may not use boundary/software package provenance"
            )
        body_ids = {body.id for body in bodies}
        for connector in connectors:
            if connector.body is not None and connector.body not in body_ids:
                raise PhysicalSpecError(
                    f"connector {connector.id!r} references unknown body {connector.body!r}"
                )
            if self.physical_role == "part" and connector.body is None:
                raise PhysicalSpecError(
                    f"part connector {connector.id!r} must have an explicit body and local_pose"
                )
            if self.physical_role != "part" and connector.body is not None:
                raise PhysicalSpecError(
                    f"{self.physical_role} connector {connector.id!r} must be explicitly nonspatial"
                )
            if (
                expected_nonphysical_provenance is not None
                and connector.provenance.kind != expected_nonphysical_provenance
            ):
                raise PhysicalSpecError(
                    f"{self.physical_role} connector {connector.id!r} requires "
                    f"provenance.kind {expected_nonphysical_provenance!r}"
                )
        bound_ports = [connector.model_port for connector in connectors if connector.model_port]
        port_duplicates = _duplicates(bound_ports)
        if port_duplicates:
            raise PhysicalSpecError(
                "component package binds a model port more than once: "
                + ", ".join(port_duplicates)
            )
        if any(
            not isinstance(binding, PhysicalParameterBindingSpec)
            for binding in parameter_bindings
        ):
            raise PhysicalSpecError(
                "component package parameter_bindings must contain only "
                "PhysicalParameterBindingSpec values"
            )
        binding_duplicates = _duplicates(
            binding.model_parameter for binding in parameter_bindings
        )
        if binding_duplicates:
            raise PhysicalSpecError(
                "component package binds a model parameter more than once: "
                + ", ".join(binding_duplicates)
            )
        if self.physical_role != "part" and parameter_bindings:
            raise PhysicalSpecError(
                f"physical_role {self.physical_role!r} may not declare physical "
                "parameter bindings"
            )
        object.__setattr__(self, "bodies", bodies)
        object.__setattr__(self, "connectors", connectors)
        object.__setattr__(self, "parameter_bindings", parameter_bindings)
        for binding in parameter_bindings:
            self.measure_parameter(binding)

    @property
    def body_map(self) -> dict[str, BodySpec]:
        return {body.id: body for body in self.bodies}

    @property
    def connector_map(self) -> dict[str, PhysicalConnectorSpec]:
        return {connector.id: connector for connector in self.connectors}

    def connector(self, name: str) -> PhysicalConnectorSpec:
        try:
            return self.connector_map[name]
        except KeyError as exc:
            raise PhysicalSpecError(
                f"component package {self.id!r} has no physical connector {name!r}"
            ) from exc

    @property
    def parameter_binding_map(self) -> dict[str, PhysicalParameterBindingSpec]:
        return {
            binding.model_parameter: binding for binding in self.parameter_bindings
        }

    def measure_parameter(self, binding: PhysicalParameterBindingSpec) -> float:
        """Evaluate one allow-listed geometric measurement in its declared unit."""

        measure = binding.measure
        value_m: float
        if isinstance(measure, SolidRadiusMeasureSpec):
            try:
                body = self.body_map[measure.body]
            except KeyError as exc:
                raise PhysicalSpecError(
                    f"parameter {binding.model_parameter!r} solid-radius measure "
                    f"references unknown body {measure.body!r}"
                ) from exc
            try:
                solid = next(item for item in body.solids if item.id == measure.solid)
            except StopIteration as exc:
                raise PhysicalSpecError(
                    f"parameter {binding.model_parameter!r} solid-radius measure "
                    f"references unknown solid {measure.body}.{measure.solid}"
                ) from exc
            if solid.geometry.kind not in {"cylinder", "sphere"}:
                raise PhysicalSpecError(
                    f"parameter {binding.model_parameter!r} solid-radius measure "
                    f"requires cylinder/sphere geometry, got {solid.geometry.kind!r}"
                )
            axis_index = {"x": 0, "y": 1, "z": 2}[measure.axis]
            if solid.geometry.kind == "cylinder" and measure.axis == "z":
                raise PhysicalSpecError(
                    f"parameter {binding.model_parameter!r} cannot measure a "
                    "cylinder radius along its local z axis"
                )
            value_m = 0.5 * solid.geometry.dimensions_m[axis_index]
        else:
            assert isinstance(measure, ConnectorDistanceMeasureSpec)
            try:
                first = self.connector_map[measure.first_connector]
                second = self.connector_map[measure.second_connector]
            except KeyError as exc:
                raise PhysicalSpecError(
                    f"parameter {binding.model_parameter!r} connector-distance "
                    f"measure references unknown connector {exc.args[0]!r}"
                ) from exc
            if not first.spatial or not second.spatial:
                raise PhysicalSpecError(
                    f"parameter {binding.model_parameter!r} connector-distance "
                    "measure requires two spatial connectors"
                )
            first_pose = _body_connector_pose(self, first)
            second_pose = _body_connector_pose(self, second)
            value_m = math.sqrt(
                sum(
                    (left - right) ** 2
                    for left, right in zip(
                        first_pose.translation_m,
                        second_pose.translation_m,
                        strict=True,
                    )
                )
            )
            if value_m <= 0:
                raise PhysicalSpecError(
                    f"parameter {binding.model_parameter!r} connector-distance "
                    "measure must be positive"
                )
        try:
            return parse_unit("m").convert_value_to(
                value_m, parse_unit(binding.unit)
            )
        except UnitError as exc:  # guarded by binding construction
            raise PhysicalSpecError(
                f"could not convert physical parameter measurement: {exc}"
            ) from exc

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComponentPackageSpec":
        data = _mapping(value, "component package")
        names = (
            "format",
            "id",
            "version",
            "physical_role",
            "model",
            "bodies",
            "connectors",
            "parameter_bindings",
            "provenance",
        )
        _keys(data, names, "component package", names)
        return cls(
            _text(data["format"], "component_package.format"),
            _identifier(data["id"], "component_package.id"),
            _text(data["version"], "component_package.version"),
            _text(data["physical_role"], "component_package.physical_role"),
            ModelReferenceSpec.from_dict(_mapping(data["model"], "component_package.model")),
            tuple(
                BodySpec.from_dict(_mapping(item, f"component_package.bodies[{index}]"))
                for index, item in enumerate(
                    _sequence(data["bodies"], "component_package.bodies")
                )
            ),
            tuple(
                PhysicalConnectorSpec.from_dict(
                    _mapping(item, f"component_package.connectors[{index}]")
                )
                for index, item in enumerate(
                    _sequence(data["connectors"], "component_package.connectors")
                )
            ),
            tuple(
                PhysicalParameterBindingSpec.from_dict(
                    _mapping(item, f"component_package.parameter_bindings[{index}]")
                )
                for index, item in enumerate(
                    _sequence(
                        data["parameter_bindings"],
                        "component_package.parameter_bindings",
                    )
                )
            ),
            ProvenanceSpec.from_dict(
                _mapping(data["provenance"], "component_package.provenance")
            ),
        )

    @classmethod
    def from_json(cls, source: str) -> "ComponentPackageSpec":
        return cls.from_dict(_json_loads(source, "component package"))


class ComponentPackageRegistry(Mapping[str, ComponentPackageSpec]):
    """Immutable registry keyed by package ID with duplicate rejection."""

    __slots__ = ("_packages",)

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError("ComponentPackageRegistry is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, packages: Iterable[ComponentPackageSpec] = ()) -> None:
        values: dict[str, ComponentPackageSpec] = {}
        for package in packages:
            if not isinstance(package, ComponentPackageSpec):
                raise TypeError("package registry accepts ComponentPackageSpec values")
            if package.id in values:
                raise PhysicalSpecError(f"duplicate component package id {package.id!r}")
            values[package.id] = package
        self._packages = FrozenDict(values)

    def __getitem__(self, key: str) -> ComponentPackageSpec:
        return self._packages[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._packages)

    def __len__(self) -> int:
        return len(self._packages)

    def with_package(self, package: ComponentPackageSpec) -> "ComponentPackageRegistry":
        if package.id in self:
            raise PhysicalSpecError(f"duplicate component package id {package.id!r}")
        return ComponentPackageRegistry((*self.values(), package))

    @classmethod
    def from_dicts(
        cls, values: Iterable[Mapping[str, Any]]
    ) -> "ComponentPackageRegistry":
        return cls(ComponentPackageSpec.from_dict(value) for value in values)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComponentPackageRegistry":
        data = _mapping(value, "component package registry")
        names = ("format", "packages")
        _keys(data, names, "component package registry", names)
        if data["format"] != "component-package-registry-1":
            raise PhysicalSpecError(
                "component package registry format must be "
                "'component-package-registry-1'"
            )
        return cls.from_dicts(
            _mapping(item, f"component package registry.packages[{index}]")
            for index, item in enumerate(
                _sequence(data["packages"], "component package registry.packages")
            )
        )

    @classmethod
    def from_json(cls, source: str) -> "ComponentPackageRegistry":
        return cls.from_dict(_json_loads(source, "component package registry"))

    @classmethod
    def load_package(cls, path: str | Path) -> ComponentPackageSpec:
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise PhysicalSpecError(f"could not read component package {source}: {exc}") from exc
        return ComponentPackageSpec.from_json(text)

    @classmethod
    def load(cls, path: str | Path) -> "ComponentPackageRegistry":
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise PhysicalSpecError(
                f"could not read component package registry {source}: {exc}"
            ) from exc
        return cls.from_json(text)

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "ComponentPackageRegistry":
        return cls(cls.load_package(path) for path in paths)

    @classmethod
    def load_directory(
        cls, path: str | Path, *, pattern: str = "*.component.json"
    ) -> "ComponentPackageRegistry":
        directory = Path(path)
        if not directory.is_dir():
            raise PhysicalSpecError(f"component package directory does not exist: {directory}")
        paths = sorted(directory.rglob(pattern))
        if not paths:
            raise PhysicalSpecError(
                f"component package directory {directory} contains no files matching {pattern!r}"
            )
        return cls(cls.load_package(package_path) for package_path in paths)


@dataclass(frozen=True, slots=True)
class PhysicalComponentInstance(StrictRecord):
    id: str
    package: str

    def __post_init__(self) -> None:
        _identifier(self.id, "component.id")
        _identifier(self.package, "component.package")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalComponentInstance":
        data = _mapping(value, "component")
        if "id" not in data or "package" not in data:
            missing = sorted({"id", "package"} - set(data))
            raise PhysicalSpecError(
                "component physical resolution is missing field(s): " + ", ".join(missing)
            )
        # A full contraption component may contain parameters, condition, and
        # provenance.  Those remain hash-bound but are interpreted by other
        # layers; this physical projection consumes only id/package.
        return cls(
            _identifier(data["id"], "component.id"),
            _identifier(data["package"], "component.package"),
        )


@dataclass(frozen=True, slots=True)
class ConnectorRef(StrictRecord):
    component: str
    connector: str

    def __post_init__(self) -> None:
        _identifier(self.component, "connector_ref.component")
        _identifier(self.connector, "connector_ref.connector")

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | str) -> "ConnectorRef":
        if isinstance(value, str):
            try:
                component, connector = value.rsplit(".", 1)
            except ValueError as exc:
                raise PhysicalSpecError(
                    "connector reference must be 'component.connector'"
                ) from exc
            return cls(
                _identifier(component, "connector_ref.component"),
                _identifier(connector, "connector_ref.connector"),
            )
        data = _mapping(value, "connector reference")
        _keys(
            data,
            ("component", "connector"),
            "connector reference",
            ("component", "connector"),
        )
        return cls(
            _identifier(data["component"], "connector_ref.component"),
            _identifier(data["connector"], "connector_ref.connector"),
        )

    @property
    def key(self) -> str:
        return f"{self.component}.{self.connector}"


@dataclass(frozen=True, slots=True)
class JointCoordinateBindingSpec(StrictRecord):
    """Bind one PMDL angle state to a revolute joint's physical angle."""

    state: str
    joint_angle_at_state_zero_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "state", _state_reference(self.state, "joint coordinate binding.state")
        )
        object.__setattr__(
            self,
            "joint_angle_at_state_zero_rad",
            _number(
                self.joint_angle_at_state_zero_rad,
                "joint coordinate binding.joint_angle_at_state_zero_rad",
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JointCoordinateBindingSpec":
        data = _mapping(value, "joint coordinate binding")
        names = ("state", "joint_angle_at_state_zero_rad")
        _keys(data, names, "joint coordinate binding", names)
        return cls(
            _state_reference(data["state"], "joint coordinate binding.state"),
            _number(
                data["joint_angle_at_state_zero_rad"],
                "joint coordinate binding.joint_angle_at_state_zero_rad",
            ),
        )


@dataclass(frozen=True, slots=True)
class MechanicalAttachmentSpec(StrictRecord):
    id: str
    parent: ConnectorRef
    child: ConnectorRef
    kind: str
    behavior_binding: str
    coordinate: str | None = None
    zero_angle_rad: float = 0.0
    coordinate_bindings: tuple[JointCoordinateBindingSpec, ...] = ()
    domain: str | None = None
    metadata: FrozenDict[Any] = FrozenDict()

    def __post_init__(self) -> None:
        _identifier(self.id, "attachment.id")
        if self.kind not in _ATTACHMENT_KINDS:
            raise PhysicalSpecError(
                f"attachment.kind must be one of {sorted(_ATTACHMENT_KINDS)}"
            )
        if self.behavior_binding not in _BEHAVIOR_BINDINGS:
            raise PhysicalSpecError(
                "attachment.behavior_binding must be one of "
                f"{sorted(_BEHAVIOR_BINDINGS)}"
            )
        if self.parent.component == self.child.component:
            raise PhysicalSpecError(
                f"attachment {self.id!r} may not connect a component to itself"
            )
        coordinate_bindings = tuple(self.coordinate_bindings)
        if any(
            not isinstance(binding, JointCoordinateBindingSpec)
            for binding in coordinate_bindings
        ):
            raise PhysicalSpecError(
                "attachment.coordinate_bindings must contain only "
                "JointCoordinateBindingSpec values"
            )
        duplicate_states = _duplicates(binding.state for binding in coordinate_bindings)
        if duplicate_states:
            raise PhysicalSpecError(
                f"attachment {self.id!r} binds joint state(s) more than once: "
                + ", ".join(duplicate_states)
            )
        if self.kind == "revolute":
            if self.coordinate is None:
                raise PhysicalSpecError(
                    f"revolute attachment {self.id!r} requires a coordinate"
                )
            _identifier(self.coordinate, "attachment.coordinate")
            if not coordinate_bindings:
                raise PhysicalSpecError(
                    f"revolute attachment {self.id!r} requires coordinate_bindings"
                )
            if coordinate_bindings[0].state != self.coordinate:
                raise PhysicalSpecError(
                    f"revolute attachment {self.id!r} primary coordinate "
                    f"{self.coordinate!r} must equal first coordinate binding state "
                    f"{coordinate_bindings[0].state!r}"
                )
            endpoint_components = {self.parent.component, self.child.component}
            invalid_components = sorted(
                {
                    split_state_reference(binding.state)[0]
                    for binding in coordinate_bindings
                    if split_state_reference(binding.state)[0]
                    not in endpoint_components
                }
            )
            if invalid_components:
                raise PhysicalSpecError(
                    f"revolute attachment {self.id!r} coordinate bindings must "
                    f"reference endpoint components; invalid={invalid_components}"
                )
        elif self.coordinate is not None:
            raise PhysicalSpecError(
                f"fixed attachment {self.id!r} may not declare a coordinate"
            )
        _number(self.zero_angle_rad, "attachment.zero_angle_rad")
        if self.kind == "fixed" and self.zero_angle_rad != 0.0:
            raise PhysicalSpecError(
                f"fixed attachment {self.id!r} may not declare a nonzero zero_angle_rad"
            )
        if self.kind == "fixed" and coordinate_bindings:
            raise PhysicalSpecError(
                f"fixed attachment {self.id!r} requires coordinate_bindings: []"
            )
        if self.kind == "revolute" and not math.isclose(
            coordinate_bindings[0].joint_angle_at_state_zero_rad,
            self.zero_angle_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise PhysicalSpecError(
                f"revolute attachment {self.id!r} zero_angle_rad must equal the "
                "first coordinate binding's joint_angle_at_state_zero_rad"
            )
        if self.domain is not None:
            _identifier(self.domain, "attachment.domain")
        metadata = _freeze_json(self.metadata, "attachment.metadata")
        if not isinstance(metadata, FrozenDict):
            raise PhysicalSpecError("attachment.metadata must be an object")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "coordinate_bindings", coordinate_bindings)

    @classmethod
    def from_connection(cls, value: Mapping[str, Any]) -> "MechanicalAttachmentSpec":
        data = _mapping(value, "attachment connection")
        outer_names = ("id", "kind", "endpoints", "joint", "domain", "metadata")
        _keys(
            data,
            outer_names,
            "attachment connection",
            ("id", "kind", "endpoints", "joint"),
        )
        if data.get("kind") != "attachment":
            raise PhysicalSpecError("mechanical attachment connection kind must be 'attachment'")
        endpoints = _sequence(data["endpoints"], "attachment.endpoints")
        if len(endpoints) != 2:
            raise PhysicalSpecError(
                f"attachment {data.get('id')!r} must have exactly two ordered endpoints; "
                f"got {len(endpoints)}"
            )
        joint = _mapping(data["joint"], "attachment.joint")
        names = (
            "kind",
            "behavior_binding",
            "coordinate",
            "zero_angle_rad",
            "coordinate_bindings",
        )
        _keys(
            joint,
            names,
            "attachment.joint",
            ("kind", "behavior_binding", "coordinate_bindings"),
        )
        return cls(
            _identifier(data["id"], "attachment.id"),
            ConnectorRef.from_value(endpoints[0]),
            ConnectorRef.from_value(endpoints[1]),
            _text(joint["kind"], "attachment.joint.kind"),
            _text(joint["behavior_binding"], "attachment.joint.behavior_binding"),
            None
            if joint.get("coordinate") is None
            else _identifier(joint["coordinate"], "attachment.joint.coordinate"),
            _number(joint.get("zero_angle_rad", 0.0), "attachment.joint.zero_angle_rad"),
            tuple(
                JointCoordinateBindingSpec.from_dict(
                    _mapping(item, f"attachment.joint.coordinate_bindings[{index}]")
                )
                for index, item in enumerate(
                    _sequence(
                        joint["coordinate_bindings"],
                        "attachment.joint.coordinate_bindings",
                    )
                )
            ),
            None
            if data.get("domain") is None
            else _identifier(data["domain"], "attachment.domain"),
            _freeze_json(
                _mapping(data.get("metadata", {}), "attachment.metadata"),
                "attachment.metadata",
            ),
        )


def _as_package_registry(
    packages: ComponentPackageRegistry
    | Mapping[str, ComponentPackageSpec | Mapping[str, Any]]
    | Iterable[ComponentPackageSpec],
) -> ComponentPackageRegistry:
    if isinstance(packages, ComponentPackageRegistry):
        return packages
    if isinstance(packages, Mapping):
        parsed: list[ComponentPackageSpec] = []
        for key, value in packages.items():
            package = (
                value
                if isinstance(value, ComponentPackageSpec)
                else ComponentPackageSpec.from_dict(_mapping(value, f"packages.{key}"))
            )
            if key != package.id:
                raise PhysicalSpecError(
                    f"package registry key {key!r} does not match package id {package.id!r}"
                )
            parsed.append(package)
        return ComponentPackageRegistry(parsed)
    return ComponentPackageRegistry(packages)


def _validate_physical_parameter_overrides(
    component_values: Sequence[Any],
    components: Sequence[PhysicalComponentInstance],
    packages: Mapping[str, ComponentPackageSpec],
) -> None:
    for index, (raw_component, component) in enumerate(
        zip(component_values, components, strict=True)
    ):
        data = _mapping(raw_component, f"contraption.components[{index}]")
        parameters = _mapping(
            data.get("parameters", {}), f"contraption.components[{index}].parameters"
        )
        package = packages[component.package]
        for binding in package.parameter_bindings:
            if binding.model_parameter not in parameters:
                continue
            raw_override = parameters[binding.model_parameter]
            override_unit = binding.unit
            if isinstance(raw_override, Mapping):
                override = _mapping(
                    raw_override,
                    f"component {component.id!r} parameter {binding.model_parameter!r}",
                )
                if "value" not in override:
                    raise PhysicalSpecError(
                        f"component {component.id!r} parameter "
                        f"{binding.model_parameter!r} override must contain numeric value"
                    )
                raw_value = override["value"]
                if override.get("unit") is not None:
                    override_unit = _text(
                        override["unit"],
                        f"component {component.id!r} parameter "
                        f"{binding.model_parameter!r}.unit",
                    )
            else:
                raw_value = raw_override
            value = _number(
                raw_value,
                f"component {component.id!r} parameter {binding.model_parameter!r}",
            )
            try:
                value = parse_unit(override_unit).convert_value_to(
                    value, parse_unit(binding.unit)
                )
            except UnitError as exc:
                raise PhysicalSpecError(
                    f"component {component.id!r} parameter "
                    f"{binding.model_parameter!r} unit mismatch: {exc}"
                ) from exc
            measured = package.measure_parameter(binding)
            error = abs(value - measured)
            if error > binding.absolute_tolerance:
                raise PhysicalSpecError(
                    f"component {component.id!r} parameter "
                    f"{binding.model_parameter!r}={value:.17g} {binding.unit} "
                    f"disagrees with package physical measure={measured:.17g} "
                    f"{binding.unit}; absolute_error={error:.17g}, "
                    f"tolerance={binding.absolute_tolerance:.17g}"
                )


def _connector_for(
    reference: ConnectorRef,
    component_map: Mapping[str, PhysicalComponentInstance],
    packages: Mapping[str, ComponentPackageSpec],
) -> tuple[ComponentPackageSpec, PhysicalConnectorSpec]:
    try:
        instance = component_map[reference.component]
    except KeyError as exc:
        raise PhysicalAssemblyError(
            f"connector {reference.key!r} references unknown component {reference.component!r}"
        ) from exc
    try:
        package = packages[instance.package]
    except KeyError as exc:
        raise PhysicalAssemblyError(
            f"component {instance.id!r} references missing package {instance.package!r}"
        ) from exc
    try:
        connector = package.connector_map[reference.connector]
    except KeyError as exc:
        raise PhysicalAssemblyError(
            f"component {instance.id!r} package {package.id!r} has no connector "
            f"{reference.connector!r}"
        ) from exc
    return package, connector


def _validate_attachment_connectors(
    attachment: MechanicalAttachmentSpec,
    component_map: Mapping[str, PhysicalComponentInstance],
    packages: Mapping[str, ComponentPackageSpec],
) -> tuple[PhysicalConnectorSpec, PhysicalConnectorSpec]:
    _parent_package, parent = _connector_for(attachment.parent, component_map, packages)
    _child_package, child = _connector_for(attachment.child, component_map, packages)
    if not parent.spatial or not child.spatial:
        raise ConnectorCompatibilityError(
            f"attachment {attachment.id!r} requires spatial connectors; "
            f"{attachment.parent.key} spatial={parent.spatial}, "
            f"{attachment.child.key} spatial={child.spatial}"
        )
    if parent.kinematics is not None or child.kinematics is not None:
        raise ConnectorCompatibilityError(
            f"attachment {attachment.id!r} may not use connector-local kinematics; "
            "counter-rotating frames are non-material power/signal interfaces"
        )
    if parent.domain != child.domain:
        raise ConnectorCompatibilityError(
            f"attachment {attachment.id!r} connects incompatible domains "
            f"{parent.domain!r} and {child.domain!r}"
        )
    if parent.domain not in _MECHANICAL_DOMAINS:
        raise ConnectorCompatibilityError(
            f"attachment {attachment.id!r} requires mechanical connectors, got {parent.domain!r}"
        )
    if attachment.domain is not None and attachment.domain != parent.domain:
        raise ConnectorCompatibilityError(
            f"attachment {attachment.id!r} declares domain {attachment.domain!r}, "
            f"but its connectors use {parent.domain!r}"
        )
    if parent.interface != child.interface:
        raise ConnectorCompatibilityError(
            f"attachment {attachment.id!r} connects incompatible interfaces "
            f"{parent.interface!r} and {child.interface!r}"
        )
    if attachment.kind == "revolute" and parent.interface != "rotational-shaft":
        raise ConnectorCompatibilityError(
            f"revolute attachment {attachment.id!r} requires interface "
            f"'rotational-shaft', got {parent.interface!r}"
        )
    bound = (parent.model_port is not None, child.model_port is not None)
    if attachment.behavior_binding == "pmdl" and bound != (True, True):
        raise ConnectorCompatibilityError(
            f"behavior-bearing attachment {attachment.id!r} requires non-null model_port "
            "on both connectors"
        )
    if attachment.behavior_binding == "kinematic_only" and bound != (False, False):
        raise ConnectorCompatibilityError(
            f"kinematic-only attachment {attachment.id!r} requires model_port:null "
            "on both connectors"
        )
    return parent, child


def _validate_and_canonicalize_connections(
    connections: Sequence[Any],
    attachments: Sequence[MechanicalAttachmentSpec],
    component_map: Mapping[str, PhysicalComponentInstance],
    packages: Mapping[str, ComponentPackageSpec],
) -> tuple[FrozenDict[Any], ...]:
    attachment_map = {attachment.id: attachment for attachment in attachments}
    result: list[FrozenDict[Any]] = []
    for index, raw in enumerate(connections):
        connection = _mapping(raw, f"connections[{index}]")
        kind = _text(connection.get("kind"), f"connections[{index}].kind")
        if kind not in {"power", "signal", "attachment", "constraint"}:
            raise PhysicalSpecError(
                f"connections[{index}].kind has unsupported value {kind!r}"
            )
        allowed = {"id", "kind", "endpoints", "domain", "metadata"}
        if kind == "attachment":
            allowed.add("joint")
        _keys(
            connection,
            allowed,
            f"connections[{index}]",
            ("id", "kind", "endpoints")
            + (("joint",) if kind == "attachment" else ()),
        )
        connection_id = _identifier(connection["id"], f"connections[{index}].id")
        endpoint_values = _sequence(
            connection["endpoints"], f"connections[{index}].endpoints"
        )
        endpoints = tuple(ConnectorRef.from_value(value) for value in endpoint_values)
        if len(endpoints) < 2:
            raise ConnectorCompatibilityError(
                f"{kind} connection {connection_id!r} requires at least two endpoints"
            )
        duplicate_endpoints = _duplicates(endpoint.key for endpoint in endpoints)
        if duplicate_endpoints:
            raise ConnectorCompatibilityError(
                f"{kind} connection {connection_id!r} repeats endpoint(s): "
                + ", ".join(duplicate_endpoints)
            )
        domains: set[str] = set()
        interfaces: set[str] = set()
        for reference in endpoints:
            _package, connector = _connector_for(reference, component_map, packages)
            if kind in {"power", "signal"} and connector.model_port is None:
                raise ConnectorCompatibilityError(
                    f"{kind} endpoint {reference.key!r} must bind a non-null model_port"
                )
            domains.add(connector.domain)
            interfaces.add(connector.interface)
        if kind != "constraint" and len(domains) > 1:
            raise ConnectorCompatibilityError(
                f"{kind} connection {connection_id!r} spans incompatible "
                f"connector domains: {sorted(domains)}"
            )
        if kind != "constraint" and len(interfaces) > 1:
            raise ConnectorCompatibilityError(
                f"{kind} connection {connection_id!r} spans incompatible physical "
                f"interfaces: {sorted(interfaces)}"
            )
        declared_domain = connection.get("domain")
        if kind in {"power", "signal", "attachment"} and declared_domain is None:
            raise ConnectorCompatibilityError(
                f"{kind} connection {connection_id!r} requires an explicit domain"
            )
        if declared_domain is not None:
            declared_domain = _identifier(
                declared_domain, f"connections[{index}].domain"
            )
            if kind != "constraint" and domains != {declared_domain}:
                raise ConnectorCompatibilityError(
                    f"{kind} connection {connection_id!r} declares domain "
                    f"{declared_domain!r}, but its connectors use {sorted(domains)}"
                )
        metadata = _canonical_json_value(
            _mapping(connection.get("metadata", {}), f"connections[{index}].metadata"),
            f"connections[{index}].metadata",
        )
        if metadata:
            raise PhysicalSpecError(
                f"connections[{index}].metadata must be empty; physical semantics "
                "require typed connection fields rather than opaque metadata"
            )
        normalized: dict[str, Any] = {
            "id": connection_id,
            "kind": kind,
            "endpoints": [endpoint.to_dict() for endpoint in endpoints],
            "domain": declared_domain,
            "metadata": metadata,
        }
        if kind == "attachment":
            try:
                attachment = attachment_map[connection_id]
            except KeyError as exc:  # defensive: parsing and canonicalization must agree
                raise PhysicalSpecError(
                    f"attachment connection {connection_id!r} was not parsed"
                ) from exc
            normalized["joint"] = {
                "kind": attachment.kind,
                "behavior_binding": attachment.behavior_binding,
                "coordinate": attachment.coordinate,
                "zero_angle_rad": attachment.zero_angle_rad,
                "coordinate_bindings": [
                    binding.to_dict()
                    for binding in attachment.coordinate_bindings
                ],
            }
        frozen = _freeze_json(normalized, f"connections[{index}]")
        assert isinstance(frozen, FrozenDict)
        result.append(frozen)
    return tuple(result)


def _body_connector_pose(
    package: ComponentPackageSpec, connector: PhysicalConnectorSpec
) -> TransformSpec:
    if connector.body is None or connector.local_pose is None:
        raise ConnectorCompatibilityError(
            f"connector {connector.id!r} is nonspatial and has no body pose"
        )
    body = package.body_map[connector.body]
    return body.local_pose.compose(connector.local_pose)


def _joint_transform(
    attachment: MechanicalAttachmentSpec,
    joint_coordinates: Mapping[str, float],
) -> TransformSpec:
    if attachment.kind == "fixed":
        return TransformSpec.identity()
    assert attachment.coordinate is not None
    if attachment.coordinate not in joint_coordinates:
        raise AssemblyUnderconstrainedError(
            f"revolute attachment {attachment.id!r} requires joint coordinate "
            f"{attachment.coordinate!r}"
        )
    angle = _number(
        joint_coordinates[attachment.coordinate],
        f"joint_coordinates.{attachment.coordinate}",
    )
    return TransformSpec.rotation_about_z(attachment.zero_angle_rad + angle)


def _validated_joint_coordinates(
    attachments: Sequence[MechanicalAttachmentSpec],
    values: Mapping[str, float],
) -> dict[str, float]:
    source = _mapping(values, "joint_coordinates")
    coordinates = {
        _identifier(name, "joint_coordinates key"): _number(
            value, f"joint_coordinates.{name}"
        )
        for name, value in source.items()
    }
    required = {
        attachment.coordinate
        for attachment in attachments
        if attachment.kind == "revolute" and attachment.coordinate is not None
    }
    missing = sorted(required - set(coordinates))
    extra = sorted(set(coordinates) - required)
    if missing:
        raise AssemblyUnderconstrainedError(
            "missing revolute joint coordinate(s): " + ", ".join(missing)
        )
    if extra:
        raise PhysicalSpecError(
            "joint_coordinates contains undeclared coordinate(s): " + ", ".join(extra)
        )
    return coordinates


def _pose_errors(expected: TransformSpec, actual: TransformSpec) -> tuple[float, float]:
    translation_error = math.sqrt(
        sum(
            (left - right) ** 2
            for left, right in zip(
                expected.translation_m, actual.translation_m, strict=True
            )
        )
    )
    return translation_error, expected.angular_distance(actual)


@dataclass(frozen=True, slots=True)
class ResolvedPhysicalAssembly:
    """Hash-bound physical topology plus one resolved kinematic configuration."""

    assembly_sha256: str
    contraption_id: str
    root_component: str
    root_state_binding: PlanarRootStateBindingSpec | None
    components: tuple[PhysicalComponentInstance, ...]
    attachments: tuple[MechanicalAttachmentSpec, ...]
    connections: tuple[FrozenDict[Any], ...]
    packages: FrozenDict[ComponentPackageSpec]
    component_poses: FrozenDict[TransformSpec]
    body_poses: FrozenDict[TransformSpec]
    connector_poses: FrozenDict[TransformSpec]
    scene: FrozenDict[Any]
    _root_pose: TransformSpec
    _joint_coordinates: FrozenDict[float]

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.assembly_sha256) is None:
            raise PhysicalSpecError("resolved assembly has an invalid assembly_sha256")

    def with_configuration(
        self,
        *,
        root_pose: TransformSpec | Mapping[str, Any] | None = None,
        joint_coordinates: Mapping[str, float] | None = None,
        translation_tolerance_m: float = 1e-9,
        angular_tolerance_rad: float = 1e-9,
    ) -> "ResolvedPhysicalAssembly":
        """Recompute poses while preserving the source assembly hash."""

        resolved_root = self._root_pose if root_pose is None else _coerce_transform(root_pose)
        coordinates = _validated_joint_coordinates(
            self.attachments,
            self._joint_coordinates if joint_coordinates is None else joint_coordinates,
        )
        component_poses = _solve_component_poses(
            self.components,
            self.attachments,
            self.packages,
            self.root_component,
            resolved_root,
            coordinates,
        )
        validate_connector_coincidence(
            self.components,
            self.attachments,
            self.packages,
            component_poses,
            coordinates,
            translation_tolerance_m=translation_tolerance_m,
            angular_tolerance_rad=angular_tolerance_rad,
        )
        body_poses, connector_poses = _resolved_spatial_poses(
            self.components, self.packages, component_poses, coordinates
        )
        validate_mechanical_power_connection_coincidence(
            self.connections,
            connector_poses,
            translation_tolerance_m=translation_tolerance_m,
            angular_tolerance_rad=angular_tolerance_rad,
        )
        scene = _scene(
            self.assembly_sha256,
            self.contraption_id,
            self.components,
            self.connections,
            self.packages,
            body_poses,
            connector_poses,
        )
        return replace(
            self,
            component_poses=FrozenDict(component_poses),
            body_poses=FrozenDict(body_poses),
            connector_poses=FrozenDict(connector_poses),
            scene=_freeze_json(scene, "resolved scene"),
            _root_pose=resolved_root,
            _joint_coordinates=FrozenDict(coordinates),
        )

    def component_pose(self, component_id: str) -> TransformSpec:
        try:
            return self.component_poses[component_id]
        except KeyError as exc:
            raise KeyError(f"unknown or nonspatial component {component_id!r}") from exc

    def body_pose(self, component_id: str, body_id: str) -> TransformSpec:
        key = f"{component_id}/{body_id}"
        try:
            return self.body_poses[key]
        except KeyError as exc:
            raise KeyError(f"unknown physical body {key!r}") from exc

    def connector_pose(self, component_id: str, connector_id: str) -> TransformSpec:
        key = f"{component_id}.{connector_id}"
        try:
            return self.connector_poses[key]
        except KeyError as exc:
            raise KeyError(f"unknown or nonspatial connector {key!r}") from exc


def _coerce_transform(value: TransformSpec | Mapping[str, Any]) -> TransformSpec:
    return value if isinstance(value, TransformSpec) else TransformSpec.from_dict(value)


def _solve_component_poses(
    components: Sequence[PhysicalComponentInstance],
    attachments: Sequence[MechanicalAttachmentSpec],
    packages: Mapping[str, ComponentPackageSpec],
    root_component: str,
    root_pose: TransformSpec,
    joint_coordinates: Mapping[str, float],
) -> dict[str, TransformSpec]:
    component_map = {component.id: component for component in components}
    spatial_ids = {
        component.id
        for component in components
        if packages[component.package].physical_role == "part"
    }
    if root_component not in spatial_ids:
        raise AssemblyUnderconstrainedError(
            f"physical root {root_component!r} must reference a spatial part component"
        )

    parent_set = {identifier: identifier for identifier in spatial_ids}

    def find(identifier: str) -> str:
        while parent_set[identifier] != identifier:
            parent_set[identifier] = parent_set[parent_set[identifier]]
            identifier = parent_set[identifier]
        return identifier

    adjacency: dict[str, list[tuple[str, MechanicalAttachmentSpec]]] = {
        identifier: [] for identifier in spatial_ids
    }
    for attachment in attachments:
        left, right = attachment.parent.component, attachment.child.component
        if left not in spatial_ids or right not in spatial_ids:
            raise ConnectorCompatibilityError(
                f"attachment {attachment.id!r} endpoints must both be spatial part components"
            )
        _validate_attachment_connectors(attachment, component_map, packages)
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            raise AssemblyCycleError(
                f"attachment {attachment.id!r} creates a closed/cyclic physical constraint; "
                "closed-loop solving is not supported by this resolver"
            )
        parent_set[right_root] = left_root
        adjacency[left].append((right, attachment))
        adjacency[right].append((left, attachment))

    root_partition = find(root_component)
    unreachable = sorted(identifier for identifier in spatial_ids if find(identifier) != root_partition)
    if unreachable:
        raise AssemblyUnderconstrainedError(
            f"physical part(s) are not constrained to root {root_component!r}: {unreachable}"
        )

    poses: dict[str, TransformSpec] = {root_component: root_pose}
    queue: deque[str] = deque([root_component])
    while queue:
        current = queue.popleft()
        current_pose = poses[current]
        for neighbor, attachment in adjacency[current]:
            if neighbor in poses:
                continue
            parent_package, parent_connector = _connector_for(
                attachment.parent, component_map, packages
            )
            child_package, child_connector = _connector_for(
                attachment.child, component_map, packages
            )
            parent_local = _body_connector_pose(parent_package, parent_connector)
            child_local = _body_connector_pose(child_package, child_connector)
            joint = _joint_transform(attachment, joint_coordinates)
            if current == attachment.parent.component:
                connector_world = current_pose.compose(parent_local)
                poses[neighbor] = connector_world.compose(joint).compose(child_local.inverse())
            else:
                connector_world = current_pose.compose(child_local)
                poses[neighbor] = (
                    connector_world.compose(joint.inverse()).compose(parent_local.inverse())
                )
            queue.append(neighbor)
    return poses


def validate_connector_coincidence(
    components: Sequence[PhysicalComponentInstance],
    attachments: Sequence[MechanicalAttachmentSpec],
    packages: Mapping[str, ComponentPackageSpec],
    component_poses: Mapping[str, TransformSpec],
    joint_coordinates: Mapping[str, float],
    *,
    translation_tolerance_m: float = 1e-9,
    angular_tolerance_rad: float = 1e-9,
) -> None:
    """Fail if any configured attachment violates its connector-frame equation."""

    translation_tolerance = _number(
        translation_tolerance_m, "translation_tolerance_m"
    )
    angular_tolerance = _number(angular_tolerance_rad, "angular_tolerance_rad")
    if translation_tolerance < 0 or angular_tolerance < 0:
        raise ValueError("connector coincidence tolerances must be nonnegative")
    component_map = {component.id: component for component in components}
    expected_components = {
        component.id
        for component in components
        if packages[component.package].physical_role == "part"
    }
    missing_components = sorted(expected_components - set(component_poses))
    extra_components = sorted(set(component_poses) - expected_components)
    if missing_components or extra_components:
        details: list[str] = []
        if missing_components:
            details.append("missing=" + repr(missing_components))
        if extra_components:
            details.append("unexpected=" + repr(extra_components))
        raise AssemblyUnderconstrainedError(
            "runtime component pose set does not match physical parts: "
            + ", ".join(details)
        )
    invalid_pose_components = sorted(
        component_id
        for component_id, component_pose in component_poses.items()
        if not isinstance(component_pose, TransformSpec)
    )
    if invalid_pose_components:
        raise PhysicalSpecError(
            "runtime component poses must be TransformSpec values; invalid component(s): "
            + ", ".join(invalid_pose_components)
        )
    for attachment in attachments:
        parent_package, parent_connector = _connector_for(
            attachment.parent, component_map, packages
        )
        child_package, child_connector = _connector_for(
            attachment.child, component_map, packages
        )
        _validate_attachment_connectors(attachment, component_map, packages)
        try:
            parent_pose = component_poses[attachment.parent.component]
            child_pose = component_poses[attachment.child.component]
        except KeyError as exc:
            raise AssemblyUnderconstrainedError(
                f"attachment {attachment.id!r} has no resolved pose for component {exc.args[0]!r}"
            ) from exc
        expected_child_connector = (
            parent_pose.compose(_body_connector_pose(parent_package, parent_connector))
            .compose(_joint_transform(attachment, joint_coordinates))
        )
        actual_child_connector = child_pose.compose(
            _body_connector_pose(child_package, child_connector)
        )
        translation_error, angular_error = _pose_errors(
            expected_child_connector, actual_child_connector
        )
        if (
            translation_error > translation_tolerance
            or angular_error > angular_tolerance
        ):
            raise ConnectorCoincidenceError(
                f"attachment {attachment.id!r} connector coincidence failed: "
                f"parent={attachment.parent.key}, child={attachment.child.key}, "
                f"translation_error_m={translation_error:.17g} "
                f"(tolerance={translation_tolerance:.17g}), "
                f"angular_error_rad={angular_error:.17g} "
                f"(tolerance={angular_tolerance:.17g})"
            )


def validate_mechanical_power_connection_coincidence(
    connections: Sequence[Mapping[str, Any]],
    connector_poses: Mapping[str, TransformSpec],
    *,
    translation_tolerance_m: float = 1e-9,
    angular_tolerance_rad: float = 1e-9,
) -> None:
    """Validate complete connector frames for scalar mechanical power nets."""

    translation_tolerance = _number(
        translation_tolerance_m, "translation_tolerance_m"
    )
    angular_tolerance = _number(angular_tolerance_rad, "angular_tolerance_rad")
    if translation_tolerance < 0 or angular_tolerance < 0:
        raise ValueError("mechanical connection tolerances must be nonnegative")
    for index, raw_connection in enumerate(connections):
        connection = _mapping(raw_connection, f"connections[{index}]")
        if (
            connection.get("kind") != "power"
            or connection.get("domain") not in _MECHANICAL_DOMAINS
        ):
            continue
        connection_id = _identifier(
            connection.get("id"), f"connections[{index}].id"
        )
        endpoints = tuple(
            ConnectorRef.from_value(endpoint)
            for endpoint in _sequence(
                connection.get("endpoints"), f"connection {connection_id!r}.endpoints"
            )
        )
        if len(endpoints) < 2:
            raise ConnectorCompatibilityError(
                f"mechanical power connection {connection_id!r} requires at least two endpoints"
            )
        try:
            reference_pose = connector_poses[endpoints[0].key]
        except KeyError as exc:
            raise ConnectorCompatibilityError(
                f"mechanical power connection {connection_id!r} endpoint "
                f"{endpoints[0].key!r} has no spatial connector pose"
            ) from exc
        for endpoint in endpoints[1:]:
            try:
                endpoint_pose = connector_poses[endpoint.key]
            except KeyError as exc:
                raise ConnectorCompatibilityError(
                    f"mechanical power connection {connection_id!r} endpoint "
                    f"{endpoint.key!r} has no spatial connector pose"
                ) from exc
            translation_error = math.sqrt(
                sum(
                    (left - right) ** 2
                    for left, right in zip(
                        reference_pose.translation_m,
                        endpoint_pose.translation_m,
                        strict=True,
                    )
                )
            )
            angular_error = reference_pose.angular_distance(endpoint_pose)
            if (
                translation_error > translation_tolerance
                or angular_error > angular_tolerance
            ):
                raise ConnectorCoincidenceError(
                    f"mechanical power connection {connection_id!r} connector "
                    f"coincidence failed: reference={endpoints[0].key}, "
                    f"endpoint={endpoint.key}, "
                    f"translation_error_m={translation_error:.17g} "
                    f"(tolerance={translation_tolerance:.17g}), "
                    f"angular_error_rad={angular_error:.17g} "
                    f"(tolerance={angular_tolerance:.17g})"
                )


def _resolved_spatial_poses(
    components: Sequence[PhysicalComponentInstance],
    packages: Mapping[str, ComponentPackageSpec],
    component_poses: Mapping[str, TransformSpec],
    joint_coordinates: Mapping[str, float],
) -> tuple[dict[str, TransformSpec], dict[str, TransformSpec]]:
    body_poses: dict[str, TransformSpec] = {}
    connector_poses: dict[str, TransformSpec] = {}
    for component in components:
        package = packages[component.package]
        if package.physical_role != "part":
            continue
        component_pose = component_poses[component.id]
        for body in package.bodies:
            body_poses[f"{component.id}/{body.id}"] = component_pose.compose(body.local_pose)
        for connector in package.connectors:
            if connector.body is None or connector.local_pose is None:
                continue
            connector_local_pose = connector.local_pose
            if connector.kinematics is not None:
                coordinate = f"{component.id}.{connector.kinematics.state}"
                if coordinate not in joint_coordinates:
                    raise AssemblyUnderconstrainedError(
                        f"connector {component.id}.{connector.id} counter-rotation "
                        f"requires joint coordinate {coordinate!r}"
                    )
                connector_local_pose = connector_local_pose.compose(
                    TransformSpec.rotation_about_z(
                        -_number(
                            joint_coordinates[coordinate],
                            f"joint_coordinates.{coordinate}",
                        )
                    )
                )
            connector_poses[f"{component.id}.{connector.id}"] = body_poses[
                f"{component.id}/{connector.body}"
            ].compose(connector_local_pose)
    return body_poses, connector_poses


def _scene(
    assembly_sha256: str,
    contraption_id: str,
    components: Sequence[PhysicalComponentInstance],
    connections: Sequence[Mapping[str, Any]],
    packages: Mapping[str, ComponentPackageSpec],
    body_poses: Mapping[str, TransformSpec],
    connector_poses: Mapping[str, TransformSpec],
) -> dict[str, Any]:
    scene_components: list[dict[str, Any]] = []
    for component in sorted(components, key=lambda item: item.id):
        package = packages[component.package]
        scene_components.append(
            {
                "id": component.id,
                "package": package.id,
                "model": package.model.id,
                "physical_role": package.physical_role,
                "bodies": [body.to_dict() for body in package.bodies],
                "connectors": [
                    {
                        key: value
                        for key, value in connector.to_dict().items()
                        if key != "kinematics"
                    }
                    for connector in package.connectors
                ],
            }
        )
    return {
        "schema": "contraption.physical-scene/v1",
        "assembly_sha256": assembly_sha256,
        "contraption_id": contraption_id,
        "components": scene_components,
        "connections": [
            _canonical_json_value(connection, "scene.connection")
            for connection in connections
        ],
        "body_poses": {
            key: body_poses[key].to_dict() for key in sorted(body_poses)
        },
        "connector_poses": {
            key: connector_poses[key].to_dict() for key in sorted(connector_poses)
        },
    }


def _hashable_contraption(
    contraption: Mapping[str, Any], root_component: str
) -> dict[str, Any]:
    """Remove only runtime configuration while retaining scenario inputs."""

    canonical = _canonical_json_value(contraption, "contraption")
    assert isinstance(canonical, dict)
    canonical.pop("joint_coordinates", None)
    root = canonical.get("physical_root")
    if isinstance(root, dict):
        canonical["physical_root"] = {
            "component": root_component,
            "state_binding": root.get("state_binding"),
        }
    canonical.pop("root_pose", None)
    return canonical


def _assembly_sha256(
    contraption: Mapping[str, Any],
    root_component: str,
    components: Sequence[PhysicalComponentInstance],
    packages: Mapping[str, ComponentPackageSpec],
) -> str:
    used_packages = sorted({component.package for component in components})
    payload = {
        "schema": "contraption.physical-assembly-closure/v1",
        "contraption": _hashable_contraption(contraption, root_component),
        "packages": {
            package_id: packages[package_id].to_dict() for package_id in used_packages
        },
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(serialized).hexdigest()


def _root_configuration(
    contraption: Mapping[str, Any],
) -> tuple[str, TransformSpec, PlanarRootStateBindingSpec | None]:
    root = _mapping(contraption.get("physical_root"), "contraption.physical_root")
    _keys(
        root,
        ("component", "pose", "state_binding"),
        "contraption.physical_root",
        ("component", "pose", "state_binding"),
    )
    component = _identifier(
        root["component"], "contraption.physical_root.component"
    )
    binding_value = root["state_binding"]
    binding = (
        None
        if binding_value is None
        else PlanarRootStateBindingSpec.from_dict(
            _mapping(binding_value, "contraption.physical_root.state_binding")
        )
    )
    if binding is not None:
        mismatched = sorted(
            {
                split_state_reference(reference)[0]
                for reference in (binding.x, binding.y, binding.yaw)
                if split_state_reference(reference)[0] != component
            }
        )
        if mismatched:
            raise PhysicalSpecError(
                f"root state_binding for component {component!r} may not reference "
                f"other component(s): {mismatched}"
            )
    return (
        component,
        TransformSpec.from_dict(
            _mapping(root["pose"], "contraption.physical_root.pose")
        ),
        binding,
    )


def resolve_physical_assembly(
    contraption: Mapping[str, Any] | Any,
    packages: ComponentPackageRegistry
    | Mapping[str, ComponentPackageSpec | Mapping[str, Any]]
    | Iterable[ComponentPackageSpec],
    joint_coordinates: Mapping[str, float] | None = None,
    *,
    translation_tolerance_m: float = 1e-9,
    angular_tolerance_rad: float = 1e-9,
) -> ResolvedPhysicalAssembly:
    """Resolve package geometry and pairwise attachments from JSON-compatible data.

    ``joint_coordinates`` maps each revolute attachment's declared coordinate
    name to its current angle in radians.  The values are configuration, not
    assembly identity, and therefore do not change ``assembly_sha256``.
    """

    raw = contraption.to_dict() if hasattr(contraption, "to_dict") else contraption
    spec = _mapping(raw, "contraption")
    contraption_id = _identifier(spec.get("id"), "contraption.id")
    component_values = _sequence(spec.get("components"), "contraption.components")
    if not component_values:
        raise PhysicalSpecError("contraption.components must not be empty")
    components = tuple(
        PhysicalComponentInstance.from_dict(
            _mapping(value, f"contraption.components[{index}]")
        )
        for index, value in enumerate(component_values)
    )
    duplicate_components = _duplicates(component.id for component in components)
    if duplicate_components:
        raise PhysicalSpecError(
            "contraption has duplicate component id(s): "
            + ", ".join(duplicate_components)
        )
    registry = _as_package_registry(packages)
    missing_packages = sorted(
        {component.package for component in components if component.package not in registry}
    )
    if missing_packages:
        raise PhysicalAssemblyError(
            "contraption references missing component package(s): "
            + ", ".join(missing_packages)
        )
    _validate_physical_parameter_overrides(
        component_values, components, registry
    )
    component_map = {component.id: component for component in components}

    connection_values = tuple(
        _sequence(spec.get("connections", []), "contraption.connections")
    )
    connection_ids = tuple(
        _identifier(
            _mapping(connection, f"contraption.connections[{index}]").get("id"),
            f"contraption.connections[{index}].id",
        )
        for index, connection in enumerate(connection_values)
    )
    duplicate_connections = _duplicates(connection_ids)
    if duplicate_connections:
        raise PhysicalSpecError(
            "contraption has duplicate connection id(s): "
            + ", ".join(duplicate_connections)
        )
    attachments = tuple(
        MechanicalAttachmentSpec.from_connection(
            _mapping(connection, f"contraption.connections[{index}]")
        )
        for index, connection in enumerate(connection_values)
        if isinstance(connection, Mapping) and connection.get("kind") == "attachment"
    )
    duplicate_attachments = _duplicates(attachment.id for attachment in attachments)
    if duplicate_attachments:
        raise PhysicalSpecError(
            "contraption has duplicate attachment id(s): "
            + ", ".join(duplicate_attachments)
        )
    connections = _validate_and_canonicalize_connections(
        connection_values, attachments, component_map, registry
    )

    root_component, root_pose, root_state_binding = _root_configuration(spec)
    coordinates = _validated_joint_coordinates(
        attachments, {} if joint_coordinates is None else joint_coordinates
    )
    component_poses = _solve_component_poses(
        components,
        attachments,
        registry,
        root_component,
        root_pose,
        coordinates,
    )
    validate_connector_coincidence(
        components,
        attachments,
        registry,
        component_poses,
        coordinates,
        translation_tolerance_m=translation_tolerance_m,
        angular_tolerance_rad=angular_tolerance_rad,
    )
    body_poses, connector_poses = _resolved_spatial_poses(
        components, registry, component_poses, coordinates
    )
    validate_mechanical_power_connection_coincidence(
        connections,
        connector_poses,
        translation_tolerance_m=translation_tolerance_m,
        angular_tolerance_rad=angular_tolerance_rad,
    )
    digest = _assembly_sha256(spec, root_component, components, registry)
    scene = _scene(
        digest,
        contraption_id,
        components,
        connections,
        registry,
        body_poses,
        connector_poses,
    )
    used_packages = {
        package_id: registry[package_id]
        for package_id in sorted({component.package for component in components})
    }
    return ResolvedPhysicalAssembly(
        digest,
        contraption_id,
        root_component,
        root_state_binding,
        components,
        attachments,
        connections,
        FrozenDict(used_packages),
        FrozenDict(component_poses),
        FrozenDict(body_poses),
        FrozenDict(connector_poses),
        _freeze_json(scene, "resolved scene"),
        root_pose,
        FrozenDict(coordinates),
    )


def resolve_configuration(
    assembly: ResolvedPhysicalAssembly,
    *,
    root_pose: TransformSpec | Mapping[str, Any] | None = None,
    joint_coordinates: Mapping[str, float] | None = None,
    translation_tolerance_m: float = 1e-9,
    angular_tolerance_rad: float = 1e-9,
) -> ResolvedPhysicalAssembly:
    """Functional spelling of :meth:`ResolvedPhysicalAssembly.with_configuration`."""

    return assembly.with_configuration(
        root_pose=root_pose,
        joint_coordinates=joint_coordinates,
        translation_tolerance_m=translation_tolerance_m,
        angular_tolerance_rad=angular_tolerance_rad,
    )


__all__ = [
    "AssemblyCycleError",
    "AssemblyUnderconstrainedError",
    "BodySpec",
    "ComponentPackageRegistry",
    "ComponentPackageSpec",
    "ConnectorDistanceMeasureSpec",
    "ConnectorCoincidenceError",
    "ConnectorCompatibilityError",
    "ConnectorRef",
    "CounterRotationKinematicsSpec",
    "GeometrySpec",
    "JointCoordinateBindingSpec",
    "MechanicalAttachmentSpec",
    "ModelReferenceSpec",
    "PhysicalAssemblyError",
    "PhysicalComponentInstance",
    "PhysicalConnectorSpec",
    "PhysicalParameterBindingSpec",
    "PhysicalSpecError",
    "PlanarRootStateBindingSpec",
    "ProvenanceSpec",
    "ResolvedPhysicalAssembly",
    "SolidSpec",
    "SolidRadiusMeasureSpec",
    "TransformSpec",
    "resolve_configuration",
    "resolve_physical_assembly",
    "split_state_reference",
    "validate_connector_coincidence",
    "validate_mechanical_power_connection_coincidence",
]
