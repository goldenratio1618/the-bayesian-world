"""Generate a display-only browser view of a resolved physical assembly.

The viewer is deliberately not a second kinematics or simulation engine.  Its
only physical input is a canonical ``ResolvedAssembly``; optional animation is
reconstructed from an actual ``SimulationResult`` through that assembly's pose
solver. Detached scenes and caller-authored frames are not admitted. Both
Python and the browser validate that every declared body has exactly one pose
in every frame; missing geometry, stale hashes, and partial trajectories fail
loudly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import html
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from ..paths import asset_root
from ..physics.resolved import ResolvedAssembly


PHYSICAL_SCENE_SCHEMA = "contraption.physical-scene/v1"
VIEWER_SCHEMA = "contraption.viewer/v2"

_ASSET_DIRECTORY = asset_root() / "web"
_ASSEMBLY_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")


class VisualizationError(ValueError):
    """Raised when canonical assembly data cannot be rendered faithfully."""


def _convert_object(value: Any, *, include_samples: bool = True) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VisualizationError("viewer data cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _convert_object(item, include_samples=include_samples)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_convert_object(item, include_samples=include_samples) for item in value]
    if hasattr(value, "tolist"):
        return _convert_object(value.tolist(), include_samples=include_samples)
    if hasattr(value, "item"):
        try:
            return _convert_object(value.item(), include_samples=include_samples)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "to_dict"):
        try:
            converted = value.to_dict(include_samples=include_samples)
        except TypeError:
            converted = value.to_dict()
        return _convert_object(converted, include_samples=include_samples)
    if is_dataclass(value):
        return _convert_object(asdict(value), include_samples=include_samples)
    raise VisualizationError(f"unsupported viewer data type: {type(value).__name__}")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    converted = _convert_object(value)
    if not isinstance(converted, Mapping):
        raise VisualizationError(f"{label} must serialize to an object")
    return converted


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VisualizationError(f"{label} must be an object")
    return value


def _list(value: Any, label: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise VisualizationError(f"{label} must be an array")
    if nonempty and not value:
        raise VisualizationError(f"{label} must not be empty")
    return value


def _keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise VisualizationError(f"{label} is missing required fields: {', '.join(missing)}")
    unknown = sorted(set(value) - required - optional)
    if unknown:
        raise VisualizationError(
            f"{label} contains unsupported fields that the viewer would ignore: "
            f"{', '.join(unknown)}"
        )


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise VisualizationError(f"{label} must be a canonical non-empty identifier")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VisualizationError(f"{label} must be a non-empty string")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisualizationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise VisualizationError(f"{label} must be finite")
    if positive and result <= 0:
        raise VisualizationError(f"{label} must be greater than zero")
    return result


def _vector(value: Any, label: str, length: int) -> list[float]:
    items = _list(value, label)
    if len(items) != length:
        raise VisualizationError(f"{label} must contain exactly {length} numbers")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(items)]


def _positive_vector(value: Any, label: str, length: int) -> list[float]:
    items = _vector(value, label, length)
    if any(item <= 0 for item in items):
        raise VisualizationError(f"{label} values must all be greater than zero")
    return items


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ASSEMBLY_HASH.fullmatch(value) is None:
        raise VisualizationError(
            f"{label} must use canonical form 'sha256:' followed by 64 lowercase hex digits"
        )
    return value


def _pose(value: Any, label: str) -> dict[str, Any]:
    pose = _mapping(value, label)
    _keys(
        pose,
        required={"translation_m", "rotation_quaternion_wxyz"},
        optional=set(),
        label=label,
    )
    quaternion = _vector(
        pose["rotation_quaternion_wxyz"],
        f"{label}.rotation_quaternion_wxyz",
        4,
    )
    norm = math.sqrt(sum(item * item for item in quaternion))
    if abs(norm - 1.0) > 1e-9:
        raise VisualizationError(
            f"{label}.rotation_quaternion_wxyz must be normalized (norm={norm:.12g})"
        )
    first_nonzero = next((item for item in quaternion if abs(item) > 1e-15), 0.0)
    if first_nonzero < 0:
        raise VisualizationError(
            f"{label}.rotation_quaternion_wxyz must use the canonical quaternion sign"
        )
    return {
        "translation_m": _vector(pose["translation_m"], f"{label}.translation_m", 3),
        "rotation_quaternion_wxyz": quaternion,
    }


_PROVENANCE_KINDS = {
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


def _provenance(value: Any, label: str) -> dict[str, Any]:
    provenance = _mapping(value, label)
    _keys(
        provenance,
        required={"kind", "source", "reference"},
        optional=set(),
        label=label,
    )
    kind = _text(provenance["kind"], f"{label}.kind")
    if kind not in _PROVENANCE_KINDS:
        raise VisualizationError(
            f"{label}.kind must be one of {', '.join(sorted(_PROVENANCE_KINDS))}"
        )
    result = {"kind": kind, "source": _text(provenance["source"], f"{label}.source")}
    if provenance["reference"] is not None:
        result["reference"] = _text(provenance["reference"], f"{label}.reference")
    else:
        result["reference"] = None
    return result


def _geometry(value: Any, label: str) -> dict[str, Any]:
    geometry = _mapping(value, label)
    kind = _text(geometry.get("kind"), f"{label}.kind")
    if kind in {"box", "sphere", "cylinder"}:
        _keys(
            geometry,
            required={"kind", "dimensions_m", "mesh_uri"},
            optional=set(),
            label=label,
        )
        if geometry["mesh_uri"] is not None:
            raise VisualizationError(f"{label}.mesh_uri must be null for {kind} geometry")
        return {
            "kind": kind,
            "dimensions_m": _positive_vector(
                geometry["dimensions_m"], f"{label}.dimensions_m", 3
            ),
            "mesh_uri": None,
        }
    if kind == "mesh":
        raise VisualizationError(
            f"{label} references mesh geometry, but the offline viewer cannot render a URI "
            "without embedding the exact mesh; resolve it upstream instead of substituting a box"
        )
    raise VisualizationError(
        f"{label}.kind {kind!r} is unsupported; expected box, sphere, or cylinder"
    )


def validate_physical_scene(scene: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    """Validate the canonical physical scene without inventing viewer metadata."""

    source = _object(scene, "physical_scene")
    _keys(
        source,
        required={
            "schema",
            "assembly_sha256",
            "contraption_id",
            "components",
            "connections",
            "body_poses",
            "connector_poses",
        },
        optional={"body_pose_frames"},
        label="physical_scene",
    )
    if source["schema"] != PHYSICAL_SCENE_SCHEMA:
        raise VisualizationError(
            f"physical_scene.schema must be {PHYSICAL_SCENE_SCHEMA!r}, got {source['schema']!r}"
        )
    assembly_sha256 = _hash(source["assembly_sha256"], "physical_scene.assembly_sha256")
    contraption_id = _identifier(source["contraption_id"], "physical_scene.contraption_id")

    raw_components = _list(source["components"], "physical_scene.components", nonempty=True)
    components: list[dict[str, Any]] = []
    instance_ids: set[str] = set()
    body_keys: set[str] = set()
    spatial_connector_keys: set[str] = set()
    connector_ids_by_component: dict[str, set[str]] = {}
    solid_count = 0
    for component_index, raw_component in enumerate(raw_components):
        component_label = f"physical_scene.components[{component_index}]"
        component = _mapping(raw_component, component_label)
        _keys(
            component,
            required={
                "id",
                "part",
                "model",
                "physical_role",
                "bodies",
                "connectors",
            },
            optional=set(),
            label=component_label,
        )
        instance_id = _identifier(component["id"], f"{component_label}.id")
        if instance_id in instance_ids:
            raise VisualizationError(f"duplicate component id {instance_id!r}")
        instance_ids.add(instance_id)
        physical_role = _identifier(
            component["physical_role"], f"{component_label}.physical_role"
        )
        if physical_role not in {"part", "boundary", "software"}:
            raise VisualizationError(
                f"{component_label}.physical_role {physical_role!r} is unsupported"
            )
        raw_bodies = _list(
            component["bodies"],
            f"{component_label}.bodies",
            nonempty=physical_role == "part",
        )
        if physical_role != "part" and raw_bodies:
            raise VisualizationError(
                f"{component_label} has physical_role={physical_role!r} but declares spatial bodies"
            )
        bodies: list[dict[str, Any]] = []
        body_ids: set[str] = set()
        for body_index, raw_body in enumerate(raw_bodies):
            body_label = f"{component_label}.bodies[{body_index}]"
            body = _mapping(raw_body, body_label)
            _keys(
                body,
                required={"id", "local_pose", "solids"},
                optional=set(),
                label=body_label,
            )
            body_id = _identifier(body["id"], f"{body_label}.id")
            if body_id in body_ids:
                raise VisualizationError(
                    f"component {instance_id!r} has duplicate body id {body_id!r}"
                )
            body_ids.add(body_id)
            body_key = f"{instance_id}/{body_id}"
            body_keys.add(body_key)
            raw_solids = _list(body["solids"], f"{body_label}.solids", nonempty=True)
            solids: list[dict[str, Any]] = []
            solid_ids: set[str] = set()
            for solid_index, raw_solid in enumerate(raw_solids):
                solid_label = f"{body_label}.solids[{solid_index}]"
                solid = _mapping(raw_solid, solid_label)
                _keys(
                    solid,
                    required={"id", "geometry", "local_pose", "provenance"},
                    optional=set(),
                    label=solid_label,
                )
                solid_id = _identifier(solid["id"], f"{solid_label}.id")
                if solid_id in solid_ids:
                    raise VisualizationError(
                        f"body {body_key!r} has duplicate solid id {solid_id!r}"
                    )
                solid_ids.add(solid_id)
                solids.append(
                    {
                        "id": solid_id,
                        "geometry": _geometry(solid["geometry"], f"{solid_label}.geometry"),
                        "local_pose": _pose(solid["local_pose"], f"{solid_label}.local_pose"),
                        "provenance": _provenance(
                            solid["provenance"], f"{solid_label}.provenance"
                        ),
                    }
                )
                solid_count += 1
            bodies.append(
                {
                    "id": body_id,
                    "local_pose": _pose(body["local_pose"], f"{body_label}.local_pose"),
                    "solids": solids,
                }
            )

        raw_connectors = _list(component["connectors"], f"{component_label}.connectors")
        connectors: list[dict[str, Any]] = []
        connector_ids: set[str] = set()
        for connector_index, raw_connector in enumerate(raw_connectors):
            connector_label = f"{component_label}.connectors[{connector_index}]"
            connector = _mapping(raw_connector, connector_label)
            _keys(
                connector,
                required={
                    "id",
                    "model_port",
                    "body",
                    "domain",
                    "interface",
                    "local_pose",
                    "provenance",
                    "joint_coordinate_state",
                },
                optional=set(),
                label=connector_label,
            )
            connector_id = _identifier(connector["id"], f"{connector_label}.id")
            if connector_id in connector_ids:
                raise VisualizationError(
                    f"component {instance_id!r} has duplicate connector id {connector_id!r}"
                )
            connector_ids.add(connector_id)
            model_port = connector["model_port"]
            if model_port is not None:
                model_port = _identifier(model_port, f"{connector_label}.model_port")
            body_id = connector["body"]
            if body_id is not None:
                body_id = _identifier(body_id, f"{connector_label}.body")
                if body_id not in body_ids:
                    raise VisualizationError(
                        f"{connector_label}.body references unknown body {body_id!r}"
                    )
            local_pose = connector["local_pose"]
            joint_coordinate_state = connector["joint_coordinate_state"]
            if joint_coordinate_state is not None:
                joint_coordinate_state = _identifier(
                    joint_coordinate_state,
                    f"{connector_label}.joint_coordinate_state",
                )
                if connector["interface"] != "rotational-shaft":
                    raise VisualizationError(
                        f"{connector_label}.joint_coordinate_state is only valid for "
                        "interface 'rotational-shaft'"
                    )
            if (body_id is None) != (local_pose is None):
                raise VisualizationError(
                    f"{connector_label} must declare both body and local_pose, or null both"
                )
            if physical_role == "part" and body_id is None:
                raise VisualizationError(f"spatial part {instance_id!r} has nonspatial connector")
            if physical_role != "part" and body_id is not None:
                raise VisualizationError(
                    f"nonspatial {physical_role} component {instance_id!r} has a spatial connector"
                )
            if body_id is not None:
                spatial_connector_keys.add(f"{instance_id}.{connector_id}")
            connectors.append(
                {
                    "id": connector_id,
                    "model_port": model_port,
                    "body": body_id,
                    "domain": _identifier(connector["domain"], f"{connector_label}.domain"),
                    "interface": _text(
                        connector["interface"], f"{connector_label}.interface"
                    ),
                    "local_pose": None
                    if local_pose is None
                    else _pose(local_pose, f"{connector_label}.local_pose"),
                    "provenance": _provenance(
                        connector["provenance"], f"{connector_label}.provenance"
                    ),
                    "joint_coordinate_state": joint_coordinate_state,
                }
            )
        connector_ids_by_component[instance_id] = connector_ids
        components.append(
            {
                "id": instance_id,
                "part": _identifier(component["part"], f"{component_label}.part"),
                "model": _identifier(component["model"], f"{component_label}.model"),
                "physical_role": physical_role,
                "bodies": bodies,
                "connectors": connectors,
            }
        )
    if solid_count == 0:
        raise VisualizationError("physical_scene contains no renderable physical solids")

    raw_connections = _list(source["connections"], "physical_scene.connections")
    connections: list[dict[str, Any]] = []
    connection_ids: set[str] = set()
    for connection_index, raw_connection in enumerate(raw_connections):
        connection_label = f"physical_scene.connections[{connection_index}]"
        connection = _mapping(raw_connection, connection_label)
        _keys(
            connection,
            required={"id", "kind", "domain", "endpoints", "metadata"},
            optional={"joint"},
            label=connection_label,
        )
        connection_id = _identifier(connection["id"], f"{connection_label}.id")
        if connection_id in connection_ids:
            raise VisualizationError(f"duplicate connection id {connection_id!r}")
        connection_ids.add(connection_id)
        kind = _identifier(connection["kind"], f"{connection_label}.kind")
        if kind not in {"power", "signal", "attachment", "constraint"}:
            raise VisualizationError(f"{connection_label}.kind {kind!r} is unsupported")
        domain = connection["domain"]
        if domain is not None:
            domain = _identifier(domain, f"{connection_label}.domain")
        metadata = _mapping(connection["metadata"], f"{connection_label}.metadata")
        if metadata:
            raise VisualizationError(
                f"{connection_label}.metadata must be empty; physical semantics "
                "require typed connection fields"
            )
        raw_endpoints = _list(connection["endpoints"], f"{connection_label}.endpoints")
        if len(raw_endpoints) < 2:
            raise VisualizationError(f"{connection_label}.endpoints must contain at least two endpoints")
        if kind == "attachment" and len(raw_endpoints) != 2:
            raise VisualizationError(f"{connection_label} attachment must have exactly two endpoints")
        endpoints: list[dict[str, str]] = []
        for endpoint_index, raw_endpoint in enumerate(raw_endpoints):
            endpoint_label = f"{connection_label}.endpoints[{endpoint_index}]"
            endpoint = _mapping(raw_endpoint, endpoint_label)
            _keys(
                endpoint,
                required={"component", "connector"},
                optional=set(),
                label=endpoint_label,
            )
            endpoint_component = _identifier(
                endpoint["component"], f"{endpoint_label}.component"
            )
            endpoint_connector = _identifier(
                endpoint["connector"], f"{endpoint_label}.connector"
            )
            if endpoint_component not in instance_ids:
                raise VisualizationError(
                    f"{endpoint_label} references unknown component {endpoint_component!r}"
                )
            if endpoint_connector not in connector_ids_by_component[endpoint_component]:
                raise VisualizationError(
                    f"{endpoint_label} references unknown connector "
                    f"{endpoint_component}.{endpoint_connector}"
                )
            endpoints.append(
                {"component": endpoint_component, "connector": endpoint_connector}
            )
        raw_joint = connection.get("joint")
        joint: dict[str, Any] | None = None
        if kind == "attachment":
            if raw_joint is None:
                raise VisualizationError(f"{connection_label} attachment requires a typed joint")
            joint_label = f"{connection_label}.joint"
            joint_value = _mapping(raw_joint, joint_label)
            _keys(
                joint_value,
                required={
                    "kind",
                    "behavior_binding",
                    "coordinate",
                    "zero_angle_rad",
                    "coordinate_bindings",
                },
                optional=set(),
                label=joint_label,
            )
            joint_kind = _identifier(joint_value["kind"], f"{joint_label}.kind")
            behavior = _identifier(
                joint_value["behavior_binding"], f"{joint_label}.behavior_binding"
            )
            coordinate = joint_value["coordinate"]
            if coordinate is not None:
                coordinate = _identifier(coordinate, f"{joint_label}.coordinate")
            zero_angle = _number(joint_value["zero_angle_rad"], f"{joint_label}.zero_angle_rad")
            coordinate_bindings: list[dict[str, Any]] = []
            binding_states: set[str] = set()
            for binding_index, raw_binding in enumerate(
                _list(
                    joint_value["coordinate_bindings"],
                    f"{joint_label}.coordinate_bindings",
                )
            ):
                binding_label = f"{joint_label}.coordinate_bindings[{binding_index}]"
                binding = _mapping(raw_binding, binding_label)
                _keys(
                    binding,
                    required={"state", "joint_angle_at_state_zero_rad"},
                    optional=set(),
                    label=binding_label,
                )
                state = _identifier(binding["state"], f"{binding_label}.state")
                if state in binding_states:
                    raise VisualizationError(
                        f"{joint_label}.coordinate_bindings repeats state {state!r}"
                    )
                binding_states.add(state)
                coordinate_bindings.append(
                    {
                        "state": state,
                        "joint_angle_at_state_zero_rad": _number(
                            binding["joint_angle_at_state_zero_rad"],
                            f"{binding_label}.joint_angle_at_state_zero_rad",
                        ),
                    }
                )
            if joint_kind not in {"fixed", "revolute"}:
                raise VisualizationError(f"{joint_label}.kind {joint_kind!r} is unsupported")
            if behavior not in {"kinematic_only", "pmdl"}:
                raise VisualizationError(
                    f"{joint_label}.behavior_binding {behavior!r} is unsupported"
                )
            if joint_kind == "revolute" and (
                coordinate is None or not coordinate_bindings
            ):
                raise VisualizationError(
                    f"{joint_label} revolute joint requires coordinate and coordinate_bindings"
                )
            if joint_kind == "revolute" and (
                coordinate_bindings[0]["state"] != coordinate
                or abs(
                    coordinate_bindings[0]["joint_angle_at_state_zero_rad"]
                    - zero_angle
                )
                > 1e-12
            ):
                raise VisualizationError(
                    f"{joint_label} primary coordinate/zero angle must match its first binding"
                )
            if joint_kind == "fixed" and (
                coordinate is not None
                or zero_angle != 0.0
                or coordinate_bindings
            ):
                raise VisualizationError(
                    f"{joint_label} fixed joint requires null coordinate, zero angle, and no bindings"
                )
            joint = {
                "kind": joint_kind,
                "behavior_binding": behavior,
                "coordinate": coordinate,
                "zero_angle_rad": zero_angle,
                "coordinate_bindings": coordinate_bindings,
            }
        elif raw_joint is not None:
            raise VisualizationError(f"{connection_label} non-attachment connection cannot have joint")
        normalized_connection: dict[str, Any] = {
            "id": connection_id,
            "kind": kind,
            "domain": domain,
            "endpoints": endpoints,
            "metadata": dict(metadata),
        }
        if kind == "attachment":
            normalized_connection["joint"] = joint
        connections.append(normalized_connection)

    def normalize_pose_map(
        value: Any, label: str, expected_keys: set[str]
    ) -> dict[str, Any]:
        raw_poses = _mapping(value, label)
        actual_keys = set(raw_poses)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if extra:
                details.append(f"unknown {', '.join(extra)}")
            raise VisualizationError(
                f"{label} must exactly match declared bodies ({'; '.join(details)})"
            )
        return {
            key: _pose(raw_poses[key], f"{label}[{key!r}]")
            for key in sorted(raw_poses)
        }

    body_poses = normalize_pose_map(
        source["body_poses"], "physical_scene.body_poses", set(body_keys)
    )
    connector_poses = normalize_pose_map(
        source["connector_poses"],
        "physical_scene.connector_poses",
        set(spatial_connector_keys),
    )
    normalized: dict[str, Any] = {
        "schema": PHYSICAL_SCENE_SCHEMA,
        "assembly_sha256": assembly_sha256,
        "contraption_id": contraption_id,
        "components": components,
        "connections": connections,
        "body_poses": body_poses,
        "connector_poses": connector_poses,
    }
    if "body_pose_frames" in source:
        wrapper_label = "physical_scene.body_pose_frames"
        wrapper = _mapping(source["body_pose_frames"], wrapper_label)
        _keys(
            wrapper,
            required={"assembly_sha256", "frames"},
            optional=set(),
            label=wrapper_label,
        )
        frame_hash = _hash(wrapper["assembly_sha256"], f"{wrapper_label}.assembly_sha256")
        if frame_hash != assembly_sha256:
            raise VisualizationError(
                "body pose frame assembly hash mismatch: "
                f"frames have {frame_hash}, scene has {assembly_sha256}"
            )
        raw_frames = _list(wrapper["frames"], f"{wrapper_label}.frames", nonempty=True)
        frames: list[dict[str, Any]] = []
        previous_time = -math.inf
        for frame_index, raw_frame in enumerate(raw_frames):
            frame_label = f"{wrapper_label}.frames[{frame_index}]"
            frame = _mapping(raw_frame, frame_label)
            _keys(
                frame,
                required={"time_s", "body_poses", "connector_poses"},
                optional=set(),
                label=frame_label,
            )
            time_s = _number(frame["time_s"], f"{frame_label}.time_s")
            if time_s < 0:
                raise VisualizationError(f"{frame_label}.time_s must be non-negative")
            if time_s <= previous_time:
                raise VisualizationError(
                    "physical_scene.body_pose_frames time_s values must increase strictly"
                )
            previous_time = time_s
            frames.append(
                {
                    "time_s": time_s,
                    "body_poses": normalize_pose_map(
                        frame["body_poses"],
                        f"{frame_label}.body_poses",
                        set(body_keys),
                    ),
                    "connector_poses": normalize_pose_map(
                        frame["connector_poses"],
                        f"{frame_label}.connector_poses",
                        set(spatial_connector_keys),
                    ),
                }
            )
        if frames[0]["body_poses"] != body_poses:
            raise VisualizationError(
                "physical_scene.body_poses must equal the first hash-bound body pose frame"
            )
        if frames[0]["connector_poses"] != connector_poses:
            raise VisualizationError(
                "physical_scene.connector_poses must equal the first hash-bound connector pose frame"
            )
        normalized["body_pose_frames"] = {
            "assembly_sha256": frame_hash,
            "frames": frames,
        }
    return normalized


def _assets() -> tuple[str, str, str]:
    template_path = _ASSET_DIRECTORY / "viewer.html"
    script_path = _ASSET_DIRECTORY / "viewer.js"
    style_path = _ASSET_DIRECTORY / "style.css"
    missing = [str(path) for path in (template_path, script_path, style_path) if not path.is_file()]
    if missing:
        raise VisualizationError(f"viewer assets are missing: {', '.join(missing)}")
    return (
        template_path.read_text(encoding="utf-8"),
        script_path.read_text(encoding="utf-8"),
        style_path.read_text(encoding="utf-8"),
    )


def _script_json(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _live_configuration(value: Mapping[str, Any] | None) -> dict[str, str] | None:
    if value is None:
        return None
    live = _mapping(value, "live viewer configuration")
    required = {"schema_endpoint", "simulate_endpoint"}
    if set(live) != required:
        raise VisualizationError(
            "live viewer configuration must contain exactly "
            "schema_endpoint and simulate_endpoint"
        )
    result: dict[str, str] = {}
    for name in sorted(required):
        endpoint = _text(live[name], f"live viewer {name}")
        parsed = urlsplit(endpoint)
        if (
            not endpoint.startswith("/")
            or endpoint.startswith("//")
            or "\\" in endpoint
            or parsed.scheme
            or parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise VisualizationError(
                f"live viewer {name} must be a same-origin absolute path without "
                "query, fragment, or backslash"
            )
        result[name] = endpoint
    return result


@dataclass(frozen=True)
class VisualizationArtifact:
    """Standalone page plus inspectable assets and its exact assembly hash."""

    title: str
    assembly_sha256: str
    html: str
    javascript: str
    stylesheet: str
    data: Mapping[str, Any]

    @property
    def data_json(self) -> str:
        return json.dumps(self.data, indent=2, sort_keys=True, allow_nan=False) + "\n"

    @property
    def files(self) -> Mapping[str, str]:
        return {
            "index.html": self.html,
            "viewer.js": self.javascript,
            "style.css": self.stylesheet,
            "viewer-data.json": self.data_json,
        }

    def write(self, destination: str | Path) -> Mapping[str, Path]:
        path = Path(destination)
        if path.suffix.lower() in {".html", ".htm"}:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.html, encoding="utf-8")
            return {path.name: path}
        if path.suffix:
            raise VisualizationError("viewer output must be an HTML file or directory")
        path.mkdir(parents=True, exist_ok=True)
        results: dict[str, Path] = {}
        for filename, contents in self.files.items():
            target = path / filename
            target.write_text(contents, encoding="utf-8")
            results[filename] = target
        return results


def generate_viewer(
    assembly: ResolvedAssembly,
    result: Any | None = None,
    *,
    sample_index: int = 0,
    output: str | Path | None = None,
    title: str | None = None,
    live: Mapping[str, Any] | None = None,
) -> VisualizationArtifact:
    """Create a viewer only from the canonical resolved assembly projection.

    A detached physical-scene mapping is deliberately not an input: its asserted
    assembly hash cannot prove that its geometry or poses still match the source
    closure.  Static data is always read from ``assembly.scene``.  Animated
    frames are reconstructed from one exact sample of an actual
    :class:`~contraption.physics.simulator.SimulationResult` by the assembly's physical
    resolver.
    """

    if not isinstance(assembly, ResolvedAssembly):
        raise TypeError(
            "generate_viewer requires a ResolvedAssembly; detached scene mappings "
            "cannot prove canonical artifact provenance"
        )
    raw_scene = _convert_object(assembly.scene)
    if not isinstance(raw_scene, Mapping):  # defensive: resolver scenes are mappings
        raise VisualizationError("resolved assembly scene must be an object")
    resolved_scene = dict(raw_scene)
    if result is None:
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise VisualizationError("sample_index must be an integer")
        if sample_index != 0:
            raise VisualizationError(
                "sample_index is meaningful only when an actual SimulationResult is supplied"
            )
    else:
        frames = assembly.body_pose_frames(result, sample_index=sample_index)
        resolved_scene["body_pose_frames"] = _convert_object(frames)

    scene = validate_physical_scene(resolved_scene)
    assembly_sha256 = str(scene["assembly_sha256"])
    if assembly_sha256 != assembly.assembly_sha256:
        raise VisualizationError(
            "resolved assembly scene hash mismatch: scene has "
            f"{assembly_sha256}, assembly has {assembly.assembly_sha256}"
        )
    default_title = str(scene["contraption_id"])
    page_title = default_title if title is None else str(title)
    if not page_title.strip():
        raise VisualizationError("viewer title must not be empty")
    payload: dict[str, Any] = {
        "schema": VIEWER_SCHEMA,
        "title": page_title,
        "assembly_sha256": assembly_sha256,
        "scene": scene,
    }
    live_configuration = _live_configuration(live)
    if live_configuration is not None:
        payload["live"] = live_configuration
    template, script, style = _assets()
    required_markers = (
        "@@TITLE@@",
        "@@STYLE@@",
        "@@DATA@@",
        "@@SCRIPT@@",
        "@@CONNECT_SRC@@",
    )
    missing = [marker for marker in required_markers if marker not in template]
    if missing:
        raise VisualizationError(f"viewer template is missing markers: {', '.join(missing)}")
    escaped_title = html.escape(page_title, quote=True).encode(
        "ascii", "xmlcharrefreplace"
    ).decode("ascii")
    standalone = (
        template.replace("@@STYLE@@", style)
        .replace("@@DATA@@", _script_json(payload))
        .replace("@@SCRIPT@@", script)
        .replace("@@TITLE@@", escaped_title)
        .replace("@@CONNECT_SRC@@", "'self'" if live_configuration else "'none'")
    )
    artifact = VisualizationArtifact(
        page_title,
        assembly_sha256,
        standalone,
        script,
        style,
        payload,
    )
    if output is not None:
        artifact.write(output)
    return artifact


generate_visualization = generate_viewer
build_viewer = generate_viewer


__all__ = [
    "PHYSICAL_SCENE_SCHEMA",
    "VIEWER_SCHEMA",
    "VisualizationArtifact",
    "VisualizationError",
    "build_viewer",
    "generate_viewer",
    "generate_visualization",
    "validate_physical_scene",
]
