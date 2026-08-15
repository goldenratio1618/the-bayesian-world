"""Optical simulation bridge for an exact resolved assembly closure.

This module is the optical peer of the electrical/mechanical simulator entry
point. It admits only a :class:`ResolvedAssembly`, consumes its already verified
part-instantiation registry and resolver-produced poses, and renders canonical
shape-artifact surfaces. Detached scene dictionaries are deliberately not an
input because they cannot prove the same filesystem/hash closure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from contraption.physics.physical import TransformSpec
from contraption.physics.resolved import ResolvedAssembly
from contraption.shape import ShapeArtifact

from .renderer import MeshInstance, NumpyOpticalBackend, RuntimeMaterial, RuntimeScene
from .schemas import ObservationArtifact, OpticalLight, OpticalScene, OpticalSensor, Pose
from .simulation import AsyncOpticalSimulator
from .torch_backend import TorchOpticalBackend, TorchOpticsUnavailable

if TYPE_CHECKING:
    from contraption.physics.simulator import SimulationResult


class AssemblyOpticalError(ValueError):
    """Raised when an assembly cannot provide an exact optical scene/capture."""


@dataclass(frozen=True, slots=True)
class BoundOpticalSensor:
    component_id: str
    descriptor: OpticalSensor
    source_descriptor_sha256: str
    mount_connector: str
    pose: Pose

    @property
    def id(self) -> str:
        return f"{self.component_id}:{self.descriptor.id}"


@dataclass(frozen=True, slots=True)
class AssemblyOpticalFrame:
    assembly_id: str
    assembly_sha256: str
    sample_index: int
    time_index: int
    time_s: float
    scene: RuntimeScene
    sensors: tuple[BoundOpticalSensor, ...]
    external_scene_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class AssemblyOpticalCapture:
    frame: AssemblyOpticalFrame
    backend: str
    device: str
    observations: tuple[ObservationArtifact, ...]
    observation_paths: tuple[Path, ...]
    sensor_descriptor_paths: tuple[Path, ...]
    report_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "assembly-optical-capture-1",
            "assembly_id": self.frame.assembly_id,
            "assembly_sha256": self.frame.assembly_sha256,
            "sample_index": self.frame.sample_index,
            "time_index": self.frame.time_index,
            "time_s": self.frame.time_s,
            "runtime_scene_sha256": self.frame.scene.artifact_sha256,
            "external_scene_sha256": self.frame.external_scene_sha256,
            "backend": self.backend,
            "device": self.device,
            "sensors": [
                {
                    "binding_id": sensor.id,
                    "sensor_id": sensor.descriptor.id,
                    "sensor_sha256": sensor.descriptor.artifact_sha256,
                    "source_sensor_sha256": sensor.source_descriptor_sha256,
                    "mount_connector": sensor.mount_connector,
                    "pose_sha256": sensor.pose.artifact_sha256,
                    "descriptor_path": str(path),
                }
                for sensor, path in zip(
                    self.frame.sensors, self.sensor_descriptor_paths, strict=True
                )
            ],
            "observations": [
                {
                    "id": observation.id,
                    "artifact_sha256": observation.artifact_sha256,
                    "path": str(path),
                }
                for observation, path in zip(
                    self.observations, self.observation_paths, strict=True
                )
            ],
        }


def _transform(value: TransformSpec | Mapping[str, Any], context: str) -> TransformSpec:
    try:
        return value if isinstance(value, TransformSpec) else TransformSpec.from_dict(value)
    except Exception as exc:
        raise AssemblyOpticalError(f"invalid {context}: {exc}") from exc


def _matrix(value: TransformSpec | Mapping[str, Any], context: str) -> tuple[float, ...]:
    pose = _transform(value, context)
    w, x, y, z = pose.rotation_quaternion_wxyz
    rotation = (
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    )
    tx, ty, tz = pose.translation_m
    return (
        rotation[0], rotation[1], rotation[2], tx,
        rotation[3], rotation[4], rotation[5], ty,
        rotation[6], rotation[7], rotation[8], tz,
        0.0, 0.0, 0.0, 1.0,
    )


def _surface_uncertainty_m(surface: Any) -> float:
    uncertainty = surface.uncertainty
    if uncertainty.distribution == "fixed":
        return 0.0
    parameters = uncertainty.parameters
    for name in ("standard_deviation_m", "std_m", "std"):
        value = parameters.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0:
            return float(value)
    if uncertainty.distribution == "uniform":
        lower, upper = parameters.get("lower_m"), parameters.get("upper_m")
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in (lower, upper)) and float(lower) <= float(upper):
            return (float(upper) - float(lower)) / math.sqrt(12.0)
    # An authored but unquantified uncertainty must not be represented as exact.
    return max(surface.bounds_m[index + 3] - surface.bounds_m[index] for index in range(3)) * 0.01


def _configured_sensor(
    sensor: OpticalSensor,
    *,
    resolution_px: tuple[int, int] | None,
) -> OpticalSensor:
    if resolution_px is None or tuple(resolution_px) == sensor.resolution_px:
        return sensor
    if len(resolution_px) != 2 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in resolution_px):
        raise AssemblyOpticalError("sensor resolution override must contain positive integer width/height")
    width, height = resolution_px
    source_width, source_height = sensor.resolution_px
    scale_x, scale_y = width / source_width, height / source_height
    metadata = dict(sensor.metadata)
    metadata["acquisition_override"] = {
        "source_sensor_sha256": sensor.artifact_sha256,
        "source_resolution_px": list(sensor.resolution_px),
        "resolution_px": [width, height],
        "intrinsics_scaling": "independent pixel-axis scale",
    }
    return replace(
        sensor,
        resolution_px=(width, height),
        focal_length_px=(sensor.focal_length_px[0] * scale_x, sensor.focal_length_px[1] * scale_y),
        principal_point_px=(sensor.principal_point_px[0] * scale_x, sensor.principal_point_px[1] * scale_y),
        metadata=metadata,
    )


def _pose_maps(
    assembly: ResolvedAssembly,
    result: "SimulationResult | None",
    *,
    sample_index: int,
    time_index: int,
) -> tuple[int, float, Mapping[str, Any], Mapping[str, Any]]:
    if result is None:
        if sample_index != 0 or time_index != 0:
            raise AssemblyOpticalError("static assembly capture requires sample_index=time_index=0")
        return 0, 0.0, assembly.physical.body_poses, assembly.physical.connector_poses
    frames = assembly.body_pose_frames(result, sample_index=sample_index)
    values = frames["frames"]
    resolved_index = time_index if time_index >= 0 else len(values) + time_index
    if resolved_index < 0 or resolved_index >= len(values):
        raise AssemblyOpticalError(f"time_index {time_index} is outside the simulation trajectory")
    frame = values[resolved_index]
    return (
        resolved_index,
        float(frame["time_s"]),
        frame["body_poses"],
        frame["connector_poses"],
    )


def build_assembly_optical_frame(
    assembly: ResolvedAssembly,
    *,
    result: "SimulationResult | None" = None,
    sample_index: int = 0,
    time_index: int = 0,
    sensor_ids: Sequence[str] = (),
    sensor_resolution_px: tuple[int, int] | None = None,
    external_scene: OpticalScene | str | Path | None = None,
    lights: Sequence[OpticalLight] = (),
    environment_linear_rgb: tuple[float, float, float] | None = None,
) -> AssemblyOpticalFrame:
    """Build one exact optical scene/sensor frame from a resolved closure."""

    if not isinstance(assembly, ResolvedAssembly):
        raise TypeError("build_assembly_optical_frame requires ResolvedAssembly")
    resolved_time_index, time_s, body_poses, connector_poses = _pose_maps(
        assembly, result, sample_index=sample_index, time_index=time_index
    )
    requested_sensors = set(sensor_ids)
    external_manifest: OpticalScene | None = None
    external_sha256: str | None = None
    instances: list[MeshInstance] = []
    if external_scene is not None:
        external_manifest = (
            OpticalScene.load(external_scene, verify_shapes=True)
            if isinstance(external_scene, (str, Path))
            else external_scene
        )
        if not isinstance(external_manifest, OpticalScene):
            raise TypeError("external_scene must be a loaded OpticalScene or manifest path")
        # A programmatically constructed scene has no manifest root and therefore
        # cannot prove its referenced shape closure. Requiring resolution here is
        # what makes --optical-scene an evidence-bearing input rather than an
        # unverified runtime convenience.
        external_manifest.verify()
        external_runtime = RuntimeScene.from_manifest(external_manifest)
        external_sha256 = external_manifest.artifact_sha256
        instances.extend(
            replace(item, id=f"external:{item.id}")
            for item in external_runtime.instances
        )
    sensors: list[BoundOpticalSensor] = []
    segmentation_id = max((item.segmentation_id for item in instances), default=0) + 1
    scene = assembly.scene
    for component in sorted(scene["components"], key=lambda item: item["id"]):
        component_id = str(component["id"])
        try:
            instantiation = assembly.instantiations[str(component["part"])]
        except KeyError as exc:
            raise AssemblyOpticalError(
                f"resolved component {component_id!r} has no retained verified instantiation"
            ) from exc
        for body in sorted(component["bodies"], key=lambda item: item["id"]):
            body_id = str(body["id"])
            body_key = f"{component_id}/{body_id}"
            if body_key not in body_poses:
                raise AssemblyOpticalError(f"pose frame is missing body {body_key!r}")
            world_from_body = _transform(body_poses[body_key], f"body pose {body_key}")
            for solid in sorted(body["solids"], key=lambda item: item["id"]):
                solid_id = str(solid["id"])
                geometry = solid["geometry"]
                if geometry["kind"] != "shape":
                    raise AssemblyOpticalError(
                        f"assembly solid {component_id}/{body_id}/{solid_id} is not a canonical shape"
                    )
                manifest = (instantiation.directory / Path(*str(geometry["shape_uri"]).split("/"))).resolve()
                root = instantiation.directory.resolve()
                if manifest != root and root not in manifest.parents:
                    raise AssemblyOpticalError("shape manifest escapes its verified instantiation")
                payload = manifest.read_bytes()
                digest = "sha256:" + hashlib.sha256(payload).hexdigest()
                if digest != geometry["shape_sha256"]:
                    raise AssemblyOpticalError(
                        f"shape manifest digest changed after assembly resolution for {component_id}/{body_id}/{solid_id}"
                    )
                artifact = ShapeArtifact.load(manifest, verify_content=True)
                surface_id = str(geometry["surface_id"])
                surface = next((item for item in artifact.surfaces if item.id == surface_id), None)
                if surface is None or "ray_trace" not in surface.purposes:
                    raise AssemblyOpticalError(
                        f"shape {artifact.id!r} surface {surface_id!r} is absent or not ray-trace capable"
                    )
                mesh = artifact.load_surface(surface_id)
                material_by_id = {
                    item.id: RuntimeMaterial.from_shape(item)
                    for item in artifact.optical_materials
                }
                missing_materials = [item for item in surface.material_ids if item not in material_by_id]
                if missing_materials:
                    raise AssemblyOpticalError(
                        f"surface {surface_id!r} has unresolved optical materials {missing_materials}"
                    )
                materials = tuple(material_by_id[item] for item in surface.material_ids)
                if not materials:
                    materials = (RuntimeMaterial("unqualified-default"),)
                world_from_solid = world_from_body.compose(
                    _transform(solid["local_pose"], f"solid pose {component_id}/{body_id}/{solid_id}")
                )
                instances.append(
                    MeshInstance(
                        f"{component_id}/{body_id}/{solid_id}",
                        mesh,
                        segmentation_id,
                        materials,
                        _matrix(world_from_solid, "world-from-solid pose"),
                        _surface_uncertainty_m(surface),
                    )
                )
                segmentation_id += 1

        for binding in instantiation.static.optical_sensors:
            qualified_id = f"{component_id}:{binding.id}"
            if requested_sensors and binding.id not in requested_sensors and qualified_id not in requested_sensors:
                continue
            descriptor_path = (instantiation.directory / binding.descriptor_uri).resolve()
            instantiation_root = instantiation.directory.resolve()
            if (
                descriptor_path != instantiation_root
                and instantiation_root not in descriptor_path.parents
            ):
                raise AssemblyOpticalError(
                    f"optical sensor descriptor escapes its verified instantiation: {qualified_id}"
                )
            raw_digest = "sha256:" + hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
            if raw_digest != binding.descriptor_sha256:
                raise AssemblyOpticalError(f"optical sensor descriptor changed after resolution: {qualified_id}")
            source = OpticalSensor.load(descriptor_path)
            configured = _configured_sensor(source, resolution_px=sensor_resolution_px)
            mount = f"{component_id}.{binding.pose_connector}"
            if mount not in connector_poses:
                raise AssemblyOpticalError(f"pose frame is missing optical connector {mount!r}")
            sensors.append(
                BoundOpticalSensor(
                    component_id,
                    configured,
                    source.artifact_sha256,
                    mount,
                    Pose(_matrix(connector_poses[mount], f"connector pose {mount}")),
                )
            )
    if requested_sensors:
        matched = {item.descriptor.id for item in sensors} | {item.id for item in sensors}
        missing = sorted(requested_sensors - matched)
        if missing:
            raise AssemblyOpticalError(f"requested optical sensors were not found: {missing}")
    if not sensors:
        raise AssemblyOpticalError("resolved assembly has no selected optical sensors")
    external_lights = () if external_manifest is None else external_manifest.lights
    merged_lights = tuple(external_lights) + tuple(lights)
    light_ids = [item.id for item in merged_lights]
    if len(light_ids) != len(set(light_ids)):
        raise AssemblyOpticalError("external and assembly optical light IDs must be unique")
    environment = (
        environment_linear_rgb
        if environment_linear_rgb is not None
        else (
            external_manifest.environment_linear_rgb
            if external_manifest is not None
            else (0.05, 0.05, 0.05)
        )
    )
    runtime = RuntimeScene(
        f"{scene['contraption_id']}:{assembly.assembly_sha256}:external={external_sha256 or 'none'}:sample={sample_index}:time={time_s:.17g}",
        tuple(instances),
        merged_lights,
        environment,
    )
    return AssemblyOpticalFrame(
        str(scene["contraption_id"]), assembly.assembly_sha256, sample_index,
        resolved_time_index, time_s, runtime, tuple(sensors), external_sha256
    )


def _backend(name: str, device: str | None) -> Any:
    if name not in {"auto", "numpy", "torch"}:
        raise AssemblyOpticalError("optical backend must be auto, numpy, or torch")
    if name == "numpy":
        if device not in {None, "cpu"}:
            raise AssemblyOpticalError("the NumPy optical backend supports only CPU")
        return NumpyOpticalBackend()
    if name == "torch":
        return TorchOpticalBackend(device=device or "auto")
    try:
        candidate = TorchOpticalBackend(device=device or "auto")
    except TorchOpticsUnavailable:
        if device not in {None, "cpu"}:
            raise
        return NumpyOpticalBackend()
    return candidate


def _safe_filename(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not result:
        raise AssemblyOpticalError("artifact ID cannot form a safe filename")
    return result


def _capture_frame(
    frame: AssemblyOpticalFrame,
    output: str | Path,
    *,
    backend: str,
    device: str | None,
    seed: int,
) -> AssemblyOpticalCapture:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AssemblyOpticalError("optical capture seed must be a nonnegative integer")
    root = Path(output).resolve()
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise AssemblyOpticalError(
                f"optical capture output directory must be empty: {root}"
            )
    sensor_root = root / "sensors"
    observation_root = root / "observations"
    sensor_root.mkdir(parents=True, exist_ok=True)
    observation_root.mkdir(parents=True, exist_ok=True)
    engine = _backend(backend, device)
    observation_paths: list[Path] = []
    sensor_paths: list[Path] = []
    observations: list[ObservationArtifact] = []
    stems = tuple(_safe_filename(item.id) for item in frame.sensors)
    if len(stems) != len(set(stems)):
        raise AssemblyOpticalError(
            "selected optical sensor IDs collide after filesystem-safe encoding"
        )
    with AsyncOpticalSimulator(observation_root, backend=engine) as simulator:
        for index, (bound, stem) in enumerate(
            zip(frame.sensors, stems, strict=True)
        ):
            sensor_path = sensor_root / f"{stem}.optical.json"
            bound.descriptor.write(sensor_path)
            sensor_paths.append(sensor_path)
            capture_id = f"{stem}-frame-{frame.time_index:08d}"
            observation = simulator.capture(
                frame.scene,
                bound.descriptor,
                bound.pose,
                frame_index=frame.time_index,
                requested_at_s=frame.time_s,
                id=capture_id,
                seed=seed + index,
                assembly_id=frame.assembly_id,
                assembly_sha256=frame.assembly_sha256.removeprefix("sha256:"),
                assembly_mount_connector=bound.mount_connector,
            )
            observations.append(observation)
            observation_paths.append(observation_root / f"{capture_id}.optical-observation.json")
    backend_name = getattr(engine, "name", type(engine).__name__)
    backend_device = str(getattr(engine, "device", "cpu"))
    provisional = AssemblyOpticalCapture(
        frame, backend_name, backend_device, tuple(observations),
        tuple(observation_paths), tuple(sensor_paths), root / "report.json"
    )
    report_path = root / "report.json"
    report_path.write_text(
        json.dumps(provisional.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return provisional


def capture_assembly(
    assembly: ResolvedAssembly,
    output: str | Path,
    *,
    sensor_ids: Sequence[str] = (),
    sensor_resolution_px: tuple[int, int] | None = None,
    backend: str = "auto",
    device: str | None = None,
    seed: int = 0,
    external_scene: OpticalScene | str | Path | None = None,
    lights: Sequence[OpticalLight] = (),
    environment_linear_rgb: tuple[float, float, float] | None = None,
) -> AssemblyOpticalCapture:
    frame = build_assembly_optical_frame(
        assembly,
        sensor_ids=sensor_ids,
        sensor_resolution_px=sensor_resolution_px,
        external_scene=external_scene,
        lights=lights,
        environment_linear_rgb=environment_linear_rgb,
    )
    return _capture_frame(frame, output, backend=backend, device=device, seed=seed)


def capture_result(
    assembly: ResolvedAssembly,
    result: "SimulationResult",
    output: str | Path,
    *,
    sample_index: int = 0,
    time_index: int = -1,
    sensor_ids: Sequence[str] = (),
    sensor_resolution_px: tuple[int, int] | None = None,
    backend: str = "auto",
    device: str | None = None,
    seed: int = 0,
    external_scene: OpticalScene | str | Path | None = None,
    lights: Sequence[OpticalLight] = (),
    environment_linear_rgb: tuple[float, float, float] | None = None,
) -> AssemblyOpticalCapture:
    frame = build_assembly_optical_frame(
        assembly,
        result=result,
        sample_index=sample_index,
        time_index=time_index,
        sensor_ids=sensor_ids,
        sensor_resolution_px=sensor_resolution_px,
        external_scene=external_scene,
        lights=lights,
        environment_linear_rgb=environment_linear_rgb,
    )
    return _capture_frame(frame, output, backend=backend, device=device, seed=seed)


__all__ = [
    "AssemblyOpticalCapture", "AssemblyOpticalError", "AssemblyOpticalFrame",
    "BoundOpticalSensor", "build_assembly_optical_frame", "capture_assembly",
    "capture_result",
]
