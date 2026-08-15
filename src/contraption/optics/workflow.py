"""Evidence-bound workflows that turn optical observations into 3-D maps."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from contraption.shape import (
    ContentReference as ShapeContentReference,
    OpticalMaterial,
    PhysicalField,
    ShapeArtifact,
    ShapeUncertainty,
    SourceRepresentation,
    SurfaceRepresentation,
    VolumeRepresentation,
)

from .assembly import AssemblyOpticalCapture
from .reconstruction import ReconstructionError, SparseBayesianReconstruction
from .schemas import ObservationArtifact, OpticalSensor, ReconstructionState
from .surface import PosteriorSurface, extract_posterior_surface


@dataclass(frozen=True, slots=True)
class ReconstructionArtifact:
    """A persisted posterior plus its canonical shape-volume reference."""

    state: ReconstructionState
    state_path: Path
    shape_volume: VolumeRepresentation
    shape_volume_path: Path
    shape_surface: SurfaceRepresentation
    shape_surface_path: Path
    surface_uncertainty: PhysicalField
    surface_uncertainty_path: Path
    shape_artifact: ShapeArtifact
    shape_artifact_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "optical-reconstruction-run-1",
            "reconstruction_state": {
                "path": str(self.state_path),
                "artifact_sha256": self.state.artifact_sha256,
                "format": self.state.format,
                "update_count": self.state.update_count,
                "block_count": len(self.state.blocks),
                "observation_sha256": list(self.state.observation_sha256),
            },
            "shape_volume": {
                "path": str(self.shape_volume_path),
                "value": self.shape_volume.to_dict(),
            },
            "shape_surface": {
                "path": str(self.shape_surface_path),
                "value": self.shape_surface.to_dict(),
            },
            "surface_uncertainty": {
                "path": str(self.surface_uncertainty_path),
                "value": self.surface_uncertainty.to_dict(),
            },
            "shape_artifact": {
                "path": str(self.shape_artifact_path),
                "artifact_sha256": self.shape_artifact.artifact_sha256,
                "format": self.shape_artifact.format,
                "source_count": len(self.shape_artifact.sources),
                "surface_ids": [item.id for item in self.shape_artifact.surfaces],
                "volume_ids": [item.id for item in self.shape_artifact.volumes],
            },
        }


def _copy_file(source: Path, target: Path) -> None:
    if target.exists():
        raise ReconstructionError(f"reconstruction evidence would overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _observation_content_path(
    manifest_path: Path,
    observation_output: Any,
) -> Path:
    root = manifest_path.parent.resolve()
    target = (
        root / Path(*PurePosixPath(observation_output.content.uri).parts)
    ).resolve()
    if target != root and root not in target.parents:
        raise ReconstructionError("observation payload escapes its artifact directory")
    return target


def _copy_evidence(
    root: Path,
    sensors_by_hash: Mapping[str, OpticalSensor],
    sensor_manifest_paths: Mapping[str, Path],
    observations: Sequence[ObservationArtifact],
    observation_manifest_paths: Sequence[Path],
) -> tuple[
    tuple[SourceRepresentation, ...],
    dict[str, Any],
]:
    if len(observations) != len(observation_manifest_paths):
        raise ReconstructionError("observation evidence path count is inconsistent")
    sources: list[SourceRepresentation] = []
    sensor_records: list[dict[str, str]] = []
    for sensor_sha256 in sorted(
        {item.sensor_sha256 for item in observations}
    ):
        sensor = sensors_by_hash.get(sensor_sha256)
        source_path = sensor_manifest_paths.get(sensor_sha256)
        if sensor is None or source_path is None:
            raise ReconstructionError(
                f"exact descriptor evidence is absent for sensor {sensor_sha256}"
            )
        source_path = source_path.resolve()
        loaded = OpticalSensor.load(source_path)
        if loaded.artifact_sha256 != sensor_sha256 or loaded != sensor:
            raise ReconstructionError(
                f"sensor descriptor evidence does not match digest {sensor_sha256}"
            )
        target = (
            root
            / "evidence"
            / "sensors"
            / f"{sensor_sha256}.optical-sensor.json"
        )
        _copy_file(source_path, target)
        reference = ShapeContentReference.from_path(
            target,
            relative_to=root,
            media_type="application/vnd.contraption.optical-sensor+json",
        )
        source_id = f"sensor-{sensor_sha256}"
        sources.append(
            SourceRepresentation(
                id=source_id,
                format="optical_sensor",
                content=reference,
                metres_per_source_unit=1.0,
                provenance={
                    "sensor_id": sensor.id,
                    "sensor_sha256": sensor_sha256,
                },
            )
        )
        sensor_records.append(
            {
                "source_id": source_id,
                "sensor_id": sensor.id,
                "sensor_sha256": sensor_sha256,
                "content_uri": reference.uri,
                "content_sha256": reference.sha256,
            }
        )

    observation_records: list[dict[str, Any]] = []
    for index, (observation, source_manifest) in enumerate(
        zip(observations, observation_manifest_paths, strict=True)
    ):
        source_manifest = source_manifest.resolve()
        loaded = ObservationArtifact.load(source_manifest, verify_content=True)
        if loaded.artifact_sha256 != observation.artifact_sha256:
            raise ReconstructionError(
                f"observation evidence changed for {observation.id!r}"
            )
        evidence_root = root / "evidence" / f"observation-{index:04d}"
        target_manifest = evidence_root / source_manifest.name
        _copy_file(source_manifest, target_manifest)
        payload_references: list[ShapeContentReference] = []
        for output in observation.outputs:
            source_payload = _observation_content_path(source_manifest, output)
            target_payload = evidence_root / Path(
                *PurePosixPath(output.content.uri).parts
            )
            _copy_file(source_payload, target_payload)
            reference = ShapeContentReference.from_path(
                target_payload,
                relative_to=root,
                media_type=output.content.media_type,
            )
            if (
                reference.sha256 != output.content.sha256
                or reference.byte_length != output.content.byte_length
            ):
                raise ReconstructionError(
                    f"copied observation payload changed for {observation.id!r}/{output.name}"
                )
            payload_references.append(reference)
        copied = ObservationArtifact.load(target_manifest, verify_content=True)
        if copied.artifact_sha256 != observation.artifact_sha256:
            raise ReconstructionError(
                f"copied observation manifest changed for {observation.id!r}"
            )
        manifest_reference = ShapeContentReference.from_path(
            target_manifest,
            relative_to=root,
            media_type="application/vnd.contraption.optical-observation+json",
        )
        sources.append(
            SourceRepresentation(
                id=f"observation-{index:04d}",
                format="optical_observation",
                content=manifest_reference,
                metres_per_source_unit=1.0,
                provenance={
                    "observation_id": observation.id,
                    "observation_sha256": observation.artifact_sha256,
                    "sensor_id": observation.sensor_id,
                    "sensor_sha256": observation.sensor_sha256,
                    "scene_sha256": observation.scene_sha256,
                    "assembly_id": observation.assembly_id,
                    "assembly_sha256": observation.assembly_sha256,
                    "assembly_frame": observation.assembly_frame,
                    "mount_connector": observation.mount_connector,
                    "mount_transform_sha256": observation.mount_transform_sha256,
                },
            )
        )
        observation_records.append(
            {
                "source_id": f"observation-{index:04d}",
                "observation_id": observation.id,
                "observation_sha256": observation.artifact_sha256,
                "manifest_uri": manifest_reference.uri,
                "manifest_content_sha256": manifest_reference.sha256,
                "payload_uris": [item.uri for item in payload_references],
            }
        )
    return (
        tuple(sources),
        {"sensors": sensor_records, "observations": observation_records},
    )


def _write_surface_uncertainty(path: Path, surface: PosteriorSurface) -> None:
    """Write a stable little-endian NPY field without admitting object data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.lib.format.write_array(
            stream,
            np.asarray(
                surface.vertex_position_standard_deviation_m, dtype="<f4"
            ),
            version=(2, 0),
            allow_pickle=False,
        )


def _surface_record(
    root: Path,
    extracted: PosteriorSurface,
    surface_path: Path,
    uncertainty_path: Path,
) -> tuple[SurfaceRepresentation, PhysicalField, OpticalMaterial]:
    low, high = extracted.mesh.bounds_m
    uncertainty_field = PhysicalField(
        id="surface-position-standard-deviation",
        quantity="surface_position_standard_deviation",
        unit="m",
        representation="per_vertex",
        content=ShapeContentReference.from_path(
            uncertainty_path,
            relative_to=root,
            media_type="application/vnd.numpy.npy",
        ),
    )
    material = OpticalMaterial(
        id="posterior-color",
        model="lambertian",
        base_color_linear_rgba=(1.0, 1.0, 1.0, 1.0),
        roughness=1.0,
        uncertainty=ShapeUncertainty(
            "empirical",
            {
                "quantity": "linear_rgb_mean",
                "representation": "ctmesh_vertex_rgba",
                "source": "Gaussian color posterior in reconstruction volume",
            },
            "reconstruction-color-posterior",
        ),
        provenance={
            "kind": "derived",
            "source": "Bayesian color posterior mean",
        },
    )
    surface = SurfaceRepresentation(
        id="reconstruction-surface",
        kind="ctmesh",
        content=ShapeContentReference.from_path(
            surface_path,
            relative_to=root,
            media_type="application/vnd.contraption.ctmesh",
        ),
        purposes=("render", "ray_trace", "analysis"),
        vertex_count=len(extracted.mesh.vertices_m),
        triangle_count=len(extracted.mesh.triangles),
        bounds_m=tuple(float(item) for item in (*low, *high)),
        watertight=extracted.mesh.watertight,
        manifold=extracted.manifold,
        material_ids=(material.id,),
        uncertainty=ShapeUncertainty(
            "empirical",
            {
                "field_id": uncertainty_field.id,
                "quantity": uncertainty_field.quantity,
                "unit": uncertainty_field.unit,
                "includes_voxel_quantization": True,
            },
            "reconstruction-surface-position",
        ),
    )
    return surface, uncertainty_field, material


def _write_result(
    reconstruction: SparseBayesianReconstruction,
    sensors_by_hash: Mapping[str, OpticalSensor],
    sensor_manifest_paths: Mapping[str, Path],
    observations: Sequence[ObservationArtifact],
    observation_manifest_paths: Sequence[Path],
    output: str | Path,
    *,
    surface_occupancy_threshold: float,
    surface_maximum_abs_tsdf: float,
    surface_maximum_occupied_voxels: int,
    surface_maximum_triangles: int,
) -> ReconstructionArtifact:
    root = Path(output).resolve()
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise ReconstructionError(
                f"reconstruction output directory must be empty: {root}"
            )
    extracted = extract_posterior_surface(
        reconstruction,
        occupancy_threshold=surface_occupancy_threshold,
        maximum_abs_tsdf=surface_maximum_abs_tsdf,
        maximum_occupied_voxels=surface_maximum_occupied_voxels,
        maximum_triangles=surface_maximum_triangles,
    )
    state_path = root / "reconstruction.state.json"
    volume_path = root / "shape-volume.json"
    surface_path = root / "canonical.surface.ctmesh"
    uncertainty_path = root / "surface.position-standard-deviation.npy"
    artifact_path = root / "shape.artifact.json"
    report_path = root / "report.json"
    for target in (
        state_path,
        volume_path,
        surface_path,
        uncertainty_path,
        artifact_path,
        report_path,
    ):
        if target.exists():
            raise ReconstructionError(f"reconstruction output would overwrite {target}")
    state = reconstruction.save(root)
    volume = state.as_shape_volume(manifest_path=state_path)
    extracted.mesh.write(surface_path)
    _write_surface_uncertainty(uncertainty_path, extracted)
    surface, uncertainty_field, material = _surface_record(
        root, extracted, surface_path, uncertainty_path
    )
    sources, evidence = _copy_evidence(
        root,
        sensors_by_hash,
        sensor_manifest_paths,
        observations,
        observation_manifest_paths,
    )
    shape_artifact = ShapeArtifact(
        id=f"{reconstruction.id}.shape",
        version="1.0.0",
        sources=sources,
        surfaces=(surface,),
        volumes=(volume,),
        optical_materials=(material,),
        physical_fields=(uncertainty_field,),
        caches=(),
        provenance={
            "kind": "bayesian_optical_reconstruction",
            "reconstruction_state_artifact_sha256": state.artifact_sha256,
            "observation_sha256": list(state.observation_sha256),
        },
        metadata={
            "evidence": evidence,
            "posterior": "independent Bernoulli occupancy and Gaussian TSDF/color",
            "surface_extraction": {
                "method": extracted.method,
                "frame": "canonical world frame in metres",
                "occupancy_threshold": extracted.occupancy_threshold,
                "maximum_abs_normalized_tsdf": extracted.maximum_abs_tsdf,
                "selected_voxel_count": extracted.occupied_voxel_count,
                "boundary_face_count": extracted.boundary_face_count,
                "vertex_count": len(extracted.mesh.vertices_m),
                "triangle_count": len(extracted.mesh.triangles),
                "known_limit": (
                    "voxel-boundary discretization; no sub-voxel isosurface interpolation"
                ),
            },
        },
    )
    shape_artifact.write(artifact_path)
    volume_path.write_text(
        json.dumps(volume.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    verified_shape = ShapeArtifact.load(artifact_path, verify_content=True)
    ReconstructionState.load(state_path, verify_content=True)
    result = ReconstructionArtifact(
        state,
        state_path,
        volume,
        volume_path,
        surface,
        surface_path,
        uncertainty_field,
        uncertainty_path,
        verified_shape,
        artifact_path,
    )
    report_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _fuse(
    sensors_by_hash: Mapping[str, OpticalSensor],
    sensor_manifest_paths: Mapping[str, Path],
    observations: Sequence[ObservationArtifact],
    observation_manifest_paths: Sequence[Path],
    output: str | Path,
    *,
    id: str,
    voxel_size_m: float,
    block_size: int,
    origin_world_m: tuple[float, float, float],
    truncation_distance_m: float | None,
    pixel_stride: int,
    surface_occupancy_threshold: float,
    surface_maximum_abs_tsdf: float,
    surface_maximum_occupied_voxels: int,
    surface_maximum_triangles: int,
) -> ReconstructionArtifact:
    if not observations:
        raise ReconstructionError("at least one optical observation is required")
    assembly_bindings = {
        (item.assembly_id, item.assembly_sha256, item.assembly_frame)
        for item in observations
        if item.assembly_id is not None
    }
    if len(assembly_bindings) > 1:
        raise ReconstructionError("observations belong to different resolved assemblies or frames")
    if assembly_bindings and any(item.assembly_id is None for item in observations):
        raise ReconstructionError("assembly-bound and unbound observations cannot share one reconstruction")
    if any(item.assembly_frame != "world" for item in observations):
        raise ReconstructionError(
            "Bayesian reconstruction currently requires observation poses in the canonical world frame"
        )
    reconstruction = SparseBayesianReconstruction(
        id,
        voxel_size_m=voxel_size_m,
        block_size=block_size,
        origin_world_m=origin_world_m,
        truncation_distance_m=truncation_distance_m,
    )
    for observation in observations:
        sensor = sensors_by_hash.get(observation.sensor_sha256)
        if sensor is None:
            raise ReconstructionError(
                f"no exact optical sensor descriptor matches observation {observation.id!r}"
            )
        reconstruction.update_observation(sensor, observation, pixel_stride=pixel_stride)
    return _write_result(
        reconstruction,
        sensors_by_hash,
        sensor_manifest_paths,
        observations,
        observation_manifest_paths,
        output,
        surface_occupancy_threshold=surface_occupancy_threshold,
        surface_maximum_abs_tsdf=surface_maximum_abs_tsdf,
        surface_maximum_occupied_voxels=surface_maximum_occupied_voxels,
        surface_maximum_triangles=surface_maximum_triangles,
    )


def reconstruct_observations(
    sensor_manifests: Sequence[str | Path],
    observation_manifests: Sequence[str | Path],
    output: str | Path,
    *,
    id: str = "optical-reconstruction",
    voxel_size_m: float = 0.01,
    block_size: int = 8,
    origin_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    truncation_distance_m: float | None = None,
    pixel_stride: int = 1,
    surface_occupancy_threshold: float = 0.55,
    surface_maximum_abs_tsdf: float = 0.5,
    surface_maximum_occupied_voxels: int = 250_000,
    surface_maximum_triangles: int = 2_000_000,
) -> ReconstructionArtifact:
    """Load, verify, and fuse one or more observation artifacts.

    Sensor lookup is by canonical descriptor SHA-256, never by a mutable file
    name or merely human-readable sensor ID. Observation NPY payload hashes are
    verified by :meth:`ObservationArtifact.load` before any Bayesian update.
    """

    sensors: dict[str, OpticalSensor] = {}
    sensor_paths: dict[str, Path] = {}
    for manifest in sensor_manifests:
        manifest_path = Path(manifest).resolve()
        sensor = OpticalSensor.load(manifest_path)
        existing = sensors.get(sensor.artifact_sha256)
        if existing is not None and existing != sensor:
            raise ReconstructionError("sensor descriptor digest collision")
        sensors[sensor.artifact_sha256] = sensor
        sensor_paths.setdefault(sensor.artifact_sha256, manifest_path)
    if not sensors:
        raise ReconstructionError("at least one optical sensor descriptor is required")
    observation_paths = tuple(Path(manifest).resolve() for manifest in observation_manifests)
    observations = tuple(
        ObservationArtifact.load(manifest, verify_content=True)
        for manifest in observation_paths
    )
    return _fuse(
        sensors,
        sensor_paths,
        observations,
        observation_paths,
        output,
        id=id,
        voxel_size_m=voxel_size_m,
        block_size=block_size,
        origin_world_m=origin_world_m,
        truncation_distance_m=truncation_distance_m,
        pixel_stride=pixel_stride,
        surface_occupancy_threshold=surface_occupancy_threshold,
        surface_maximum_abs_tsdf=surface_maximum_abs_tsdf,
        surface_maximum_occupied_voxels=surface_maximum_occupied_voxels,
        surface_maximum_triangles=surface_maximum_triangles,
    )


def reconstruct_capture(
    capture: AssemblyOpticalCapture,
    output: str | Path,
    *,
    id: str = "optical-reconstruction",
    voxel_size_m: float = 0.01,
    block_size: int = 8,
    origin_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    truncation_distance_m: float | None = None,
    pixel_stride: int = 1,
    surface_occupancy_threshold: float = 0.55,
    surface_maximum_abs_tsdf: float = 0.5,
    surface_maximum_occupied_voxels: int = 250_000,
    surface_maximum_triangles: int = 2_000_000,
) -> ReconstructionArtifact:
    """Fuse a capture result without weakening its assembly/hash bindings."""

    sensors = {
        item.descriptor.artifact_sha256: item.descriptor
        for item in capture.frame.sensors
    }
    sensor_paths = {
        item.descriptor.artifact_sha256: path.resolve()
        for item, path in zip(
            capture.frame.sensors, capture.sensor_descriptor_paths, strict=True
        )
    }
    return _fuse(
        sensors,
        sensor_paths,
        capture.observations,
        tuple(path.resolve() for path in capture.observation_paths),
        output,
        id=id,
        voxel_size_m=voxel_size_m,
        block_size=block_size,
        origin_world_m=origin_world_m,
        truncation_distance_m=truncation_distance_m,
        pixel_stride=pixel_stride,
        surface_occupancy_threshold=surface_occupancy_threshold,
        surface_maximum_abs_tsdf=surface_maximum_abs_tsdf,
        surface_maximum_occupied_voxels=surface_maximum_occupied_voxels,
        surface_maximum_triangles=surface_maximum_triangles,
    )


__all__ = [
    "ReconstructionArtifact",
    "reconstruct_capture",
    "reconstruct_observations",
]
