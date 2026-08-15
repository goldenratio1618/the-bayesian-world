"""Backend-neutral runtime scene records and the exact NumPy optical backend."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

from contraption.shape import OpticalMaterial as ShapeOpticalMaterial
from contraption.shape import ShapeArtifact, TriangleMesh

from .rays import RayBundle, camera_rays, intersect_triangles, transform_vertices
from .schemas import OpticalLight, OpticalScene, OpticalSensor, Pose


class OpticalRenderError(ValueError):
    """Raised when a runtime scene cannot be rendered without ambiguity."""


@dataclass(frozen=True, slots=True)
class RuntimeMaterial:
    id: str
    base_color_linear_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    roughness: float = 0.5
    metallic: float = 0.0
    transmission: float = 0.0
    refractive_index: float = 1.5
    emission_linear_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_shape(cls, material: ShapeOpticalMaterial) -> "RuntimeMaterial":
        return cls(
            id=material.id,
            base_color_linear_rgb=tuple(material.base_color_linear_rgba[:3]),
            roughness=material.roughness,
            metallic=material.metallic,
            transmission=material.transmission,
            refractive_index=material.refractive_index,
            emission_linear_rgb=tuple(material.emission_linear_rgb),
        )

    def __post_init__(self) -> None:
        if not self.id:
            raise OpticalRenderError("runtime material requires an ID")
        for name in ("base_color_linear_rgb", "emission_linear_rgb"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise OpticalRenderError(f"material {name} must contain three nonnegative values")
            object.__setattr__(self, name, tuple(float(item) for item in value))
        for name in ("roughness", "metallic", "transmission"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise OpticalRenderError(f"material {name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        refractive_index = float(self.refractive_index)
        if not math.isfinite(refractive_index) or refractive_index < 1:
            raise OpticalRenderError("material refractive index must be at least 1")
        object.__setattr__(self, "refractive_index", refractive_index)


@dataclass(frozen=True, slots=True)
class MeshInstance:
    id: str
    mesh: TriangleMesh
    segmentation_id: int
    materials: tuple[RuntimeMaterial, ...] = (RuntimeMaterial("default"),)
    transform_world_from_object_row_major: tuple[float, ...] = tuple(np.eye(4).reshape(-1))
    surface_uncertainty_m: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.id
            or isinstance(self.segmentation_id, bool)
            or not isinstance(self.segmentation_id, int)
            or self.segmentation_id < 1
        ):
            raise OpticalRenderError("mesh instances need a nonempty ID and positive segmentation ID")
        matrix = np.asarray(
            self.transform_world_from_object_row_major, dtype=float
        )
        if matrix.size != 16 or not np.all(np.isfinite(matrix)):
            raise OpticalRenderError("mesh instance transform must be a finite 4x4 matrix")
        matrix = matrix.reshape(4, 4)
        tolerance = 1e-9
        if not np.allclose(
            matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=tolerance
        ):
            raise OpticalRenderError(
                "mesh instance transform must have homogeneous final row [0, 0, 0, 1]"
            )
        rotation = matrix[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=tolerance
        ):
            raise OpticalRenderError(
                "mesh instance transform rotation must be orthonormal (no scale or shear)"
            )
        if not math.isclose(
            float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=tolerance
        ):
            raise OpticalRenderError(
                "mesh instance transform must be a right-handed proper rotation"
            )
        object.__setattr__(
            self,
            "transform_world_from_object_row_major",
            tuple(float(item) for item in matrix.reshape(-1)),
        )
        if not self.materials or len({item.id for item in self.materials}) != len(self.materials):
            raise OpticalRenderError("mesh instance materials must be nonempty with unique IDs")
        if self.mesh.face_material is not None and int(self.mesh.face_material.max(initial=0)) >= len(self.materials):
            raise OpticalRenderError("mesh face_material index exceeds the material table")
        surface_uncertainty = float(self.surface_uncertainty_m)
        if not math.isfinite(surface_uncertainty) or surface_uncertainty < 0:
            raise OpticalRenderError("surface uncertainty must be finite and nonnegative")
        object.__setattr__(self, "surface_uncertainty_m", surface_uncertainty)


@dataclass(frozen=True, slots=True)
class RuntimeScene:
    id: str
    instances: tuple[MeshInstance, ...]
    lights: tuple[OpticalLight, ...] = ()
    environment_linear_rgb: tuple[float, float, float] = (0.02, 0.02, 0.02)

    def __post_init__(self) -> None:
        if not self.id or not self.instances:
            raise OpticalRenderError("runtime scene needs an ID and at least one mesh instance")
        if len({item.id for item in self.instances}) != len(self.instances):
            raise OpticalRenderError("runtime scene instance IDs must be unique")
        if len({item.segmentation_id for item in self.instances}) != len(self.instances):
            raise OpticalRenderError("runtime scene segmentation IDs must be unique")
        environment = np.asarray(self.environment_linear_rgb, dtype=float)
        if environment.shape != (3,) or np.any(environment < 0) or not np.all(np.isfinite(environment)):
            raise OpticalRenderError("runtime scene environment must contain three nonnegative values")
        object.__setattr__(
            self,
            "environment_linear_rgb",
            tuple(float(item) for item in environment),
        )

    @classmethod
    def from_manifest(cls, scene: OpticalScene | str | Path) -> "RuntimeScene":
        manifest = OpticalScene.load(scene) if isinstance(scene, (str, Path)) else scene
        instances: list[MeshInstance] = []
        for item in manifest.objects:
            shape_path = manifest.resolve_shape(item)
            shape = ShapeArtifact.load(shape_path)
            surface = next((candidate for candidate in shape.surfaces if candidate.id == item.surface_id), None)
            if surface is None:
                if item.surface_id is not None:
                    raise OpticalRenderError(
                        f"shape {shape.id!r} has no explicitly selected surface "
                        f"{item.surface_id!r}"
                    )
                surface = shape.surface_for("ray_trace")
            elif "ray_trace" not in surface.purposes:
                raise OpticalRenderError(
                    f"shape {shape.id!r} surface {surface.id!r} is not authored "
                    "for ray tracing"
                )
            mesh = shape.load_surface(surface.id)
            by_id = {material.id: RuntimeMaterial.from_shape(material) for material in shape.optical_materials}
            missing_materials = [
                material_id
                for material_id in surface.material_ids
                if material_id not in by_id
            ]
            if missing_materials:
                raise OpticalRenderError(
                    f"shape {shape.id!r} surface {surface.id!r} references missing "
                    f"optical materials {missing_materials}"
                )
            if not surface.material_ids:
                detail = (
                    " and its CTMESH contains face_material indices"
                    if mesh.face_material is not None
                    else ""
                )
                raise OpticalRenderError(
                    f"shape {shape.id!r} surface {surface.id!r} has no authored "
                    f"optical material table{detail}; refusing an invented default"
                )
            materials = tuple(by_id[material_id] for material_id in surface.material_ids)
            instances.append(
                MeshInstance(
                    item.id,
                    mesh,
                    item.segmentation_id,
                    materials,
                    item.transform_world_from_object_row_major,
                    item.surface_uncertainty_m,
                )
            )
        return cls(manifest.id, tuple(instances), manifest.lights, manifest.environment_linear_rgb)

    @property
    def artifact_sha256(self) -> str:
        """Canonical digest of every scene field consumed by optical backends.

        Text is UTF-8 with an unsigned 64-bit byte length. Numeric scalars and
        array elements use explicit little-endian encodings. Collection counts
        and optional-value markers make the stream unambiguous without relying
        on Python ``repr`` or JSON number formatting.
        """

        digest = hashlib.sha256()

        def write_bytes(value: bytes) -> None:
            digest.update(struct.pack("<Q", len(value)))
            digest.update(value)

        def write_text(value: str) -> None:
            write_bytes(value.encode("utf-8"))

        def write_count(value: int) -> None:
            digest.update(struct.pack("<Q", value))

        def write_integer(value: int) -> None:
            digest.update(struct.pack("<q", value))

        def write_float(value: float) -> None:
            digest.update(struct.pack("<d", float(value)))

        def write_array(value: Any) -> None:
            array = np.asarray(value)
            little_dtype = array.dtype.newbyteorder("<")
            array = np.ascontiguousarray(array.astype(little_dtype, copy=False))
            write_text(array.dtype.str)
            write_count(array.ndim)
            for dimension in array.shape:
                write_count(int(dimension))
            write_bytes(array.tobytes())

        def write_float_array(value: Any) -> None:
            write_array(np.asarray(value, dtype="<f8"))

        def write_optional_array(value: Any | None) -> None:
            digest.update(struct.pack("<?", value is not None))
            if value is not None:
                write_array(value)

        def write_optional_float_array(value: Any | None) -> None:
            digest.update(struct.pack("<?", value is not None))
            if value is not None:
                write_float_array(value)

        write_text("contraption.runtime-optical-scene/v1")
        write_text(self.id)
        write_count(len(self.instances))
        for instance in self.instances:
            write_text(instance.id)
            write_array(instance.mesh.vertices_m)
            write_array(instance.mesh.triangles)
            write_optional_array(instance.mesh.vertex_normals)
            write_optional_array(instance.mesh.vertex_rgba_linear)
            write_optional_array(instance.mesh.face_material)
            write_integer(instance.segmentation_id)
            write_count(len(instance.materials))
            for material in instance.materials:
                write_text(material.id)
                write_float_array(material.base_color_linear_rgb)
                write_float(material.roughness)
                write_float(material.metallic)
                write_float(material.transmission)
                write_float(material.refractive_index)
                write_float_array(material.emission_linear_rgb)
            write_float_array(instance.transform_world_from_object_row_major)
            write_float(instance.surface_uncertainty_m)
        write_count(len(self.lights))
        for light in self.lights:
            write_text(light.id)
            write_text(light.kind)
            write_float_array(light.color_linear_rgb)
            write_float(light.intensity)
            write_optional_float_array(light.position_m)
            write_optional_float_array(light.direction_world)
        write_float_array(self.environment_linear_rgb)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledScene:
    vertices_m: np.ndarray
    triangles: np.ndarray
    face_material: np.ndarray
    face_segmentation: np.ndarray
    face_uncertainty_m: np.ndarray
    materials: tuple[RuntimeMaterial, ...]
    lights: tuple[OpticalLight, ...]
    environment_linear_rgb: np.ndarray


def compile_scene(scene: RuntimeScene) -> CompiledScene:
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    face_material: list[np.ndarray] = []
    face_segmentation: list[np.ndarray] = []
    face_uncertainty: list[np.ndarray] = []
    materials: list[RuntimeMaterial] = []
    vertex_offset = 0
    material_offset = 0
    for instance in scene.instances:
        current_vertices = transform_vertices(instance.mesh.vertices_m, instance.transform_world_from_object_row_major)
        vertices.append(current_vertices)
        triangles.append(instance.mesh.triangles.astype(np.int64) + vertex_offset)
        local_material = instance.mesh.face_material
        if local_material is None:
            local_material = np.zeros(len(instance.mesh.triangles), dtype=np.int64)
        face_material.append(np.asarray(local_material, dtype=np.int64) + material_offset)
        face_segmentation.append(np.full(len(instance.mesh.triangles), instance.segmentation_id, dtype=np.int32))
        face_uncertainty.append(np.full(len(instance.mesh.triangles), instance.surface_uncertainty_m, dtype=np.float64))
        materials.extend(instance.materials)
        vertex_offset += len(current_vertices)
        material_offset += len(instance.materials)
    return CompiledScene(
        np.concatenate(vertices), np.concatenate(triangles), np.concatenate(face_material),
        np.concatenate(face_segmentation), np.concatenate(face_uncertainty), tuple(materials),
        scene.lights, np.asarray(scene.environment_linear_rgb, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class RenderProducts:
    rgb_linear: np.ndarray
    depth_m: np.ndarray
    segmentation: np.ndarray
    uncertainty: np.ndarray

    def as_dict(self, outputs: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        selected = outputs or ("rgb_linear", "depth_m", "segmentation", "uncertainty")
        return {name: getattr(self, name) for name in selected}


class NumpyOpticalBackend:
    """Exact primary/shadow ray CPU backend with deterministic sensor physics."""

    name = "numpy-exact"
    hardware_accelerated = False
    differentiable = False

    def __init__(self, *, cast_shadows: bool = True, ray_batch_size: int = 4096, triangle_batch_size: int = 2048) -> None:
        self.cast_shadows = bool(cast_shadows)
        self.ray_batch_size = ray_batch_size
        self.triangle_batch_size = triangle_batch_size

    def render(
        self,
        scene: RuntimeScene | CompiledScene,
        sensor: OpticalSensor,
        pose: Pose = Pose(),
        *,
        frame_index: int = 0,
        seed: int | None = None,
        apply_noise: bool = True,
    ) -> RenderProducts:
        compiled = compile_scene(scene) if isinstance(scene, RuntimeScene) else scene
        rays = camera_rays(sensor, pose)
        hits = intersect_triangles(
            rays,
            compiled.vertices_m,
            compiled.triangles,
            near_m=sensor.near_clip_m,
            far_m=sensor.far_clip_m,
            ray_batch_size=self.ray_batch_size,
            triangle_batch_size=self.triangle_batch_size,
        )
        count = len(rays.origins_m)
        hit = hits.hit
        radiance = np.broadcast_to(compiled.environment_linear_rgb, (count, 3)).copy()
        segmentation = np.full(count, -1, dtype=np.int32)
        uncertainty = np.full(count, np.inf, dtype=np.float64)
        if np.any(hit):
            selected_face = hits.triangle_index[hit]
            material_index = compiled.face_material[selected_face]
            base = np.asarray([compiled.materials[index].base_color_linear_rgb for index in material_index])
            emission = np.asarray([compiled.materials[index].emission_linear_rgb for index in material_index])
            roughness = np.asarray([compiled.materials[index].roughness for index in material_index])
            metallic = np.asarray([compiled.materials[index].metallic for index in material_index])
            transmission = np.asarray([compiled.materials[index].transmission for index in material_index])
            refractive_index = np.asarray(
                [compiled.materials[index].refractive_index for index in material_index]
            )
            position = hits.position_world_m[hit]
            normal = hits.geometric_normal_world[hit].copy()
            view = -rays.directions_world[hit]
            flip = np.einsum("ij,ij->i", normal, view) < 0
            normal[flip] *= -1
            shaded = base * compiled.environment_linear_rgb[None, :] + emission
            for light in compiled.lights:
                if light.kind == "point":
                    light_vector = np.asarray(light.position_m)[None, :] - position
                    squared_distance = np.maximum(np.einsum("ij,ij->i", light_vector, light_vector), 1e-12)
                    light_direction = light_vector / np.sqrt(squared_distance)[:, None]
                    attenuation = light.intensity / (4.0 * math.pi * squared_distance)
                    shadow_far = np.sqrt(squared_distance) - 2e-6
                else:
                    light_direction = np.broadcast_to(-np.asarray(light.direction_world), position.shape).copy()
                    attenuation = np.full(len(position), light.intensity)
                    shadow_far = np.full(len(position), sensor.far_clip_m)
                visible = np.ones(len(position), dtype=bool)
                if self.cast_shadows:
                    shadow_rays = RayBundle(position + normal * 1e-6, light_direction)
                    shadow_hits = intersect_triangles(
                        shadow_rays, compiled.vertices_m, compiled.triangles,
                        near_m=1e-7, far_m=sensor.far_clip_m,
                        ray_batch_size=self.ray_batch_size, triangle_batch_size=self.triangle_batch_size,
                    )
                    visible = (~shadow_hits.hit) | (shadow_hits.distance_m >= shadow_far)
                cosine = np.maximum(np.einsum("ij,ij->i", normal, light_direction), 0.0)
                light_rgb = np.asarray(light.color_linear_rgb)[None, :]
                diffuse = base * (1.0 - metallic[:, None]) * cosine[:, None] / math.pi
                halfway = light_direction + view
                halfway /= np.maximum(np.linalg.norm(halfway, axis=1, keepdims=True), 1e-12)
                shininess = np.maximum(2.0 / np.maximum(roughness, 1e-3) ** 2 - 2.0, 1.0)
                specular_strength = np.maximum(np.einsum("ij,ij->i", normal, halfway), 0.0) ** shininess
                dielectric_f0 = (
                    (refractive_index - 1.0) / (refractive_index + 1.0)
                ) ** 2
                f0 = dielectric_f0[:, None] * (1.0 - metallic[:, None]) + base * metallic[:, None]
                contribution = (diffuse + f0 * specular_strength[:, None]) * light_rgb * attenuation[:, None]
                shaded += contribution * visible[:, None]
            # A one-surface energy split keeps transmissive materials distinct
            # while secondary refraction remains the responsibility of a path backend.
            shaded = shaded * (1.0 - transmission[:, None]) + compiled.environment_linear_rgb * transmission[:, None]
            radiance[hit] = shaded
            segmentation[hit] = compiled.face_segmentation[selected_face]
            incidence = np.maximum(np.abs(np.einsum("ij,ij->i", normal, view)), 1e-3)
            surface_sigma = compiled.face_uncertainty_m[selected_face] / incidence
            uncertainty[hit] = np.sqrt(surface_sigma**2 + sensor.noise.depth_noise_std_m**2)

        rgb = np.maximum(radiance * sensor.exposure_duration_s, 0.0)
        depth = hits.distance_m.copy()
        if apply_noise and sensor.noise.model != "none":
            mixed_seed = (sensor.noise.seed ^ (0 if seed is None else int(seed)) ^ (int(frame_index) * 0x9E3779B1)) & 0xFFFFFFFFFFFFFFFF
            generator = np.random.default_rng(mixed_seed)
            if sensor.noise.shot_noise_scale > 0 or sensor.noise.read_noise_std_linear > 0:
                sigma = np.sqrt(np.maximum(rgb, 0.0) * sensor.noise.shot_noise_scale + sensor.noise.read_noise_std_linear**2)
                rgb = np.maximum(rgb + generator.normal(0.0, sigma), 0.0)
            if sensor.noise.depth_noise_std_m > 0 and np.any(hit):
                depth[hit] += generator.normal(0.0, sensor.noise.depth_noise_std_m, int(np.sum(hit)))
            if sensor.noise.depth_quantization_m > 0 and np.any(hit):
                quantum = sensor.noise.depth_quantization_m
                depth[hit] = np.round(depth[hit] / quantum) * quantum
            if sensor.noise.dropout_probability > 0:
                dropout = hit & (generator.random(count) < sensor.noise.dropout_probability)
                depth[dropout] = np.inf
                segmentation[dropout] = -1
                uncertainty[dropout] = np.inf
                rgb[dropout] = 0

        height, width = rays.image_shape or (1, count)
        return RenderProducts(
            np.asarray(rgb.reshape((height, width, 3)), dtype=np.float32),
            np.asarray(depth.reshape((height, width)), dtype=np.float32),
            segmentation.reshape((height, width)),
            np.asarray(uncertainty.reshape((height, width)), dtype=np.float32),
        )


__all__ = [
    "CompiledScene", "MeshInstance", "NumpyOpticalBackend", "OpticalRenderError",
    "RenderProducts", "RuntimeMaterial", "RuntimeScene", "compile_scene",
]
