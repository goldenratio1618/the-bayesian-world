"""Optional Torch optical backend with CPU/CUDA and differentiable ray tracing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .renderer import CompiledScene, RuntimeScene, compile_scene
from .schemas import OpticalSensor, Pose


class TorchOpticsUnavailable(RuntimeError):
    """Raised when a requested Torch device/backend is unavailable."""


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise TorchOpticsUnavailable("Torch optics requires the optional 'gpu' package extra") from exc
    return torch


def rotation_vector_matrix(vector: Any) -> Any:
    """Differentiable Rodrigues exponential map for a 3-vector."""
    torch = require_torch()
    if tuple(vector.shape) != (3,):
        raise ValueError("rotation vector must have shape [3]")
    x, y, z = vector.unbind()
    zero = torch.zeros((), device=vector.device, dtype=vector.dtype)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    theta = torch.linalg.vector_norm(vector)
    # torch.sinc(x) = sin(pi*x)/(pi*x), including the stable zero limit.
    a = torch.sinc(theta / math.pi)
    b = 0.5 * torch.sinc(theta / (2.0 * math.pi)) ** 2
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype)
    return identity + a * skew + b * (skew @ skew)


@dataclass(frozen=True, slots=True)
class TorchScene:
    vertices: Any
    triangles: Any
    face_material: Any
    face_segmentation: Any
    face_uncertainty: Any
    material_base_color: Any
    material_roughness: Any
    material_metallic: Any
    material_transmission: Any
    material_refractive_index: Any
    material_emission: Any
    light_kind: Any
    light_position: Any
    light_direction: Any
    light_color: Any
    light_intensity: Any
    environment: Any

    def differentiable_state(self) -> dict[str, Any]:
        """Clone every continuous physical quantity without detaching gradients."""
        return {
            "geometry.vertices": self.vertices.clone(),
            "materials.base_color": self.material_base_color.clone(),
            "materials.roughness": self.material_roughness.clone(),
            "materials.metallic": self.material_metallic.clone(),
            "materials.transmission": self.material_transmission.clone(),
            "materials.refractive_index": self.material_refractive_index.clone(),
            "materials.emission": self.material_emission.clone(),
            "lights.position": self.light_position.clone(),
            "lights.direction": self.light_direction.clone(),
            "lights.color": self.light_color.clone(),
            "lights.intensity": self.light_intensity.clone(),
            "environment.radiance": self.environment.clone(),
        }


@dataclass(frozen=True, slots=True)
class TorchRenderProducts:
    rgb_linear: Any
    depth_m: Any
    segmentation: Any
    uncertainty: Any

    def as_dict(self, outputs: tuple[str, ...] | None = None) -> dict[str, Any]:
        selected = outputs or ("rgb_linear", "depth_m", "segmentation", "uncertainty")
        return {name: getattr(self, name) for name in selected}

    def numpy(self) -> dict[str, np.ndarray]:
        return {name: value.detach().cpu().numpy() for name, value in self.as_dict().items()}


class TorchOpticalBackend:
    """Vectorized differentiable primary-ray backend.

    Nearest-triangle selection is piecewise differentiable: gradients propagate
    through the selected intersection, material, camera, and light equations.
    Visibility changes remain discontinuous, as in conventional differentiable
    ray tracers, and can be handled by multi-start inference or a smooth loss.
    """

    name = "torch-differentiable"
    differentiable = True

    def __init__(
        self,
        *,
        device: str = "auto",
        dtype: str = "float32",
        cast_shadows: bool = False,
        ray_batch_size: int = 65536,
        triangle_batch_size: int = 2048,
    ) -> None:
        torch = require_torch()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise TorchOpticsUnavailable("CUDA was requested but torch.cuda.is_available() is false")
        if dtype not in {"float32", "float64"}:
            raise ValueError("Torch optical dtype must be float32 or float64")
        if ray_batch_size < 1 or triangle_batch_size < 1:
            raise ValueError("ray and triangle batch sizes must be positive")
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.cast_shadows = bool(cast_shadows)
        self.ray_batch_size = ray_batch_size
        self.triangle_batch_size = triangle_batch_size
        self.hardware_accelerated = self.device.type == "cuda"

    def tensor(self, value: Any, *, dtype: Any | None = None) -> Any:
        torch = require_torch()
        return torch.as_tensor(value, device=self.device, dtype=self.dtype if dtype is None else dtype)

    def compile(self, scene: RuntimeScene | CompiledScene) -> TorchScene:
        torch = require_torch()
        compiled = compile_scene(scene) if isinstance(scene, RuntimeScene) else scene
        materials = compiled.materials
        positions: list[tuple[float, float, float]] = []
        directions: list[tuple[float, float, float]] = []
        kinds: list[int] = []
        colors: list[tuple[float, float, float]] = []
        intensities: list[float] = []
        for light in compiled.lights:
            kinds.append(0 if light.kind == "point" else 1)
            positions.append(light.position_m or (0.0, 0.0, 0.0))
            directions.append(light.direction_world or (0.0, 0.0, -1.0))
            colors.append(light.color_linear_rgb)
            intensities.append(light.intensity)
        return TorchScene(
            vertices=self.tensor(compiled.vertices_m),
            triangles=self.tensor(compiled.triangles, dtype=torch.long),
            face_material=self.tensor(compiled.face_material, dtype=torch.long),
            face_segmentation=self.tensor(compiled.face_segmentation, dtype=torch.long),
            face_uncertainty=self.tensor(compiled.face_uncertainty_m),
            material_base_color=self.tensor([item.base_color_linear_rgb for item in materials]),
            material_roughness=self.tensor([item.roughness for item in materials]),
            material_metallic=self.tensor([item.metallic for item in materials]),
            material_transmission=self.tensor([item.transmission for item in materials]),
            material_refractive_index=self.tensor([item.refractive_index for item in materials]),
            material_emission=self.tensor([item.emission_linear_rgb for item in materials]),
            light_kind=self.tensor(kinds, dtype=torch.long),
            light_position=self.tensor(positions).reshape((-1, 3)),
            light_direction=self.tensor(directions).reshape((-1, 3)),
            light_color=self.tensor(colors).reshape((-1, 3)),
            light_intensity=self.tensor(intensities),
            environment=self.tensor(compiled.environment_linear_rgb),
        )

    def _state(self, scene: TorchScene, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
        result = scene.differentiable_state()
        if overrides:
            unknown = set(overrides) - set(result)
            if unknown:
                raise ValueError(f"unknown Torch optical state fields: {sorted(unknown)}")
            result.update(overrides)
        return result

    def _camera_rays(
        self,
        sensor: OpticalSensor,
        pose: Pose,
        *,
        camera_translation_delta: Any | None,
        camera_rotation_vector: Any | None,
        focal_length_px: Any | None,
        principal_point_px: Any | None,
    ) -> tuple[Any, Any]:
        torch = require_torch()
        width, height = sensor.resolution_px
        y, x = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=self.dtype) + 0.5,
            torch.arange(width, device=self.device, dtype=self.dtype) + 0.5,
            indexing="ij",
        )
        focal = self.tensor(sensor.focal_length_px) if focal_length_px is None else focal_length_px
        principal = self.tensor(sensor.principal_point_px) if principal_point_px is None else principal_point_px
        directions = torch.stack(((x - principal[0]) / focal[0], (y - principal[1]) / focal[1], torch.ones_like(x)), dim=-1).reshape((-1, 3))
        directions = torch.nn.functional.normalize(directions, dim=-1)
        matrix = self.tensor(pose.transform_world_from_sensor_row_major).reshape(4, 4)
        rotation = matrix[:3, :3]
        if camera_rotation_vector is not None:
            rotation = rotation_vector_matrix(camera_rotation_vector) @ rotation
        translation = matrix[:3, 3]
        if camera_translation_delta is not None:
            translation = translation + camera_translation_delta
        directions = directions @ rotation.T
        origins = translation.expand_as(directions)
        return origins, directions

    def _intersect_batch(self, origins: Any, directions: Any, vertices: Any, triangles: Any, near: float, far: float) -> tuple[Any, Any, Any, Any]:
        torch = require_torch()
        triangle_vertices = vertices[triangles]
        ray_count = origins.shape[0]
        best_t = torch.full((ray_count,), float("inf"), device=self.device, dtype=self.dtype)
        best_face = torch.full((ray_count,), -1, device=self.device, dtype=torch.long)
        best_u = torch.zeros((ray_count,), device=self.device, dtype=self.dtype)
        best_v = torch.zeros((ray_count,), device=self.device, dtype=self.dtype)
        rows = torch.arange(ray_count, device=self.device)
        epsilon = 1e-8 if self.dtype == torch.float32 else 1e-12
        for start in range(0, triangles.shape[0], self.triangle_batch_size):
            stop = min(triangles.shape[0], start + self.triangle_batch_size)
            current = triangle_vertices[start:stop]
            v0 = current[:, 0]
            edge1 = current[:, 1] - v0
            edge2 = current[:, 2] - v0
            pvec = torch.cross(directions[:, None, :].expand(-1, len(current), -1), edge2[None, :, :].expand(ray_count, -1, -1), dim=-1)
            determinant = torch.einsum("tj,rtj->rt", edge1, pvec)
            valid = torch.abs(determinant) > epsilon
            inverse = torch.where(valid, 1.0 / determinant, torch.zeros_like(determinant))
            tvec = origins[:, None, :] - v0[None, :, :]
            u = torch.einsum("rtj,rtj->rt", tvec, pvec) * inverse
            valid = valid & (u >= 0) & (u <= 1)
            qvec = torch.cross(tvec, edge1[None, :, :].expand(ray_count, -1, -1), dim=-1)
            v = torch.einsum("rj,rtj->rt", directions, qvec) * inverse
            valid = valid & (v >= 0) & ((u + v) <= 1)
            distance = torch.einsum("tj,rtj->rt", edge2, qvec) * inverse
            valid = valid & (distance >= near) & (distance <= far)
            candidate = torch.where(valid, distance, torch.full_like(distance, float("inf")))
            candidate_t, candidate_index = candidate.min(dim=1)
            improve = candidate_t < best_t
            chosen_u = u[rows, candidate_index]
            chosen_v = v[rows, candidate_index]
            best_t = torch.where(improve, candidate_t, best_t)
            best_face = torch.where(improve, candidate_index + start, best_face)
            best_u = torch.where(improve, chosen_u, best_u)
            best_v = torch.where(improve, chosen_v, best_v)
        return best_t, best_face, best_u, best_v

    def _intersect(self, origins: Any, directions: Any, vertices: Any, triangles: Any, near: float, far: float) -> tuple[Any, Any, Any, Any]:
        torch = require_torch()
        pieces = [[], [], [], []]
        for start in range(0, origins.shape[0], self.ray_batch_size):
            stop = min(origins.shape[0], start + self.ray_batch_size)
            current = self._intersect_batch(origins[start:stop], directions[start:stop], vertices, triangles, near, far)
            for destination, value in zip(pieces, current, strict=True):
                destination.append(value)
        return tuple(torch.cat(items, dim=0) for items in pieces)

    def render(
        self,
        scene: RuntimeScene | CompiledScene | TorchScene,
        sensor: OpticalSensor,
        pose: Pose = Pose(),
        *,
        state: Mapping[str, Any] | None = None,
        camera_translation_delta: Any | None = None,
        camera_rotation_vector: Any | None = None,
        focal_length_px: Any | None = None,
        principal_point_px: Any | None = None,
        frame_index: int = 0,
        seed: int | None = None,
        apply_noise: bool = False,
    ) -> TorchRenderProducts:
        torch = require_torch()
        torch_scene = scene if isinstance(scene, TorchScene) else self.compile(scene)
        values = self._state(torch_scene, state)
        vertices = values["geometry.vertices"]
        origins, directions = self._camera_rays(
            sensor, pose,
            camera_translation_delta=camera_translation_delta,
            camera_rotation_vector=camera_rotation_vector,
            focal_length_px=focal_length_px,
            principal_point_px=principal_point_px,
        )
        distance, face, _u, _v = self._intersect(origins, directions, vertices, torch_scene.triangles, sensor.near_clip_m, sensor.far_clip_m)
        hit = face >= 0
        safe_face = torch.clamp(face, min=0)
        selected_triangles = vertices[torch_scene.triangles[safe_face]]
        normal = torch.cross(selected_triangles[:, 1] - selected_triangles[:, 0], selected_triangles[:, 2] - selected_triangles[:, 0], dim=-1)
        normal = torch.nn.functional.normalize(normal, dim=-1)
        view = -directions
        normal = torch.where((normal * view).sum(dim=-1, keepdim=True) < 0, -normal, normal)
        safe_distance = torch.where(hit, distance, torch.zeros_like(distance))
        position = origins + directions * safe_distance[:, None]
        material_index = torch_scene.face_material[safe_face]
        base = values["materials.base_color"][material_index]
        roughness = values["materials.roughness"][material_index]
        metallic = values["materials.metallic"][material_index]
        transmission = values["materials.transmission"][material_index]
        refractive_index = values["materials.refractive_index"][material_index]
        emission = values["materials.emission"][material_index]
        environment = values["environment.radiance"]
        shaded = base * environment + emission
        light_position = values["lights.position"]
        light_direction_state = values["lights.direction"]
        light_color = values["lights.color"]
        light_intensity = values["lights.intensity"]
        for index in range(torch_scene.light_kind.numel()):
            if int(torch_scene.light_kind[index]) == 0:
                light_vector = light_position[index] - position
                squared_distance = torch.clamp((light_vector * light_vector).sum(dim=-1), min=1e-12)
                light_direction = light_vector / torch.sqrt(squared_distance)[:, None]
                attenuation = light_intensity[index] / (4.0 * math.pi * squared_distance)
            else:
                light_direction = -torch.nn.functional.normalize(light_direction_state[index], dim=-1).expand_as(position)
                attenuation = light_intensity[index].expand(position.shape[0])
            visible = torch.ones_like(attenuation)
            if self.cast_shadows:
                shadow_t, shadow_face, _su, _sv = self._intersect(position + normal * 1e-5, light_direction, vertices, torch_scene.triangles, 1e-6, sensor.far_clip_m)
                if int(torch_scene.light_kind[index]) == 0:
                    visible = ((shadow_face < 0) | (shadow_t >= torch.sqrt(squared_distance) - 2e-5)).to(self.dtype)
                else:
                    visible = (shadow_face < 0).to(self.dtype)
            cosine = torch.clamp((normal * light_direction).sum(dim=-1), min=0)
            diffuse = base * (1.0 - metallic[:, None]) * cosine[:, None] / math.pi
            halfway = torch.nn.functional.normalize(light_direction + view, dim=-1)
            shininess = torch.clamp(2.0 / torch.clamp(roughness, min=1e-3) ** 2 - 2.0, min=1.0)
            specular_strength = torch.clamp((normal * halfway).sum(dim=-1), min=0) ** shininess
            dielectric_f0 = ((refractive_index - 1.0) / (refractive_index + 1.0)) ** 2
            f0 = dielectric_f0[:, None] * (1.0 - metallic[:, None]) + base * metallic[:, None]
            shaded = shaded + (diffuse + f0 * specular_strength[:, None]) * light_color[index] * attenuation[:, None] * visible[:, None]
        shaded = shaded * (1.0 - transmission[:, None]) + environment * transmission[:, None]
        radiance = torch.where(hit[:, None], shaded, environment.expand_as(shaded))
        rgb = torch.clamp(radiance * sensor.exposure_duration_s, min=0)
        depth = torch.where(hit, distance, torch.full_like(distance, float("inf")))
        segmentation = torch.where(hit, torch_scene.face_segmentation[safe_face], torch.full_like(face, -1))
        incidence = torch.clamp(torch.abs((normal * view).sum(dim=-1)), min=1e-3)
        uncertainty = torch.sqrt((torch_scene.face_uncertainty[safe_face] / incidence) ** 2 + sensor.noise.depth_noise_std_m**2)
        uncertainty = torch.where(hit, uncertainty, torch.full_like(uncertainty, float("inf")))
        if apply_noise and sensor.noise.model != "none":
            mixed_seed = (sensor.noise.seed ^ (0 if seed is None else int(seed)) ^ (int(frame_index) * 0x9E3779B1)) & 0x7FFFFFFFFFFFFFFF
            generator = torch.Generator(device=self.device)
            generator.manual_seed(mixed_seed)
            sigma_rgb = torch.sqrt(torch.clamp(rgb, min=0) * sensor.noise.shot_noise_scale + sensor.noise.read_noise_std_linear**2)
            rgb = torch.clamp(rgb + torch.randn(rgb.shape, generator=generator, device=self.device, dtype=self.dtype) * sigma_rgb, min=0)
            if sensor.noise.depth_noise_std_m > 0:
                depth_noise = torch.randn(depth.shape, generator=generator, device=self.device, dtype=self.dtype) * sensor.noise.depth_noise_std_m
                depth = torch.where(hit, depth + depth_noise, depth)
            if sensor.noise.depth_quantization_m > 0:
                depth = torch.where(hit, torch.round(depth / sensor.noise.depth_quantization_m) * sensor.noise.depth_quantization_m, depth)
            if sensor.noise.dropout_probability > 0:
                dropout = hit & (torch.rand(depth.shape, generator=generator, device=self.device, dtype=self.dtype) < sensor.noise.dropout_probability)
                depth = torch.where(dropout, torch.full_like(depth, float("inf")), depth)
                segmentation = torch.where(dropout, torch.full_like(segmentation, -1), segmentation)
                uncertainty = torch.where(dropout, torch.full_like(uncertainty, float("inf")), uncertainty)
                rgb = torch.where(dropout[:, None], torch.zeros_like(rgb), rgb)
        width, height = sensor.resolution_px
        return TorchRenderProducts(
            rgb.reshape((height, width, 3)),
            depth.reshape((height, width)),
            segmentation.reshape((height, width)),
            uncertainty.reshape((height, width)),
        )


__all__ = [
    "TorchOpticalBackend", "TorchOpticsUnavailable", "TorchRenderProducts", "TorchScene",
    "require_torch", "rotation_vector_matrix",
]
