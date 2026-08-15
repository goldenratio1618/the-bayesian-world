"""Deterministic pinhole rays and exact triangle-surface intersections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .schemas import OpticalSensor, Pose


class RayTracingError(ValueError):
    """Raised for invalid ray batches or surface topology."""


def _vectors(value: Any, context: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise RayTracingError(f"{context} must have finite shape [N, 3]")
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class RayBundle:
    origins_m: np.ndarray
    directions_world: np.ndarray
    image_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        origins = _vectors(self.origins_m, "ray origins")
        directions = _vectors(self.directions_world, "ray directions")
        if origins.shape != directions.shape:
            raise RayTracingError("ray origin/direction arrays must have equal shape")
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths <= 1e-15):
            raise RayTracingError("ray directions must be nonzero")
        directions = directions / lengths[:, None]
        if self.image_shape is not None:
            if len(self.image_shape) != 2 or math.prod(self.image_shape) != len(origins):
                raise RayTracingError("ray image shape does not match the ray count")
        object.__setattr__(self, "origins_m", origins)
        object.__setattr__(self, "directions_world", directions)


@dataclass(frozen=True, slots=True)
class RayHits:
    distance_m: np.ndarray
    triangle_index: np.ndarray
    barycentric: np.ndarray
    position_world_m: np.ndarray
    geometric_normal_world: np.ndarray

    @property
    def hit(self) -> np.ndarray:
        return self.triangle_index >= 0


def camera_rays(
    sensor: OpticalSensor,
    pose: Pose = Pose(),
    *,
    pixel_offsets_xy: np.ndarray | None = None,
) -> RayBundle:
    """Generate one +Z-forward pinhole ray through every pixel center."""

    width, height = sensor.resolution_px
    if pixel_offsets_xy is None:
        offsets = np.full((height, width, 2), 0.5, dtype=np.float64)
    else:
        offsets = np.asarray(pixel_offsets_xy, dtype=np.float64)
        if offsets.shape != (height, width, 2) or not np.all(np.isfinite(offsets)):
            raise RayTracingError("pixel offsets must have shape [height, width, 2]")
    grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    fx, fy = sensor.focal_length_px
    cx, cy = sensor.principal_point_px
    sensor_directions = np.stack(
        (
            (grid_x + offsets[..., 0] - cx) / fx,
            (grid_y + offsets[..., 1] - cy) / fy,
            np.ones((height, width), dtype=np.float64),
        ),
        axis=-1,
    ).reshape((-1, 3))
    sensor_directions /= np.linalg.norm(sensor_directions, axis=1, keepdims=True)
    transform = np.asarray(pose.transform_world_from_sensor_row_major, dtype=np.float64).reshape((4, 4))
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    directions = sensor_directions @ rotation.T
    origins = np.broadcast_to(translation, directions.shape).copy()
    return RayBundle(origins, directions, (height, width))


def transform_vertices(vertices_m: Any, transform_row_major: Any) -> np.ndarray:
    vertices = _vectors(vertices_m, "vertices")
    matrix = np.asarray(transform_row_major, dtype=np.float64)
    if matrix.size != 16 or not np.all(np.isfinite(matrix)):
        raise RayTracingError("vertex transform must be a finite 4x4 matrix")
    matrix = matrix.reshape((4, 4))
    homogeneous = np.concatenate((vertices, np.ones((len(vertices), 1))), axis=1)
    transformed = homogeneous @ matrix.T
    if np.any(np.abs(transformed[:, 3]) <= 1e-15):
        raise RayTracingError("vertex transform produced a point at infinity")
    return transformed[:, :3] / transformed[:, 3:4]


def intersect_triangles(
    rays: RayBundle,
    vertices_m: Any,
    triangles: Any,
    *,
    near_m: float = 0.0,
    far_m: float = math.inf,
    ray_batch_size: int = 4096,
    triangle_batch_size: int = 2048,
    determinant_epsilon: float = 1e-12,
) -> RayHits:
    """Return nearest two-sided Möller–Trumbore intersections.

    The calculation examines every triangle. Batching only bounds temporary
    memory and does not change the geometric result or tie-breaking order.
    """

    vertices = _vectors(vertices_m, "vertices")
    faces = np.asarray(triangles)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in {"i", "u"}:
        raise RayTracingError("triangles must have integer shape [M, 3]")
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) < 1 or int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise RayTracingError("triangle topology is empty or out of range")
    if not math.isfinite(float(near_m)) or near_m < 0 or far_m <= near_m:
        raise RayTracingError("ray clipping range must satisfy 0 <= near < far")
    if ray_batch_size < 1 or triangle_batch_size < 1:
        raise RayTracingError("ray/triangle batch sizes must be positive")

    count = len(rays.origins_m)
    best_t = np.full(count, np.inf, dtype=np.float64)
    best_triangle = np.full(count, -1, dtype=np.int32)
    best_u = np.zeros(count, dtype=np.float64)
    best_v = np.zeros(count, dtype=np.float64)
    all_triangles = vertices[faces]

    for ray_start in range(0, count, ray_batch_size):
        ray_stop = min(count, ray_start + ray_batch_size)
        origin = rays.origins_m[ray_start:ray_stop]
        direction = rays.directions_world[ray_start:ray_stop]
        local_best = best_t[ray_start:ray_stop]
        local_triangle = best_triangle[ray_start:ray_stop]
        local_u = best_u[ray_start:ray_stop]
        local_v = best_v[ray_start:ray_stop]
        for triangle_start in range(0, len(faces), triangle_batch_size):
            triangle_stop = min(len(faces), triangle_start + triangle_batch_size)
            current = all_triangles[triangle_start:triangle_stop]
            v0 = current[:, 0]
            edge1 = current[:, 1] - v0
            edge2 = current[:, 2] - v0
            pvec = np.cross(direction[:, None, :], edge2[None, :, :])
            determinant = np.einsum("tj,rtj->rt", edge1, pvec)
            valid = np.abs(determinant) > determinant_epsilon
            inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=valid)
            tvec = origin[:, None, :] - v0[None, :, :]
            u = np.einsum("rtj,rtj->rt", tvec, pvec) * inverse
            valid &= (u >= 0.0) & (u <= 1.0)
            qvec = np.cross(tvec, edge1[None, :, :])
            v = np.einsum("rj,rtj->rt", direction, qvec) * inverse
            valid &= (v >= 0.0) & ((u + v) <= 1.0)
            distance = np.einsum("tj,rtj->rt", edge2, qvec) * inverse
            valid &= (distance >= near_m) & (distance <= far_m)
            candidate = np.where(valid, distance, np.inf)
            candidate_index = np.argmin(candidate, axis=1)
            row = np.arange(len(origin))
            candidate_t = candidate[row, candidate_index]
            improve = candidate_t < local_best
            local_best[improve] = candidate_t[improve]
            local_triangle[improve] = (triangle_start + candidate_index[improve]).astype(np.int32)
            local_u[improve] = u[row[improve], candidate_index[improve]]
            local_v[improve] = v[row[improve], candidate_index[improve]]
        best_t[ray_start:ray_stop] = local_best
        best_triangle[ray_start:ray_stop] = local_triangle
        best_u[ray_start:ray_stop] = local_u
        best_v[ray_start:ray_stop] = local_v

    hit = best_triangle >= 0
    positions = np.full((count, 3), np.nan, dtype=np.float64)
    normals = np.full((count, 3), np.nan, dtype=np.float64)
    positions[hit] = rays.origins_m[hit] + rays.directions_world[hit] * best_t[hit, None]
    if np.any(hit):
        selected = all_triangles[best_triangle[hit]]
        selected_normals = np.cross(selected[:, 1] - selected[:, 0], selected[:, 2] - selected[:, 0])
        selected_normals /= np.linalg.norm(selected_normals, axis=1, keepdims=True)
        normals[hit] = selected_normals
    barycentric = np.stack((1.0 - best_u - best_v, best_u, best_v), axis=1)
    barycentric[~hit] = np.nan
    best_t[~hit] = np.inf
    return RayHits(best_t, best_triangle, barycentric, positions, normals)


__all__ = ["RayBundle", "RayHits", "RayTracingError", "camera_rays", "intersect_triangles", "transform_vertices"]
