"""Deterministic surfaces extracted from sparse optical posteriors.

The extractor deliberately uses a bounded voxel-boundary method.  It is less
smooth than marching cubes, but has no optional dependency, produces a closed
metric surface for every selected occupancy cell, and is byte deterministic.
The sparse reconstruction remains the canonical mutable posterior; this module
creates an immutable CTMESH-ready analysis/render/ray-tracing projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from contraption.shape import TriangleMesh
from contraption.shape.mesh import MeshError

from .reconstruction import ReconstructionError

if TYPE_CHECKING:
    from .reconstruction import SparseBayesianReconstruction


_FACE_CORNERS: tuple[
    tuple[tuple[int, int, int], tuple[tuple[int, int, int], ...]], ...
] = (
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
)


@dataclass(frozen=True, slots=True)
class PosteriorSurface:
    """An immutable metric surface and its posterior-derived annotations."""

    mesh: TriangleMesh
    vertex_position_standard_deviation_m: np.ndarray
    occupied_voxel_count: int
    boundary_face_count: int
    occupancy_threshold: float
    maximum_abs_tsdf: float
    manifold: bool
    method: str = "occupancy-tsdf-voxel-boundary-v1"

    def __post_init__(self) -> None:
        uncertainty = np.asarray(
            self.vertex_position_standard_deviation_m, dtype=np.float32
        )
        if uncertainty.shape != (len(self.mesh.vertices_m),):
            raise ReconstructionError(
                "surface position uncertainty must contain one value per vertex"
            )
        if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty < 0.0):
            raise ReconstructionError(
                "surface position uncertainty must be finite and nonnegative"
            )
        uncertainty = np.ascontiguousarray(uncertainty)
        uncertainty.setflags(write=False)
        object.__setattr__(
            self, "vertex_position_standard_deviation_m", uncertainty
        )


@dataclass(frozen=True, slots=True)
class _SelectedVoxel:
    color_linear_rgb: tuple[float, float, float]
    position_standard_deviation_m: float


def _validate_options(
    occupancy_threshold: float,
    maximum_abs_tsdf: float,
    maximum_occupied_voxels: int,
    maximum_triangles: int,
) -> None:
    if (
        not math.isfinite(occupancy_threshold)
        or not 0.0 < occupancy_threshold < 1.0
    ):
        raise ReconstructionError("surface occupancy threshold must be in (0, 1)")
    if (
        not math.isfinite(maximum_abs_tsdf)
        or not 0.0 <= maximum_abs_tsdf <= 1.0
    ):
        raise ReconstructionError("maximum absolute normalized TSDF must be in [0, 1]")
    for name, value in (
        ("maximum_occupied_voxels", maximum_occupied_voxels),
        ("maximum_triangles", maximum_triangles),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ReconstructionError(f"{name} must be a positive integer")


def _selected_voxels(
    reconstruction: "SparseBayesianReconstruction",
    *,
    occupancy_threshold: float,
    maximum_abs_tsdf: float,
    maximum_occupied_voxels: int,
) -> dict[tuple[int, int, int], _SelectedVoxel]:
    selected: dict[tuple[int, int, int], _SelectedVoxel] = {}
    quantization_variance = reconstruction.voxel_size_m**2 / 12.0
    for block_index, block in sorted(reconstruction.blocks.items()):
        probability = 1.0 / (1.0 + np.exp(-block.occupancy_log_odds))
        mask = (
            (probability >= occupancy_threshold)
            & (np.abs(block.tsdf_mean) <= maximum_abs_tsdf)
            & (block.tsdf_precision > 0.0)
            & (block.update_count > 0)
        )
        for local_array in np.argwhere(mask):
            local = tuple(int(item) for item in local_array)
            global_index_array = (
                np.asarray(block_index, dtype=np.int64)
                * reconstruction.block_size
                + local_array
            )
            global_index = tuple(int(item) for item in global_index_array)
            color = np.clip(block.color_mean[local], 0.0, 1.0)
            tsdf_variance_m2 = (
                reconstruction.truncation_distance_m**2
                / float(block.tsdf_precision[local])
            )
            selected[global_index] = _SelectedVoxel(
                tuple(float(item) for item in color),
                math.sqrt(tsdf_variance_m2 + quantization_variance),
            )
            if len(selected) > maximum_occupied_voxels:
                raise ReconstructionError(
                    "posterior surface exceeds maximum_occupied_voxels="
                    f"{maximum_occupied_voxels}; increase the explicit bound or "
                    "use a coarser reconstruction"
                )
    if not selected:
        raise ReconstructionError(
            "posterior contains no occupancy/TSDF cells eligible for surface extraction"
        )
    return selected


def _is_closed_vertex_manifold(triangles: np.ndarray, vertex_count: int) -> bool:
    """Check edge incidence and each vertex link without external geometry tools."""

    edge_counts: dict[tuple[int, int], int] = {}
    incident: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for triangle in triangles:
        a, b, c = (int(item) for item in triangle)
        for first, second in ((a, b), (b, c), (c, a)):
            edge = (min(first, second), max(first, second))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        incident[a].append((b, c))
        incident[b].append((c, a))
        incident[c].append((a, b))
    if any(count != 2 for count in edge_counts.values()):
        return False
    for links in incident:
        adjacency: dict[int, set[int]] = {}
        for first, second in links:
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
        if not adjacency or any(len(neighbours) != 2 for neighbours in adjacency.values()):
            return False
        pending = [next(iter(adjacency))]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(adjacency[current] - visited)
        if len(visited) != len(adjacency):
            return False
    return True


def extract_posterior_surface(
    reconstruction: "SparseBayesianReconstruction",
    *,
    occupancy_threshold: float = 0.55,
    maximum_abs_tsdf: float = 0.5,
    maximum_occupied_voxels: int = 250_000,
    maximum_triangles: int = 2_000_000,
) -> PosteriorSurface:
    """Extract a deterministic boundary mesh from the sparse posterior.

    Cells must satisfy both an occupancy probability threshold and normalized
    TSDF support threshold. Each face next to an unselected cell becomes two
    triangles with outward winding. Grid corners are shared, and vertex color
    and uncertainty are deterministic averages of the selected cells touching
    that corner. Coordinates remain in the reconstruction's canonical world
    frame: ``origin_world_m + grid_corner * voxel_size_m``.
    """

    _validate_options(
        occupancy_threshold,
        maximum_abs_tsdf,
        maximum_occupied_voxels,
        maximum_triangles,
    )
    selected = _selected_voxels(
        reconstruction,
        occupancy_threshold=occupancy_threshold,
        maximum_abs_tsdf=maximum_abs_tsdf,
        maximum_occupied_voxels=maximum_occupied_voxels,
    )
    selected_indices = set(selected)
    vertex_indices: dict[tuple[int, int, int], int] = {}
    vertex_cells: list[set[tuple[int, int, int]]] = []
    triangles: list[tuple[int, int, int]] = []
    face_count = 0

    def vertex_index(
        corner: tuple[int, int, int], voxel: tuple[int, int, int]
    ) -> int:
        result = vertex_indices.get(corner)
        if result is None:
            result = len(vertex_indices)
            vertex_indices[corner] = result
            vertex_cells.append(set())
        vertex_cells[result].add(voxel)
        return result

    for voxel in sorted(selected):
        for direction, relative_corners in _FACE_CORNERS:
            neighbour = tuple(
                voxel[axis] + direction[axis] for axis in range(3)
            )
            if neighbour in selected_indices:
                continue
            face_count += 1
            if face_count * 2 > maximum_triangles:
                raise ReconstructionError(
                    "posterior surface exceeds maximum_triangles="
                    f"{maximum_triangles}; increase the explicit bound or "
                    "use a coarser reconstruction"
                )
            corners = tuple(
                tuple(voxel[axis] + relative[axis] for axis in range(3))
                for relative in relative_corners
            )
            a, b, c, d = (vertex_index(corner, voxel) for corner in corners)
            triangles.extend(((a, b, c), (a, c, d)))

    ordered_corners: list[tuple[int, int, int] | None] = [None] * len(vertex_indices)
    for corner, index in vertex_indices.items():
        ordered_corners[index] = corner
    vertices_m = np.asarray(
        [
            reconstruction.origin_world_m
            + np.asarray(corner, dtype=float) * reconstruction.voxel_size_m
            for corner in ordered_corners
            if corner is not None
        ],
        dtype=float,
    )
    colors: list[tuple[float, float, float, float]] = []
    uncertainties: list[float] = []
    for cells in vertex_cells:
        ordered_cells = sorted(cells)
        colors.append(
            tuple(
                sum(selected[cell].color_linear_rgb[channel] for cell in ordered_cells)
                / len(ordered_cells)
                for channel in range(3)
            )
            + (1.0,)
        )
        uncertainties.append(
            sum(
                selected[cell].position_standard_deviation_m
                for cell in ordered_cells
            )
            / len(ordered_cells)
        )
    topology = np.asarray(triangles, dtype=np.uint32)
    mesh_without_normals = TriangleMesh(
        vertices_m,
        topology,
        vertex_rgba_linear=np.asarray(colors, dtype=float),
        face_material=np.zeros(len(topology), dtype=np.uint32),
    )
    try:
        mesh = mesh_without_normals.with_computed_normals()
    except MeshError:
        # Point-touching occupied components can make the area-weighted normal
        # cancel at their shared grid corner. Keep their exact topology and use
        # a deterministic first-face normal only for those singular vertices.
        face_vertices = vertices_m[topology]
        face_normals = np.cross(
            face_vertices[:, 1] - face_vertices[:, 0],
            face_vertices[:, 2] - face_vertices[:, 0],
        )
        normals = np.zeros_like(vertices_m)
        first_normal: dict[int, np.ndarray] = {}
        for face_index, triangle in enumerate(topology):
            for vertex in triangle:
                index = int(vertex)
                normals[index] += face_normals[face_index]
                first_normal.setdefault(index, face_normals[face_index])
        lengths = np.linalg.norm(normals, axis=1)
        for index in np.flatnonzero(lengths <= 1e-18):
            normals[index] = first_normal[int(index)]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        mesh = TriangleMesh(
            vertices_m,
            topology,
            normals,
            np.asarray(colors, dtype=float),
            np.zeros(len(topology), dtype=np.uint32),
        )
    return PosteriorSurface(
        mesh=mesh,
        vertex_position_standard_deviation_m=np.asarray(
            uncertainties, dtype=np.float32
        ),
        occupied_voxel_count=len(selected),
        boundary_face_count=face_count,
        occupancy_threshold=float(occupancy_threshold),
        maximum_abs_tsdf=float(maximum_abs_tsdf),
        manifold=_is_closed_vertex_manifold(topology, len(vertices_m)),
    )


__all__ = ["PosteriorSurface", "extract_posterior_surface"]
