"""Mechanical quantities derived directly from canonical closed surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .mesh import TriangleMesh


class MassPropertyError(ValueError):
    """Raised when a surface cannot define finite solid mass properties."""


@dataclass(frozen=True, slots=True)
class MassProperties:
    density_kg_m3: float
    volume_m3: float
    mass_kg: float
    center_of_mass_m: np.ndarray
    inertia_kg_m2: np.ndarray

    def __post_init__(self) -> None:
        density = float(self.density_kg_m3)
        volume = float(self.volume_m3)
        mass = float(self.mass_kg)
        center = np.asarray(self.center_of_mass_m, dtype=float)
        inertia = np.asarray(self.inertia_kg_m2, dtype=float)
        if not all(math.isfinite(value) and value > 0.0 for value in (density, volume, mass)):
            raise MassPropertyError("density, volume, and mass must be finite and positive")
        if center.shape != (3,) or inertia.shape != (3, 3) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(inertia)):
            raise MassPropertyError("mass-property arrays have invalid shape or values")
        if not np.allclose(inertia, inertia.T, atol=1e-12, rtol=1e-10):
            raise MassPropertyError("inertia tensor must be symmetric")
        eigenvalues = np.linalg.eigvalsh((inertia + inertia.T) / 2.0)
        if np.any(eigenvalues < -max(1e-15, float(np.max(np.abs(eigenvalues))) * 1e-10)):
            raise MassPropertyError("inertia tensor is not positive semidefinite")
        object.__setattr__(self, "density_kg_m3", density)
        object.__setattr__(self, "volume_m3", volume)
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "center_of_mass_m", center)
        object.__setattr__(self, "inertia_kg_m2", (inertia + inertia.T) / 2.0)


def mass_properties(mesh: TriangleMesh, density_kg_m3: float) -> MassProperties:
    """Integrate a constant-density solid bounded by an oriented triangle mesh.

    The surface is decomposed into signed tetrahedra with the origin as the
    fourth vertex.  Integrals are exact for the piecewise-linear boundary and
    independent of the origin chosen for decomposition.
    """

    density = float(density_kg_m3)
    if not math.isfinite(density) or density <= 0.0:
        raise MassPropertyError("density_kg_m3 must be finite and positive")
    if not mesh.closed_oriented_manifold:
        raise MassPropertyError(
            "mass properties require a closed, consistently oriented manifold canonical surface"
        )

    faces = mesh.vertices_m[mesh.triangles]
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    signed_volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = float(np.sum(signed_volumes))
    scale = max(float(np.prod(np.maximum(np.asarray(mesh.dimensions_m), 1e-30))), 1e-30)
    if abs(volume) <= scale * 1e-12:
        raise MassPropertyError("canonical surface encloses zero or numerically unstable volume")
    orientation = 1.0 if volume > 0.0 else -1.0
    weights = signed_volumes * orientation
    volume *= orientation

    tetra_centers = (a + b + c) / 4.0
    first_moment = np.einsum("i,ij->j", weights, tetra_centers)
    center = first_moment / volume

    second_moment = np.zeros((3, 3), dtype=float)
    for weight, va, vb, vc in zip(weights, a, b, c, strict=True):
        total = va + vb + vc
        second_moment += (weight / 20.0) * (
            np.outer(total, total) + np.outer(va, va) + np.outer(vb, vb) + np.outer(vc, vc)
        )
    inertia_origin_per_density = np.trace(second_moment) * np.eye(3) - second_moment
    mass = density * volume
    inertia_origin = density * inertia_origin_per_density
    shift = mass * ((center @ center) * np.eye(3) - np.outer(center, center))
    inertia_center = inertia_origin - shift
    return MassProperties(density, volume, mass, center, inertia_center)
