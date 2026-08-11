"""Public entry points for catalog-backed contraption assemblies."""

from .catalog.instantiations import (
    ModelInstantiationSpec,
    PartInstantiation,
    PartInstantiationRegistry,
    StaticPartSpec,
)
from .physics.assembly import AssembledPMDLSystem, AssemblyError, NetworkInvariantError
from .physics.physical import PhysicalAssemblyError, ResolvedPhysicalAssembly, TransformSpec
from .physics.resolved import ResolutionError, ResolvedAssembly, resolve_assembly

__version__ = "0.2.0"

__all__ = [
    "AssembledPMDLSystem",
    "AssemblyError",
    "ModelInstantiationSpec",
    "NetworkInvariantError",
    "PartInstantiation",
    "PartInstantiationRegistry",
    "PhysicalAssemblyError",
    "ResolutionError",
    "ResolvedAssembly",
    "ResolvedPhysicalAssembly",
    "StaticPartSpec",
    "TransformSpec",
    "resolve_assembly",
]
