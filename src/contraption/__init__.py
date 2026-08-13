"""Public entry points for catalog-backed contraption assemblies."""

from .catalog.instantiations import (
    ModelInstantiationSpec,
    PartInstantiation,
    PartInstantiationRegistry,
    StaticPartSpec,
)
from .loading import ContraptionLoadError, load_contraption
from .physics.assembly import AssembledPMDLSystem, AssemblyError, NetworkInvariantError
from .physics.physical import PhysicalAssemblyError, ResolvedPhysicalAssembly, TransformSpec
from .physics.resolved import (
    ResolutionError,
    ResolvedAssembly,
    ResolvedController,
    ResolvedControllerOutputBinding,
    ResolvedExplicitInputBinding,
    ResolvedVerification,
    ResolvedVerificationInputBinding,
    resolve_assembly,
)

__version__ = "0.3.0"

__all__ = [
    "AssembledPMDLSystem",
    "AssemblyError",
    "ContraptionLoadError",
    "ModelInstantiationSpec",
    "NetworkInvariantError",
    "PartInstantiation",
    "PartInstantiationRegistry",
    "PhysicalAssemblyError",
    "ResolutionError",
    "ResolvedAssembly",
    "ResolvedController",
    "ResolvedControllerOutputBinding",
    "ResolvedExplicitInputBinding",
    "ResolvedPhysicalAssembly",
    "ResolvedVerification",
    "ResolvedVerificationInputBinding",
    "StaticPartSpec",
    "TransformSpec",
    "load_contraption",
    "resolve_assembly",
]
