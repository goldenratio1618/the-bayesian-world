"""Public entry points for canonical contraption assembly artifacts.

Component packages and PMDL remain inert data until :func:`resolve_assembly`
verifies their exact-hash closure.  Simulation, visualization, build planning,
and C99 compilation consume that returned :class:`ResolvedAssembly`; raw
parallel representations are intentionally not public entry points.
"""

from .assembly import (
    AssembledPMDLSystem,
    AssemblyError,
    NetworkInvariantError,
)
from .build import BuildInstructionError, BuildPlan, generate_build_instructions
from .compiler import compile_resolved_assembly
from .live import LiveRequestError, LiveScannerApplication, serve_live_scanner
from .physical import (
    ComponentPackageRegistry,
    ComponentPackageSpec,
    PhysicalAssemblyError,
    ResolvedPhysicalAssembly,
    TransformSpec,
)
from .resolved import ResolutionError, ResolvedAssembly, resolve_assembly

__version__ = "0.1.0"

__all__ = [
    "AssembledPMDLSystem",
    "AssemblyError",
    "BuildInstructionError",
    "BuildPlan",
    "ComponentPackageRegistry",
    "ComponentPackageSpec",
    "NetworkInvariantError",
    "LiveRequestError",
    "LiveScannerApplication",
    "PhysicalAssemblyError",
    "ResolutionError",
    "ResolvedAssembly",
    "ResolvedPhysicalAssembly",
    "TransformSpec",
    "compile_resolved_assembly",
    "generate_build_instructions",
    "resolve_assembly",
    "serve_live_scanner",
]
