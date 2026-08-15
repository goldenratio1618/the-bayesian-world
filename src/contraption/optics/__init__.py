"""Optical simulation, differentiable inference, and Bayesian reconstruction."""

from .assembly import (
    AssemblyOpticalCapture,
    AssemblyOpticalError,
    AssemblyOpticalFrame,
    BoundOpticalSensor,
    build_assembly_optical_frame,
    capture_assembly,
    capture_result,
)
from .inverse import DifferentiableConstraint, InverseProblemError, InverseResult, InverseView, Likelihood, OpticalInverseProblem, ParameterSpec, Prior
from .rays import RayBundle, RayHits, RayTracingError, camera_rays, intersect_triangles
from .reconstruction import ReconstructionError, SparseBayesianReconstruction, VoxelBlock
from .renderer import MeshInstance, NumpyOpticalBackend, OpticalRenderError, RenderProducts, RuntimeMaterial, RuntimeScene
from .schemas import (
    ContentReference,
    ObservationArtifact,
    ObservationOutput,
    OpticalLight,
    OpticalScene,
    OpticalSchemaError,
    OpticalSensor,
    Pose,
    ReconstructionBlockReference,
    ReconstructionState,
    SceneObject,
    SensorNoise,
    SpectralChannel,
    WirePayloadSpec,
)
from .surface import PosteriorSurface, extract_posterior_surface
from .simulation import AsyncOpticalSimulator, OpticalSimulationError, PendingCapture
from .torch_backend import TorchOpticalBackend, TorchOpticsUnavailable
from .wire import WireFrame, WirePayloadError, decode_wire_frame, encode_wire_frame
from .workflow import ReconstructionArtifact, reconstruct_capture, reconstruct_observations

__all__ = [
    "AssemblyOpticalCapture", "AssemblyOpticalError", "AssemblyOpticalFrame",
    "AsyncOpticalSimulator", "BoundOpticalSensor", "ContentReference", "DifferentiableConstraint", "InverseProblemError", "InverseResult",
    "InverseView", "Likelihood", "MeshInstance", "NumpyOpticalBackend", "ObservationArtifact",
    "ObservationOutput", "OpticalInverseProblem", "OpticalLight", "OpticalRenderError",
    "OpticalScene", "OpticalSchemaError", "OpticalSensor", "OpticalSimulationError",
    "ParameterSpec", "PendingCapture", "Pose", "PosteriorSurface", "Prior", "RayBundle", "RayHits",
    "RayTracingError", "ReconstructionArtifact", "ReconstructionBlockReference", "ReconstructionError",
    "ReconstructionState", "RenderProducts", "RuntimeMaterial", "RuntimeScene", "SceneObject",
    "SensorNoise", "SparseBayesianReconstruction", "SpectralChannel", "TorchOpticalBackend",
    "TorchOpticsUnavailable", "VoxelBlock", "WireFrame", "WirePayloadError", "WirePayloadSpec",
    "build_assembly_optical_frame", "camera_rays", "capture_assembly", "capture_result",
    "decode_wire_frame", "encode_wire_frame", "extract_posterior_surface", "intersect_triangles", "reconstruct_capture",
    "reconstruct_observations",
]
