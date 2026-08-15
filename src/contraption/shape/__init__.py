"""Canonical, source-independent physical shape artifacts.

Raw CAD and scan files are immutable evidence. The public classes exported by
this package describe normalized representations shared by mechanics, optics,
reconstruction, and visualization.
"""

from .artifacts import (
    ContentReference,
    DerivedMassProperties,
    OpticalMaterial,
    OpticalMaterialLibrary,
    PhysicalField,
    ShapeArtifact,
    ShapeArtifactError,
    SpectralOpticalSample,
    ShapeUncertainty,
    SourceRepresentation,
    SurfaceRepresentation,
    VolumeRepresentation,
)
from .ingestion import ImportResult, TessellatedShape, import_shape
from .mechanics import MassProperties, mass_properties
from .mesh import TriangleMesh

__all__ = [
    "ContentReference",
    "DerivedMassProperties",
    "ImportResult",
    "MassProperties",
    "OpticalMaterial",
    "OpticalMaterialLibrary",
    "PhysicalField",
    "ShapeArtifact",
    "ShapeArtifactError",
    "ShapeUncertainty",
    "SourceRepresentation",
    "SpectralOpticalSample",
    "SurfaceRepresentation",
    "TessellatedShape",
    "TriangleMesh",
    "VolumeRepresentation",
    "import_shape",
    "mass_properties",
]
