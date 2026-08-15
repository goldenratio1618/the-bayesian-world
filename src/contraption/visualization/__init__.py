"""Viewer generation and live visualization services."""

from .viewer import (
    VisualizationArtifact,
    VisualizationError,
    generate_viewer,
    validate_physical_scene,
)
from .render_bundle import (
    RENDER_BUNDLE_SCHEMA,
    TRIANGLE_SURFACE_SCHEMA,
    RenderBundleError,
    content_sha256,
    materialize_render_bundle,
    normalize_render_bundle,
    optical_sensors_from_registry,
    sensor_view_from_descriptor,
    shape_artifacts_from_registry,
)
from .optical_views import OpticalViewError, observation_view_from_artifact

__all__ = [
    "VisualizationArtifact",
    "VisualizationError",
    "RENDER_BUNDLE_SCHEMA",
    "TRIANGLE_SURFACE_SCHEMA",
    "RenderBundleError",
    "OpticalViewError",
    "content_sha256",
    "generate_viewer",
    "materialize_render_bundle",
    "normalize_render_bundle",
    "optical_sensors_from_registry",
    "observation_view_from_artifact",
    "sensor_view_from_descriptor",
    "shape_artifacts_from_registry",
    "validate_physical_scene",
]
