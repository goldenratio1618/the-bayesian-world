"""Scanner-specific scene reconstruction for the generic viewer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..applications.scanner import ScannerRuntimeError
from ..physics.resolved import ResolvedAssembly
from ..physics.simulator import SimulationResult


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def scanner_physical_scene(
    assembly: ResolvedAssembly,
    result: SimulationResult,
    *,
    sample_index: int | None = None,
) -> dict[str, Any]:
    """Reconstruct a viewer scene from an exact simulated sample."""

    if result.metadata.get("assembly_sha256") != assembly.assembly_sha256:
        raise ScannerRuntimeError("simulation result was produced from a different assembly")
    if sample_index is None:
        sample_index = result.metadata.get("pose_frame_sample_index")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise ScannerRuntimeError(
            "pose_frame_sample_index must be an integer identifying an actual sample"
        )
    frames = assembly.body_pose_frames(result, sample_index=sample_index)
    scene = _plain(assembly.scene)
    scene["body_pose_frames"] = _plain(frames)
    return scene


__all__ = ["scanner_physical_scene"]
