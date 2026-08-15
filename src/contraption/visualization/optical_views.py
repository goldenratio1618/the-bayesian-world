"""Deterministic, evidence-bound optical products for the offline viewer."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import struct
from typing import Any, Mapping
import zlib

import numpy as np


class OpticalViewError(ValueError):
    """Raised when optical evidence is stale, detached, or not displayable."""


def _digest(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OpticalViewError(f"{label} must be a lowercase SHA-256 digest")
    return "sha256:" + value


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _png_rgba(value: np.ndarray) -> bytes:
    rgba = np.ascontiguousarray(value, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise OpticalViewError("derived display raster must have shape [height, width, 4]")
    height, width, _channels = rgba.shape
    rows = b"".join(b"\0" + rgba[row].tobytes(order="C") for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(rows, level=9))
        + _chunk(b"IEND", b"")
    )


def _rgba(rgb: np.ndarray) -> np.ndarray:
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate((np.asarray(rgb, dtype=np.uint8), alpha), axis=2)


def _rgb_display(value: np.ndarray) -> tuple[np.ndarray, str, list[float] | None]:
    if value.ndim != 3 or value.shape[2] != 3:
        raise OpticalViewError("rgb_linear output must have shape [height, width, 3]")
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise OpticalViewError("rgb_linear output must be finite and non-negative")
    linear = np.clip(value, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return _rgba(np.rint(srgb * 255.0)), "linear-rgb-clamped-to-srgb8", None


def _depth_display(
    value: np.ndarray, near_m: float, far_m: float
) -> tuple[np.ndarray, str, list[float]]:
    if value.ndim != 2 or np.any(np.isnan(value)) or np.any(np.isneginf(value)):
        raise OpticalViewError("depth_m output must be a NaN-free [height, width] array")
    finite = np.isfinite(value)
    if np.any(value[finite] < 0):
        raise OpticalViewError("finite depth_m values must be non-negative")
    visible = finite & (value >= near_m) & (value <= far_m)
    intensity = np.zeros(value.shape, dtype=np.float64)
    intensity[visible] = (far_m - value[visible]) / (far_m - near_m)
    gray = np.rint(np.clip(intensity, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _rgba(np.repeat(gray[:, :, None], 3, axis=2)), "depth-near-white-far-black", [near_m, far_m]


def _segmentation_display(value: np.ndarray) -> tuple[np.ndarray, str, None]:
    if value.ndim != 2:
        raise OpticalViewError("segmentation output must have shape [height, width]")
    labels = np.asarray(value, dtype=np.int64)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    foreground = labels >= 0
    unsigned = labels[foreground].astype(np.uint64)
    rgb[..., 0][foreground] = ((unsigned * 37 + 17) % 251 + 4).astype(np.uint8)
    rgb[..., 1][foreground] = ((unsigned * 67 + 53) % 251 + 4).astype(np.uint8)
    rgb[..., 2][foreground] = ((unsigned * 97 + 101) % 251 + 4).astype(np.uint8)
    return _rgba(rgb), "stable-integer-label-colors", None


def _uncertainty_display(value: np.ndarray) -> tuple[np.ndarray, str, list[float]]:
    if value.ndim != 2 or np.any(np.isnan(value)) or np.any(np.isneginf(value)):
        raise OpticalViewError("uncertainty output must be a NaN-free [height, width] array")
    finite = np.isfinite(value)
    if np.any(value[finite] < 0):
        raise OpticalViewError("finite uncertainty values must be non-negative")
    maximum = float(np.max(value[finite])) if np.any(finite) else 0.0
    scale = maximum if maximum > 0 else 1.0
    normalized = np.zeros(value.shape, dtype=np.float64)
    normalized[finite] = np.log1p(value[finite]) / math.log1p(scale)
    normalized = np.clip(normalized, 0.0, 1.0)
    rgb = np.stack(
        (
            np.rint(255.0 * normalized),
            np.rint(255.0 * np.sqrt(normalized)),
            np.rint(255.0 * (1.0 - normalized)),
        ),
        axis=2,
    ).astype(np.uint8)
    rgb[~finite] = (255, 0, 255)
    return _rgba(rgb), "uncertainty-log-blue-yellow-infinite-magenta", [0.0, maximum]


def _matrix_from_pose(value: Mapping[str, Any]) -> tuple[float, ...]:
    translation = tuple(float(item) for item in value["translation_m"])
    w, x, y, z = (float(item) for item in value["rotation_quaternion_wxyz"])
    return (
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), translation[0],
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), translation[1],
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), translation[2],
        0.0, 0.0, 0.0, 1.0,
    )


def _frame_connector_pose(scene: Mapping[str, Any], frame_index: int, connector: str) -> Mapping[str, Any]:
    if "body_pose_frames" in scene:
        frames = scene["body_pose_frames"]["frames"]
        if frame_index >= len(frames):
            raise OpticalViewError("observation frame_index is outside the physical trajectory")
        connector_poses = frames[frame_index]["connector_poses"]
    else:
        if frame_index != 0:
            raise OpticalViewError("a static physical scene admits only observation frame_index 0")
        connector_poses = scene["connector_poses"]
    try:
        return connector_poses[connector]
    except KeyError as exc:
        raise OpticalViewError(
            f"observation mount_connector {connector!r} has no physical pose at its frame"
        ) from exc


def observation_view_from_artifact(
    observation: Any,
    sensor: Any,
    *,
    assembly_sha256: str,
    assembly_id: str,
    scene: Mapping[str, Any],
    expected_scene_sha256: str | None = None,
    viewer_sensor_id: str | None = None,
    assembly_mount_connector: str | None = None,
) -> dict[str, Any]:
    """Derive PNG views while retaining exact optical evidence identities."""

    if getattr(observation, "format", None) != "optical-observation-1":
        raise OpticalViewError("observation must use optical-observation-1")
    if getattr(sensor, "format", None) != "optical-sensor-1":
        raise OpticalViewError("sensor must use optical-sensor-1")
    if observation.sensor_id != sensor.id or observation.sensor_sha256 != sensor.artifact_sha256:
        raise OpticalViewError("observation sensor identity/hash does not match its descriptor")
    expected_assembly = assembly_sha256.removeprefix("sha256:")
    if observation.assembly_id != assembly_id or observation.assembly_sha256 != expected_assembly:
        raise OpticalViewError("observation is not bound to the exact rendered assembly")
    if observation.assembly_frame != "world":
        raise OpticalViewError("observation assembly_frame must be 'world'")
    effective_connector = assembly_mount_connector or sensor.mount_connector
    if not effective_connector or observation.mount_connector != effective_connector:
        raise OpticalViewError("observation mount_connector does not match its sensor descriptor")
    if observation.mount_transform_sha256 != observation.pose.artifact_sha256:
        raise OpticalViewError("observation mount transform hash does not match its exact pose")
    if expected_scene_sha256 is not None and observation.scene_sha256 != expected_scene_sha256.removeprefix("sha256:"):
        raise OpticalViewError("observation optical-scene hash is stale")

    physical_pose = _frame_connector_pose(scene, observation.frame_index, effective_connector)
    expected_matrix = _matrix_from_pose(physical_pose)
    actual_matrix = tuple(float(item) for item in observation.pose.transform_world_from_sensor_row_major)
    if any(abs(expected - actual) > 1e-9 for expected, actual in zip(expected_matrix, actual_matrix)):
        raise OpticalViewError("observation pose differs from the exact physical connector pose")

    arrays = observation.load_arrays()
    outputs = {item.name: item for item in observation.outputs}
    if set(outputs) != set(sensor.outputs) or set(arrays) != set(outputs):
        raise OpticalViewError("observation products do not exactly match the sensor outputs")
    width, height = sensor.resolution_px
    converters = {
        "rgb_linear": lambda value: _rgb_display(value),
        "depth_m": lambda value: _depth_display(value, sensor.near_clip_m, sensor.far_clip_m),
        "segmentation": lambda value: _segmentation_display(value),
        "uncertainty": lambda value: _uncertainty_display(value),
    }
    modes = {
        "rgb_linear": "rgb",
        "depth_m": "depth",
        "segmentation": "segmentation",
        "uncertainty": "uncertainty",
    }
    layers: dict[str, Any] = {}
    observation_digest = _digest(observation.artifact_sha256, "observation.artifact_sha256")
    for name in sensor.outputs:
        output = outputs[name]
        value = np.asarray(arrays[name])
        if str(value.dtype) != output.dtype or tuple(value.shape) != output.shape:
            raise OpticalViewError(f"observation output metadata mismatch for {name!r}")
        expected_shape = (height, width, 3) if name == "rgb_linear" else (height, width)
        if tuple(value.shape) != expected_shape:
            raise OpticalViewError(
                f"observation output {name!r} shape {tuple(value.shape)} does not match "
                f"sensor resolution {expected_shape}"
            )
        rgba, transform, display_range = converters[name](value)
        png = _png_rgba(rgba)
        layers[modes[name]] = {
            "kind": "raster",
            "sha256": "sha256:" + hashlib.sha256(png).hexdigest(),
            "source_observation_sha256": observation_digest,
            "source_output_sha256": _digest(output.content.sha256, f"{name}.content.sha256"),
            "source_output_media_type": output.content.media_type,
            "source_output_dtype": output.dtype,
            "source_output_shape": list(output.shape),
            "display_transform": transform,
            "display_range": display_range,
            "media_type": "image/png",
            "width_px": width,
            "height_px": height,
            "data_base64": base64.b64encode(png).decode("ascii"),
        }
    return {
        "id": str(observation.id),
        "artifact_sha256": observation_digest,
        "frame_index": int(observation.frame_index),
        "sensor": str(viewer_sensor_id or sensor.id),
        "sensor_descriptor_sha256": _digest(sensor.artifact_sha256, "sensor.artifact_sha256"),
        "optical_scene_sha256": _digest(observation.scene_sha256, "observation.scene_sha256"),
        "assembly_sha256": "sha256:" + expected_assembly,
        "assembly_id": assembly_id,
        "assembly_frame": "world",
        "mount_connector": str(effective_connector),
        "mount_transform_sha256": _digest(observation.mount_transform_sha256, "observation.mount_transform_sha256"),
        "pose_world_from_sensor_row_major": list(actual_matrix),
        "layers": layers,
    }


__all__ = ["OpticalViewError", "observation_view_from_artifact"]
