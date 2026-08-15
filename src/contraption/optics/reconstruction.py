"""Sparse Bayesian occupancy and TSDF reconstruction for scanner observations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping
import zlib

import numpy as np

from ..strict_json import loads_strict_json

from .rays import camera_rays
from .schemas import (
    ContentReference,
    ObservationArtifact,
    OpticalSchemaError,
    OpticalSensor,
    Pose,
    ReconstructionBlockReference,
    ReconstructionState,
)


_MAGIC = b"SVOXBLK2\n"
_HEADER_LENGTH = struct.Struct("<I")


class ReconstructionError(ValueError):
    """Raised when probabilistic volume state or an update is invalid."""


@dataclass(slots=True)
class VoxelBlock:
    occupancy_log_odds: np.ndarray
    tsdf_mean: np.ndarray
    tsdf_precision: np.ndarray
    color_mean: np.ndarray
    color_precision: np.ndarray
    update_count: np.ndarray
    index: tuple[int, int, int]

    @classmethod
    def empty(
        cls,
        block_size: int,
        prior_log_odds: float,
        index: tuple[int, int, int],
    ) -> "VoxelBlock":
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or not 2 <= block_size <= 64
        ):
            raise ReconstructionError("voxel block size must be in [2, 64]")
        shape = (block_size, block_size, block_size)
        return cls(
            np.full(shape, prior_log_odds, dtype=np.float32),
            np.ones(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape + (3,), dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype="<u4"),
            index,
        )

    @property
    def block_size(self) -> int:
        return int(self.occupancy_log_odds.shape[0])

    def validate(self) -> None:
        if (
            len(self.index) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in self.index)
        ):
            raise ReconstructionError("voxel block index must contain three integers")
        size = self.block_size
        shape = (size, size, size)
        arrays = {
            "occupancy_log_odds": (self.occupancy_log_odds, shape),
            "tsdf_mean": (self.tsdf_mean, shape),
            "tsdf_precision": (self.tsdf_precision, shape),
            "color_mean": (self.color_mean, shape + (3,)),
            "color_precision": (self.color_precision, shape),
            "update_count": (self.update_count, shape),
        }
        if not 2 <= size <= 64:
            raise ReconstructionError("voxel block size must be in [2, 64]")
        for name, (value, expected) in arrays.items():
            if value.shape != expected:
                raise ReconstructionError(f"voxel block {name} has wrong shape")
            if name != "update_count" and not np.all(np.isfinite(value)):
                raise ReconstructionError(f"voxel block {name} contains NaN or infinity")
        if np.any(self.tsdf_precision < 0) or np.any(self.color_precision < 0):
            raise ReconstructionError("voxel precisions may not be negative")
        if np.any(self.tsdf_mean < -1.0) or np.any(self.tsdf_mean > 1.0):
            raise ReconstructionError("normalized TSDF means must be in [-1, 1]")
        if self.update_count.dtype != np.dtype("<u4"):
            raise ReconstructionError("voxel update counts must use uint32")

    def to_bytes(self) -> bytes:
        self.validate()
        ordered = (
            ("occupancy_log_odds", self.occupancy_log_odds, "<f4"),
            ("tsdf_mean", self.tsdf_mean, "<f4"),
            ("tsdf_precision", self.tsdf_precision, "<f4"),
            ("color_mean", self.color_mean, "<f4"),
            ("color_precision", self.color_precision, "<f4"),
            ("update_count", self.update_count, "<u4"),
        )
        body = bytearray()
        descriptors: list[dict[str, Any]] = []
        for name, value, dtype in ordered:
            raw = np.ascontiguousarray(value, dtype=dtype).tobytes()
            descriptors.append({"name": name, "dtype": dtype, "shape": list(value.shape), "offset": len(body), "byte_length": len(raw)})
            body.extend(raw)
        compressed = zlib.compress(bytes(body), level=6)
        header = {
            "schema": "contraption.sparse-voxel-block/v2",
            "index": list(self.index),
            "block_size": self.block_size,
            "arrays": descriptors,
            "uncompressed_byte_length": len(body),
            "uncompressed_crc32": zlib.crc32(body) & 0xFFFFFFFF,
        }
        header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return _MAGIC + _HEADER_LENGTH.pack(len(header_bytes)) + header_bytes + compressed

    @classmethod
    def from_bytes(cls, payload: bytes) -> "VoxelBlock":
        if not payload.startswith(_MAGIC) or len(payload) < len(_MAGIC) + _HEADER_LENGTH.size:
            raise ReconstructionError("not a sparse voxel block payload")
        start = len(_MAGIC)
        (header_length,) = _HEADER_LENGTH.unpack(payload[start : start + _HEADER_LENGTH.size])
        header_start = start + _HEADER_LENGTH.size
        body_start = header_start + header_length
        if not 0 < header_length <= 65_536 or body_start >= len(payload):
            raise ReconstructionError("sparse voxel block header escapes payload")
        try:
            header = loads_strict_json(payload[header_start:body_start])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReconstructionError(f"invalid sparse voxel block: {exc}") from exc
        required_header = {
            "schema", "index", "block_size", "arrays",
            "uncompressed_byte_length", "uncompressed_crc32",
        }
        if set(header) != required_header or header["schema"] != "contraption.sparse-voxel-block/v2":
            raise ReconstructionError("unsupported sparse voxel block schema")
        block_size = header["block_size"]
        index = header["index"]
        if isinstance(block_size, bool) or not isinstance(block_size, int) or not 2 <= block_size <= 64:
            raise ReconstructionError("invalid sparse voxel block size")
        if not isinstance(index, list) or len(index) != 3 or any(isinstance(item, bool) or not isinstance(item, int) for item in index):
            raise ReconstructionError("invalid sparse voxel block index")
        expected_uncompressed = 32 * block_size**3
        if header["uncompressed_byte_length"] != expected_uncompressed:
            raise ReconstructionError("invalid sparse voxel block byte length")
        try:
            decompressor = zlib.decompressobj()
            body = decompressor.decompress(
                payload[body_start:], expected_uncompressed + 1
            )
        except zlib.error as exc:
            raise ReconstructionError(f"invalid sparse voxel block: {exc}") from exc
        if (
            not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
            or len(body) != expected_uncompressed
        ):
            raise ReconstructionError("invalid sparse voxel block framing")
        if len(body) != header["uncompressed_byte_length"] or zlib.crc32(body) & 0xFFFFFFFF != header["uncompressed_crc32"]:
            raise ReconstructionError("sparse voxel block checksum mismatch")
        arrays: dict[str, np.ndarray] = {}
        allowed = {
            "occupancy_log_odds": "<f4", "tsdf_mean": "<f4", "tsdf_precision": "<f4",
            "color_mean": "<f4", "color_precision": "<f4", "update_count": "<u4",
        }
        expected_names = list(allowed)
        if not isinstance(header["arrays"], list) or [item.get("name") if isinstance(item, dict) else None for item in header["arrays"]] != expected_names:
            raise ReconstructionError("sparse voxel arrays are not in canonical order")
        expected_offset = 0
        for descriptor in header["arrays"]:
            if set(descriptor) != {"name", "dtype", "shape", "offset", "byte_length"}:
                raise ReconstructionError("invalid sparse voxel array descriptor")
            name = descriptor["name"]
            if name not in allowed or descriptor["dtype"] != allowed[name] or name in arrays:
                raise ReconstructionError("unexpected sparse voxel array")
            offset, length = descriptor["offset"], descriptor["byte_length"]
            expected_shape = (
                (block_size, block_size, block_size, 3)
                if name == "color_mean"
                else (block_size, block_size, block_size)
            )
            expected_length = math.prod(expected_shape) * 4
            if offset != expected_offset or length != expected_length or descriptor["shape"] != list(expected_shape):
                raise ReconstructionError("sparse voxel array is not canonically packed")
            dtype = np.dtype(descriptor["dtype"])
            shape = tuple(descriptor["shape"])
            if math.prod(shape) * dtype.itemsize != length:
                raise ReconstructionError("sparse voxel array shape/length mismatch")
            arrays[name] = np.frombuffer(body, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape).copy()
            expected_offset += length
        if set(arrays) != set(allowed):
            raise ReconstructionError("sparse voxel block is missing required arrays")
        if expected_offset != len(body):
            raise ReconstructionError("sparse voxel block has unreferenced bytes")
        result = cls(**arrays, index=tuple(index))
        result.validate()
        if result.block_size != header["block_size"]:
            raise ReconstructionError("sparse voxel block size metadata mismatch")
        return result


class SparseBayesianReconstruction:
    """Mutable sparse map with conjugate Gaussian TSDF/color updates.

    Occupancy uses a bounded log-odds inverse sensor model. TSDF and color use
    independent Gaussian posteriors represented by mean/precision, so multiple
    observations can be fused in any order up to floating-point roundoff.
    """

    def __init__(
        self,
        id: str,
        *,
        voxel_size_m: float = 0.005,
        block_size: int = 8,
        origin_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        truncation_distance_m: float | None = None,
        occupancy_prior_probability: float = 0.5,
        occupied_probability: float = 0.7,
        free_probability: float = 0.35,
        min_log_odds: float = -8.0,
        max_log_odds: float = 8.0,
    ) -> None:
        if not id:
            raise ReconstructionError("reconstruction requires an ID")
        if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise ReconstructionError("voxel_size_m must be finite and positive")
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or not 2 <= block_size <= 64
        ):
            raise ReconstructionError("block_size must be an integer in [2, 64]")
        origin = np.asarray(origin_world_m, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ReconstructionError("origin_world_m must contain three finite values")
        truncation = 4.0 * voxel_size_m if truncation_distance_m is None else float(truncation_distance_m)
        if not math.isfinite(truncation) or truncation < voxel_size_m:
            raise ReconstructionError("truncation distance must be at least one voxel")
        for name, probability in (("prior", occupancy_prior_probability), ("occupied", occupied_probability), ("free", free_probability)):
            if not 0 < probability < 1:
                raise ReconstructionError(f"{name} occupancy probability must be in (0, 1)")
        if occupied_probability <= occupancy_prior_probability or free_probability >= occupancy_prior_probability:
            raise ReconstructionError("inverse sensor probabilities must move occupied/free evidence in opposite directions")
        if (
            not math.isfinite(min_log_odds)
            or not math.isfinite(max_log_odds)
            or min_log_odds >= max_log_odds
        ):
            raise ReconstructionError("minimum log odds must be below maximum")
        self.id = id
        self.voxel_size_m = float(voxel_size_m)
        self.block_size = int(block_size)
        self.origin_world_m = origin
        self.truncation_distance_m = truncation
        self.occupancy_prior_probability = float(occupancy_prior_probability)
        self.occupied_probability = float(occupied_probability)
        self.free_probability = float(free_probability)
        self.prior_log_odds = math.log(occupancy_prior_probability / (1.0 - occupancy_prior_probability))
        self.occupied_increment = math.log(occupied_probability / (1.0 - occupied_probability)) - self.prior_log_odds
        self.free_increment = math.log(free_probability / (1.0 - free_probability)) - self.prior_log_odds
        self.min_log_odds = float(min_log_odds)
        self.max_log_odds = float(max_log_odds)
        self.blocks: dict[tuple[int, int, int], VoxelBlock] = {}
        self.update_count = 0
        self.observation_sha256: list[str] = []

    def _split_index(self, voxel_index: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        block = np.floor_divide(voxel_index, self.block_size).astype(np.int64)
        local = np.mod(voxel_index, self.block_size).astype(np.int64)
        return tuple(int(item) for item in block), tuple(int(item) for item in local)

    def _block(self, index: tuple[int, int, int]) -> VoxelBlock:
        block = self.blocks.get(index)
        if block is None:
            block = VoxelBlock.empty(self.block_size, self.prior_log_odds, index)
            self.blocks[index] = block
        return block

    def _update_voxel(
        self,
        index: np.ndarray,
        *,
        occupancy_increment: float | None,
        tsdf: float,
        tsdf_precision: float,
        color: np.ndarray | None = None,
        color_precision: float = 0.0,
    ) -> None:
        block_index, local = self._split_index(index)
        block = self._block(block_index)
        if occupancy_increment is not None:
            block.occupancy_log_odds[local] = np.clip(block.occupancy_log_odds[local] + occupancy_increment, self.min_log_odds, self.max_log_odds)
        if tsdf_precision > 0:
            old_precision = float(block.tsdf_precision[local])
            new_precision = old_precision + tsdf_precision
            block.tsdf_mean[local] = (float(block.tsdf_mean[local]) * old_precision + tsdf * tsdf_precision) / new_precision
            block.tsdf_precision[local] = new_precision
        if color is not None and color_precision > 0:
            old_precision = float(block.color_precision[local])
            new_precision = old_precision + color_precision
            block.color_mean[local] = (block.color_mean[local] * old_precision + color * color_precision) / new_precision
            block.color_precision[local] = new_precision
        block.update_count[local] += 1

    def update_depth(
        self,
        sensor: OpticalSensor,
        depth_m: Any,
        pose: Pose,
        *,
        uncertainty_m: Any | None = None,
        rgb_linear: Any | None = None,
        pixel_stride: int = 1,
        observation_sha256: str | None = None,
    ) -> None:
        depth = np.asarray(depth_m, dtype=float)
        width, height = sensor.resolution_px
        if depth.shape != (height, width):
            raise ReconstructionError("depth array shape does not match the optical sensor")
        if pixel_stride < 1:
            raise ReconstructionError("pixel_stride must be positive")
        if uncertainty_m is None:
            sigma = np.full_like(depth, max(sensor.noise.depth_noise_std_m, self.voxel_size_m / 2.0))
        else:
            sigma = np.asarray(uncertainty_m, dtype=float)
            if sigma.shape != depth.shape:
                raise ReconstructionError("uncertainty shape must match depth")
        color = None if rgb_linear is None else np.asarray(rgb_linear, dtype=float)
        if color is not None and (color.shape != (height, width, 3) or not np.all(np.isfinite(color))):
            raise ReconstructionError("RGB array must have finite shape [height, width, 3]")
        rays = camera_rays(sensor, pose)
        for y in range(0, height, pixel_stride):
            for x in range(0, width, pixel_stride):
                measured = float(depth[y, x])
                measured_sigma = float(sigma[y, x])
                if not math.isfinite(measured) or not sensor.near_clip_m <= measured <= sensor.far_clip_m:
                    continue
                if not math.isfinite(measured_sigma) or measured_sigma <= 0:
                    continue
                flat = y * width + x
                origin = rays.origins_m[flat]
                direction = rays.directions_world[flat]
                stop = min(measured + self.truncation_distance_m, sensor.far_clip_m)
                distances = np.arange(sensor.near_clip_m, stop + 0.5 * self.voxel_size_m, self.voxel_size_m)
                points = origin[None, :] + direction[None, :] * distances[:, None]
                voxels = np.floor((points - self.origin_world_m) / self.voxel_size_m).astype(np.int64)
                if len(voxels) > 1:
                    keep = np.ones(len(voxels), dtype=bool)
                    keep[1:] = np.any(voxels[1:] != voxels[:-1], axis=1)
                    voxels, distances = voxels[keep], distances[keep]
                normalized_sigma = max(measured_sigma / self.truncation_distance_m, 1e-6)
                precision = min(1.0 / (normalized_sigma * normalized_sigma), 1e8)
                surface_voxel = int(np.argmin(np.abs(distances - measured)))
                for index, (voxel, distance) in enumerate(zip(voxels, distances, strict=True)):
                    signed_distance = float(np.clip((measured - distance) / self.truncation_distance_m, -1.0, 1.0))
                    occupancy_increment: float | None = None
                    if distance < measured - 0.5 * self.voxel_size_m:
                        occupancy_increment = self.free_increment
                    elif abs(distance - measured) <= 0.75 * self.voxel_size_m:
                        occupancy_increment = self.occupied_increment
                    pixel_color = color[y, x] if color is not None and index == surface_voxel else None
                    self._update_voxel(
                        voxel,
                        occupancy_increment=occupancy_increment,
                        tsdf=signed_distance,
                        tsdf_precision=precision,
                        color=pixel_color,
                        color_precision=1.0 / max(1e-4, sensor.noise.read_noise_std_linear**2 + 1e-4),
                    )
        self.update_count += 1
        if observation_sha256 is not None:
            if len(observation_sha256) != 64 or any(character not in "0123456789abcdef" for character in observation_sha256):
                raise ReconstructionError("observation digest must be SHA-256")
            self.observation_sha256.append(observation_sha256)

    def update_observation(self, sensor: OpticalSensor, observation: ObservationArtifact, *, pixel_stride: int = 1) -> None:
        if observation.sensor_id != sensor.id or observation.sensor_sha256 != sensor.artifact_sha256:
            raise ReconstructionError("observation sensor binding does not match the supplied sensor")
        arrays = observation.load_arrays()
        if "depth_m" not in arrays:
            raise ReconstructionError("Bayesian reconstruction requires a depth_m observation product")
        self.update_depth(
            sensor,
            arrays["depth_m"],
            observation.pose,
            uncertainty_m=arrays.get("uncertainty"),
            rgb_linear=arrays.get("rgb_linear"),
            pixel_stride=pixel_stride,
            observation_sha256=observation.artifact_sha256,
        )

    def voxel_posterior(self, voxel_index: tuple[int, int, int]) -> dict[str, Any]:
        index = np.asarray(voxel_index, dtype=np.int64)
        block_index, local = self._split_index(index)
        block = self.blocks.get(block_index)
        if block is None:
            return {"occupancy_probability": self.occupancy_prior_probability, "tsdf_mean": 1.0, "tsdf_standard_deviation_m": math.inf, "color_mean": (0.0, 0.0, 0.0), "update_count": 0}
        log_odds = float(block.occupancy_log_odds[local])
        probability = 1.0 / (1.0 + math.exp(-log_odds))
        precision = float(block.tsdf_precision[local])
        return {
            "occupancy_probability": probability,
            "tsdf_mean": float(block.tsdf_mean[local]),
            "tsdf_standard_deviation_m": math.inf if precision <= 0 else self.truncation_distance_m / math.sqrt(precision),
            "color_mean": tuple(float(item) for item in block.color_mean[local]),
            "update_count": int(block.update_count[local]),
        }

    def surface_points(
        self,
        *,
        occupancy_threshold: float = 0.5,
        maximum_abs_tsdf: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        standard_deviations: list[float] = []
        for block_index, block in sorted(self.blocks.items()):
            probability = 1.0 / (1.0 + np.exp(-block.occupancy_log_odds))
            mask = (probability >= occupancy_threshold) & (np.abs(block.tsdf_mean) <= maximum_abs_tsdf) & (block.tsdf_precision > 0)
            for local in np.argwhere(mask):
                global_index = np.asarray(block_index) * self.block_size + local
                points.append(self.origin_world_m + (global_index + 0.5) * self.voxel_size_m)
                local_tuple = tuple(int(item) for item in local)
                colors.append(block.color_mean[local_tuple])
                standard_deviations.append(self.truncation_distance_m / math.sqrt(float(block.tsdf_precision[local_tuple])))
        if not points:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))
        return np.asarray(points), np.asarray(colors), np.asarray(standard_deviations)

    def expected_information_gain(
        self,
        sensor: OpticalSensor,
        pose: Pose,
        *,
        maximum_distance_m: float | None = None,
        pixel_stride: int = 8,
    ) -> float:
        """Approximate next-best-view score from Bernoulli occupancy entropy."""
        if pixel_stride < 1:
            raise ReconstructionError("pixel_stride must be positive")
        maximum = min(sensor.far_clip_m, maximum_distance_m or sensor.far_clip_m)
        rays = camera_rays(sensor, pose)
        width, height = sensor.resolution_px
        visited: set[tuple[int, int, int]] = set()
        score = 0.0
        for y in range(0, height, pixel_stride):
            for x in range(0, width, pixel_stride):
                flat = y * width + x
                for distance in np.arange(sensor.near_clip_m, maximum, self.block_size * self.voxel_size_m):
                    point = rays.origins_m[flat] + rays.directions_world[flat] * distance
                    voxel = tuple(np.floor((point - self.origin_world_m) / self.voxel_size_m).astype(np.int64))
                    if voxel in visited:
                        continue
                    visited.add(voxel)
                    probability = self.voxel_posterior(voxel)["occupancy_probability"]
                    score += -(probability * math.log(max(probability, 1e-12)) + (1 - probability) * math.log(max(1 - probability, 1e-12)))
        return score

    def rank_candidate_views(self, sensor: OpticalSensor, poses: Iterable[Pose], **kwargs: Any) -> list[tuple[float, Pose]]:
        return sorted(((self.expected_information_gain(sensor, pose, **kwargs), pose) for pose in poses), key=lambda item: item[0], reverse=True)

    def save(self, directory: str | Path) -> ReconstructionState:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        references: list[ReconstructionBlockReference] = []
        for index, block in sorted(self.blocks.items()):
            if block.index != index:
                raise ReconstructionError("voxel block dictionary/index binding mismatch")
            name = "block_" + "_".join(str(item) for item in index) + ".svox"
            target = root / name
            target.write_bytes(block.to_bytes())
            references.append(ReconstructionBlockReference(index, ContentReference.from_path(target, relative_to=root, media_type="application/vnd.contraption.sparse-voxel-block")))
        state = ReconstructionState(
            id=self.id,
            voxel_size_m=self.voxel_size_m,
            block_size=self.block_size,
            origin_world_m=tuple(float(item) for item in self.origin_world_m),
            truncation_distance_m=self.truncation_distance_m,
            occupancy_prior_probability=self.occupancy_prior_probability,
            occupied_probability=self.occupied_probability,
            free_probability=self.free_probability,
            min_log_odds=self.min_log_odds,
            max_log_odds=self.max_log_odds,
            update_count=self.update_count,
            blocks=tuple(references),
            observation_sha256=tuple(self.observation_sha256),
            metadata={"posterior": "independent Bernoulli occupancy and Gaussian TSDF/color"},
        )
        state.write(root / "reconstruction.state.json")
        return state

    @classmethod
    def load(cls, path: str | Path) -> "SparseBayesianReconstruction":
        state = ReconstructionState.load(path)
        result = cls(
            state.id,
            voxel_size_m=state.voxel_size_m,
            block_size=state.block_size,
            origin_world_m=state.origin_world_m,
            truncation_distance_m=state.truncation_distance_m,
            occupancy_prior_probability=state.occupancy_prior_probability,
            occupied_probability=state.occupied_probability,
            free_probability=state.free_probability,
            min_log_odds=state.min_log_odds,
            max_log_odds=state.max_log_odds,
        )
        result.update_count = state.update_count
        result.observation_sha256 = list(state.observation_sha256)
        for reference in state.blocks:
            block = VoxelBlock.from_bytes(state.resolve(reference.content).read_bytes())
            if block.index != reference.index:
                raise ReconstructionError("voxel block payload/index binding mismatch")
            result.blocks[reference.index] = block
        return result


__all__ = ["ReconstructionError", "SparseBayesianReconstruction", "VoxelBlock"]
