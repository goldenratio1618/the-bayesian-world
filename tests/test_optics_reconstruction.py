from __future__ import annotations

import json

import numpy as np
import pytest

from contraption.optics import (
    OpticalSchemaError,
    OpticalSensor,
    Pose,
    ReconstructionState,
    ReconstructionError,
    SensorNoise,
    SparseBayesianReconstruction,
    VoxelBlock,
    extract_posterior_surface,
)


def _sensor() -> OpticalSensor:
    return OpticalSensor(
        "depth-camera",
        (1, 1),
        (1.0, 1.0),
        (0.5, 0.5),
        near_clip_m=0.1,
        far_clip_m=5.0,
        noise=SensorNoise("none", depth_noise_std_m=0.01),
    )


def test_voxel_block_binary_round_trip() -> None:
    block = VoxelBlock.empty(4, 0.0, (2, -1, 7))
    block.occupancy_log_odds[1, 2, 3] = 1.25
    block.tsdf_mean[1, 2, 3] = -0.2
    block.color_mean[1, 2, 3] = (0.1, 0.2, 0.3)
    block.update_count[1, 2, 3] = 7
    payload = block.to_bytes()
    assert payload.startswith(b"SVOXBLK2\n")
    decoded = VoxelBlock.from_bytes(payload)
    assert decoded.index == (2, -1, 7)
    assert np.array_equal(decoded.occupancy_log_odds, block.occupancy_log_odds)
    assert np.array_equal(decoded.tsdf_mean, block.tsdf_mean)
    assert np.array_equal(decoded.color_mean, block.color_mean)
    assert np.array_equal(decoded.update_count, block.update_count)
    with pytest.raises(ReconstructionError, match="framing"):
        VoxelBlock.from_bytes(block.to_bytes() + b"trailing")


def test_sparse_bayesian_update_and_persistence(tmp_path) -> None:
    reconstruction = SparseBayesianReconstruction(
        "scan",
        voxel_size_m=0.1,
        block_size=4,
        origin_world_m=(-1.0, -1.0, 0.0),
        truncation_distance_m=0.3,
    )
    sensor = _sensor()
    depth = np.asarray([[1.0]], dtype=np.float32)
    uncertainty = np.asarray([[0.02]], dtype=np.float32)
    rgb = np.asarray([[[0.8, 0.2, 0.1]]], dtype=np.float32)
    reconstruction.update_depth(sensor, depth, Pose(), uncertainty_m=uncertainty, rgb_linear=rgb, observation_sha256="c" * 64)
    free = reconstruction.voxel_posterior((10, 10, 5))
    surface = reconstruction.voxel_posterior((10, 10, 10))
    assert free["occupancy_probability"] < 0.5
    assert surface["occupancy_probability"] > 0.5
    assert surface["tsdf_standard_deviation_m"] < float("inf")
    assert surface["color_mean"][0] > surface["color_mean"][1]
    assert reconstruction.blocks

    state = reconstruction.save(tmp_path)
    state.verify()
    assert state.format == "reconstruction-state-1"
    assert state.observation_sha256 == ("c" * 64,)
    loaded = SparseBayesianReconstruction.load(tmp_path / "reconstruction.state.json")
    assert loaded.voxel_posterior((10, 10, 10))["occupancy_probability"] == surface["occupancy_probability"]

    manifest = tmp_path / "reconstruction.state.json"
    value = json.loads(manifest.read_text())
    value["blocks"][0]["index"][0] += 1
    manifest.write_text(json.dumps(value))
    with pytest.raises(OpticalSchemaError, match="index mismatch"):
        ReconstructionState.load(manifest)


def test_entropy_next_best_view_prefers_more_unobserved_volume() -> None:
    reconstruction = SparseBayesianReconstruction("scan", voxel_size_m=0.2, block_size=4)
    sensor = _sensor()
    short = reconstruction.expected_information_gain(sensor, Pose(), maximum_distance_m=1.0, pixel_stride=1)
    long = reconstruction.expected_information_gain(sensor, Pose(), maximum_distance_m=3.0, pixel_stride=1)
    assert long > short


def _single_voxel_reconstruction() -> SparseBayesianReconstruction:
    reconstruction = SparseBayesianReconstruction(
        "surface",
        voxel_size_m=0.2,
        block_size=4,
        origin_world_m=(1.0, -2.0, 3.0),
        truncation_distance_m=0.4,
    )
    reconstruction._update_voxel(
        np.asarray((2, -1, 3), dtype=np.int64),
        occupancy_increment=reconstruction.occupied_increment,
        tsdf=0.0,
        tsdf_precision=400.0,
        color=np.asarray((0.8, 0.2, 0.1)),
        color_precision=100.0,
    )
    return reconstruction


def test_posterior_voxel_boundary_surface_is_metric_closed_and_deterministic() -> None:
    reconstruction = _single_voxel_reconstruction()
    first = extract_posterior_surface(reconstruction)
    second = extract_posterior_surface(reconstruction)

    assert first.method == "occupancy-tsdf-voxel-boundary-v1"
    assert first.occupied_voxel_count == 1
    assert first.boundary_face_count == 6
    assert len(first.mesh.vertices_m) == 8
    assert len(first.mesh.triangles) == 12
    assert first.mesh.watertight
    assert first.manifold
    assert first.mesh.to_bytes() == second.mesh.to_bytes()
    assert np.array_equal(
        first.vertex_position_standard_deviation_m,
        second.vertex_position_standard_deviation_m,
    )
    # Voxel (2, -1, 3) occupies grid-corner range [index, index+1].
    assert np.allclose(first.mesh.bounds_m[0], (1.4, -2.2, 3.6))
    assert np.allclose(first.mesh.bounds_m[1], (1.6, -2.0, 3.8))
    assert np.allclose(first.mesh.vertex_rgba_linear[:, :3], (0.8, 0.2, 0.1))
    expected_uncertainty = np.sqrt(0.4**2 / 400.0 + 0.2**2 / 12.0)
    assert np.allclose(
        first.vertex_position_standard_deviation_m, expected_uncertainty
    )


def test_posterior_surface_extraction_fails_closed_for_empty_or_excessive_maps() -> None:
    empty = SparseBayesianReconstruction("empty", voxel_size_m=0.1)
    with pytest.raises(ReconstructionError, match="no occupancy/TSDF cells"):
        extract_posterior_surface(empty)
    reconstruction = _single_voxel_reconstruction()
    with pytest.raises(ReconstructionError, match="maximum_triangles=11"):
        extract_posterior_surface(reconstruction, maximum_triangles=11)
    reconstruction._update_voxel(
        np.asarray((3, -1, 3), dtype=np.int64),
        occupancy_increment=reconstruction.occupied_increment,
        tsdf=0.0,
        tsdf_precision=400.0,
    )
    with pytest.raises(ReconstructionError, match="maximum_occupied_voxels=1"):
        extract_posterior_surface(reconstruction, maximum_occupied_voxels=1)
