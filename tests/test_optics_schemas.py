from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from contraption.optics import (
    ObservationArtifact,
    OpticalSchemaError,
    OpticalScene,
    OpticalSensor,
    Pose,
    SceneObject,
    SensorNoise,
    WireFrame,
    WirePayloadError,
    camera_rays,
    decode_wire_frame,
    encode_wire_frame,
)


def test_sensor_round_trip_and_canonical_pixel_frame() -> None:
    sensor = OpticalSensor(
        "scanner-camera",
        (3, 3),
        (2.0, 2.0),
        (1.5, 1.5),
        display_name="Scanner camera",
        mount_connector="camera_mount",
    )
    assert OpticalSensor.from_dict(sensor.to_dict()) == sensor
    rays = camera_rays(sensor)
    directions = rays.directions_world.reshape((3, 3, 3))
    # Sensor imaging coordinates are +X right, +Y down, +Z forward.
    assert directions[1, 2, 0] > directions[1, 0, 0]
    assert directions[2, 1, 1] > directions[0, 1, 1]
    assert np.all(directions[..., 2] > 0)


def test_scanner_catalog_sensor_descriptor_is_valid_and_mount_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    sensor = OpticalSensor.load(
        root / "model_catalog/optical/cameras/powered_rotational_cameras/instantiations/scanner_camera/sensor.optical.json"
    )
    assert sensor.id == "scanner.camera.optical"
    assert sensor.mount_connector == "optical_axis"
    assert sensor.resolution_px == (1920, 1080)


def test_optical_sensor_rejects_duplicate_json_fields(tmp_path) -> None:
    sensor = OpticalSensor("camera", (2, 1), (2, 2), (1, 0.5))
    path = tmp_path / "sensor.json"
    encoded = json.dumps(sensor.to_dict(), separators=(",", ":"))
    path.write_text('{"id":"ambiguous",' + encoded[1:], encoding="utf-8")
    with pytest.raises(OpticalSchemaError, match="duplicate JSON field 'id'"):
        OpticalSensor.load(path)


def test_pose_digest_uses_exact_little_endian_float64_matrix() -> None:
    pose = Pose()
    expected = hashlib.sha256(np.asarray(pose.transform_world_from_sensor_row_major, dtype="<f8").tobytes()).hexdigest()
    assert pose.artifact_sha256 == expected


@pytest.mark.parametrize(
    "transform",
    [
        (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1),
        (2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        (1, 0.1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
    ],
)
def test_pose_rejects_reflections_scale_and_shear(transform) -> None:
    with pytest.raises(OpticalSchemaError):
        Pose(transform)


def test_observation_optional_binding_fields_round_trip(tmp_path) -> None:
    sensor = OpticalSensor("camera", (2, 1), (2.0, 2.0), (1.0, 0.5), noise=SensorNoise("none"))
    arrays = {
        "rgb_linear": np.zeros((1, 2, 3), dtype=np.float32),
        "depth_m": np.ones((1, 2), dtype=np.float32),
        "segmentation": np.ones((1, 2), dtype=np.int32),
        "uncertainty": np.full((1, 2), 0.01, dtype=np.float32),
    }
    manifest = tmp_path / "frame.observation.json"
    observation = ObservationArtifact.from_arrays(
        path=manifest,
        arrays=arrays,
        id="frame-0",
        sensor=sensor,
        scene_sha256="a" * 64,
        frame_index=0,
        requested_at_s=0.0,
        exposure_started_at_s=0.0,
        ready_at_s=sensor.exposure_duration_s,
        pose=Pose(),
        seed=7,
    )
    encoded = observation.to_dict()
    assert encoded["assembly_frame"] == "world"
    assert "assembly_id" not in encoded
    assert "mount_connector" not in encoded
    loaded = ObservationArtifact.load(manifest)
    assert loaded.to_dict() == encoded
    assert all(np.array_equal(arrays[name], value) for name, value in loaded.load_arrays().items())


def test_observation_rejects_partial_assembly_binding(tmp_path) -> None:
    sensor = OpticalSensor("camera", (1, 1), (1, 1), (0.5, 0.5), outputs=("depth_m",))
    with pytest.raises(OpticalSchemaError, match="require assembly_id"):
        ObservationArtifact.from_arrays(
            path=tmp_path / "partial.json", arrays={"depth_m": np.ones((1, 1))}, id="partial",
            sensor=sensor, scene_sha256="a" * 64, frame_index=0, requested_at_s=0,
            exposure_started_at_s=0, ready_at_s=sensor.exposure_duration_s, pose=Pose(), seed=0,
            assembly_id="scanner",
        )


def test_observation_rejects_wrong_product_shape_before_writing(tmp_path) -> None:
    sensor = OpticalSensor("camera", (2, 1), (2, 2), (1, 0.5), outputs=("depth_m",))
    manifest = tmp_path / "wrong-shape.json"
    with pytest.raises(OpticalSchemaError, match="expected"):
        ObservationArtifact.from_arrays(
            path=manifest, arrays={"depth_m": np.ones((2, 1))}, id="wrong-shape",
            sensor=sensor, scene_sha256="a" * 64, frame_index=0, requested_at_s=0,
            exposure_started_at_s=0, ready_at_s=sensor.exposure_duration_s, pose=Pose(), seed=0,
        )
    assert not manifest.exists()
    assert not list(tmp_path.glob("*.npy"))


def test_observation_verify_rejects_npy_header_metadata_tamper(tmp_path) -> None:
    sensor = OpticalSensor(
        "camera", (2, 1), (2, 2), (1, 0.5), outputs=("depth_m",)
    )
    manifest = tmp_path / "frame.json"
    ObservationArtifact.from_arrays(
        path=manifest,
        arrays={"depth_m": np.ones((1, 2), dtype=np.float32)},
        id="frame",
        sensor=sensor,
        scene_sha256="a" * 64,
        frame_index=0,
        requested_at_s=0,
        exposure_started_at_s=0,
        ready_at_s=sensor.exposure_duration_s,
        pose=Pose(),
        seed=0,
    )
    value = json.loads(manifest.read_text())
    value["outputs"][0]["shape"] = [2, 1]
    manifest.write_text(json.dumps(value))
    with pytest.raises(OpticalSchemaError, match="shape mismatch"):
        ObservationArtifact.load(manifest)


def test_strict_schemas_reject_unknown_fields() -> None:
    value = OpticalSensor("camera", (2, 2), (2, 2), (1, 1)).to_dict()
    value["surprise"] = True
    with pytest.raises(OpticalSchemaError, match="unknown keys"):
        OpticalSensor.from_dict(value)


def test_scene_object_transform_is_not_a_sensor_pose_alias() -> None:
    transform = (1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4, 0, 0, 0, 1)
    scene = OpticalScene(
        "scene",
        (SceneObject("part", "part/shape.artifact.json", "d" * 64, 4, transform),),
    )
    encoded = scene.to_dict()
    assert encoded["objects"][0]["transform_world_from_object_row_major"] == list(transform)
    assert "pose" not in encoded["objects"][0]
    assert OpticalScene.from_dict(encoded).to_dict() == encoded


def test_wire_payload_is_bounded_versioned_and_checksummed() -> None:
    frame = WireFrame("depth_m", 9, 123456, b"depth bytes")
    assert decode_wire_frame(encode_wire_frame(frame)) == frame
    corrupt = bytearray(encode_wire_frame(frame))
    corrupt[-1] ^= 1
    with pytest.raises(WirePayloadError, match="CRC"):
        decode_wire_frame(bytes(corrupt))
