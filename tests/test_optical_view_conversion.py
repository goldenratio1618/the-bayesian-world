from __future__ import annotations

import base64
import hashlib

import numpy as np
import pytest

from contraption.optics.schemas import (
    ObservationArtifact,
    OpticalSchemaError,
    OpticalSensor,
    Pose,
    SensorNoise,
)
from contraption.visualization.optical_views import (
    OpticalViewError,
    observation_view_from_artifact,
)


ASSEMBLY_DIGEST = "a" * 64
SCENE_DIGEST = "b" * 64
IDENTITY_POSE = {
    "translation_m": [0.0, 0.0, 0.0],
    "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
}


def _sensor() -> OpticalSensor:
    return OpticalSensor(
        "camera.optical",
        (2, 2),
        (2.0, 2.0),
        (1.0, 1.0),
        near_clip_m=0.1,
        far_clip_m=5.0,
        mount_connector="optical_axis",
        noise=SensorNoise("none"),
    )


def _observation(tmp_path, *, pose: Pose = Pose()) -> tuple[OpticalSensor, ObservationArtifact]:
    sensor = _sensor()
    observation = ObservationArtifact.from_arrays(
        path=tmp_path / "frame.observation.json",
        arrays={
            "rgb_linear": np.asarray(
                [[[0.0, 0.5, 1.0], [1.0, 0.0, 0.0]], [[0.1, 0.2, 0.3], [2.0, 1.0, 0.0]]],
                dtype=np.float32,
            ),
            "depth_m": np.asarray([[0.1, 5.0], [np.inf, 1.0]], dtype=np.float32),
            "segmentation": np.asarray([[-1, 0], [1, 42]], dtype=np.int32),
            "uncertainty": np.asarray([[0.0, 0.01], [np.inf, 0.1]], dtype=np.float32),
        },
        id="frame-0",
        sensor=sensor,
        scene_sha256=SCENE_DIGEST,
        frame_index=0,
        requested_at_s=0.0,
        exposure_started_at_s=0.0,
        ready_at_s=sensor.exposure_duration_s,
        pose=pose,
        seed=11,
        assembly_id="scanner",
        assembly_sha256=ASSEMBLY_DIGEST,
        mount_connector="camera.optical_axis",
        mount_transform_sha256=pose.artifact_sha256,
    )
    return sensor, observation


def _scene() -> dict:
    return {
        "contraption_id": "scanner",
        "connector_poses": {"camera.optical_axis": IDENTITY_POSE},
    }


def test_observation_conversion_preserves_source_evidence_and_derives_deterministic_pngs(tmp_path) -> None:
    sensor, observation = _observation(tmp_path)
    first = observation_view_from_artifact(
        observation,
        sensor,
        assembly_sha256="sha256:" + ASSEMBLY_DIGEST,
        assembly_id="scanner",
        scene=_scene(),
        expected_scene_sha256="sha256:" + SCENE_DIGEST,
        viewer_sensor_id="camera.camera.optical",
        assembly_mount_connector="camera.optical_axis",
    )
    repeat = observation_view_from_artifact(
        observation,
        sensor,
        assembly_sha256="sha256:" + ASSEMBLY_DIGEST,
        assembly_id="scanner",
        scene=_scene(),
        expected_scene_sha256=SCENE_DIGEST,
        viewer_sensor_id="camera.camera.optical",
        assembly_mount_connector="camera.optical_axis",
    )
    assert first == repeat
    assert first["artifact_sha256"] == "sha256:" + observation.artifact_sha256
    assert first["sensor_descriptor_sha256"] == "sha256:" + sensor.artifact_sha256
    assert first["mount_transform_sha256"] == "sha256:" + Pose().artifact_sha256
    assert set(first["layers"]) == {"rgb", "depth", "segmentation", "uncertainty"}
    outputs = {item.name: item for item in observation.outputs}
    for source_name, mode in {
        "rgb_linear": "rgb",
        "depth_m": "depth",
        "segmentation": "segmentation",
        "uncertainty": "uncertainty",
    }.items():
        layer = first["layers"][mode]
        png = base64.b64decode(layer["data_base64"], validate=True)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert layer["sha256"] == "sha256:" + hashlib.sha256(png).hexdigest()
        assert layer["source_output_sha256"] == "sha256:" + outputs[source_name].content.sha256
        assert layer["source_output_dtype"] == outputs[source_name].dtype
        assert layer["source_output_shape"] == list(outputs[source_name].shape)


def test_observation_conversion_rejects_pose_or_content_tampering(tmp_path) -> None:
    translated = Pose((1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
    sensor, observation = _observation(tmp_path, pose=translated)
    with pytest.raises(OpticalViewError, match="physical connector pose"):
        observation_view_from_artifact(
            observation,
            sensor,
            assembly_sha256="sha256:" + ASSEMBLY_DIGEST,
            assembly_id="scanner",
            scene=_scene(),
            viewer_sensor_id="camera.camera.optical",
            assembly_mount_connector="camera.optical_axis",
        )

    sensor, observation = _observation(tmp_path / "second")
    output_path = observation._resolve(observation.outputs[0].content)
    output_path.write_bytes(output_path.read_bytes() + b"tamper")
    with pytest.raises(OpticalSchemaError, match="hash mismatch"):
        observation_view_from_artifact(
            observation,
            sensor,
            assembly_sha256="sha256:" + ASSEMBLY_DIGEST,
            assembly_id="scanner",
            scene=_scene(),
            viewer_sensor_id="camera.camera.optical",
            assembly_mount_connector="camera.optical_axis",
        )
