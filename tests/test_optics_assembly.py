from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from contraption import load_contraption
from contraption.cli import _trajectory_result, main
from contraption.optics import (
    ObservationArtifact,
    OpticalLight,
    OpticalScene,
    ReconstructionState,
    RuntimeScene,
    SceneObject,
    build_assembly_optical_frame,
    capture_assembly,
)
from contraption.shape import (
    ContentReference,
    OpticalMaterial,
    ShapeArtifact,
    ShapeArtifactError,
    SourceRepresentation,
    SurfaceRepresentation,
    TriangleMesh,
    VolumeRepresentation,
)


ROOT = Path(__file__).resolve().parents[1]
SCANNER_SPEC = ROOT / "assembled_contraptions" / "scanner" / "contraption.json"


def _external_target(tmp_path: Path, scanner) -> Path:
    source_path = tmp_path / "target.procedural"
    source_path.write_text("two-triangle optical calibration target\n", encoding="utf-8")
    mesh = TriangleMesh(
        np.asarray(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ],
            dtype=float,
        ),
        np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.uint32),
        face_material=np.asarray([0, 0], dtype=np.uint32),
    )
    mesh_path = mesh.write(tmp_path / "target.ctmesh")
    artifact = ShapeArtifact(
        id="scanner-test-target",
        version="1.0.0",
        sources=(
            SourceRepresentation(
                "fixture-source",
                "procedural",
                ContentReference.from_path(
                    source_path, relative_to=tmp_path, media_type="text/plain"
                ),
                1.0,
            ),
        ),
        surfaces=(
            SurfaceRepresentation(
                "optical-surface",
                "ctmesh",
                ContentReference.from_path(
                    mesh_path,
                    relative_to=tmp_path,
                    media_type="application/vnd.contraption.ctmesh",
                ),
                ("ray_trace", "render"),
                4,
                2,
                (-2.0, -2.0, 0.0, 2.0, 2.0, 0.0),
                False,
                True,
                ("blue-paint",),
            ),
        ),
        optical_materials=(
            OpticalMaterial(
                "blue-paint",
                model="lambertian",
                base_color_linear_rgba=(0.02, 0.1, 0.8, 1.0),
                roughness=0.8,
                double_sided=True,
            ),
        ),
    )
    shape_path = artifact.write(tmp_path / "target.shape.json")

    static_frame = build_assembly_optical_frame(
        scanner, sensor_resolution_px=(16, 12)
    )
    camera = static_frame.sensors[0]
    world_from_sensor = np.asarray(
        camera.pose.transform_world_from_sensor_row_major, dtype=float
    ).reshape(4, 4)
    sensor_from_target = np.eye(4)
    sensor_from_target[2, 3] = 0.6
    world_from_target = world_from_sensor @ sensor_from_target
    camera_position = tuple(float(item) for item in world_from_sensor[:3, 3])
    scene = OpticalScene(
        "scanner-external-target",
        (
            SceneObject(
                "calibration-target",
                shape_path.name,
                artifact.artifact_sha256,
                9001,
                tuple(float(item) for item in world_from_target.reshape(-1)),
                "optical-surface",
                0.003,
            ),
        ),
        (
            OpticalLight(
                "scanner-headlight",
                "point",
                (1.0, 1.0, 1.0),
                20.0,
                position_m=camera_position,
            ),
        ),
        (0.05, 0.05, 0.05),
        {"fixture": "explicit external geometry, never object_bounding_cube"},
    )
    scene_path = scene.write(tmp_path / "target.optical-scene.json")
    assert OpticalScene.load(scene_path).objects[0].segmentation_id == 9001
    return scene_path


def test_scanner_simulation_capture_and_reconstruction_end_to_end(
    tmp_path: Path, capsys
) -> None:
    scanner = load_contraption(SCANNER_SPEC)
    scene_path = _external_target(tmp_path, scanner)
    run = tmp_path / "run"
    exit_code = main(
        [
            "simulate",
            "--spec",
            str(SCANNER_SPEC),
            "--duration",
            "0.02",
            "--dt",
            "0.01",
            "--samples",
            "1",
            "--seed",
            "71",
            "--output",
            str(run),
            "--optical-capture",
            "--optical-scene",
            str(scene_path),
            "--optical-width",
            "16",
            "--optical-height",
            "12",
            "--optical-backend",
            "numpy",
        ]
    )
    assert exit_code == 0, capsys.readouterr().err
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    optical_report_path = Path(report["artifacts"]["optical_capture"])
    optical_report = json.loads(optical_report_path.read_text(encoding="utf-8"))
    assert optical_report["external_scene_sha256"] == OpticalScene.load(
        scene_path
    ).artifact_sha256
    assert optical_report["backend"] == "numpy-exact"
    sensor_path = next((run / "optical" / "sensors").glob("*.optical.json"))
    observation_path = next(
        (run / "optical" / "observations").glob("*.optical-observation.json")
    )
    observation = ObservationArtifact.load(observation_path)
    assert observation.assembly_id == scanner.scene["contraption_id"]
    assert observation.assembly_sha256 == scanner.assembly_sha256.removeprefix(
        "sha256:"
    )
    assert observation.mount_connector == "camera.optical_axis"
    assert observation.mount_transform_sha256 == observation.pose.artifact_sha256
    products = observation.load_arrays()
    assert products["depth_m"].shape == (12, 16)
    assert np.any(products["segmentation"] == 9001)

    trajectory = _trajectory_result(scanner, run / "trajectory.json")
    final_pose = scanner.body_pose_frames(trajectory, sample_index=0)["frames"][-1][
        "connector_poses"
    ]["camera.optical_axis"]
    expected_matrix = build_assembly_optical_frame(
        scanner,
        result=trajectory,
        sample_index=0,
        time_index=-1,
        sensor_resolution_px=(16, 12),
        external_scene=scene_path,
    ).sensors[0].pose
    assert expected_matrix == observation.pose
    assert np.allclose(
        final_pose["translation_m"],
        np.asarray(observation.pose.transform_world_from_sensor_row_major).reshape(4, 4)[
            :3, 3
        ],
    )

    reconstruction_root = tmp_path / "reconstruction"
    exit_code = main(
        [
            "optical-reconstruct",
            "--sensor",
            str(sensor_path),
            "--observation",
            str(observation_path),
            "--output",
            str(reconstruction_root),
            "--id",
            "scanner-map",
            "--voxel-size",
            "0.025",
        ]
    )
    assert exit_code == 0, capsys.readouterr().err
    state = ReconstructionState.load(
        reconstruction_root / "reconstruction.state.json"
    )
    assert state.update_count == 1
    assert state.blocks
    assert state.observation_sha256 == (observation.artifact_sha256,)
    volume = VolumeRepresentation.from_dict(
        json.loads(
            (reconstruction_root / "shape-volume.json").read_text(encoding="utf-8")
        )
    )
    assert volume.kind == "sparse_tsdf"
    assert volume.purposes == ("reconstruction", "ray_trace")
    assert volume.content.sha256 == hashlib.sha256(
        (reconstruction_root / "reconstruction.state.json").read_bytes()
    ).hexdigest()
    shape_path = reconstruction_root / "shape.artifact.json"
    shape = ShapeArtifact.load(shape_path, verify_content=True)
    assert shape.format == "shape-artifact-1"
    assert len(shape.surfaces) == 1
    surface = shape.surfaces[0]
    assert surface.id == "reconstruction-surface"
    assert surface.kind == "ctmesh"
    assert surface.purposes == ("render", "ray_trace", "analysis")
    assert surface.material_ids == ("posterior-color",)
    assert surface.vertex_count > 0
    assert surface.triangle_count > 0
    mesh = shape.load_surface("render")
    assert len(mesh.vertices_m) == surface.vertex_count
    assert len(mesh.triangles) == surface.triangle_count
    assert mesh.vertex_rgba_linear is not None
    assert mesh.face_material is not None
    uncertainty_field = next(
        item
        for item in shape.physical_fields
        if item.id == "surface-position-standard-deviation"
    )
    uncertainty = np.load(
        shape.resolve(uncertainty_field.content), allow_pickle=False
    )
    assert uncertainty.shape == (surface.vertex_count,)
    assert np.all(np.isfinite(uncertainty))
    assert np.all(uncertainty >= 0.0)
    assert shape.volumes == (volume,)
    assert shape.caches == ()
    assert {item.format for item in shape.sources} == {
        "optical_sensor",
        "optical_observation",
    }
    reconstruction_report = json.loads(
        (reconstruction_root / "report.json").read_text(encoding="utf-8")
    )
    assert reconstruction_report["shape_artifact"] == {
        "path": str(shape_path),
        "artifact_sha256": shape.artifact_sha256,
        "format": "shape-artifact-1",
        "source_count": 2,
        "surface_ids": ["reconstruction-surface"],
        "volume_ids": ["reconstruction"],
    }
    assert reconstruction_report["shape_surface"] == {
        "path": str(reconstruction_root / "canonical.surface.ctmesh"),
        "value": surface.to_dict(),
    }
    assert reconstruction_report["surface_uncertainty"] == {
        "path": str(
            reconstruction_root / "surface.position-standard-deviation.npy"
        ),
        "value": uncertainty_field.to_dict(),
    }

    # The generated shape is directly materialized by the standard optical
    # scene path; no scan-specific or volume-to-render adapter is involved.
    reconstructed_scene = OpticalScene(
        "reconstructed-scene",
        (
            SceneObject(
                "reconstructed-object",
                shape_path.name,
                shape.artifact_sha256,
                7101,
                tuple(float(item) for item in np.eye(4).reshape(-1)),
                surface.id,
                float(np.max(uncertainty)),
            ),
        ),
    )
    reconstructed_scene_path = reconstructed_scene.write(
        reconstruction_root / "reconstructed.optical-scene.json"
    )
    runtime = RuntimeScene.from_manifest(reconstructed_scene_path)
    assert len(runtime.instances) == 1
    assert runtime.instances[0].mesh.to_bytes() == mesh.to_bytes()
    assert runtime.instances[0].materials[0].id == "posterior-color"

    # Identical evidence and settings yield identical canonical posterior
    # surface bytes and field bytes in a fresh output directory.
    deterministic_root = tmp_path / "reconstruction-repeat"
    exit_code = main(
        [
            "optical-reconstruct",
            "--sensor",
            str(sensor_path),
            "--observation",
            str(observation_path),
            "--output",
            str(deterministic_root),
            "--id",
            "scanner-map",
            "--voxel-size",
            "0.025",
        ]
    )
    assert exit_code == 0, capsys.readouterr().err
    assert (deterministic_root / "canonical.surface.ctmesh").read_bytes() == (
        reconstruction_root / "canonical.surface.ctmesh"
    ).read_bytes()
    assert (
        deterministic_root / "surface.position-standard-deviation.npy"
    ).read_bytes() == (
        reconstruction_root / "surface.position-standard-deviation.npy"
    ).read_bytes()

    # The unified artifact verifies the complete nested evidence/posterior
    # closure, not only its observation and reconstruction JSON manifests.
    copied_output = reconstruction_root / shape.metadata["evidence"]["observations"][0][
        "payload_uris"
    ][0]
    original_output = copied_output.read_bytes()
    copied_output.write_bytes(original_output + b"tampered")
    with pytest.raises(ShapeArtifactError, match="transitive optical source"):
        ShapeArtifact.load(shape_path, verify_content=True)
    copied_output.write_bytes(original_output)
    ShapeArtifact.load(shape_path, verify_content=True)

    copied_block = state.resolve(state.blocks[0].content)
    original_block = copied_block.read_bytes()
    copied_block.write_bytes(original_block + b"tampered")
    with pytest.raises(ShapeArtifactError, match="transitive sparse TSDF"):
        ShapeArtifact.load(shape_path, verify_content=True)
    copied_block.write_bytes(original_block)
    ShapeArtifact.load(shape_path, verify_content=True)


def test_external_scene_segmentation_is_preserved_and_assembly_ids_do_not_collide(
    tmp_path: Path,
) -> None:
    scanner = load_contraption(SCANNER_SPEC)
    scene_path = _external_target(tmp_path, scanner)
    frame = build_assembly_optical_frame(
        scanner,
        sensor_resolution_px=(8, 6),
        external_scene=scene_path,
    )
    by_id = {item.id: item.segmentation_id for item in frame.scene.instances}
    assert by_id["external:calibration-target"] == 9001
    assert all(
        segmentation > 9001
        for name, segmentation in by_id.items()
        if not name.startswith("external:")
    )
    assert frame.external_scene_sha256 == OpticalScene.load(scene_path).artifact_sha256


def test_torch_cpu_assembly_capture_persists_numpy_observation_payloads(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    scanner = load_contraption(SCANNER_SPEC)
    scene_path = _external_target(tmp_path, scanner)
    capture = capture_assembly(
        scanner,
        tmp_path / "torch-capture",
        sensor_resolution_px=(4, 3),
        external_scene=scene_path,
        backend="torch",
        device="cpu",
        seed=8,
    )
    assert capture.backend == "torch-differentiable"
    assert capture.device == "cpu"
    arrays = capture.observations[0].load_arrays()
    assert arrays["rgb_linear"].dtype == np.float32
    assert arrays["segmentation"].dtype == np.int32
