from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from contraption.optics import (
    AsyncOpticalSimulator,
    MeshInstance,
    NumpyOpticalBackend,
    OpticalLight,
    OpticalRenderError,
    OpticalScene,
    OpticalSensor,
    Pose,
    RuntimeMaterial,
    RuntimeScene,
    SceneObject,
    SensorNoise,
)
from contraption.shape import (
    ContentReference,
    OpticalMaterial,
    ShapeArtifact,
    SourceRepresentation,
    SurfaceRepresentation,
    TriangleMesh,
)
from contraption.shape.artifacts import ShapeArtifactError


def _front_facing_square(z: float = 2.0) -> TriangleMesh:
    vertices = np.asarray(
        [
            [-1.0, -1.0, z],
            [1.0, -1.0, z],
            [1.0, 1.0, z],
            [-1.0, 1.0, z],
        ]
    )
    # Winding points toward the camera; the backend is two-sided regardless.
    return TriangleMesh(vertices, np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.uint32))


def _scene() -> RuntimeScene:
    return RuntimeScene(
        "square-scene",
        (
            MeshInstance(
                "square",
                _front_facing_square(),
                17,
                (RuntimeMaterial("red", (0.8, 0.1, 0.1), roughness=0.7),),
                surface_uncertainty_m=0.002,
            ),
        ),
        (OpticalLight("headlight", "point", (1, 1, 1), 100.0, position_m=(0, 0, 0)),),
        (0.01, 0.01, 0.01),
    )


def _sensor(noise: SensorNoise | None = None) -> OpticalSensor:
    return OpticalSensor(
        "camera",
        (5, 5),
        (5.0, 5.0),
        (2.5, 2.5),
        near_clip_m=0.1,
        far_clip_m=10.0,
        exposure_duration_s=1.0,
        noise=noise or SensorNoise("none"),
    )


def test_numpy_backend_returns_all_optical_products() -> None:
    products = NumpyOpticalBackend().render(_scene(), _sensor())
    center = (2, 2)
    assert products.rgb_linear.shape == (5, 5, 3)
    assert np.isclose(products.depth_m[center], 2.0)
    assert products.segmentation[center] == 17
    assert np.isclose(products.uncertainty[center], 0.002)
    assert products.rgb_linear[center][0] > products.rgb_linear[center][1]


def test_seeded_noise_is_bitwise_reproducible_and_frame_specific() -> None:
    noise = SensorNoise("gaussian_poisson", seed=91, read_noise_std_linear=0.02, depth_noise_std_m=0.001)
    backend, scene, sensor = NumpyOpticalBackend(), _scene(), _sensor(noise)
    first = backend.render(scene, sensor, frame_index=3, seed=8)
    repeat = backend.render(scene, sensor, frame_index=3, seed=8)
    different = backend.render(scene, sensor, frame_index=4, seed=8)
    assert np.array_equal(first.rgb_linear, repeat.rgb_linear)
    assert np.array_equal(first.depth_m, repeat.depth_m)
    assert not np.array_equal(first.rgb_linear, different.rgb_linear)


def test_async_simulator_writes_hash_verified_observation(tmp_path) -> None:
    sensor = _sensor()
    with AsyncOpticalSimulator(tmp_path, max_workers=2) as simulator:
        pending = simulator.submit_capture(
            _scene(), sensor, Pose(), frame_index=4, requested_at_s=3.0,
            assembly_id="scanner", assembly_sha256="b" * 64, assembly_mount_connector="camera.optical_axis",
        )
        assert pending.ready_at_s == 4.0
        observation = pending.result()
    assert observation.assembly_id == "scanner"
    assert observation.assembly_sha256 == "b" * 64
    assert observation.mount_connector == "camera.optical_axis"
    assert observation.mount_transform_sha256 == Pose().artifact_sha256
    arrays = observation.load_arrays()
    assert set(arrays) == set(sensor.outputs)
    assert arrays["depth_m"].shape == (5, 5)


def _replace_only_instance(scene: RuntimeScene, **changes) -> RuntimeScene:
    return replace(scene, instances=(replace(scene.instances[0], **changes),))


def test_runtime_scene_hash_covers_every_render_affecting_field() -> None:
    scene = _scene()
    instance = scene.instances[0]
    material = instance.materials[0]
    light = scene.lights[0]
    moved = np.eye(4)
    moved[0, 3] = 0.01
    changed_mesh = TriangleMesh(
        instance.mesh.vertices_m
        + np.asarray([[0.0, 0.0, 1e-10], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        instance.mesh.triangles,
    )
    material_variants = (
        replace(material, base_color_linear_rgb=(0.7, 0.1, 0.1)),
        replace(material, roughness=0.6),
        replace(material, metallic=0.1),
        replace(material, transmission=0.1),
        replace(material, refractive_index=1.8),
        replace(material, emission_linear_rgb=(0.01, 0.0, 0.0)),
    )
    variants = [
        _replace_only_instance(scene, mesh=changed_mesh),
        _replace_only_instance(scene, segmentation_id=18),
        _replace_only_instance(
            scene,
            transform_world_from_object_row_major=tuple(moved.reshape(-1)),
        ),
        _replace_only_instance(scene, surface_uncertainty_m=0.003),
        replace(scene, lights=(replace(light, color_linear_rgb=(0.9, 1.0, 1.0)),)),
        replace(scene, lights=(replace(light, intensity=99.0),)),
        replace(scene, lights=(replace(light, position_m=(0.0, 0.0, 0.1)),)),
        replace(
            scene,
            lights=(
                OpticalLight(
                    light.id,
                    "directional",
                    light.color_linear_rgb,
                    light.intensity,
                    direction_world=(0.0, 0.0, 1.0),
                ),
            ),
        ),
        replace(
            scene,
            lights=(
                OpticalLight(
                    light.id,
                    "directional",
                    light.color_linear_rgb,
                    light.intensity,
                    direction_world=(0.0, 1.0, 1.0),
                ),
            ),
        ),
        replace(scene, environment_linear_rgb=(0.02, 0.01, 0.01)),
    ]
    variants.extend(
        _replace_only_instance(scene, materials=(variant,))
        for variant in material_variants
    )
    digests = {scene.artifact_sha256, *(item.artifact_sha256 for item in variants)}
    assert len(digests) == len(variants) + 1
    assert _scene().artifact_sha256 == scene.artifact_sha256
    numerically_equivalent = replace(
        scene,
        lights=(
            OpticalLight(
                light.id,
                light.kind,
                (1.0, 1.0, 1.0),
                100,
                position_m=(0.0, 0.0, 0.0),
            ),
        ),
    )
    assert numerically_equivalent.artifact_sha256 == scene.artifact_sha256

    # The former concatenated-text scheme could not distinguish these
    # scene/instance ID boundaries ("a" + "bc" versus "ab" + "c").
    boundary_one = replace(
        scene, id="a", instances=(replace(instance, id="bc"),)
    )
    boundary_two = replace(
        scene, id="ab", instances=(replace(instance, id="c"),)
    )
    assert boundary_one.artifact_sha256 != boundary_two.artifact_sha256

    materials = (material, replace(material, id="blue"))
    indexed_mesh_one = TriangleMesh(
        instance.mesh.vertices_m,
        instance.mesh.triangles,
        face_material=np.asarray((0, 1), dtype=np.uint32),
    )
    indexed_mesh_two = TriangleMesh(
        instance.mesh.vertices_m,
        instance.mesh.triangles,
        face_material=np.asarray((1, 0), dtype=np.uint32),
    )
    indexed_scene_one = _replace_only_instance(
        scene, mesh=indexed_mesh_one, materials=materials
    )
    indexed_scene_two = _replace_only_instance(
        scene, mesh=indexed_mesh_two, materials=materials
    )
    assert indexed_scene_one.artifact_sha256 != indexed_scene_two.artifact_sha256

    uncertain = _replace_only_instance(scene, surface_uncertainty_m=0.02)
    original_products = NumpyOpticalBackend().render(scene, _sensor())
    uncertain_products = NumpyOpticalBackend().render(uncertain, _sensor())
    assert uncertain.artifact_sha256 != scene.artifact_sha256
    assert uncertain_products.uncertainty[2, 2] > original_products.uncertainty[2, 2]


@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (np.diag((2.0, 1.0, 1.0, 1.0)), "no scale or shear"),
        (np.diag((-1.0, 1.0, 1.0, 1.0)), "right-handed"),
        (
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.1, 1.0],
                ]
            ),
            "homogeneous final row",
        ),
    ),
)
def test_mesh_instance_rejects_non_rigid_or_left_handed_transforms(
    matrix: np.ndarray, message: str
) -> None:
    with pytest.raises(OpticalRenderError, match=message):
        MeshInstance(
            "invalid",
            _front_facing_square(),
            1,
            transform_world_from_object_row_major=tuple(matrix.reshape(-1)),
        )


def _shape_scene_manifest(
    tmp_path,
    *,
    purposes: tuple[str, ...],
    material_ids: tuple[str, ...],
    include_material: bool,
):
    source_path = tmp_path / "source.procedural"
    source_path.write_text("fixture\n", encoding="utf-8")
    mesh = TriangleMesh(
        _front_facing_square().vertices_m,
        _front_facing_square().triangles,
        face_material=np.zeros(2, dtype=np.uint32),
    )
    mesh_path = mesh.write(tmp_path / "surface.ctmesh")
    shape = ShapeArtifact(
        id="manifest-shape",
        version="1.0.0",
        sources=(
            SourceRepresentation(
                "source",
                "procedural",
                ContentReference.from_path(
                    source_path, relative_to=tmp_path, media_type="text/plain"
                ),
                1.0,
            ),
        ),
        surfaces=(
            SurfaceRepresentation(
                "surface",
                "ctmesh",
                ContentReference.from_path(
                    mesh_path,
                    relative_to=tmp_path,
                    media_type="application/vnd.contraption.ctmesh",
                ),
                purposes,
                len(mesh.vertices_m),
                len(mesh.triangles),
                tuple(float(item) for item in (*mesh.bounds_m[0], *mesh.bounds_m[1])),
                mesh.watertight,
                True,
                material_ids,
            ),
        ),
        optical_materials=(OpticalMaterial("paint"),) if include_material else (),
    )
    shape_path = shape.write(tmp_path / "shape.artifact.json")
    scene = OpticalScene(
        "manifest-scene",
        (
            SceneObject(
                "object",
                shape_path.name,
                shape.artifact_sha256,
                1,
                surface_id="surface",
            ),
        ),
    )
    return scene.write(tmp_path / "scene.optical.json")


def test_runtime_scene_manifest_requires_ray_trace_surface(tmp_path) -> None:
    manifest = _shape_scene_manifest(
        tmp_path,
        purposes=("render",),
        material_ids=("paint",),
        include_material=True,
    )
    with pytest.raises(OpticalRenderError, match="not authored for ray tracing"):
        RuntimeScene.from_manifest(manifest)


def test_runtime_scene_manifest_rejects_unbound_face_material_table(tmp_path) -> None:
    manifest = _shape_scene_manifest(
        tmp_path,
        purposes=("ray_trace",),
        material_ids=(),
        include_material=False,
    )
    with pytest.raises(ShapeArtifactError, match="face-material index"):
        RuntimeScene.from_manifest(manifest)


def _dielectric_scene(refractive_index: float) -> RuntimeScene:
    return RuntimeScene(
        "dielectric",
        (
            MeshInstance(
                "surface",
                _front_facing_square(),
                1,
                (
                    RuntimeMaterial(
                        "dielectric",
                        (0.0, 0.0, 0.0),
                        roughness=0.5,
                        refractive_index=refractive_index,
                    ),
                ),
            ),
        ),
        (
            OpticalLight(
                "headlight",
                "point",
                (1.0, 1.0, 1.0),
                100.0,
                position_m=(0.0, 0.0, 0.0),
            ),
        ),
        (0.0, 0.0, 0.0),
    )


def test_numpy_dielectric_fresnel_is_refractive_index_sensitive() -> None:
    backend = NumpyOpticalBackend(cast_shadows=False)
    low = backend.render(_dielectric_scene(1.1), _sensor(), apply_noise=False)
    high = backend.render(_dielectric_scene(2.0), _sensor(), apply_noise=False)
    assert np.all(high.rgb_linear[2, 2] > low.rgb_linear[2, 2])
