from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
from pathlib import Path
import struct

import pytest
import numpy as np

from contraption import load_contraption
from contraption.catalog import validate_optical_sensors
from contraption.optics import capture_result
from contraption.shape.artifacts import (
    ContentReference,
    OpticalMaterial,
    ShapeArtifact,
    SourceRepresentation,
    SurfaceRepresentation,
)
from contraption.shape.mesh import TriangleMesh
from contraption.visualization.render_bundle import (
    RENDER_BUNDLE_SCHEMA,
    TRIANGLE_SURFACE_SCHEMA,
    RenderBundleError,
    content_sha256,
    materialize_render_bundle,
    optical_sensors_from_registry,
)
from contraption.visualization.viewer import VisualizationError, generate_viewer


ROOT = Path(__file__).resolve().parents[1]
ASSEMBLY_HASH = "sha256:" + "a" * 64
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pose_matrix(pose: dict) -> list[float]:
    tx, ty, tz = pose["translation_m"]
    w, x, y, z = pose["rotation_quaternion_wxyz"]
    return [
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w), tx,
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w), ty,
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y), tz,
        0.0, 0.0, 0.0, 1.0,
    ]


def _box_surface(
    dimensions: list[float],
    source: str,
    manifest_sha256: str | None = None,
    surface_id: str = "render",
) -> dict:
    x, y, z = (item / 2.0 for item in dimensions)
    surface = {
        "schema": TRIANGLE_SURFACE_SCHEMA,
        "sha256": _hash_text("pending"),
        "shape_manifest_sha256": manifest_sha256 or _hash_text(f"manifest/{source}"),
        "shape_artifact_sha256": _hash_text(source),
        "shape_id": "fixture.shape",
        "surface_id": surface_id,
        "source_surface_sha256": _hash_text(f"surface/{source}"),
        "vertices_m": [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        "triangles": [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        "vertex_normals": None,
        "vertex_rgba_linear": None,
        "materials": [
            {
                "id": "housing",
                "base_color_linear_rgba": [0.18, 0.55, 0.72, 1.0],
                "optical_material_sha256": _hash_text(f"optical/{source}"),
            }
        ],
        "triangle_materials": [0] * 12,
        "vertex_uncertainty_m": [0.0001] * 8,
    }
    surface["sha256"] = content_sha256(surface)
    return surface


def _render_bundle(assembly) -> dict:
    surfaces: dict[str, dict] = {}
    bindings: list[dict] = []
    for component in assembly.scene["components"]:
        for body in component["bodies"]:
            for solid in body["solids"]:
                source = f"{component['id']}/{body['id']}/{solid['id']}"
                surface = _box_surface(
                    list(solid["geometry"]["dimensions_m"]),
                    source,
                    solid["geometry"]["shape_sha256"],
                    solid["geometry"]["surface_id"],
                )
                surfaces[surface["sha256"]] = surface
                bindings.append(
                    {
                        "component": component["id"],
                        "body": body["id"],
                        "solid": solid["id"],
                        "surface_sha256": surface["sha256"],
                    }
                )
    observation_sha256 = _hash_text("scanner/observation/0")
    raster = {
        "kind": "raster",
        "sha256": "sha256:" + hashlib.sha256(PNG_1X1).hexdigest(),
        "source_observation_sha256": observation_sha256,
        "source_output_sha256": _hash_text("scanner/rgb/0.npy"),
        "source_output_media_type": "application/vnd.numpy.npy",
        "source_output_dtype": "float32",
        "source_output_shape": [1, 1, 3],
        "display_transform": "linear-rgb-clamped-to-srgb8",
        "display_range": None,
        "media_type": "image/png",
        "width_px": 1,
        "height_px": 1,
        "data_base64": base64.b64encode(PNG_1X1).decode("ascii"),
    }
    bundle = {
        "schema": RENDER_BUNDLE_SCHEMA,
        "sha256": _hash_text("pending-bundle"),
        "assembly_sha256": assembly.assembly_sha256,
        "surfaces": surfaces,
        "solid_bindings": bindings,
        "sensors": [
            {
                "id": "scanner_camera",
                "display_name": "Scanner camera",
                "connector": "camera.optical_axis",
                "projection": {
                    "kind": "pinhole",
                    "resolution_px": [1, 1],
                    "focal_length_px": [1.0, 1.0],
                    "principal_point_px": [0.5, 0.5],
                    "clipping_m": [0.01, 20.0],
                },
                "descriptor_sha256": _hash_text("scanner/camera/descriptor"),
            }
        ],
        "observations": [
            {
                "id": "scanner-observation-0",
                "artifact_sha256": observation_sha256,
                "frame_index": 0,
                "sensor": "scanner_camera",
                "sensor_descriptor_sha256": _hash_text("scanner/camera/descriptor"),
                "optical_scene_sha256": _hash_text("scanner/optical-scene"),
                "assembly_id": assembly.scene["contraption_id"],
                "assembly_sha256": assembly.assembly_sha256,
                "assembly_frame": "world",
                "mount_connector": "camera.optical_axis",
                "mount_transform_sha256": "sha256:" + hashlib.sha256(
                    struct.pack("<16d", *_pose_matrix(assembly.scene["connector_poses"]["camera.optical_axis"]))
                ).hexdigest(),
                "pose_world_from_sensor_row_major": _pose_matrix(
                    assembly.scene["connector_poses"]["camera.optical_axis"]
                ),
                "layers": {
                    "rgb": raster,
                    "depth": {**raster, "source_output_sha256": _hash_text("scanner/depth/0.npy"), "source_output_shape": [1, 1], "display_transform": "depth-near-white-far-black", "display_range": [0.01, 20.0]},
                    "segmentation": {**raster, "source_output_sha256": _hash_text("scanner/segmentation/0.npy"), "source_output_dtype": "int32", "source_output_shape": [1, 1], "display_transform": "stable-integer-label-colors"},
                    "uncertainty": {**raster, "source_output_sha256": _hash_text("scanner/uncertainty/0.npy"), "source_output_shape": [1, 1], "display_transform": "uncertainty-log-blue-yellow-infinite-magenta", "display_range": [0.0, 0.1]},
                    "reconstruction": {
                        "kind": "surface",
                        "source_observation_sha256": _hash_text("scanner/reconstruction/0"),
                        "surface_sha256": bindings[0]["surface_sha256"],
                        "world_pose": {
                            "translation_m": [0.0, 0.0, 0.0],
                            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    },
                },
            }
        ],
    }
    bundle["sha256"] = content_sha256(bundle)
    return bundle


@pytest.fixture(scope="module")
def scanner():
    return load_contraption(
        ROOT / "assembled_contraptions" / "scanner" / "contraption.json"
    )


def _rehash(bundle: dict) -> dict:
    bundle["sha256"] = content_sha256(bundle)
    return bundle


def test_complete_bundle_materializes_detailed_surfaces_and_optical_views(scanner) -> None:
    bundle = _render_bundle(scanner)
    artifact = generate_viewer(scanner, render_bundle=bundle)
    assert artifact.data["schema"] == "contraption.viewer/v3"
    assert artifact.data["render_bundle"]["assembly_sha256"] == scanner.assembly_sha256
    assert len(artifact.data["render_bundle"]["solid_bindings"]) == sum(
        len(body["solids"])
        for component in scanner.scene["components"]
        for body in component["bodies"]
    )
    assert 'id="viewpoint"' in artifact.html
    assert 'id="view-mode"' in artifact.html
    assert "inverseTransformPose" in artifact.javascript
    assert "No hash-bound" in artifact.javascript
    assert "refusing a bounding-box substitute" in artifact.javascript
    assert "camera.optical_axis" in artifact.data_json


def test_scanner_viewer_contains_fixed_icosahedron_target(scanner) -> None:
    artifact = generate_viewer(scanner)
    scene = artifact.data["scene"]
    target = next(
        component for component in scene["components"]
        if component["id"] == "scan-target"
    )
    assert target["part"] == "scanner.icosahedron.v1"
    assert scene["body_poses"]["scan-target/body"]["translation_m"] == [
        0.0,
        0.0,
        0.35,
    ]
    binding = next(
        item for item in artifact.data["render_bundle"]["solid_bindings"]
        if item["component"] == "scan-target"
    )
    surface = artifact.data["render_bundle"]["surfaces"][binding["surface_sha256"]]
    assert len(surface["vertices_m"]) == 12
    assert len(surface["triangles"]) == 20


def test_viewer_accepts_exact_configured_sensor_from_capture(
    scanner, tmp_path: Path
) -> None:
    capture = capture_result(
        scanner,
        None,
        tmp_path / "capture",
        time_index=0,
        sensor_resolution_px=(16, 9),
        backend="numpy",
        device="cpu",
        seed=17,
    )
    artifact = generate_viewer(
        scanner,
        optical_sensors=capture.frame.sensors,
        optical_observations=capture.observations,
    )
    bundle = artifact.data["render_bundle"]
    assert bundle["sensors"][0]["projection"]["resolution_px"] == [16, 9]
    assert bundle["observations"][0]["frame_index"] == 0


def test_bundle_is_complete_and_bound_to_the_exact_assembly(scanner) -> None:
    missing = _render_bundle(scanner)
    missing["solid_bindings"].pop()
    with pytest.raises(VisualizationError, match="bind every physical solid"):
        generate_viewer(scanner, render_bundle=_rehash(missing))

    stale = _render_bundle(scanner)
    stale["assembly_sha256"] = ASSEMBLY_HASH
    with pytest.raises(VisualizationError, match="assembly hash mismatch"):
        generate_viewer(scanner, render_bundle=_rehash(stale))


def test_optical_pov_requires_an_authored_spatial_optical_connector(scanner) -> None:
    bundle = _render_bundle(scanner)
    bundle["sensors"][0]["connector"] = "camera.supply_p"
    with pytest.raises(VisualizationError, match="not a declared spatial optical connector"):
        generate_viewer(scanner, render_bundle=_rehash(bundle))


def _bound_camera(scanner):
    for component in scanner.scene["components"]:
        instantiation = scanner.instantiations[component["part"]]
        if instantiation.static.optical_sensors:
            return component, instantiation
    raise AssertionError("scanner fixture has no bound optical sensor")


def test_registry_sensor_discovery_uses_only_verified_static_bindings(
    scanner, tmp_path: Path
) -> None:
    component, instantiation = _bound_camera(scanner)
    verified = validate_optical_sensors(
        instantiation.static,
        instantiation.directory,
        scanner.component_models[component["id"]],
    )
    assert [sensor.id for sensor in verified] == [
        instantiation.static.optical_sensors[0].id
    ]
    discovered = optical_sensors_from_registry(
        scene=scanner.scene,
        registry=scanner.instantiations,
        component_models=scanner.component_models,
    )
    assert [(component_id, sensor.id) for component_id, sensor in discovered] == [
        (component["id"], instantiation.static.optical_sensors[0].id)
    ]

    # A descriptor present beside static.part has no authority without an exact
    # static optical_sensors binding.
    binding = instantiation.static.optical_sensors[0]
    source = instantiation.directory / binding.descriptor_uri
    (tmp_path / "sensor.optical.json").write_bytes(source.read_bytes())
    unbound = replace(instantiation.static, optical_sensors=())
    replacement = replace(instantiation, static=unbound, directory=tmp_path)
    registry = dict(scanner.instantiations)
    registry[component["part"]] = replacement
    assert optical_sensors_from_registry(
        scene=scanner.scene,
        registry=registry,
        component_models=scanner.component_models,
    ) == []


def test_registry_sensor_discovery_rechecks_binding_and_pmdl_closure(
    scanner, tmp_path: Path
) -> None:
    component, instantiation = _bound_camera(scanner)
    binding = instantiation.static.optical_sensors[0]
    source = instantiation.directory / binding.descriptor_uri
    (tmp_path / "sensor.optical.json").write_bytes(source.read_bytes())

    cases = (
        (
            replace(
                instantiation.static,
                optical_sensors=(
                    replace(binding, descriptor_sha256="sha256:" + "0" * 64),
                ),
            ),
            "descriptor hash mismatch",
        ),
        (
            replace(
                instantiation.static,
                optical_sensors=(replace(binding, body="missing_body"),),
            ),
            "references unknown body",
        ),
        (
            replace(
                instantiation.static,
                optical_sensors=(
                    replace(binding, artifact_port="missing_observation_stream"),
                ),
            ),
            "requires an output contraption/optical-observation@1 artifact port",
        ),
    )
    for static, message in cases:
        registry = dict(scanner.instantiations)
        registry[component["part"]] = replace(
            instantiation, static=static, directory=tmp_path
        )
        with pytest.raises(RenderBundleError, match=message):
            optical_sensors_from_registry(
                scene=scanner.scene,
                registry=registry,
                component_models=scanner.component_models,
            )

    wrong_models = dict(scanner.component_models)
    wrong_models[component["id"]] = next(
        model
        for component_id, model in scanner.component_models.items()
        if component_id != component["id"]
    )
    with pytest.raises(RenderBundleError, match="PMDL identity/hash"):
        optical_sensors_from_registry(
            scene=scanner.scene,
            registry=scanner.instantiations,
            component_models=wrong_models,
        )


def test_render_bundle_rejects_reflected_sensor_pose(scanner) -> None:
    bundle = _render_bundle(scanner)
    observation = bundle["observations"][0]
    reflected = list(observation["pose_world_from_sensor_row_major"])
    reflected[0:3] = [-item for item in reflected[0:3]]
    observation["pose_world_from_sensor_row_major"] = reflected
    observation["mount_transform_sha256"] = "sha256:" + hashlib.sha256(
        struct.pack("<16d", *reflected)
    ).hexdigest()
    with pytest.raises(VisualizationError, match="right-handed with determinant"):
        generate_viewer(scanner, render_bundle=_rehash(bundle))


def test_raster_bytes_are_verified_and_never_silently_replaced(scanner) -> None:
    bundle = _render_bundle(scanner)
    bundle["observations"][0]["layers"]["rgb"]["sha256"] = _hash_text("stale")
    with pytest.raises(VisualizationError, match="sha256 is stale or incorrect"):
        generate_viewer(scanner, render_bundle=_rehash(bundle))


def test_surface_extent_must_match_the_bound_physical_solid(scanner) -> None:
    bundle = _render_bundle(scanner)
    digest = bundle["solid_bindings"][0]["surface_sha256"]
    surface = bundle["surfaces"].pop(digest)
    surface["vertices_m"][0][0] *= 2.0
    surface["sha256"] = content_sha256(surface)
    bundle["surfaces"][surface["sha256"]] = surface
    bundle["solid_bindings"][0]["surface_sha256"] = surface["sha256"]
    bundle["observations"][0]["layers"]["reconstruction"]["surface_sha256"] = surface["sha256"]
    with pytest.raises(VisualizationError, match="surface extent axis"):
        generate_viewer(scanner, render_bundle=_rehash(bundle))


def test_shape_artifact_ctmesh_is_hash_verified_and_materialized(tmp_path: Path) -> None:
    source_path = tmp_path / "source.procedural"
    source_path.write_text("deterministic fixture", encoding="utf-8")
    fixture = _box_surface([0.2, 0.1, 0.05], "materializer")
    mesh = TriangleMesh(fixture["vertices_m"], fixture["triangles"]).with_computed_normals()
    mesh_path = mesh.write(tmp_path / "render.ctmesh")
    artifact = ShapeArtifact(
        id="fixture.shape",
        version="1.0.0",
        sources=(
            SourceRepresentation(
                "fixture_source",
                "procedural",
                ContentReference.from_path(
                    source_path,
                    relative_to=tmp_path,
                    media_type="text/plain",
                ),
                1.0,
            ),
        ),
        surfaces=(
            SurfaceRepresentation(
                "render",
                "ctmesh",
                ContentReference.from_path(
                    mesh_path,
                    relative_to=tmp_path,
                    media_type="application/vnd.contraption.ctmesh",
                ),
                ("render", "ray_trace"),
                len(mesh.vertices_m),
                len(mesh.triangles),
                (-0.1, -0.05, -0.025, 0.1, 0.05, 0.025),
                True,
                True,
                ("paint",),
            ),
        ),
        optical_materials=(
            OpticalMaterial(
                id="paint",
                base_color_linear_rgba=(0.1, 0.4, 0.8, 1.0),
            ),
        ),
    )
    manifest = artifact.write(tmp_path / "shape.json")
    pose = {
        "translation_m": [0.0, 0.0, 0.0],
        "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    scene = {
        "schema": "contraption.physical-scene/v1",
        "assembly_sha256": ASSEMBLY_HASH,
        "contraption_id": "fixture",
        "components": [
            {
                "id": "part",
                "part": "fixture.part",
                "model": "fixture.model",
                "physical_role": "part",
                "bodies": [
                    {
                        "id": "body",
                        "local_pose": pose,
                        "solids": [
                            {
                                "id": "solid",
                                "geometry": {
                                    "kind": "shape",
                                    "dimensions_m": [0.2, 0.1, 0.05],
                                    "shape_uri": "shape.json",
                                    "shape_sha256": "sha256:" + hashlib.sha256(
                                        manifest.read_bytes()
                                    ).hexdigest(),
                                    "surface_id": "render",
                                },
                                "local_pose": pose,
                                "provenance": {
                                    "kind": "derived",
                                    "source": "fixture",
                                    "reference": None,
                                },
                            }
                        ],
                    }
                ],
                "connectors": [],
            }
        ],
        "connections": [],
        "body_poses": {"part/body": pose},
        "connector_poses": {},
    }
    bundle = materialize_render_bundle(
        assembly_sha256=ASSEMBLY_HASH,
        scene=scene,
        solid_shapes={"part/body/solid": manifest},
    )
    surface = next(iter(bundle["surfaces"].values()))
    assert surface["shape_artifact_sha256"] == "sha256:" + artifact.artifact_sha256
    assert surface["source_surface_sha256"] == "sha256:" + artifact.surfaces[0].content.sha256
    assert surface["materials"][0]["id"] == "paint"
    assert np.allclose(surface["vertices_m"], mesh.vertices_m)

    mesh_path.write_bytes(mesh_path.read_bytes() + b"tamper")
    with pytest.raises(RenderBundleError, match="content length mismatch"):
        materialize_render_bundle(
            assembly_sha256=ASSEMBLY_HASH,
            scene=scene,
            solid_shapes={"part/body/solid": manifest},
        )
