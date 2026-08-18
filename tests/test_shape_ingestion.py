from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from contraption.part_import.agents import ModelingAgent
from contraption.part_import.deterministic_assets import bundle_staged_plan, stage_plan
from contraption.shape import (
    OpticalMaterial,
    ShapeArtifact,
    TessellatedShape,
    import_shape,
    mass_properties,
)
from contraption.shape.artifacts import ShapeArtifactError
from contraption.shape.mesh import MeshError, TriangleMesh


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "shape"


def test_existing_3d_surface_import_is_source_independent_and_mechanical(tmp_path: Path) -> None:
    result = import_shape(
        FIXTURES / "plain_tetrahedron.stl",
        tmp_path / "shape",
        artifact_id="fixture.tetrahedron",
        metres_per_source_unit=0.01,
        density_kg_m3=1200.0,
    )
    artifact = ShapeArtifact.load(result.manifest_path)
    mesh = artifact.load_surface("analysis")
    assert mesh.watertight
    assert len(mesh.triangles) == 4
    assert np.allclose(mesh.dimensions_m, (0.01, 0.01, 0.01))
    properties = mass_properties(mesh, 1200.0)
    assert properties.volume_m3 == pytest.approx(1.0e-6 / 6.0)
    assert properties.center_of_mass_m == pytest.approx((0.0025, 0.0025, 0.0025))
    assert artifact.derived_mass_properties is not None
    assert artifact.sources[0].format == "stl"
    assert artifact.surfaces[0].content.uri == "canonical.ctmesh"
    assert artifact.caches[0].uri == "runtime.glb"
    assert artifact.surfaces[0].uncertainty.distribution == "uniform"
    assert artifact.surfaces[0].uncertainty.parameters["lower_m"] < 0
    assert artifact.derived_mass_properties.uncertainty == artifact.surfaces[0].uncertainty

    # Runtime behavior never depends on the source bytes after canonicalization.
    result.imported_sources[0].write_text("mutated source evidence", encoding="utf-8")
    with pytest.raises(ShapeArtifactError, match="(hash|length) mismatch"):
        ShapeArtifact.load(result.manifest_path)
    mesh_again = type(mesh).read(result.manifest_path.parent / "canonical.ctmesh")
    assert np.array_equal(mesh_again.triangles, mesh.triangles)


def test_shape_manifest_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    result = import_shape(
        FIXTURES / "plain_tetrahedron.stl",
        tmp_path / "shape",
        artifact_id="fixture.tetrahedron",
        metres_per_source_unit=1.0,
    )
    source = result.manifest_path.read_text(encoding="utf-8")
    result.manifest_path.write_text(
        '{"id":"ambiguous",' + source.lstrip()[1:], encoding="utf-8"
    )
    with pytest.raises(ShapeArtifactError, match="duplicate JSON field 'id'"):
        ShapeArtifact.load(result.manifest_path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("vertex_count", 5, "vertex-count"),
        ("triangle_count", 5, "triangle-count"),
        ("bounds_m", [0, 0, 0, 2, 1, 1], "bounds"),
        ("watertight", False, "watertight"),
        ("manifold", False, "manifold"),
    ],
)
def test_shape_verify_rejects_surface_manifest_tamper(
    tmp_path: Path, field: str, replacement, message: str
) -> None:
    result = import_shape(
        FIXTURES / "plain_tetrahedron.stl",
        tmp_path / "shape",
        artifact_id="fixture.tetrahedron",
        metres_per_source_unit=1.0,
    )
    value = json.loads(result.manifest_path.read_text())
    value["surfaces"][0][field] = replacement
    result.manifest_path.write_text(json.dumps(value))
    with pytest.raises(ShapeArtifactError, match=message):
        ShapeArtifact.load(result.manifest_path)


def test_shape_verify_rejects_derived_mass_property_tamper(tmp_path: Path) -> None:
    result = import_shape(
        FIXTURES / "plain_tetrahedron.stl",
        tmp_path / "shape",
        artifact_id="fixture.tetrahedron",
        metres_per_source_unit=0.01,
        density_kg_m3=1200.0,
    )
    value = json.loads(result.manifest_path.read_text())
    value["derived_mass_properties"]["mass_kg"] *= 1.1
    result.manifest_path.write_text(json.dumps(value))
    with pytest.raises(ShapeArtifactError, match="derived mass"):
        ShapeArtifact.load(result.manifest_path)


def test_distinct_optical_physics_are_imported_without_an_agent(tmp_path: Path) -> None:
    result = import_shape(
        FIXTURES / "optical_cube.obj",
        tmp_path / "glass",
        artifact_id="fixture.optical-glass",
        metres_per_source_unit=0.02,
    )
    material = result.artifact.optical_materials[0]
    assert material.id == "optical-glass"
    assert material.model == "dielectric"
    assert material.transmission == pytest.approx(0.92)
    assert material.refractive_index == pytest.approx(1.52)
    assert material.roughness < 0.1
    assert material.provenance["format"] == "mtl"
    assert material.uncertainty.distribution == "uniform"
    assert {path.suffix for path in result.imported_sources} == {".obj", ".mtl"}


def test_external_3d_backend_must_bundle_available_optical_properties(tmp_path: Path) -> None:
    source = tmp_path / "vendor.glb"
    source.write_bytes(b"vendor geometry evidence")
    mesh = TriangleMesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 2]],
        face_material=[0],
    ).with_computed_normals()

    def tessellate(path: Path, scale: float) -> TessellatedShape:
        assert path == source.resolve()
        assert scale == 0.001
        return TessellatedShape(
            mesh,
            (
                OpticalMaterial(
                    "vendor-glass",
                    model="dielectric",
                    transmission=0.9,
                    refractive_index=1.51,
                ),
            ),
        )

    result = import_shape(
        source,
        tmp_path / "imported",
        metres_per_source_unit=0.001,
        tessellator=tessellate,
    )
    assert result.artifact.optical_materials[0].id == "vendor-glass"
    assert result.artifact.optical_materials[0].refractive_index == pytest.approx(1.51)

    with pytest.raises(ShapeArtifactError, match="must return TessellatedShape"):
        import_shape(
            source,
            tmp_path / "invalid",
            tessellator=lambda _path, _scale: mesh,
        )


def test_ctmesh_decoder_rejects_noncanonical_or_ambiguous_bytes() -> None:
    mesh = TriangleMesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2]],
    ).with_computed_normals()
    payload = mesh.to_bytes()
    assert np.array_equal(TriangleMesh.from_bytes(payload).triangles, mesh.triangles)

    with pytest.raises(MeshError, match="trailing or unreferenced"):
        TriangleMesh.from_bytes(payload + b"ignored")

    magic_length = len(b"CTMESH1\n")
    header_length = struct.unpack_from("<I", payload, magic_length)[0]
    header_start = magic_length + 4
    body_start = header_start + header_length
    header = json.loads(payload[header_start:body_start])
    header["arrays"][0]["offset"] = 4
    altered_header = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    altered = (
        payload[:magic_length]
        + struct.pack("<I", len(altered_header))
        + altered_header
        + payload[body_start:]
    )
    with pytest.raises(MeshError, match="not canonically packed"):
        TriangleMesh.from_bytes(altered)

    with pytest.raises(MeshError, match="zero-area or collinear"):
        TriangleMesh(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0, 1, 2]],
        )


def test_host_bundles_deterministic_shape_with_luna_text_outputs(tmp_path: Path) -> None:
    component_root = tmp_path / "input"
    component_root.mkdir()
    for name in ("optical_cube.obj", "optical_cube.mtl"):
        (component_root / name).write_bytes((FIXTURES / name).read_bytes())
    catalog_directory = "electrical/resistors/fixed_resistors/instantiations/generic-100ohm-resistor"
    component = component_root / "part.json"
    component.write_text(
        json.dumps(
            {
                "manufacturer": "fixture",
                "deterministic_ingestion": {
                    "format": "deterministic-part-ingestion-1",
                    "shapes": [
                        {
                            "source": "optical_cube.obj",
                            "catalog_directory": catalog_directory,
                            "body": "body",
                            "solid": "envelope",
                            "artifact_id": "fixture.luna-glass",
                            "metres_per_source_unit": 0.01,
                            "density_kg_m3": 2500.0,
                            "surface_uncertainty": {
                                "distribution": "normal",
                                "parameters": {"standard_deviation_m": 0.0002},
                                "correlation_group": "fixture-metrology",
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    plan = stage_plan(component, run_root / "deterministic-assets")
    assert plan is not None
    proposed = run_root / "proposed"
    source_part = ROOT / "model_catalog" / catalog_directory
    target_part = proposed / catalog_directory
    target_part.mkdir(parents=True)
    # These bytes stand in for Luna's validated text-only structured response.
    (target_part / "static.part").write_bytes((source_part / "static.part").read_bytes())
    (target_part / "v1.model").write_bytes((source_part / "v1.model").read_bytes())

    written = bundle_staged_plan(proposed, plan)
    assert written
    manifest = target_part / "shape" / "envelope" / "shape.artifact.json"
    artifact = ShapeArtifact.load(manifest)
    assert artifact.optical_materials[0].model == "dielectric"
    assert artifact.surfaces[0].uncertainty.distribution == "normal"
    assert artifact.surfaces[0].uncertainty.parameters["standard_deviation_m"] == 0.0002
    assert artifact.surfaces[0].uncertainty.correlation_group == "fixture-metrology"
    static = json.loads((target_part / "static.part").read_text(encoding="utf-8"))
    geometry = static["bodies"][0]["solids"][0]["geometry"]
    assert geometry["kind"] == "shape"
    assert geometry["shape_sha256"] == "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest()
    ModelingAgent.validate_artifacts(proposed, catalog_root=ROOT / "model_catalog")


def test_host_bundles_deterministic_optical_sensor_with_luna_text_outputs(
    tmp_path: Path,
) -> None:
    component_root = tmp_path / "input"
    component_root.mkdir()
    catalog_directory = (
        "optical/cameras/powered_rotational_cameras/instantiations/scanner_camera"
    )
    source_part = ROOT / "model_catalog" / catalog_directory
    source_descriptor = source_part / "sensor.optical.json"
    staged_descriptor = component_root / "calibrated-camera.optical.json"
    staged_descriptor.write_bytes(source_descriptor.read_bytes())
    component = component_root / "part.json"
    component.write_text(
        json.dumps(
            {
                "manufacturer": "fixture",
                "deterministic_ingestion": {
                    "format": "deterministic-part-ingestion-1",
                    "optical_sensors": [
                        {
                            "source": staged_descriptor.name,
                            "catalog_directory": catalog_directory,
                            "body": "camera",
                            "pose_connector": "optical_axis",
                            "artifact_port": "optical_observation",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "run"
    plan = stage_plan(component, run_root / "deterministic-assets")
    assert plan is not None
    proposed = run_root / "proposed"
    target_part = proposed / catalog_directory
    target_part.mkdir(parents=True)
    (target_part / "static.part").write_bytes((source_part / "static.part").read_bytes())
    (target_part / "v1.model").write_bytes((source_part / "v1.model").read_bytes())

    written = bundle_staged_plan(proposed, plan)
    bundled_descriptor = target_part / "sensor.optical.json"
    assert bundled_descriptor in written
    assert bundled_descriptor.read_bytes() == source_descriptor.read_bytes()
    static = json.loads((target_part / "static.part").read_text(encoding="utf-8"))
    assert static["optical_sensors"] == [
        {
            "id": "scanner.camera.optical",
            "body": "camera",
            "pose_connector": "optical_axis",
            "artifact_port": "optical_observation",
            "descriptor_uri": "sensor.optical.json",
            "descriptor_sha256": "sha256:"
            + hashlib.sha256(bundled_descriptor.read_bytes()).hexdigest(),
        }
    ]
    ModelingAgent.validate_artifacts(proposed, catalog_root=ROOT / "model_catalog")


def test_shipped_physical_solids_use_detailed_uncertainty_aware_shapes() -> None:
    manifests: list[Path] = []
    for static_path in (ROOT / "model_catalog").rglob("static.part"):
        static = json.loads(static_path.read_text(encoding="utf-8"))
        for body in static.get("bodies", []):
            for solid in body.get("solids", []):
                geometry = solid["geometry"]
                assert geometry["kind"] == "shape"
                manifest = static_path.parent / geometry["shape_uri"]
                artifact = ShapeArtifact.load(manifest, verify_content=True)
                surface = next(
                    item
                    for item in artifact.surfaces
                    if item.id == geometry["surface_id"]
                )
                assert surface.kind == "ctmesh"
                assert {"analysis", "ray_trace", "render", "collision"} <= set(
                    surface.purposes
                )
                if artifact.provenance.get("kind") == "deterministic-procedural":
                    assert surface.uncertainty.distribution != "fixed"
                    assert surface.uncertainty.parameters["standard_deviation_m"] > 0
                    assert all(
                        material.uncertainty.distribution != "fixed"
                        for material in artifact.optical_materials
                    )
                    assert artifact.metadata["geometric_fidelity"].endswith(
                        "not vendor CAD"
                    )
                manifests.append(manifest)
    assert manifests and len(manifests) == len(set(manifests))
