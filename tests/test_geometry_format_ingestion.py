from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from contraption.part_import import deterministic_assets as deterministic_assets_module
from contraption.part_import.archive_ingestion import (
    DeterministicArchiveError,
    extract_shape_archive,
)
from contraption.part_import.deterministic_assets import (
    DeterministicAssetError,
    bundle_staged_plan,
    modeling_context_paths,
    stage_plan,
)
from contraption.part_import.dxf_ingestion import DeterministicDxfError, extract_dxf
from contraption.shape import ShapeArtifact, TessellatedShape, TriangleMesh
from contraption.shape.backends import (
    GeometryBackendError,
    TrimeshTessellator,
    automatic_tessellator,
    backend_identity,
    native_ply_tessellator,
)


ROOT = Path(__file__).resolve().parents[1]
SHAPE_FIXTURES = ROOT / "tests" / "fixtures" / "shape"


def _dxf(*entity_pairs: tuple[int, str]) -> bytes:
    pairs = [
        (0, "SECTION"),
        (2, "HEADER"),
        (9, "$INSUNITS"),
        (70, "4"),
        (0, "ENDSEC"),
        (0, "SECTION"),
        (2, "ENTITIES"),
        *entity_pairs,
        (0, "ENDSEC"),
        (0, "EOF"),
    ]
    return ("\n".join(value for pair in pairs for value in (str(pair[0]), pair[1])) + "\n").encode()


def _component(
    source: str,
    *,
    archive_member: str | None = None,
    source_format: str | None = None,
) -> dict:
    if source_format is not None:
        return {
            "deterministic_ingestion": {
                "format": "deterministic-part-ingestion-1",
                "documents": [{"source": source, "source_format": source_format}],
            }
        }
    shape = {
        "source": source,
        "catalog_directory": "mechanical/test/instantiations/fixture",
        "body": "body",
        "solid": "solid",
        "artifact_id": "fixture.geometry",
        "metres_per_source_unit": 1.0,
    }
    if archive_member is not None:
        shape["archive_member"] = archive_member
    return {
        "deterministic_ingestion": {
            "format": "deterministic-part-ingestion-1",
            "shapes": [shape],
        }
    }


def _candidate(root: Path) -> Path:
    target = root / "mechanical/test/instantiations/fixture"
    target.mkdir(parents=True)
    (target / "static.part").write_text(
        json.dumps(
            {
                "bodies": [
                    {
                        "id": "body",
                        "solids": [{"id": "solid", "geometry": {"kind": "box"}}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return target


def test_ascii_dxf_extraction_preserves_every_pair_and_units(tmp_path: Path) -> None:
    source = tmp_path / "outline.dxf"
    source.write_bytes(
        _dxf(
            (0, "LINE"),
            (8, "CUT"),
            (10, "1.25"),
            (20, "-2.5"),
            (11, "3.0"),
            (21, "4.0"),
        )
    )

    extracted = extract_dxf(source)

    assert extracted["format"] == "deterministic-dxf-extraction-1"
    assert extracted["parser"]["lossless_group_pairs"] is True
    assert extracted["drawing_units"] == {"code": 4, "name": "millimetre"}
    assert extracted["entity_counts"] == {"LINE": 1}
    assert extracted["pairs"][-1] == {"code": 0, "value": "EOF"}

    component = tmp_path / "component.json"
    component.write_text(json.dumps(_component(source.name, source_format="dxf")))
    plan = stage_plan(component, tmp_path / "staged")
    contexts = modeling_context_paths(plan)
    assert len(contexts) == 1
    assert json.loads(contexts[0].read_text())["pairs"] == extracted["pairs"]
    assert not tuple(plan.parent.rglob("*.dxf"))


def test_dxf_rejects_non_nfkc_and_binary_sources(tmp_path: Path) -> None:
    noncanonical = tmp_path / "noncanonical.dxf"
    noncanonical.write_bytes(_dxf((0, "TEXT"), (1, "\u212b")))
    with pytest.raises(DeterministicDxfError, match="NFKC"):
        extract_dxf(noncanonical)

    binary = tmp_path / "binary.dxf"
    binary.write_bytes(b"AutoCAD Binary DXF\r\n\x1a\x00")
    with pytest.raises(DeterministicDxfError, match="binary DXF"):
        extract_dxf(binary)


@pytest.mark.parametrize("unsafe_name", ["../escape.stl", "mesh\\escape.stl"])
def test_zip_rejects_unsafe_paths_and_cleans_destination(
    tmp_path: Path, unsafe_name: str
) -> None:
    source = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(unsafe_name, b"solid empty\nendsolid empty\n")
    destination = tmp_path / "extracted"

    with pytest.raises(DeterministicArchiveError, match="(unsafe|POSIX)"):
        extract_shape_archive(source, destination, member=unsafe_name)

    assert not destination.exists()


def test_zip_cleans_destination_after_unexpected_extraction_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "valid.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mesh.stl", b"solid empty\nendsolid empty\n")
    destination = tmp_path / "extracted"

    def fail_validation(*_args, **_kwargs):
        raise LookupError("fixture failure")

    monkeypatch.setattr(
        "contraption.part_import.archive_ingestion._validated_infos",
        fail_validation,
    )
    with pytest.raises(DeterministicArchiveError, match="ZIP extraction failed"):
        extract_shape_archive(source, destination, member="mesh.stl")

    assert not destination.exists()


def test_zip_hash_validation_and_extraction_share_one_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "shape.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mesh.stl", b"original-snapshot")
    original = source.read_bytes()
    replacement = tmp_path / "replacement.zip"
    with zipfile.ZipFile(
        replacement, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("mesh.stl", b"replacement-path-bytes")
    replacement_bytes = replacement.read_bytes()
    real_zip_file = zipfile.ZipFile
    opened = False

    def swap_path_then_open(snapshot, *args, **kwargs):
        nonlocal opened
        opened = True
        assert isinstance(snapshot, io.BytesIO)
        source.write_bytes(replacement_bytes)
        return real_zip_file(snapshot, *args, **kwargs)

    monkeypatch.setattr(
        "contraption.part_import.archive_ingestion.zipfile.ZipFile",
        swap_path_then_open,
    )
    extracted = extract_shape_archive(
        source,
        tmp_path / "extracted",
        member="mesh.stl",
    )

    assert opened is True
    assert extracted.selected_path.read_bytes() == b"original-snapshot"
    assert extracted.archive_sha256 == hashlib.sha256(original).hexdigest()


def test_scanned_shape_zip_is_canonicalized_before_bundle(tmp_path: Path) -> None:
    source = tmp_path / "scan.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "meshes/model.stl",
            (SHAPE_FIXTURES / "plain_tetrahedron.stl").read_bytes(),
        )
        archive.writestr("metadata.pbtxt", b"name: 'fixture'\n")
    component = tmp_path / "component.json"
    component.write_text(
        json.dumps(_component(source.name, archive_member="meshes/model.stl"))
    )

    plan = stage_plan(component, tmp_path / "staged")
    staged = json.loads(plan.read_text())
    shape = staged["shapes"][0]
    assert shape["archive"]["sha256"]
    assert shape["backend"] == {"id": "contraption-native-stl", "version": "1"}
    assert "canonical.ctmesh" in shape["prepared_sha256"]
    assert modeling_context_paths(plan) == ()

    target = _candidate(tmp_path / "candidate")
    bundle_staged_plan(tmp_path / "candidate", plan)
    artifact = ShapeArtifact.load(target / "shape/solid/shape.artifact.json")
    assert artifact.provenance["backend"] == shape["backend"]
    assert artifact.provenance["archive"]["member"] == "meshes/model.stl"


def test_staged_artifact_provenance_must_bind_backend(tmp_path: Path) -> None:
    source = tmp_path / "tetra.ply"
    source.write_bytes(_ply())
    component = tmp_path / "component.json"
    component.write_text(json.dumps(_component(source.name)), encoding="utf-8")
    plan = stage_plan(component, tmp_path / "staged")
    staged = json.loads(plan.read_text(encoding="utf-8"))
    manifest = plan.parent / staged["shapes"][0]["prepared_root"] / "shape.artifact.json"
    artifact = json.loads(manifest.read_text(encoding="utf-8"))
    artifact["provenance"]["backend"]["id"] = "tampered-backend"
    manifest.write_text(json.dumps(artifact), encoding="utf-8")
    staged["shapes"][0]["prepared_sha256"]["shape.artifact.json"] = __import__(
        "hashlib"
    ).sha256(manifest.read_bytes()).hexdigest()
    plan.write_text(json.dumps(staged), encoding="utf-8")

    with pytest.raises(DeterministicAssetError, match="artifact provenance changed"):
        modeling_context_paths(plan)


def test_textured_scanned_object_zip_fails_before_luna(tmp_path: Path) -> None:
    source = tmp_path / "textured.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "meshes/model.obj",
            "mtllib model.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        )
        archive.writestr("meshes/model.mtl", "newmtl scanned\nmap_Kd texture.png\n")
        archive.writestr("materials/textures/texture.png", b"not-decoded")
    component = tmp_path / "component.json"
    component.write_text(
        json.dumps(_component(source.name, archive_member="meshes/model.obj"))
    )

    with pytest.raises(DeterministicAssetError, match="texture/UV"):
        stage_plan(component, tmp_path / "staged")


def _ply(face_property: str = "property list uchar int vertex_indices") -> bytes:
    return (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 4\n"
        f"{face_property}\n"
        "end_header\n"
        "0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        "3 0 2 1\n3 0 1 3\n3 1 2 3\n3 2 0 3\n"
    ).encode()


def test_native_ply_is_exact_bounded_and_versioned(tmp_path: Path) -> None:
    source = tmp_path / "tetra.ply"
    source.write_bytes(_ply())

    converted = native_ply_tessellator(source, 0.001)
    assert len(converted.mesh.vertices_m) == 4
    assert len(converted.mesh.triangles) == 4
    automatic = automatic_tessellator(source)
    assert automatic is not None
    assert backend_identity(automatic) == {
        "id": "contraption-native-ply",
        "version": "1",
    }


@pytest.mark.parametrize(
    "property_line",
    [
        "property list float int vertex_indices",
        "property list uchar float vertex_indices",
    ],
)
def test_ply_rejects_float_counts_or_indices(
    tmp_path: Path, property_line: str
) -> None:
    source = tmp_path / "invalid.ply"
    source.write_bytes(_ply(property_line))
    with pytest.raises(GeometryBackendError, match="integer types"):
        native_ply_tessellator(source, 1.0)


def test_ply_rejects_unpreserved_vertex_properties(tmp_path: Path) -> None:
    source = tmp_path / "intensity.ply"
    source.write_bytes(
        _ply().replace(
            b"property float z\n",
            b"property float z\nproperty float intensity\n",
        )
    )

    with pytest.raises(GeometryBackendError, match="cannot be preserved.*intensity"):
        native_ply_tessellator(source, 1.0)


@pytest.mark.parametrize("uri", ["https://example.invalid/mesh.bin", "../escape.bin"])
def test_gltf_rejects_unsafe_uri_before_importing_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uri: str,
) -> None:
    source = tmp_path / "unsafe.gltf"
    source.write_text(
        json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": uri}]}),
        encoding="utf-8",
    )
    imported = False

    def forbidden_import(_name: str):
        nonlocal imported
        imported = True
        raise AssertionError("trimesh import must follow URI validation")

    monkeypatch.setattr("contraption.shape.backends.importlib.import_module", forbidden_import)

    with pytest.raises(GeometryBackendError, match="(local relative URI|escapes)"):
        TrimeshTessellator()(source, 1.0)
    assert imported is False


def test_vrml_rejects_active_content_before_importing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "active.vrml"
    source.write_text("#VRML V2.0 utf8\nScript { }\n", encoding="utf-8")
    imported = False

    def forbidden_import(_name: str):
        nonlocal imported
        imported = True
        raise AssertionError("trimesh import must follow VRML preflight")

    monkeypatch.setattr("contraption.shape.backends.importlib.import_module", forbidden_import)

    with pytest.raises(GeometryBackendError, match="executable, external, or textured"):
        TrimeshTessellator()(source, 1.0)
    assert imported is False


class _FakeTextureVisual:
    kind = "texture"
    uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    material = object()


class _FakeGeometry:
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    faces = [[0, 1, 2]]
    visual = _FakeTextureVisual()

    def copy(self):
        return self

    def apply_transform(self, _transform) -> None:
        return None


class _FakeGraph:
    nodes_geometry = ["node"]

    def get(self, _node):
        return (
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "geometry",
        )


class _FakeScene:
    graph = _FakeGraph()
    geometry = {"geometry": _FakeGeometry()}


class _FakeTrimeshModule:
    @staticmethod
    def load(*_args, **_kwargs):
        return _FakeScene()


def test_scene_rejects_uv_coordinates_without_texture_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "uv.gltf"
    source.write_text(json.dumps({"asset": {"version": "2.0"}}), encoding="utf-8")
    monkeypatch.setattr(
        "contraption.shape.backends.importlib.import_module",
        lambda _name: _FakeTrimeshModule(),
    )

    with pytest.raises(GeometryBackendError, match="UV coordinates"):
        TrimeshTessellator()(source, 1.0)


class _VersionedStepBackend:
    backend_id = "fixture-step-kernel"
    backend_version = "9.1.2"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _path: Path, scale: float) -> TessellatedShape:
        self.calls += 1
        return TessellatedShape(
            TriangleMesh(
                [[0, 0, 0], [scale, 0, 0], [0, scale, 0]],
                [[0, 1, 2]],
            ).with_computed_normals()
        )


class _VersionedGltfBackend:
    backend_id = "fixture-gltf-kernel"
    backend_version = "2.0.0"

    def __call__(self, path: Path, scale: float) -> TessellatedShape:
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["buffers"] == [{"uri": "scene.bin"}]
        return TessellatedShape(
            TriangleMesh(
                [[0, 0, 0], [scale, 0, 0], [0, scale, 0]],
                [[0, 1, 2]],
            ).with_computed_normals(),
            (),
            (path.parent / "scene.bin",),
        )


def test_shape_conversion_and_evidence_use_private_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "scene.gltf"
    original = json.dumps(
        {"asset": {"version": "2.0"}, "buffers": [{"uri": "scene.bin"}]}
    )
    source.write_text(original, encoding="utf-8")
    (tmp_path / "scene.bin").write_bytes(b"original-linked-snapshot")
    component = tmp_path / "component.json"
    component.write_text(json.dumps(_component(source.name)), encoding="utf-8")
    real_snapshot = deterministic_assets_module._snapshot_regular_file
    mutated = False

    def snapshot_then_mutate_original(
        source_path: Path, target: Path, context: str
    ) -> None:
        nonlocal mutated
        real_snapshot(source_path, target, context)
        if source_path.resolve() == source.resolve():
            source.write_text(
                json.dumps(
                    {
                        "asset": {"version": "2.0"},
                        "buffers": [{"uri": "https://example.invalid/swapped.bin"}],
                    }
                ),
                encoding="utf-8",
            )
            mutated = True

    monkeypatch.setattr(
        deterministic_assets_module,
        "_snapshot_regular_file",
        snapshot_then_mutate_original,
    )
    plan = stage_plan(
        component,
        tmp_path / "staged",
        tessellator=_VersionedGltfBackend(),
    )

    assert mutated is True
    staged = json.loads(plan.read_text(encoding="utf-8"))
    staged_source = plan.parent / staged["shapes"][0]["source"]
    assert staged_source.read_text(encoding="utf-8") == original
    artifact = ShapeArtifact.load(
        plan.parent
        / staged["shapes"][0]["prepared_root"]
        / "shape.artifact.json"
    )
    source_payloads = {
        item.content.uri: artifact.resolve(item.content).read_bytes()
        for item in artifact.sources
    }
    assert any(payload == original.encode("utf-8") for payload in source_payloads.values())
    assert any(payload == b"original-linked-snapshot" for payload in source_payloads.values())


def test_explicit_cad_backend_runs_only_before_luna(tmp_path: Path) -> None:
    source = tmp_path / "fixture.step"
    source.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n")
    component = tmp_path / "component.json"
    component.write_text(json.dumps(_component(source.name)))
    backend = _VersionedStepBackend()

    plan = stage_plan(component, tmp_path / "staged", tessellator=backend)
    staged = json.loads(plan.read_text())
    assert staged["shapes"][0]["backend"] == {
        "id": "fixture-step-kernel",
        "version": "9.1.2",
    }
    assert backend.calls == 1

    target = _candidate(tmp_path / "candidate")
    bundle_staged_plan(tmp_path / "candidate", plan)
    assert backend.calls == 1
    assert ShapeArtifact.load(target / "shape/solid/shape.artifact.json")


def test_missing_cad_backend_fails_during_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixture.step"
    source.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n")
    component = tmp_path / "component.json"
    component.write_text(json.dumps(_component(source.name)))
    monkeypatch.setattr(
        "contraption.part_import.deterministic_assets.automatic_tessellator",
        lambda _source: None,
    )

    with pytest.raises(DeterministicAssetError, match="before Luna dispatch"):
        stage_plan(component, tmp_path / "staged")
