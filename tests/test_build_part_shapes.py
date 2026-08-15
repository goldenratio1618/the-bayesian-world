from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = runpy.run_path(
    str(ROOT / "scripts" / "build_part_shapes.py"),
    run_name="contraption_build_part_shapes_test",
)
ShapeBuildError = BUILDER["ShapeBuildError"]
build = BUILDER["build"]
detailed_mesh = BUILDER["detailed_mesh"]


def _static_part(part_id: str, solid_id: str) -> dict:
    return {
        "id": part_id,
        "version": "1.0.0",
        "bodies": [
            {
                "id": "body",
                "solids": [
                    {
                        "id": solid_id,
                        "geometry": {
                            "kind": "box",
                            "dimensions_m": [0.01, 0.02, 0.03],
                        },
                    }
                ],
            }
        ],
    }


def test_detailed_shape_builder_rejects_unknown_parts_and_solids() -> None:
    with pytest.raises(ShapeBuildError, match="no reviewed detailed-shape recipe"):
        detailed_mesh("unknown.part", "envelope", (0.01, 0.02, 0.03))

    with pytest.raises(ShapeBuildError, match="covers solid 'envelope'"):
        detailed_mesh(
            "generic-100ohm-resistor",
            "unexpected_solid",
            (0.01, 0.02, 0.03),
        )


def test_scanner_target_is_a_regular_twenty_face_icosahedron() -> None:
    mesh, features = detailed_mesh(
        "scanner.icosahedron",
        "icosahedron",
        (0.12, 0.12, 0.12),
    )

    assert len(mesh.vertices_m) == 12
    assert len(mesh.triangles) == 20
    assert mesh.watertight
    assert mesh.manifold
    assert mesh.dimensions_m == pytest.approx((0.12, 0.12, 0.12))
    assert set(mesh.face_material.tolist()) == {8}
    assert "twenty triangular scan faces" in features


def test_catalog_build_preflights_all_recipes_before_writing(tmp_path: Path) -> None:
    known = tmp_path / "a-known"
    unknown = tmp_path / "z-unknown"
    known.mkdir()
    unknown.mkdir()
    (known / "static.part").write_text(
        json.dumps(_static_part("generic-100ohm-resistor", "envelope")),
        encoding="utf-8",
    )
    (unknown / "static.part").write_text(
        json.dumps(_static_part("unreviewed.new-part", "envelope")),
        encoding="utf-8",
    )

    with pytest.raises(ShapeBuildError, match="unreviewed.new-part"):
        build(tmp_path)

    assert not (known / "shape").exists()
    assert not (unknown / "shape").exists()


def test_shipped_catalog_has_a_reviewed_recipe_for_every_solid() -> None:
    assert build(ROOT / "model_catalog", check=True) == 0
