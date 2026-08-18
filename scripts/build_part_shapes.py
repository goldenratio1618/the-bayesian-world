#!/usr/bin/env python3
"""Build reproducible detailed canonical shapes for every catalog solid."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from contraption.shape.artifacts import (
    ContentReference,
    OpticalMaterial,
    ShapeArtifact,
    ShapeUncertainty,
    SourceRepresentation,
    SurfaceRepresentation,
)
from contraption.shape.mesh import TriangleMesh, box_mesh, combine_meshes, cylinder_mesh, sphere_mesh


class ShapeBuildError(ValueError):
    """Raised when the procedural catalog builder has no reviewed shape recipe."""


_SUPPORTED_SOLIDS = {
    "C1210C476K8RAC": "envelope",
    "scanner.control_board": "pcb",
    "scanner.servo_regulator": "regulator_board",
    "generic-100ohm-resistor": "envelope",
    "scanner.compute": "compute_board",
    "scanner.nimh_battery": "battery_envelope",
    "scanner.encoder_pair": "encoder_board",
    "scanner.gearmotor": "motor_case",
    "scanner.position_servo": "servo_case",
    "scanner.romi_chassis": "deck",
    "generic-camera-module": "envelope",
    "scanner.arm_linkage": "linkage",
    "scanner.wheel": "tire",
    "scanner.camera": "camera_module",
    "scanner.icosahedron": "icosahedron",
    "yageo-rc0603-10k": "package",
    "yageo-rc0603-220r": "package",
    "yageo-rc0603-47r": "package",
    "yageo_rc0603_100k": "package",
    "yageo_rc0603_100r": "envelope",
    "yageo_rc0603_10r": "envelope",
    "yageo_rc0603_1k": "package",
    "yageo_rc0603_1m": "package",
    "yageo_rc0603_47k": "package",
    "yageo_rc0603_4k7": "package",
}
_RC0603_CHIP_RESISTORS = frozenset(part for part in _SUPPORTED_SOLIDS if part.startswith("yageo"))


def _require_supported(part_id: str, solid_id: str) -> None:
    expected = _SUPPORTED_SOLIDS.get(part_id)
    if expected is None:
        raise ShapeBuildError(f"no reviewed detailed-shape recipe for part {part_id!r}")
    if solid_id != expected:
        raise ShapeBuildError(
            f"part {part_id!r} shape recipe covers solid {expected!r}, not {solid_id!r}"
        )


def transform(translation=(0.0, 0.0, 0.0), rotation: np.ndarray | None = None) -> np.ndarray:
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


_AXIS_ROTATIONS = {
    "z": np.eye(3),
    "x": np.asarray(((0, 0, 1), (0, 1, 0), (-1, 0, 0)), dtype=float),
    "y": np.asarray(((1, 0, 0), (0, 0, 1), (0, -1, 0)), dtype=float),
}


def materialized(mesh: TriangleMesh, material: int) -> TriangleMesh:
    return TriangleMesh(
        mesh.vertices_m,
        mesh.triangles,
        mesh.vertex_normals,
        mesh.vertex_rgba_linear,
        np.full(len(mesh.triangles), material, dtype=np.uint32),
    )


def box(parts, size, center=(0, 0, 0), material=0, rotation=None):
    parts.append((materialized(box_mesh(size), material), transform(center, rotation)))


def cylinder(parts, diameter, length, center=(0, 0, 0), material=0, axis="z", segments=32):
    parts.append((materialized(cylinder_mesh(diameter, length, segments), material), transform(center, _AXIS_ROTATIONS[axis])))


def sphere(parts, diameter, center=(0, 0, 0), material=0):
    parts.append((materialized(sphere_mesh(diameter), material), transform(center)))


def icosahedron_mesh(size_m: float) -> TriangleMesh:
    """Return a regular icosahedron with an exact cubic AABB of ``size_m``."""

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (0, sy, sz * phi)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
        + [
            (sx, sy * phi, 0)
            for sx in (-1, 1)
            for sy in (-1, 1)
        ]
        + [
            (sx * phi, 0, sz)
            for sx in (-1, 1)
            for sz in (-1, 1)
        ],
        dtype=float,
    )
    vertices *= float(size_m) / (2.0 * phi)
    distance_squared = np.sum(
        (vertices[:, None, :] - vertices[None, :, :]) ** 2,
        axis=2,
    )
    edge_squared = float(np.min(distance_squared[distance_squared > 0]))
    faces: list[tuple[int, int, int]] = []
    for candidate in itertools.combinations(range(len(vertices)), 3):
        pairs = (
            distance_squared[candidate[0], candidate[1]],
            distance_squared[candidate[1], candidate[2]],
            distance_squared[candidate[2], candidate[0]],
        )
        if not all(math.isclose(float(value), edge_squared, rel_tol=1e-12) for value in pairs):
            continue
        left, middle, right = candidate
        normal = np.cross(vertices[middle] - vertices[left], vertices[right] - vertices[left])
        centroid = np.mean(vertices[list(candidate)], axis=0)
        if float(np.dot(normal, centroid)) < 0:
            middle, right = right, middle
        faces.append((left, middle, right))
    if len(faces) != 20:
        raise ShapeBuildError(f"regular icosahedron construction produced {len(faces)} faces")
    return TriangleMesh(vertices, np.asarray(faces, dtype=np.uint32)).with_computed_normals()


def rotate_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)), dtype=float)


def optical_materials() -> tuple[OpticalMaterial, ...]:
    return (
        OpticalMaterial("dark-polymer", "principled", (0.018, 0.022, 0.028, 1), 0.72, uncertainty=ShapeUncertainty("normal", {"std": 0.05}), provenance={"kind": "catalog-class"}),
        OpticalMaterial("metal", "conductor", (0.52, 0.55, 0.58, 1), 0.24, 0.92, extinction_coefficient=3.1, uncertainty=ShapeUncertainty("normal", {"std": 0.08}), provenance={"kind": "catalog-class"}),
        OpticalMaterial("pcb-green", "principled", (0.015, 0.19, 0.045, 1), 0.46, uncertainty=ShapeUncertainty("normal", {"std": 0.04}), provenance={"kind": "catalog-class"}),
        OpticalMaterial("rubber", "principled", (0.008, 0.009, 0.01, 1), 0.94, uncertainty=ShapeUncertainty("normal", {"std": 0.03}), provenance={"kind": "catalog-class"}),
        OpticalMaterial("optical-glass", "dielectric", (0.96, 0.98, 1.0, 0.08), 0.035, transmission=0.94, refractive_index=1.52, absorption_per_m=(0.03, 0.02, 0.015), uncertainty=ShapeUncertainty("normal", {"refractive_index_std": 0.01}), provenance={"kind": "catalog-class"}),
        OpticalMaterial("copper", "conductor", (0.72, 0.31, 0.12, 1), 0.21, 1.0, extinction_coefficient=3.6, uncertainty=ShapeUncertainty("normal", {"std": 0.05}), provenance={"kind": "catalog-class"}),
        OpticalMaterial("ceramic", "dielectric", (0.66, 0.56, 0.38, 1), 0.61, refractive_index=1.7, uncertainty=ShapeUncertainty("uniform", {"roughness_lower": 0.45, "roughness_upper": 0.8}), provenance={"kind": "catalog-class"}),
        OpticalMaterial("battery-wrap", "principled", (0.04, 0.17, 0.48, 1), 0.58, uncertainty=ShapeUncertainty("normal", {"std": 0.05}), provenance={"kind": "catalog-class"}),
    )


def scan_target_material() -> OpticalMaterial:
    return OpticalMaterial(
        "scan-target",
        "principled",
        (0.82, 0.16, 0.035, 1),
        0.34,
        uncertainty=ShapeUncertainty("normal", {"std": 0.03}),
        provenance={"kind": "designed-scanner-target"},
    )


def detailed_mesh(part_id: str, solid_id: str, dimensions: tuple[float, float, float]) -> tuple[TriangleMesh, list[str]]:
    _require_supported(part_id, solid_id)
    x, y, z = dimensions
    parts = []
    features: list[str] = []
    if part_id == "C1210C476K8RAC":
        box(parts, (0.72*x, y, z), material=6)
        box(parts, (0.14*x, y, z), (-0.43*x, 0, 0), 1)
        box(parts, (0.14*x, y, z), (0.43*x, 0, 0), 1)
        features += ["ceramic dielectric body", "two metallized end terminations"]
    elif part_id in _RC0603_CHIP_RESISTORS:
        box(parts, (0.72*x, y, z), material=6)
        box(parts, (0.14*x, y, z), (-0.43*x, 0, 0), 1)
        box(parts, (0.14*x, y, z), (0.43*x, 0, 0), 1)
        box(parts, (0.48*x, 0.72*y, 0.08*z), (0, 0, 0.46*z), 0)
        features += ["ceramic resistor body", "two metallized end terminations", "top resistive coating"]
    elif part_id in {"scanner.control_board", "scanner.servo_regulator", "scanner.compute", "scanner.encoder_pair"}:
        board_thickness = min(z * 0.22, 0.002)
        box(parts, (x, y, board_thickness), (0, 0, -z/2 + board_thickness/2), 2)
        box(parts, (0.26*x, 0.28*y, 0.28*z), (0, 0, -z/2 + board_thickness + 0.14*z), 0)
        box(parts, (0.18*x, 0.24*y, 0.45*z), (-0.41*x, 0, -z/2 + board_thickness + 0.225*z), 1)
        box(parts, (0.16*x, 0.7*y, 0.35*z), (0.4*x, 0, -z/2 + board_thickness + 0.175*z), 0)
        for index, fy in enumerate((-0.31, 0.31)):
            cylinder(parts, min(0.14*x, 0.22*y), 0.35*z, (0.18*x, fy*y, -z/2 + board_thickness + 0.175*z), 1)
        for fx in (-0.24, 0.02, 0.27):
            box(parts, (0.08*x, 0.12*y, 0.18*z), (fx*x, -0.28*y, -z/2 + board_thickness + 0.09*z), 6)
        features += ["PCB substrate", "controller IC", "edge connector", "terminal header", "passive components"]
    elif part_id == "generic-100ohm-resistor":
        cylinder(parts, 0.36*min(y,z), 0.62*x, material=6, axis="x")
        cylinder(parts, 0.06*min(y,z), 0.19*x, (-0.405*x, 0, 0), 1, "x", 16)
        cylinder(parts, 0.06*min(y,z), 0.19*x, (0.405*x, 0, 0), 1, "x", 16)
        for offset in (-0.13, -0.04, 0.05, 0.14):
            cylinder(parts, 0.38*min(y,z), 0.025*x, (offset*x, 0, 0), 5 if offset < 0.1 else 1, "x")
        features += ["axial resistive body", "four identifying bands", "two wire leads"]
    elif part_id == "scanner.nimh_battery":
        diameter = min(y/3.15, z/2.15)
        for iy in (-1, 0, 1):
            for iz in (-0.5, 0.5):
                cylinder(parts, diameter, 0.94*x, (0, iy*diameter*1.03, iz*diameter*1.03), 7, "x")
        box(parts, (0.035*x, 0.2*y, 0.16*z), (0.4825*x, 0, 0), 1)
        features += ["six cylindrical NiMH cells", "pack terminal"]
    elif part_id == "scanner.gearmotor":
        motor_diameter = min(y, 0.72*z)
        cylinder(parts, motor_diameter, 0.5*x, (0.2*x, 0, 0), 1, "x")
        box(parts, (0.35*x, y, z), (-0.225*x, 0, 0), 1)
        cylinder(parts, 0.2*min(y,z), 0.15*x, (-0.425*x, 0, 0), 1, "x", 24)
        cylinder(parts, 0.045*min(y,z), 0.12*x, (0.44*x, 0, 0), 5, "x", 16)
        features += ["motor can", "rectangular metal gearbox", "output shaft", "rear electrical terminal"]
    elif part_id == "scanner.position_servo":
        box(parts, (0.68*x, 0.9*y, 0.72*z), (0, 0, -0.08*z), 0)
        box(parts, (x, y, 0.11*z), (0, 0, -0.25*z), 0)
        cylinder(parts, 0.24*min(x,y), 0.18*z, (0.18*x, 0, 0.41*z), 1)
        cylinder(parts, 0.38*min(x,y), 0.07*z, (0.18*x, 0, 0.465*z), 0)
        features += ["servo case", "mounting flanges", "output spline", "servo horn boss"]
    elif part_id == "scanner.romi_chassis":
        box(parts, (x, y, 0.12*z), (0, 0, -0.36*z), 0)
        box(parts, (0.68*x, 0.62*y, 0.12*z), (0, 0, -0.22*z), 0)
        for fx in (-0.38, 0.38):
            for fy in (-0.39, 0.39):
                cylinder(parts, 0.12*min(x,y), 0.48*z, (fx*x, fy*y, -0.08*z), 0)
        box(parts, (0.46*x, 0.5*y, 0.16*z), (-0.08*x, 0, 0.05*z), 7)
        sphere(parts, 0.22*z, (0.39*x, 0, 0.36*z), 1)
        features += ["sculpted deck plates", "four mounting bosses", "battery bay", "caster ball"]
    elif part_id == "scanner.arm_linkage":
        box(parts, (0.82*x, 0.46*y, 0.5*z), material=0)
        cylinder(parts, min(y,z), z, (-0.5*x + 0.5*min(y,z), 0, 0), 0)
        cylinder(parts, min(y,z), z, (0.5*x - 0.5*min(y,z), 0, 0), 0)
        cylinder(parts, 0.38*min(y,z), z, (-0.5*x + 0.5*min(y,z), 0, 0), 1)
        cylinder(parts, 0.38*min(y,z), z, (0.5*x - 0.5*min(y,z), 0, 0), 1)
        features += ["taper-equivalent linkage beam", "two pivot bosses", "two metal bushings"]
    elif part_id == "scanner.wheel":
        cylinder(parts, x, z, material=3)
        cylinder(parts, 0.25*x, z, material=0)
        for index in range(6):
            angle = index * math.pi / 3
            box(parts, (0.34*x, 0.065*x, 0.72*z), (0.19*x*math.cos(angle), 0.19*x*math.sin(angle), 0), 0, rotate_z(angle))
        cylinder(parts, 0.09*x, z, material=1, segments=24)
        features += ["rubber tire", "six-spoke polymer hub", "metal axle bore"]
    elif part_id in {"scanner.camera", "generic-camera-module"}:
        board_x = 0.66*x
        box(parts, (board_x, y, 0.12*z), (-0.5*x + 0.5*board_x, 0, -0.28*z), 2)
        box(parts, (0.2*x, 0.56*y, 0.35*z), (-0.12*x, 0, -0.04*z), 0)
        cylinder(parts, 0.52*min(y,z), 0.23*x, (0.235*x, 0, 0.06*z), 0, "x")
        cylinder(parts, 0.38*min(y,z), 0.13*x, (0.415*x, 0, 0.06*z), 1, "x")
        cylinder(parts, 0.34*min(y,z), 0.012*x, (0.494*x, 0, 0.06*z), 4, "x")
        box(parts, (0.14*x, 0.34*y, 0.1*z), (-0.36*x, 0, 0.19*z), 1)
        features += ["camera PCB", "sensor package", "multi-stage lens barrel", "glass objective", "flex connector"]
    elif part_id == "scanner.icosahedron":
        if not (math.isclose(x, y) and math.isclose(y, z)):
            raise ShapeBuildError("scanner icosahedron requires equal XYZ dimensions")
        parts.append((materialized(icosahedron_mesh(x), 8), transform()))
        features += ["regular icosahedron", "twenty triangular scan faces", "twelve vertices"]
    else:
        raise AssertionError(f"supported part {part_id!r} has no detailed-shape branch")
    return combine_meshes(parts), features


def build(catalog_root: Path, *, check: bool = False) -> int:
    materials = optical_materials()
    changed = 0
    catalog: list[tuple[Path, dict]] = []
    for static_path in sorted(catalog_root.rglob("static.part")):
        data = json.loads(static_path.read_text(encoding="utf-8"))
        for body in data["bodies"]:
            for solid in body["solids"]:
                _require_supported(data["id"], solid["id"])
        catalog.append((static_path, data))

    for static_path, data in catalog:
        static_changed = False
        for body in data["bodies"]:
            for solid in body["solids"]:
                geometry = solid["geometry"]
                root = static_path.parent / "shape" / solid["id"]
                source_path = root / "source" / "procedural-shape.json"
                if geometry.get("kind") == "shape" and source_path.is_file():
                    source_geometry = json.loads(source_path.read_text(encoding="utf-8"))
                    dimensions = tuple(float(item) for item in source_geometry["input_dimensions_m"])
                else:
                    dimensions = tuple(float(item) for item in geometry["dimensions_m"])
                mesh, features = detailed_mesh(data["id"], solid["id"], dimensions)
                shape_materials = (
                    (*materials, scan_target_material())
                    if data["id"] == "scanner.icosahedron"
                    else materials
                )
                source_value = {
                    "format": "contraption.procedural-shape/v1", "part": data["id"], "body": body["id"], "solid": solid["id"],
                    "input_dimensions_m": list(dimensions), "features": features, "builder": "scripts/build_part_shapes.py",
                    "optical_material_ids": [item.id for item in shape_materials],
                }
                source_payload = json.dumps(source_value, indent=2, sort_keys=True) + "\n"
                canonical_payload = mesh.to_bytes()
                runtime_payload = mesh.to_glb_bytes()
                low, high = mesh.bounds_m
                if not check:
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    source_path.write_text(source_payload, encoding="utf-8")
                    (root / "canonical.ctmesh").write_bytes(canonical_payload)
                    (root / "runtime.glb").write_bytes(runtime_payload)
                source_reference = ContentReference("source/procedural-shape.json", "application/vnd.contraption.procedural-shape+json", hashlib.sha256(source_payload.encode()).hexdigest(), len(source_payload.encode()))
                surface_reference = ContentReference("canonical.ctmesh", "application/vnd.contraption.ctmesh", hashlib.sha256(canonical_payload).hexdigest(), len(canonical_payload))
                runtime_reference = ContentReference("runtime.glb", "model/gltf-binary", hashlib.sha256(runtime_payload).hexdigest(), len(runtime_payload))
                artifact = ShapeArtifact(
                    id=f"{data['id']}.{body['id']}.{solid['id']}", version=data["version"],
                    sources=(SourceRepresentation("procedural-catalog-model", "procedural", source_reference, 1.0, provenance={"kind": "derived", "static_part": static_path.name}),),
                    surfaces=(SurfaceRepresentation("canonical", "ctmesh", surface_reference, ("analysis", "ray_trace", "render", "collision"), len(mesh.vertices_m), len(mesh.triangles), tuple(float(value) for value in np.concatenate((low, high))), mesh.watertight, mesh.manifold, tuple(item.id for item in shape_materials), ShapeUncertainty("normal", {
                        "standard_deviation_m": max(0.0001, 0.005 * max(dimensions)),
                        "basis": "catalog envelope and deterministic procedural feature estimate",
                        "builder_version": 1,
                    })),),
                    optical_materials=shape_materials, caches=(runtime_reference,),
                    provenance={"kind": "deterministic-procedural", "source_static_part": static_path.relative_to(catalog_root).as_posix()},
                    metadata={
                        "detailed_features": features,
                        "geometric_fidelity": "catalog-derived engineering visualization; not vendor CAD",
                    },
                )
                manifest_payload = json.dumps(artifact.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
                if not check:
                    (root / "shape.artifact.json").write_text(manifest_payload, encoding="utf-8")
                relative = (Path("shape") / solid["id"] / "shape.artifact.json").as_posix()
                new_geometry = {
                    "kind": "shape", "dimensions_m": [float(value) for value in high - low], "shape_uri": relative,
                    "shape_sha256": "sha256:" + hashlib.sha256(manifest_payload.encode()).hexdigest(), "surface_id": "canonical",
                }
                if geometry != new_geometry:
                    static_changed = True
                    solid["geometry"] = new_geometry
                if check:
                    expected = {source_path: source_payload.encode(), root / "canonical.ctmesh": canonical_payload, root / "runtime.glb": runtime_payload, root / "shape.artifact.json": manifest_payload.encode()}
                    for path, payload in expected.items():
                        if not path.is_file() or path.read_bytes() != payload:
                            raise SystemExit(f"stale generated shape artifact: {path}")
        static_payload = json.dumps(data, indent=2, sort_keys=False, allow_nan=False) + "\n"
        if check:
            if static_path.read_text(encoding="utf-8") != static_payload:
                raise SystemExit(f"stale static shape binding: {static_path}")
        elif static_changed:
            static_path.write_text(static_payload, encoding="utf-8")
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-root", type=Path, default=Path("model_catalog"))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    changed = build(arguments.catalog_root.resolve(), check=arguments.check)
    print(json.dumps({"valid": True, "changed_static_parts": changed, "check": arguments.check}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
