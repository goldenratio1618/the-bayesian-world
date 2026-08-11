from __future__ import annotations

from dataclasses import FrozenInstanceError
import copy
import json
import math
from pathlib import Path
import unittest

from contraption.catalog.instantiations import PartInstantiationRegistry
from contraption.catalog.interfaces import load_interface_catalog
from contraption.physics.dsl import ModelRegistry
from contraption.physics.physical import (
    AssemblyCycleError,
    AssemblyUnderconstrainedError,
    ResolvedPartRegistry,
    ResolvedPartSpec,
    ConnectorCoincidenceError,
    ConnectorCompatibilityError,
    PhysicalSpecError,
    TransformSpec,
    resolve_configuration,
    resolve_physical_assembly as _resolve_physical_assembly,
    validate_connector_coincidence,
)
from contraption.visualization.viewer import validate_physical_scene


MODEL_DIGEST = "sha256:" + "0" * 64


def resolve_physical_assembly(contraption, parts, *args, **kwargs):
    if not isinstance(parts, ResolvedPartRegistry):
        values = parts.values() if isinstance(parts, dict) else parts
        parts = ResolvedPartRegistry(
            value if isinstance(value, ResolvedPartSpec) else ResolvedPartSpec.from_dict(value)
            for value in values
        )
    return _resolve_physical_assembly(contraption, parts, *args, **kwargs)


def pose(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> dict[str, object]:
    return {
        "translation_m": [x, y, z],
        "rotation_quaternion_wxyz": list(quaternion),
    }


def provenance(kind: str = "estimated") -> dict[str, str]:
    return {"kind": kind, "source": "test fixture"}


def connector(
    connector_id: str,
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    domain: str = "rigid_mechanical",
    interface: str = "fixture-mount-v1",
    model_port: str | None = None,
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    kinematics: dict[str, object] | None = None,
    joint_coordinate_state: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": connector_id,
        "model_port": model_port,
        "body": "body",
        "domain": domain,
        "interface": interface,
        "local_pose": pose(x, y, z, quaternion),
        "provenance": provenance(),
    }
    if kinematics is not None:
        result["kinematics"] = kinematics
    if interface == "rotational-shaft":
        result["joint_coordinate_state"] = joint_coordinate_state
    return result


def resolved_part(
    part_id: str,
    connectors: list[dict[str, object]],
    *,
    dimensions: tuple[float, float, float] = (0.2, 0.1, 0.05),
) -> dict[str, object]:
    return {
        "format": "resolved-part-1",
        "id": part_id,
        "version": "1.0.0",
        "physical_role": "part",
        "model": {
            "id": f"{part_id}.model",
            "version": "1.0.0",
            "sha256": MODEL_DIGEST,
        },
        "bodies": [
            {
                "id": "body",
                "local_pose": pose(),
                "solids": [
                    {
                        "id": "solid",
                        "geometry": {
                            "kind": "box",
                            "dimensions_m": list(dimensions),
                        },
                        "local_pose": pose(),
                        "provenance": provenance(),
                    }
                ],
            }
        ],
        "connectors": connectors,
        "parameter_bindings": [],
        "provenance": provenance(),
    }


def boundary_part(part_id: str = "ground-boundary") -> dict[str, object]:
    return {
        "format": "resolved-part-1",
        "id": part_id,
        "version": "1.0.0",
        "physical_role": "boundary",
        "model": {
            "id": f"{part_id}.model",
            "version": "1.0.0",
            "sha256": MODEL_DIGEST,
        },
        "bodies": [],
        "connectors": [
            {
                "id": "reference",
                "model_port": "reference",
                "body": None,
                "domain": "electrical",
                "interface": "wire-terminal-v1",
                "local_pose": None,
                "provenance": provenance("boundary"),
            }
        ],
        "parameter_bindings": [],
        "provenance": provenance("boundary"),
    }


def attachment(
    attachment_id: str,
    parent: str,
    child: str,
    *,
    kind: str = "fixed",
    behavior_binding: str = "kinematic_only",
    coordinate: str | None = None,
    coordinate_bindings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    joint: dict[str, object] = {
        "kind": kind,
        "behavior_binding": behavior_binding,
        "coordinate_bindings": [],
    }
    if coordinate is not None:
        joint["coordinate"] = coordinate
        joint["coordinate_bindings"] = coordinate_bindings or [
            {"state": coordinate, "joint_angle_at_state_zero_rad": 0.0}
        ]
    return {
        "id": attachment_id,
        "kind": "attachment",
        "endpoints": [parent, child],
        "domain": "rigid_mechanical",
        "joint": joint,
    }


def contraption(
    components: list[dict[str, object]],
    connections: list[dict[str, object]],
    *,
    root: str = "base",
    root_pose: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "format": "contraption-physical-1",
        "id": "fixture",
        "name": "Fixture",
        "version": "1.0.0",
        "components": components,
        "connections": connections,
        "controls": [{"id": "controller", "settings": {"gain": 2.0}}],
        "environment": {"gravity_m_s2": [0.0, 0.0, -9.81]},
        "physical_root": {
            "component": root,
            "pose": root_pose or pose(),
            "state_binding": None,
        },
    }


class ResolvedPartParsingTests(unittest.TestCase):
    def test_resolved_part_is_strict_explicit_and_immutable(self) -> None:
        raw = resolved_part("base-package", [connector("mount")])
        package = ResolvedPartSpec.from_dict(raw)

        self.assertEqual(package.bodies[0].solids[0].geometry.dimensions_m, (0.2, 0.1, 0.05))
        self.assertIsNone(package.connectors[0].model_port)
        with self.assertRaises(FrozenInstanceError):
            package.version = "2.0.0"  # type: ignore[misc]

        extra = copy.deepcopy(raw)
        extra["viewer_transform"] = pose()
        with self.assertRaisesRegex(PhysicalSpecError, "unknown resolved part field"):
            ResolvedPartSpec.from_dict(extra)

        with self.assertRaisesRegex(PhysicalSpecError, "duplicate JSON field 'format'"):
            ResolvedPartSpec.from_json(
                '{"format":"resolved-part-1","format":"resolved-part-1"}'
            )

    def test_parts_require_real_geometry_and_nonphysical_roles_forbid_it(self) -> None:
        no_bodies = resolved_part("empty-part", [])
        no_bodies["bodies"] = []
        with self.assertRaisesRegex(PhysicalSpecError, "placeholder geometry is not permitted"):
            ResolvedPartSpec.from_dict(no_bodies)

        empty_solids = resolved_part("empty-body", [])
        empty_solids["bodies"][0]["solids"] = []  # type: ignore[index]
        with self.assertRaisesRegex(PhysicalSpecError, "at least one solid"):
            ResolvedPartSpec.from_dict(empty_solids)

        boundary = ResolvedPartSpec.from_dict(boundary_part())
        self.assertEqual(boundary.physical_role, "boundary")
        self.assertEqual(boundary.bodies, ())
        self.assertFalse(boundary.connectors[0].spatial)

        hidden_cube = boundary_part()
        hidden_cube["bodies"] = resolved_part("donor", [])["bodies"]
        with self.assertRaisesRegex(PhysicalSpecError, "may not carry hidden geometry"):
            ResolvedPartSpec.from_dict(hidden_cube)

        wrong_provenance = boundary_part()
        wrong_provenance["provenance"] = provenance("estimated")
        with self.assertRaisesRegex(PhysicalSpecError, "requires part provenance.kind"):
            ResolvedPartSpec.from_dict(wrong_provenance)

    def test_geometry_conventions_are_unambiguous(self) -> None:
        raw = resolved_part("cylinder", [])
        geometry = raw["bodies"][0]["solids"][0]["geometry"]  # type: ignore[index]
        geometry.update({"kind": "cylinder", "dimensions_m": [0.1, 0.2, 0.3]})
        with self.assertRaisesRegex(PhysicalSpecError, "equal X/Y diameters"):
            ResolvedPartSpec.from_dict(raw)

        raw = resolved_part("sphere", [])
        geometry = raw["bodies"][0]["solids"][0]["geometry"]  # type: ignore[index]
        geometry.update({"kind": "sphere", "dimensions_m": [0.1, 0.1, 0.2]})
        with self.assertRaisesRegex(PhysicalSpecError, "three equal diameters"):
            ResolvedPartSpec.from_dict(raw)

    def test_registry_rejects_duplicate_part_ids_and_is_immutable(self) -> None:
        package = ResolvedPartSpec.from_dict(resolved_part("base-package", []))
        with self.assertRaisesRegex(PhysicalSpecError, "duplicate component part id"):
            ResolvedPartRegistry([package, package])

        registry = ResolvedPartRegistry([package])
        self.assertIs(registry[package.id], registry[package.id])
        with self.assertRaisesRegex(AttributeError, "immutable"):
            registry._parts = {}  # type: ignore[assignment]

    def test_typed_physical_parameter_measures_are_strict(self) -> None:
        wheel = resolved_part("wheel-package", [])
        geometry = wheel["bodies"][0]["solids"][0]["geometry"]  # type: ignore[index]
        geometry.update({"kind": "cylinder", "dimensions_m": [0.07, 0.07, 0.02]})
        wheel["parameter_bindings"] = [
            {
                "model_parameter": "radius",
                "unit": "m",
                "absolute_tolerance": 1e-12,
                "measure": {
                    "kind": "solid_radius",
                    "body": "body",
                    "solid": "solid",
                    "axis": "x",
                },
            }
        ]
        parsed_wheel = ResolvedPartSpec.from_dict(wheel)
        binding = parsed_wheel.parameter_bindings[0]
        self.assertAlmostEqual(parsed_wheel.measure_parameter(binding), 0.035)

        chassis = resolved_part(
            "chassis-package",
            [connector("left", y=0.075), connector("right", y=-0.075)],
        )
        chassis["parameter_bindings"] = [
            {
                "model_parameter": "wheel_base",
                "unit": "m",
                "absolute_tolerance": 1e-12,
                "measure": {
                    "kind": "connector_distance",
                    "first_connector": "left",
                    "second_connector": "right",
                },
            }
        ]
        parsed_chassis = ResolvedPartSpec.from_dict(chassis)
        self.assertAlmostEqual(
            parsed_chassis.measure_parameter(parsed_chassis.parameter_bindings[0]),
            0.15,
        )

        invalid = copy.deepcopy(wheel)
        invalid["parameter_bindings"][0]["measure"]["kind"] = "python_expression"  # type: ignore[index]
        with self.assertRaisesRegex(PhysicalSpecError, "measure.kind"):
            ResolvedPartSpec.from_dict(invalid)


class TransformTests(unittest.TestCase):
    def test_composition_and_inverse_round_trip(self) -> None:
        rotation = TransformSpec.rotation_about_z(math.pi / 2.0)
        transform = TransformSpec((1.0, 2.0, 3.0), rotation.rotation_quaternion_wxyz)
        self.assertAlmostEqual(transform.apply((1.0, 0.0, 0.0))[0], 1.0)
        self.assertAlmostEqual(transform.apply((1.0, 0.0, 0.0))[1], 3.0)
        identity = transform.compose(transform.inverse())
        self.assertLess(identity.angular_distance(TransformSpec.identity()), 1e-12)
        for value in identity.translation_m:
            self.assertAlmostEqual(value, 0.0)

    def test_nonunit_and_nonfinite_transforms_fail(self) -> None:
        with self.assertRaisesRegex(PhysicalSpecError, "must be normalized"):
            TransformSpec.from_dict(pose(quaternion=(2.0, 0.0, 0.0, 0.0)))
        with self.assertRaisesRegex(PhysicalSpecError, "must be finite"):
            TransformSpec.from_dict(pose(x=float("nan")))

    def test_angular_distance_is_stable_near_identity(self) -> None:
        transform = TransformSpec.from_roll_pitch_yaw(
            (1.0, 2.0, 3.0), 0.2, -0.3, 1.1
        )
        identity = transform.inverse().compose(transform)
        self.assertLess(identity.angular_distance(TransformSpec.identity()), 1e-14)


class PhysicalAssemblyTests(unittest.TestCase):
    def test_bundled_scanner_resolves_from_canonical_json(self) -> None:
        project = Path(__file__).resolve().parents[1]
        catalog_root = project / "model_catalog"
        interfaces = load_interface_catalog(catalog_root)
        models = ModelRegistry()
        models.load_directory(catalog_root, interfaces=interfaces)
        instantiations = PartInstantiationRegistry.load_catalog(
            catalog_root, models=models
        )
        scanner = json.loads(
            (project / "examples" / "scanner_robot" / "contraption.json").read_text(
                encoding="utf-8"
            )
        )
        physical_source = dict(scanner)
        physical_source["components"] = [
            {"id": item["id"], "part": item["instantiation"]}
            for item in scanner["components"]
        ]
        assembly = resolve_physical_assembly(
            physical_source,
            instantiations.resolved_parts,
            {
                "left-wheel.angle": 0.0,
                "right-wheel.angle": 0.0,
                "arm-linkage.angle": 0.25,
                "camera.angle": 0.0,
            },
        )
        self.assertEqual(len(assembly.component_poses), 14)
        self.assertEqual(len(assembly.body_poses), 14)
        self.assertEqual(len(assembly.connections), len(scanner["connections"]))
        self.assertNotIn("electrical-reference", assembly.component_poses)
        validated_scene = validate_physical_scene(assembly.scene)
        self.assertEqual(validated_scene["assembly_sha256"], assembly.assembly_sha256)

    def test_fixed_attachment_places_child_from_connector_frames(self) -> None:
        base = ResolvedPartSpec.from_dict(
            resolved_part("base-package", [connector("mount", x=1.0)])
        )
        arm = ResolvedPartSpec.from_dict(
            resolved_part("arm-package", [connector("mount", x=-1.0)])
        )
        spec = contraption(
            [
                {"id": "base", "part": base.id, "parameters": {"mass_kg": 2.0}},
                {"id": "arm", "part": arm.id},
            ],
            [attachment("base-arm", "base.mount", "arm.mount")],
        )

        assembly = resolve_physical_assembly(spec, [base, arm])

        self.assertRegex(assembly.assembly_sha256, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(assembly.component_pose("arm").translation_m, (2.0, 0.0, 0.0))
        self.assertEqual(assembly.body_pose("arm", "body").translation_m, (2.0, 0.0, 0.0))
        self.assertEqual(assembly.connector_pose("base", "mount").translation_m, (1.0, 0.0, 0.0))
        self.assertEqual(assembly.connector_pose("arm", "mount").translation_m, (1.0, 0.0, 0.0))

        scene_component = assembly.scene["components"][0]
        self.assertIn("connectors", scene_component)
        self.assertEqual(assembly.scene["connections"][0]["kind"], "attachment")
        self.assertEqual(
            assembly.scene["connections"][0]["endpoints"][0],
            {"component": "base", "connector": "mount"},
        )
        with self.assertRaises(TypeError):
            scene_component["id"] = "mutated"

    def test_revolute_configuration_changes_pose_not_assembly_hash(self) -> None:
        base = ResolvedPartSpec.from_dict(
            resolved_part(
                "base-package",
                [connector("pivot", interface="rotational-shaft")],
            )
        )
        arm = ResolvedPartSpec.from_dict(
            resolved_part(
                "arm-package",
                [
                    connector(
                        "pivot",
                        x=-1.0,
                        interface="rotational-shaft",
                        joint_coordinate_state="shoulder_angle",
                    )
                ],
            )
        )
        spec = contraption(
            [{"id": "base", "part": base.id}, {"id": "arm", "part": arm.id}],
            [
                attachment(
                    "shoulder",
                    "base.pivot",
                    "arm.pivot",
                    kind="revolute",
                    coordinate="arm.shoulder_angle",
                )
            ],
        )
        assembly = resolve_physical_assembly(
            spec, [base, arm], {"arm.shoulder_angle": math.pi / 2.0}
        )
        arm_pose = assembly.component_pose("arm")
        self.assertAlmostEqual(arm_pose.translation_m[0], 0.0, places=12)
        self.assertAlmostEqual(arm_pose.translation_m[1], 1.0, places=12)

        reconfigured = assembly.with_configuration(
            root_pose=pose(10.0, 0.0, 0.0),
            joint_coordinates={"arm.shoulder_angle": 0.0},
        )
        functional = resolve_configuration(
            assembly, joint_coordinates={"arm.shoulder_angle": 0.0}
        )
        self.assertEqual(reconfigured.assembly_sha256, assembly.assembly_sha256)
        self.assertEqual(functional.assembly_sha256, assembly.assembly_sha256)
        self.assertEqual(reconfigured.component_pose("arm").translation_m, (11.0, 0.0, 0.0))
        self.assertEqual(functional.component_pose("arm").translation_m, (1.0, 0.0, 0.0))

        with self.assertRaisesRegex(PhysicalSpecError, "undeclared coordinate"):
            assembly.with_configuration(
                joint_coordinates={
                    "arm.shoulder_angle": 0.0,
                    "arm.typo": 0.0,
                }
            )

    def test_cycle_and_disconnected_part_fail_loudly(self) -> None:
        package = ResolvedPartSpec.from_dict(
            resolved_part(
                "node-package",
                [connector("left", x=-0.1), connector("right", x=0.1)],
            )
        )
        components = [
            {"id": name, "part": package.id} for name in ("base", "middle", "end")
        ]
        cycle = contraption(
            components,
            [
                attachment("edge-a", "base.right", "middle.left"),
                attachment("edge-b", "middle.right", "end.left"),
                attachment("edge-c", "end.right", "base.left"),
            ],
        )
        with self.assertRaisesRegex(AssemblyCycleError, "closed/cyclic"):
            resolve_physical_assembly(cycle, [package])

        disconnected = contraption(components[:2], [])
        with self.assertRaisesRegex(AssemblyUnderconstrainedError, "not constrained to root"):
            resolve_physical_assembly(disconnected, [package])

    def test_incompatible_and_unbound_connectors_fail_loudly(self) -> None:
        base = ResolvedPartSpec.from_dict(
            resolved_part("base-package", [connector("mount")])
        )
        electrical = ResolvedPartSpec.from_dict(
            resolved_part(
                "electrical-package",
                [
                    connector(
                        "mount",
                        domain="electrical",
                        interface="wire-terminal-v1",
                    )
                ],
            )
        )
        spec = contraption(
            [{"id": "base", "part": base.id}, {"id": "motor", "part": electrical.id}],
            [attachment("invalid", "base.mount", "motor.mount")],
        )
        with self.assertRaisesRegex(ConnectorCompatibilityError, r"incompatible.*domains"):
            resolve_physical_assembly(spec, [base, electrical])

        bound_base = ResolvedPartSpec.from_dict(
            resolved_part(
                "bound-base",
                [connector("mount", model_port="shaft")],
            )
        )
        bound_arm = ResolvedPartSpec.from_dict(
            resolved_part(
                "bound-arm",
                [connector("mount", model_port="shaft")],
            )
        )
        spec = contraption(
            [{"id": "base", "part": bound_base.id}, {"id": "arm", "part": bound_arm.id}],
            [attachment("invalid-binding", "base.mount", "arm.mount")],
        )
        with self.assertRaisesRegex(ConnectorCompatibilityError, "requires model_port:null"):
            resolve_physical_assembly(spec, [bound_base, bound_arm])

        unbound_wire = ResolvedPartSpec.from_dict(
            resolved_part(
                "wire-package",
                [
                    connector(
                        "terminal_a",
                        domain="electrical",
                        interface="wire-terminal-v1",
                    ),
                    connector(
                        "terminal_b",
                        domain="electrical",
                        interface="wire-terminal-v1",
                    )
                ],
            )
        )
        power_spec = contraption(
            [{"id": "base", "part": unbound_wire.id}],
            [
                {
                    "id": "power",
                    "kind": "power",
                    "endpoints": ["base.terminal_a", "base.terminal_b"],
                    "domain": "electrical",
                }
            ],
        )
        with self.assertRaisesRegex(ConnectorCompatibilityError, "non-null model_port"):
            resolve_physical_assembly(power_spec, [unbound_wire])

    def test_electrical_boundary_is_nonspatial_but_model_connected(self) -> None:
        board = ResolvedPartSpec.from_dict(
            resolved_part(
                "board-package",
                [
                    connector(
                        "ground",
                        domain="electrical",
                        interface="wire-terminal-v1",
                        model_port="ground",
                    )
                ],
            )
        )
        boundary = ResolvedPartSpec.from_dict(boundary_part())
        spec = contraption(
            [
                {"id": "base", "part": board.id},
                {"id": "reference", "part": boundary.id},
            ],
            [
                {
                    "id": "ground-net",
                    "kind": "power",
                    "endpoints": ["base.ground", "reference.reference"],
                    "domain": "electrical",
                }
            ],
        )
        assembly = resolve_physical_assembly(spec, [board, boundary])
        self.assertNotIn("reference", assembly.component_poses)
        self.assertEqual(
            [component["id"] for component in assembly.scene["components"]],
            ["base", "reference"],
        )

    def test_geometry_bound_instance_override_disagreement_fails(self) -> None:
        raw = resolved_part("wheel-package", [])
        geometry = raw["bodies"][0]["solids"][0]["geometry"]  # type: ignore[index]
        geometry.update({"kind": "cylinder", "dimensions_m": [0.07, 0.07, 0.02]})
        raw["parameter_bindings"] = [
            {
                "model_parameter": "radius",
                "unit": "m",
                "absolute_tolerance": 1e-12,
                "measure": {
                    "kind": "solid_radius",
                    "body": "body",
                    "solid": "solid",
                    "axis": "x",
                },
            }
        ]
        package = ResolvedPartSpec.from_dict(raw)
        spec = contraption(
            [{"id": "base", "part": package.id, "parameters": {"radius": 0.04}}],
            [],
        )
        with self.assertRaisesRegex(
            PhysicalSpecError, r"radius.*disagrees with part physical measure"
        ):
            resolve_physical_assembly(spec, [package])

    def test_counter_rotating_virtual_hub_preserves_mechanical_power_frame(self) -> None:
        base = ResolvedPartSpec.from_dict(
            resolved_part(
                "base-package",
                [
                    connector(
                        "shaft",
                        domain="mechanical",
                        interface="rotational-shaft",
                        model_port="shaft",
                    ),
                    connector(
                        "drive",
                        domain="mechanical",
                        interface="hub-translation",
                        model_port="drive",
                    ),
                ],
            )
        )
        wheel_raw = resolved_part(
            "wheel-package",
            [
                    connector(
                        "hub",
                    domain="mechanical",
                    interface="rotational-shaft",
                        model_port="axle",
                        joint_coordinate_state="angle",
                    ),
                connector(
                    "translation",
                    domain="mechanical",
                    interface="hub-translation",
                    model_port="translation",
                    kinematics={"kind": "counter_rotation", "state": "angle"},
                ),
            ],
        )
        wheel = ResolvedPartSpec.from_dict(wheel_raw)
        joint = attachment(
            "wheel-joint",
            "base.shaft",
            "wheel.hub",
            kind="revolute",
            behavior_binding="pmdl",
            coordinate="wheel.angle",
        )
        joint["domain"] = "mechanical"
        power = {
            "id": "hub-translation",
            "kind": "power",
            "domain": "mechanical",
            "endpoints": ["base.drive", "wheel.translation"],
        }
        spec = contraption(
            [{"id": "base", "part": base.id}, {"id": "wheel", "part": wheel.id}],
            [joint, power],
        )
        assembly = resolve_physical_assembly(
            spec, [base, wheel], {"wheel.angle": 0.7}
        )
        self.assertLess(
            assembly.connector_pose("base", "drive").angular_distance(
                assembly.connector_pose("wheel", "translation")
            ),
            1e-12,
        )
        self.assertNotIn(
            "kinematics",
            next(
                connector
                for component in assembly.scene["components"]
                if component["id"] == "wheel"
                for connector in component["connectors"]
                if connector["id"] == "translation"
            ),
        )

        no_counter = copy.deepcopy(wheel_raw)
        del no_counter["connectors"][1]["kinematics"]  # type: ignore[index]
        with self.assertRaisesRegex(
            ConnectorCoincidenceError, r"hub-translation.*angular_error_rad"
        ):
            resolve_physical_assembly(
                spec,
                [base, ResolvedPartSpec.from_dict(no_counter)],
                {"wheel.angle": 0.7},
            )

    def test_runtime_boundary_validation_detects_pose_drift(self) -> None:
        base = ResolvedPartSpec.from_dict(
            resolved_part("base-package", [connector("mount", x=1.0)])
        )
        arm = ResolvedPartSpec.from_dict(
            resolved_part("arm-package", [connector("mount", x=-1.0)])
        )
        assembly = resolve_physical_assembly(
            contraption(
                [{"id": "base", "part": base.id}, {"id": "arm", "part": arm.id}],
                [attachment("base-arm", "base.mount", "arm.mount")],
            ),
            [base, arm],
        )
        drifted = dict(assembly.component_poses)
        drifted["arm"] = TransformSpec((2.01, 0.0, 0.0))
        with self.assertRaisesRegex(
            ConnectorCoincidenceError,
            r"base-arm.*translation_error_m=.*tolerance=",
        ):
            validate_connector_coincidence(
                assembly.components,
                assembly.attachments,
                assembly.parts,
                drifted,
                {},
            )
        with self.assertRaisesRegex(AssemblyUnderconstrainedError, "missing=.*arm"):
            validate_connector_coincidence(
                assembly.components,
                assembly.attachments,
                assembly.parts,
                {"base": assembly.component_pose("base")},
                {},
            )

    def test_connection_metadata_cannot_hide_untyped_physical_semantics(self) -> None:
        base = ResolvedPartSpec.from_dict(
            resolved_part("base-package", [connector("mount")])
        )
        arm = ResolvedPartSpec.from_dict(
            resolved_part("arm-package", [connector("mount")])
        )
        connection = attachment("base-arm", "base.mount", "arm.mount")
        connection["metadata"] = {"fastener": "M3", "axis": "+z"}
        with self.assertRaisesRegex(PhysicalSpecError, "metadata must be empty"):
            resolve_physical_assembly(
                contraption(
                    [
                        {"id": "base", "part": base.id},
                        {"id": "arm", "part": arm.id},
                    ],
                    [connection],
                ),
                [base, arm],
            )
    def test_scenario_inputs_invalidate_hash_but_runtime_configuration_does_not(self) -> None:
        package = ResolvedPartSpec.from_dict(resolved_part("base-package", []))
        original = contraption(
            [{"id": "base", "part": package.id, "parameters": {"mass_kg": 2.0}}],
            [],
        )
        first = resolve_physical_assembly(original, [package])

        changed_parameter = copy.deepcopy(original)
        changed_parameter["components"][0]["parameters"]["mass_kg"] = 3.0
        changed_environment = copy.deepcopy(original)
        changed_environment["environment"]["gravity_m_s2"][2] = -1.62
        changed_control = copy.deepcopy(original)
        changed_control["controls"][0]["settings"]["gain"] = 3.0
        for changed in (changed_parameter, changed_environment, changed_control):
            self.assertNotEqual(
                resolve_physical_assembly(changed, [package]).assembly_sha256,
                first.assembly_sha256,
            )

        changed_geometry = ResolvedPartSpec.from_dict(
            resolved_part("base-package", [], dimensions=(0.3, 0.1, 0.05))
        )
        changed_model_data = resolved_part("base-package", [])
        changed_model_data["model"]["sha256"] = "sha256:" + "1" * 64  # type: ignore[index]
        changed_model = ResolvedPartSpec.from_dict(changed_model_data)
        for changed_package in (changed_geometry, changed_model):
            self.assertNotEqual(
                resolve_physical_assembly(original, [changed_package]).assembly_sha256,
                first.assembly_sha256,
            )

        moved_root = copy.deepcopy(original)
        moved_root["physical_root"]["pose"] = pose(42.0, -5.0, 1.0)
        moved = resolve_physical_assembly(moved_root, [package])
        self.assertEqual(moved.assembly_sha256, first.assembly_sha256)
        self.assertEqual(moved.component_pose("base").translation_m, (42.0, -5.0, 1.0))


if __name__ == "__main__":
    unittest.main()
