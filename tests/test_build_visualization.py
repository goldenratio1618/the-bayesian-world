from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from contraption.manufacturing.build import BuildInstructionError, generate_build_instructions
from contraption import load_contraption
from contraption.cli import (
    _trajectory_payload,
    _trajectory_result,
    build_parser,
    command_compile,
)
from contraption.physics.physical import (
    PhysicalAssemblySpec,
    ResolvedPartRegistry,
    ResolvedPartSpec,
    resolve_physical_assembly as _resolve_physical_assembly,
)
from contraption.physics.simulator import simulate
from contraption.physics.specs import FrozenDict
from contraption.visualization.viewer import (
    VisualizationError,
    generate_viewer,
    validate_physical_scene,
)


ROOT = Path(__file__).resolve().parents[1]


def resolve_physical_assembly(contraption, parts):
    values = parts.values() if isinstance(parts, dict) else parts
    registry = ResolvedPartRegistry(
        value if isinstance(value, ResolvedPartSpec) else ResolvedPartSpec.from_dict(value)
        for value in values
    )
    return _resolve_physical_assembly(
        PhysicalAssemblySpec.from_dict(contraption), registry
    )


def scanner_assembly():
    return load_contraption(
        ROOT / "assembled_contraptions" / "scanner" / "contraption.json"
    )


class BuildInstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assembly = scanner_assembly()

    def test_plan_is_deterministic_and_bound_to_both_closure_hashes(self) -> None:
        first = generate_build_instructions(self.assembly)
        second = generate_build_instructions(self.assembly)
        self.assertEqual(first, second)
        self.assertEqual(first.to_markdown(), second.to_markdown())
        self.assertEqual(first.assembly_sha256, self.assembly.assembly_sha256)
        self.assertEqual(first.pmdl_sha256, self.assembly.system.pmdl_sha256)
        self.assertIn(self.assembly.assembly_sha256, first.to_markdown())
        expected = tuple(
            {
                "id": controller.id,
                "version": controller.spec.version,
                "sha256": link.program.sha256,
            }
            for link in self.assembly.specification.controllers
            for controller in (self.assembly.controllers[link.id],)
        )
        self.assertEqual(
            first.controllers,
            expected,
        )
        self.assertIn(expected[0]["sha256"], first.to_markdown())

    def test_every_written_build_artifact_carries_the_exact_closure_hash(self) -> None:
        plan = generate_build_instructions(self.assembly)
        with tempfile.TemporaryDirectory() as directory:
            paths = plan.write(Path(directory) / "build")
            machine = json.loads(paths["build-plan.json"].read_text(encoding="utf-8"))
            human = paths["BUILD_INSTRUCTIONS.md"].read_text(encoding="utf-8")
        self.assertEqual(machine["assembly_sha256"], self.assembly.assembly_sha256)
        self.assertEqual(machine["pmdl_sha256"], self.assembly.system.pmdl_sha256)
        self.assertEqual(machine["controllers"], [dict(item) for item in plan.controllers])
        self.assertIn(self.assembly.assembly_sha256, human)
        self.assertIn(self.assembly.system.pmdl_sha256, human)

    def test_placement_and_wiring_are_derived_from_resolved_connector_poses(self) -> None:
        plan = generate_build_instructions(self.assembly)
        self.assertEqual(
            {item.body for item in plan.placements},
            set(self.assembly.physical.body_poses),
        )
        battery_bus = next(
            item for item in plan.wiring if item.connection_id == "battery-positive-bus"
        )
        self.assertEqual(battery_bus.routed_length_m, None)
        self.assertGreater(battery_bus.straight_line_lower_bound_m, 0.0)
        self.assertEqual(
            set(battery_bus.connector_world_poses), set(battery_bus.endpoints)
        )
        for endpoint in battery_bus.endpoints:
            component_id, port_id = endpoint.rsplit(".", 1)
            component = next(
                item
                for item in self.assembly.specification.components
                if item.id == component_id
            )
            expected = self.assembly.parts[component.part].connector_map[
                port_id
            ].provenance.to_dict()
            self.assertEqual(battery_bus.connector_provenance[endpoint], expected)

    def test_unrepresented_fabrication_facts_remain_release_gates(self) -> None:
        plan = generate_build_instructions(self.assembly)
        self.assertFalse(plan.build_ready)
        self.assertTrue(any("conductor type/gauge" in item for item in plan.unresolved))
        self.assertTrue(any("pose is estimated" in item for item in plan.unresolved))
        self.assertTrue(any("condition is 'unverified'" in item for item in plan.unresolved))
        dynamics = [
            item
            for item in plan.unresolved
            if item.startswith("dynamics completeness gate ")
        ]
        self.assertEqual(len(dynamics), 7)
        self.assertTrue(any("fixed_payload_mass_inertia" in item for item in dynamics))
        self.assertTrue(any("full_body_keepout" in item for item in dynamics))

    def test_build_rejects_resolved_assembly_missing_mandatory_completeness(self) -> None:
        corrupted = replace(
            self.assembly,
            specification=replace(
                self.assembly.specification,
                metadata=FrozenDict({}),
            ),
        )
        with self.assertRaisesRegex(
            BuildInstructionError, "mandatory dynamics_completeness record"
        ):
            generate_build_instructions(corrupted)

    def test_every_canonical_connection_is_accounted_for(self) -> None:
        plan = generate_build_instructions(self.assembly)
        accounted = {
            item.connection_id
            for collection in (plan.mechanical, plan.wiring, plan.model_connections)
            for item in collection
        }
        self.assertEqual(
            accounted,
            {item.id for item in self.assembly.specification.connections},
        )

    def test_mechanical_steps_preserve_every_joint_coordinate_binding(self) -> None:
        plan = generate_build_instructions(self.assembly)
        expected = {
            attachment.id: tuple(
                binding.to_dict()
                for binding in attachment.joint.coordinate_bindings
            )
            for attachment in self.assembly.physical.attachments
        }
        self.assertEqual(
            {item.connection_id: item.coordinate_bindings for item in plan.mechanical},
            expected,
        )

    def test_raw_or_parallel_representation_is_rejected(self) -> None:
        with self.assertRaisesRegex(BuildInstructionError, "ResolvedAssembly"):
            generate_build_instructions(self.assembly.specification.to_dict())


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assembly = scanner_assembly()

    @staticmethod
    def assembly_scene() -> dict:
        identity = {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        provenance = {
            "kind": "estimated",
            "source": "unit-test static part",
            "reference": None,
        }
        initial_poses = {
            "chassis/base": identity,
            "chassis/arm": {
                "translation_m": [0.0, 0.0, 0.23],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "camera/housing": {
                "translation_m": [0.0, 0.0, 0.4],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        }
        initial_connector_poses = {
            "chassis.aux_power": identity,
            "camera.power": {
                "translation_m": [0.0, 0.0, 0.4],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        }
        return {
            "schema": "contraption.physical-scene/v1",
            "assembly_sha256": "sha256:" + "a" * 64,
            "contraption_id": "scanner-test",
            "components": [
                {
                    "id": "chassis",
                    "part": "scanner.chassis.v1",
                    "model": "rigid_chassis",
                    "physical_role": "part",
                    "bodies": [
                        {
                            "id": "base",
                            "local_pose": identity,
                            "solids": [
                                {
                                    "id": "shell",
                                    "geometry": {
                                        "kind": "box",
                                        "dimensions_m": [0.35, 0.28, 0.06],
                                        "mesh_uri": None,
                                    },
                                    "local_pose": identity,
                                    "provenance": provenance,
                                }
                            ],
                        },
                        {
                            "id": "arm",
                            "local_pose": identity,
                            "solids": [
                                {
                                    "id": "boom",
                                    "geometry": {
                                        "kind": "cylinder",
                                        "dimensions_m": [0.025, 0.025, 0.32],
                                        "mesh_uri": None,
                                    },
                                    "local_pose": identity,
                                    "provenance": provenance,
                                }
                            ],
                        },
                    ],
                    "connectors": [
                        {
                            "id": "aux_power",
                            "model_port": "aux_power",
                            "body": "base",
                            "domain": "electrical",
                            "interface": "dc-barrel",
                            "local_pose": identity,
                            "provenance": provenance,
                            "joint_coordinate_state": None,
                        }
                    ],
                },
                {
                    "id": "camera",
                    "part": "scanner.camera.v1",
                    "model": "depth_camera",
                    "physical_role": "part",
                    "bodies": [
                        {
                            "id": "housing",
                            "local_pose": identity,
                            "solids": [
                                {
                                    "id": "case",
                                    "geometry": {
                                        "kind": "box",
                                        "dimensions_m": [0.09, 0.03, 0.025],
                                        "mesh_uri": None,
                                    },
                                    "local_pose": identity,
                                    "provenance": provenance,
                                }
                            ],
                        }
                    ],
                    "connectors": [
                        {
                            "id": "power",
                            "model_port": "power",
                            "body": "housing",
                            "domain": "electrical",
                            "interface": "usb-c </script><script>alert(1)</script>",
                            "local_pose": identity,
                            "provenance": provenance,
                            "joint_coordinate_state": None,
                        }
                    ],
                },
                {
                    "id": "mission",
                    "part": "scanner.mission-boundary.v1",
                    "model": "mission_boundary",
                    "physical_role": "boundary",
                    "bodies": [],
                    "connectors": [
                        {
                            "id": "request",
                            "model_port": "request",
                            "body": None,
                            "domain": "signal",
                            "interface": "mission-command",
                            "local_pose": None,
                            "provenance": {
                                "kind": "boundary",
                                "source": "unit-test mission boundary",
                                "reference": None,
                            },
                            "joint_coordinate_state": None,
                        }
                    ],
                },
            ],
            "connections": [
                {
                    "id": "camera_power",
                    "kind": "power",
                    "domain": "electrical",
                    "metadata": {},
                    "endpoints": [
                        {"component": "chassis", "connector": "aux_power"},
                        {"component": "camera", "connector": "power"},
                    ],
                }
            ],
            "body_poses": initial_poses,
            "connector_poses": initial_connector_poses,
            "body_pose_frames": {
                "assembly_sha256": "sha256:" + "a" * 64,
                "frames": [
                    {
                        "time_s": 0.0,
                        "body_poses": initial_poses,
                        "connector_poses": initial_connector_poses,
                    },
                    {
                        "time_s": 0.1,
                        "body_poses": {
                            "chassis/base": {
                                "translation_m": [0.02, 0.0, 0.0],
                                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            },
                            "chassis/arm": {
                                "translation_m": [0.02, 0.0, 0.25],
                                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            },
                            "camera/housing": {
                                "translation_m": [0.02, 0.0, 0.42],
                                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            },
                        },
                        "connector_poses": {
                            "chassis.aux_power": {
                                "translation_m": [0.02, 0.0, 0.0],
                                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            },
                            "camera.power": {
                                "translation_m": [0.02, 0.0, 0.42],
                                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            },
                        },
                    },
                ],
            },
        }

    def test_standalone_viewer_embeds_only_hash_bound_body_poses(self) -> None:
        artifact = generate_viewer(
            self.assembly,
            title="Scanner <offline>",
        )
        self.assertIn("<canvas id=\"scene\"", artifact.html)
        self.assertIn("electrical-diagram", artifact.html)
        self.assertIn("global-alpha", artifact.html)
        self.assertIn("Connector frames", artifact.html)
        self.assertIn("wire-terminal", artifact.html)
        self.assertIn("Scanner \\u003coffline\\u003e", artifact.html)
        self.assertNotIn("<script src=", artifact.html)
        self.assertNotIn("<link rel=\"stylesheet\"", artifact.html)
        self.assertIn("wheel", artifact.javascript)
        self.assertIn("pointermove", artifact.javascript)
        self.assertIn('event.button === 2 ? "rotate" : "pan"', artifact.javascript)
        self.assertIn("contextmenu", artifact.javascript)
        self.assertIn("requestAnimationFrame", artifact.javascript)
        self.assertIn("if (!initial) render();", artifact.javascript)
        self.assertIn("Right-drag to rotate", artifact.html)
        self.assertIn("rotation_quaternion_wxyz", artifact.javascript)
        self.assertNotIn("payload.specification", artifact.javascript)
        self.assertNotIn("payload.simulation", artifact.javascript)
        self.assertNotIn("payload.runtime", artifact.javascript)
        self.assertNotIn("visualBindings", artifact.javascript)
        self.assertNotIn("trajectoryState", artifact.javascript)
        self.assertNotIn("pointInFrame", artifact.javascript)
        self.assertNotIn("arm_elevation", artifact.javascript)
        self.assertNotIn(
            "transformPose(transformPose([0, 0, 0], item.connector.local_pose",
            artifact.javascript,
        )
        self.assertEqual(artifact.data["schema"], "contraption.viewer/v2")
        self.assertEqual(
            artifact.data["assembly_sha256"], self.assembly.assembly_sha256
        )
        self.assertEqual(artifact.assembly_sha256, self.assembly.assembly_sha256)
        self.assertEqual(set(artifact.data), {"schema", "title", "assembly_sha256", "scene"})
        self.assertIn("connect-src 'none'", artifact.html)

    def test_live_viewer_is_same_origin_and_display_only(self) -> None:
        artifact = generate_viewer(
            self.assembly,
            live={
                "schema_endpoint": "/api/schema",
                "simulate_endpoint": "/api/simulate",
            },
        )
        self.assertEqual(
            artifact.data["live"],
            {
                "schema_endpoint": "/api/schema",
                "simulate_endpoint": "/api/simulate",
            },
        )
        self.assertIn("connect-src 'self'", artifact.html)
        self.assertIn("fetch(payload.live.simulate_endpoint", artifact.javascript)
        self.assertIn("validatedPayload({ ...payload, scene: body })", artifact.javascript)
        self.assertIn("window.location.reload()", artifact.javascript)
        self.assertNotIn("function integrate", artifact.javascript)
        self.assertNotIn("function derivative", artifact.javascript)

    def test_live_viewer_rejects_non_same_origin_or_ambiguous_endpoints(self) -> None:
        unsafe = (
            "https://example.invalid/api/schema",
            "//example.invalid/api/schema",
            "/api/schema?version=1",
            "/api/schema#fragment",
            "/api\\schema",
        )
        for endpoint in unsafe:
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(
                VisualizationError, "same-origin"
            ):
                generate_viewer(
                    self.assembly,
                    live={
                        "schema_endpoint": endpoint,
                        "simulate_endpoint": "/api/simulate",
                    },
                )

    def test_writes_single_page_and_inspectable_bundle(self) -> None:
        artifact = generate_viewer(self.assembly)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = artifact.write(root / "scanner.html")
            self.assertEqual(set(single), {"scanner.html"})
            self.assertIn(
                "Self-contained, display-only viewer",
                (root / "scanner.html").read_text(encoding="utf-8"),
            )
            bundle = artifact.write(root / "bundle")
            self.assertEqual(
                set(bundle), {"index.html", "viewer.js", "style.css", "viewer-data.json"}
            )
            data = json.loads(
                (root / "bundle" / "viewer-data.json").read_text(encoding="utf-8")
            )
            self.assertIn(
                "chassis", {component["id"] for component in data["scene"]["components"]}
            )
            self.assertEqual(data["assembly_sha256"], self.assembly.assembly_sha256)

    def test_rejects_non_finite_browser_data(self) -> None:
        scene = self.assembly_scene()
        scene["body_pose_frames"]["frames"][0]["time_s"] = float("nan")
        with self.assertRaisesRegex(VisualizationError, "NaN"):
            validate_physical_scene(scene)

    def test_rejects_detached_scene_and_frame_hash_mismatch(self) -> None:
        with self.assertRaisesRegex(TypeError, "ResolvedAssembly"):
            generate_viewer(self.assembly_scene())
        stale_frames = self.assembly_scene()
        stale_frames["body_pose_frames"]["assembly_sha256"] = "sha256:" + "b" * 64
        with self.assertRaisesRegex(VisualizationError, "frame assembly hash mismatch"):
            validate_physical_scene(stale_frames)

    def test_rejects_incomplete_or_extra_body_pose_frames(self) -> None:
        missing = self.assembly_scene()
        del missing["body_pose_frames"]["frames"][1]["body_poses"]["camera/housing"]
        with self.assertRaisesRegex(VisualizationError, "missing camera/housing"):
            validate_physical_scene(missing)

        extra = self.assembly_scene()
        extra["body_pose_frames"]["frames"][0]["body_poses"]["ghost/body"] = {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        with self.assertRaisesRegex(VisualizationError, "unknown ghost/body"):
            validate_physical_scene(extra)

        missing_connector = self.assembly_scene()
        del missing_connector["body_pose_frames"]["frames"][1]["connector_poses"][
            "camera.power"
        ]
        with self.assertRaisesRegex(VisualizationError, "missing camera.power"):
            validate_physical_scene(missing_connector)

    def test_rejects_unnormalized_quaternion_and_unknown_scene_fields(self) -> None:
        bad_rotation = self.assembly_scene()
        bad_rotation["body_pose_frames"]["frames"][0]["body_poses"]["chassis/base"][
            "rotation_quaternion_wxyz"
        ] = [2.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(VisualizationError, "must be normalized"):
            validate_physical_scene(bad_rotation)

        ignored_metadata = self.assembly_scene()
        ignored_metadata["metadata"] = {"visualization": {"frames": {}}}
        with self.assertRaisesRegex(VisualizationError, "would ignore: metadata"):
            validate_physical_scene(ignored_metadata)

    def test_rejects_mesh_uri_instead_of_drawing_an_inaccurate_box(self) -> None:
        scene = self.assembly_scene()
        scene["components"][0]["bodies"][0]["solids"][0]["geometry"] = {
            "kind": "mesh",
            "dimensions_m": [0.35, 0.28, 0.06],
            "mesh_uri": "cad/chassis.stl",
        }
        with self.assertRaisesRegex(VisualizationError, "cannot render a URI"):
            validate_physical_scene(scene)

    def test_static_canonical_body_poses_are_sufficient(self) -> None:
        artifact = generate_viewer(self.assembly)
        self.assertNotIn("body_pose_frames", artifact.data["scene"])
        self.assertIn("scene.body_poses", artifact.javascript)

    def test_preserves_typed_attachment_without_projecting_endpoints(self) -> None:
        scene = self.assembly_scene()
        identity = {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        provenance = {
            "kind": "estimated",
            "source": "unit-test mount",
            "reference": None,
        }
        scene["components"][0]["connectors"].append(
            {
                "id": "camera_mount",
                "model_port": None,
                "body": "arm",
                "domain": "rigid_mechanical",
                "interface": "scanner-camera-mount-v1",
                "local_pose": identity,
                "provenance": provenance,
                "joint_coordinate_state": None,
            }
        )
        scene["components"][1]["connectors"].append(
            {
                "id": "mount",
                "model_port": None,
                "body": "housing",
                "domain": "rigid_mechanical",
                "interface": "scanner-camera-mount-v1",
                "local_pose": identity,
                "provenance": provenance,
                "joint_coordinate_state": None,
            }
        )
        scene["connector_poses"].update(
            {
                "chassis.camera_mount": scene["body_poses"]["chassis/arm"],
                "camera.mount": scene["body_poses"]["camera/housing"],
            }
        )
        for pose_frame in scene["body_pose_frames"]["frames"]:
            pose_frame["connector_poses"].update(
                {
                    "chassis.camera_mount": pose_frame["body_poses"]["chassis/arm"],
                    "camera.mount": pose_frame["body_poses"]["camera/housing"],
                }
            )
        attachment = {
            "id": "camera_attachment",
            "kind": "attachment",
            "domain": "rigid_mechanical",
            "metadata": {},
            "endpoints": [
                {"component": "chassis", "connector": "camera_mount"},
                {"component": "camera", "connector": "mount"},
            ],
            "joint": {
                "kind": "fixed",
                "behavior_binding": "kinematic_only",
                "coordinate": None,
                "zero_angle_rad": 0.0,
                "coordinate_bindings": [],
            },
        }
        scene["connections"].append(attachment)
        normalized = validate_physical_scene(scene)
        self.assertEqual(normalized["connections"][1], attachment)
        self.assertIsNone(
            normalized["components"][0]["connectors"][-1][
                "joint_coordinate_state"
            ]
        )
        serialized = json.dumps(normalized, sort_keys=True)
        self.assertNotIn("instance_id", serialized)
        self.assertNotIn("port_id", serialized)

        attachment["metadata"] = {"fastener": "M3"}
        with self.assertRaisesRegex(VisualizationError, "metadata must be empty"):
            validate_physical_scene(scene)

    def test_rejects_physical_only_resolver_projection(self) -> None:
        identity = {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        provenance = {"kind": "estimated", "source": "resolver-viewer integration test"}
        package = {
            "format": "resolved-part-1",
            "id": "single.part",
            "version": "1.0.0",
            "physical_role": "part",
            "model": {
                "id": "single.model",
                "version": "1.0.0",
                "sha256": "sha256:" + "0" * 64,
            },
            "bodies": [
                {
                    "id": "body",
                    "local_pose": identity,
                    "solids": [
                        {
                            "id": "solid",
                            "geometry": {"kind": "box", "dimensions_m": [0.2, 0.1, 0.05]},
                            "local_pose": identity,
                            "provenance": provenance,
                        }
                    ],
                }
            ],
            "connectors": [],
            "parameter_bindings": [],
            "provenance": provenance,
        }
        contraption = {
            "format": "contraption-physical-1",
            "id": "single",
            "name": "Single part",
            "version": "1.0.0",
            "components": [{"id": "base", "part": "single.part"}],
            "connections": [],
            "controls": [],
            "environment": {},
            "physical_root": {
                "component": "base",
                "pose": identity,
                "state_binding": None,
            },
        }
        resolved = resolve_physical_assembly(contraption, {"single.part": package})
        with self.assertRaisesRegex(TypeError, "ResolvedAssembly"):
            generate_viewer(resolved)


class CliTrajectoryArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assembly = scanner_assembly()
        cls.result = simulate(
            cls.assembly,
            duration=0.02,
            dt=0.01,
            num_samples=1,
            seed=5,
            use_model_uncertainty=False,
            process_noise=False,
        )

    def test_v2_round_trip_preserves_exact_samples_for_canonical_viewer(self) -> None:
        payload = _trajectory_payload(self.result, {"accepted": False})
        self.assertEqual(payload["schema"], "contraption.trajectory/v2")
        self.assertIn("samples", payload)
        self.assertIn("output_samples", payload)
        self.assertNotIn("state_mean", payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            restored = _trajectory_result(self.assembly, path)
        np.testing.assert_array_equal(restored.samples, self.result.samples)
        np.testing.assert_array_equal(
            restored.output_samples, self.result.output_samples
        )
        artifact = generate_viewer(self.assembly, restored, sample_index=0)
        self.assertIn("body_pose_frames", artifact.data["scene"])

    def test_detached_scene_cli_admission_is_removed(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["view", "--scene", "detached.json"])

    def test_simulation_tick_defaults_to_resolved_controller_subdivision(self) -> None:
        arguments = build_parser().parse_args(
            ["simulate", "--spec", str(ROOT / "assembled_contraptions" / "scanner" / "contraption.json")]
        )
        self.assertIsNone(arguments.dt)

    def test_v2_rejects_redundant_pose_metadata(self) -> None:
        payload = _trajectory_payload(self.result, {})
        payload["metadata"]["body_pose_frames"] = {"forged": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden redundant"):
                _trajectory_result(self.assembly, path)

    def test_v2_loader_rejects_ambiguous_or_nonfinite_json(self) -> None:
        cases = (
            ('{"schema":"first","schema":"second"}', "duplicate JSON key"),
            ('{"time":NaN}', "non-finite JSON number"),
        )
        for source, message in cases:
            with self.subTest(source=source), mock.patch.object(
                Path, "read_text", return_value=source
            ), self.assertRaisesRegex(ValueError, message):
                _trajectory_result(self.assembly, Path("ignored-trajectory.json"))

    def test_compile_cli_emits_each_resolved_controller_to_both_targets(self) -> None:
        bundle = mock.Mock()
        bundle.source_digest = "sha256:" + "c" * 64
        bundle.targets = ("c99", "verilog")
        bundle.artifacts = (
            SimpleNamespace(path="controller.h", sha256="sha256:" + "1" * 64),
            SimpleNamespace(path="controller.c", sha256="sha256:" + "2" * 64),
            SimpleNamespace(path="controller.v", sha256="sha256:" + "3" * 64),
        )
        bundle.closure = {"assembly_sha256": self.assembly.assembly_sha256}
        bundle.manifest = {"closure": bundle.closure}
        bundle.write.return_value = (
            Path("/tmp/scanner/controller.h"),
            Path("/tmp/scanner/controller.c"),
            Path("/tmp/scanner/controller.v"),
            Path("/tmp/scanner/manifest.json"),
        )
        arguments = SimpleNamespace(
            output="ignored-output",
        )
        stream = StringIO()
        with mock.patch(
            "contraption.cli._assembly_from_args", return_value=self.assembly
        ), mock.patch(
            "contraption.cli.compile_resolved_controller", return_value=bundle
        ) as compile_controller, redirect_stdout(stream):
            self.assertEqual(command_compile(arguments), 0)
        reported = json.loads(stream.getvalue())
        controller_id = next(iter(self.assembly.controllers))
        self.assertEqual(
            reported["compiled"][controller_id]["targets"], ["c99", "verilog"]
        )
        compile_controller.assert_called_once_with(
            self.assembly,
            controller_id,
            identifier=controller_id,
            targets=("c99", "verilog"),
        )
        bundle.write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
