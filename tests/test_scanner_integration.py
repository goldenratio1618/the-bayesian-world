from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest

import numpy as np

import contraption.scanner as scanner_runtime_module
from contraption.backend import NumpyBackend, TorchBackend
from contraption.resolved import ResolvedAssembly
from contraption.scanner import (
    ScannerAssemblyController,
    ScannerMission,
    ScannerRuntimeError,
    load_scanner_assembly,
    scanner_metrics,
    scanner_physical_scene,
    simulate_scanner_robot,
)
from contraption.simulator import simulate
from contraption.visualization import validate_physical_scene


ROOT = Path(__file__).resolve().parents[1]


class ScannerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assembly = load_scanner_assembly(ROOT)
        cls.mission = ScannerMission.from_assembly(cls.assembly)
        cls.controller = ScannerAssemblyController(cls.assembly)
        cls.result = simulate_scanner_robot(
            cls.assembly,
            duration=0.05,
            dt=0.05,
            num_samples=2,
            seed=7,
            visualization_sample_index=1,
        )

    def test_real_example_resolves_as_one_component_pmdl_closure(self) -> None:
        assembly = self.assembly
        self.assertIsInstance(assembly, ResolvedAssembly)
        self.assertEqual(assembly.assembly_sha256, assembly.physical.assembly_sha256)
        self.assertEqual(assembly.assembly_sha256, assembly.system.assembly_sha256)
        self.assertNotEqual(assembly.assembly_sha256, assembly.system.pmdl_sha256)
        self.assertEqual(
            {component.id for component in assembly.specification.components},
            set(assembly.component_models),
        )
        self.assertEqual(len(assembly.specification.components), 15)
        self.assertEqual(len(assembly.specification.connections), 27)
        self.assertEqual(assembly.system.balance.unknown_count, len(assembly.system.state_names))
        self.assertEqual(assembly.system.balance.equation_count, len(assembly.system.state_names))
        self.assertEqual(assembly.system.balance.structural_rank, len(assembly.system.state_names))
        self.assertEqual(len(assembly.system.control_names), 4)
        self.assertIsNotNone(assembly.controller)
        self.assertEqual(
            set(assembly.controller_output_bindings.values()),
            set(assembly.system.control_names),
        )
        self.assertEqual(assembly.controller_telemetry_outputs, ("record_video",))

    def test_controller_discovers_names_parameters_and_exact_control_inventory(self) -> None:
        assembly = self.assembly
        layout = self.controller.layout
        self.assertEqual(set(layout.output_bindings.values()), set(assembly.system.control_names))
        self.assertEqual(layout.chassis_forward_speed, f"{layout.root_component}.forward_speed")
        self.assertIn(layout.camera_optical_connector, assembly.physical.connector_poses)
        self.assertIn(layout.camera_tilt_axis_connector, assembly.physical.connector_poses)
        self.assertEqual(layout.lift_feedback, "lift-servo.position_measurement")
        self.assertEqual(layout.tilt_feedback, "tilt-servo.position_measurement")
        state = np.asarray([assembly.system.initial_state], dtype=np.float64)
        controls = self.controller.evaluate(0.0, state, NumpyBackend())
        self.assertEqual(set(controls), set(assembly.system.control_names))
        for value in controls.values():
            self.assertTrue(np.all(np.isfinite(np.asarray(value))))

    def test_emergency_stop_dominates_all_four_declared_controls(self) -> None:
        controller = ScannerAssemblyController(
            self.assembly, external_inputs={"emergency_stop": True}
        )
        state = np.asarray([self.assembly.system.initial_state], dtype=np.float64)
        for coordinate, target in (
            (controller.layout.lift_coordinate, 0.3),
            (controller.layout.tilt_coordinate, -0.2),
        ):
            attachment = next(
                item
                for item in self.assembly.physical.attachments
                if item.coordinate == coordinate
            )
            for binding in attachment.coordinate_bindings:
                state[
                    0, self.assembly.system.state_names.index(binding.state)
                ] = target - binding.joint_angle_at_state_zero_rad
        state[0, self.assembly.system.state_names.index(controller.layout.lift_feedback)] = 0.3
        state[0, self.assembly.system.state_names.index(controller.layout.tilt_feedback)] = -0.2
        controls = controller.evaluate(0.0, state, NumpyBackend())
        layout = controller.layout
        np.testing.assert_allclose(controls[layout.output_bindings["left_voltage"]], 0.0)
        np.testing.assert_allclose(controls[layout.output_bindings["right_voltage"]], 0.0)
        np.testing.assert_allclose(controls[layout.output_bindings["lift_target"]], 0.3)
        np.testing.assert_allclose(controls[layout.output_bindings["tilt_target"]], -0.2)

    def test_normal_commands_enforce_canonical_slew_rates(self) -> None:
        controller = ScannerAssemblyController(self.assembly)
        state = np.asarray([self.assembly.system.initial_state], dtype=np.float64)
        initial = controller.evaluate(0.0, state, NumpyBackend())
        later = controller.evaluate(0.05, state, NumpyBackend())
        for output_name, source in controller.layout.output_bindings.items():
            maximum_delta = controller.layout.output_slew_per_second[output_name] * 0.05
            actual_delta = float(
                np.max(
                    np.abs(
                        np.asarray(later[source]) - np.asarray(initial[source])
                    )
                )
            )
            self.assertLessEqual(actual_delta, maximum_delta + 1e-12)

    def test_simulation_uses_generic_assembled_dae_and_preserves_invariants(self) -> None:
        result = self.result
        assembly = self.assembly
        self.assertEqual(result.samples.shape, (2, 2, len(assembly.system.state_names)))
        self.assertEqual(result.metadata["simulation_scope"], "component_pmdl_resolved_assembly")
        self.assertTrue(result.metadata["pmdl_network_composed"])
        self.assertEqual(result.metadata["assembly_sha256"], assembly.assembly_sha256)
        self.assertEqual(result.metadata["pmdl_sha256"], assembly.system.pmdl_sha256)
        self.assertNotIn("body_pose_frames", result.metadata)
        self.assertEqual(result.metadata["pose_frame_sample_index"], 1)
        self.assertTrue(
            result.metadata["scanner_emergency_override"]["bypasses_slew_limit"]
        )
        self.assertEqual(
            result.metadata["scanner_emergency_override"]["drive_behavior"],
            "zero_left_and_right_voltage_immediately",
        )
        self.assertEqual(
            result.metadata["scanner_emergency_override"]["joint_behavior"],
            "hold_current_lift_and_tilt_feedback",
        )
        self.assertEqual(
            result.metadata["scanner_controller"]["telemetry_outputs"],
            ["record_video"],
        )
        assumptions = result.metadata["scanner_sensor_assumptions"]
        self.assertEqual(
            assumptions["lift_feedback"]["source"],
            "lift-servo.position_measurement",
        )
        self.assertEqual(
            assumptions["measured_speed"]["fidelity"],
            "ideal_simulated_state_estimator_proxy",
        )
        self.assertEqual(len(result.metadata["scanner_control_frames"]), len(result.time))
        backend = NumpyBackend()
        samples = np.asarray(result.samples)
        control_frames = result.metadata["scanner_control_frames"]
        for index, time_s in enumerate(np.asarray(result.time)):
            # Descriptor states at t[i] satisfy the command held over the
            # preceding interval; t[0] uses the first controller frame.
            frame = control_frames[0 if index == 0 else index - 1]
            controls = {
                name: np.asarray(value, dtype=np.float64)
                for name, value in frame["applied_controls"].items()
            }
            assembly.system.require_network_invariants(
                samples[:, index, :], controls, backend, time=float(time_s)
            )

    def test_hash_bound_pose_frames_are_complete_and_viewer_compatible(self) -> None:
        scene = scanner_physical_scene(self.assembly, self.result)
        validated = validate_physical_scene(scene)
        wrapper = validated["body_pose_frames"]
        self.assertEqual(wrapper["assembly_sha256"], self.assembly.assembly_sha256)
        self.assertEqual(len(wrapper["frames"]), len(self.result.time))
        expected_bodies = set(validated["body_poses"])
        expected_connectors = set(validated["connector_poses"])
        for frame in wrapper["frames"]:
            self.assertEqual(set(frame["body_poses"]), expected_bodies)
            self.assertEqual(set(frame["connector_poses"]), expected_connectors)
        self.assertEqual(wrapper["frames"][0]["body_poses"], validated["body_poses"])
        self.assertEqual(
            wrapper["frames"][0]["connector_poses"], validated["connector_poses"]
        )
        forged = replace(
            self.result,
            metadata={
                **self.result.metadata,
                "body_pose_frames": {
                    "assembly_sha256": self.assembly.assembly_sha256,
                    "frames": [{"forged": True}],
                },
            },
        )
        self.assertEqual(
            scanner_physical_scene(self.assembly, forged),
            scene,
        )

    def test_midrun_joint_drift_fails_at_the_offending_accepted_step(self) -> None:
        assembly = self.assembly
        attachment = next(
            item
            for item in assembly.physical.attachments
            if len(item.coordinate_bindings) > 1
        )
        redundant_state = attachment.coordinate_bindings[1].state
        drift = np.zeros(len(assembly.system.state_names), dtype=np.float64)
        drift[assembly.system.state_names.index(redundant_state)] = 0.1

        class DriftingSystem:
            state_names = assembly.system.state_names
            initial_state = assembly.system.initial_state
            default_parameters = {}
            control_names = ()

            @staticmethod
            def derivative(t, state, parameters, controls, backend):
                return state * 0.0 + backend.asarray(drift)

        drifting_assembly = replace(assembly, system=DriftingSystem())
        runtime = scanner_runtime_module._ScannerSimulationRuntime(
            drifting_assembly,
            None,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(
            ScannerRuntimeError,
            r"sample=0, step_index=1, time_s=0\.1000.*coordinate",
        ):
            simulate(
                runtime,
                duration=0.2,
                dt=0.1,
                integrator="euler",
                num_samples=2,
                use_model_uncertainty=False,
                process_noise=False,
            )

    def test_metrics_report_hashes_and_validated_physical_frames(self) -> None:
        metrics = scanner_metrics(self.assembly, self.result)
        self.assertEqual(metrics["assembly_sha256"], self.assembly.assembly_sha256)
        self.assertEqual(metrics["pmdl_sha256"], self.assembly.system.pmdl_sha256)
        self.assertEqual(
            metrics["validated_physical_configuration_count"],
            self.result.samples.shape[0] * len(self.result.time),
        )
        self.assertGreater(metrics["minimum_root_keepout_clearance_m"], 0.0)
        self.assertEqual(metrics["root_keepout_violation_probability"], 0.0)
        self.assertIn("orbit_radius_rmse", metrics["acceptance"])
        self.assertIn("camera_pointing_p95", metrics["acceptance"])
        self.assertFalse(metrics["acceptance"]["dynamics_completeness"])
        self.assertFalse(metrics["accepted"])
        self.assertEqual(metrics["dynamics_completeness"]["status"], "incomplete")
        self.assertEqual(
            {gate["id"] for gate in metrics["dynamics_completeness"]["open_gates"]},
            {
                "fixed_payload_mass_inertia",
                "moving_arm_camera_inertial_derivation",
                "servo_case_reaction_coupling",
                "caster_floor_contact",
                "full_body_keepout",
                "controller_sensor_observation_binding",
                "electrical_supply_and_fault_coupling",
            },
        )

    def test_mission_override_and_nonresolved_input_fail_loudly(self) -> None:
        with self.assertRaisesRegex(TypeError, "requires a ResolvedAssembly"):
            simulate_scanner_robot({})  # type: ignore[arg-type]
        changed = replace(self.mission, orbit_radius_m=self.mission.orbit_radius_m + 0.1)
        with self.assertRaisesRegex(ScannerRuntimeError, "assembly_sha256"):
            simulate_scanner_robot(self.assembly, changed, duration=0.05)
        with self.assertRaisesRegex(ScannerRuntimeError, "visualization_sample_index"):
            simulate_scanner_robot(
                self.assembly,
                duration=0.05,
                num_samples=1,
                visualization_sample_index=1,
            )

    def test_external_inputs_are_typed_bounded_and_recorded(self) -> None:
        with self.assertRaisesRegex(ScannerRuntimeError, "unknown scanner external"):
            ScannerAssemblyController(self.assembly, external_inputs={"mystery": 1.0})
        with self.assertRaisesRegex(ValueError, "above"):
            ScannerAssemblyController(self.assembly, external_inputs={"target_speed": 99.0})
        result = simulate_scanner_robot(
            self.assembly,
            external_inputs={"target_speed": 0.1, "orbit_radius": 0.9},
            duration=0.05,
            dt=0.05,
            num_samples=1,
        )
        self.assertEqual(result.metadata["scanner_external_inputs"]["target_speed"], 0.1)
        self.assertEqual(scanner_metrics(self.assembly, result)["target_orbit_radius_m"], 0.9)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is optional")
    def test_numpy_and_torch_controller_and_simulation_outputs_match(self) -> None:
        state = np.asarray([self.assembly.system.initial_state], dtype=np.float64)
        numpy_controller = ScannerAssemblyController(self.assembly)
        numpy_controller.evaluate(0.0, state, NumpyBackend())
        numpy_values = numpy_controller.evaluate(0.25, state, NumpyBackend())
        torch_backend = TorchBackend(device="cpu", dtype="float64")
        torch_state = torch_backend.asarray(state)
        torch_controller = ScannerAssemblyController(self.assembly)
        torch_controller.evaluate(0.0, torch_state, torch_backend)
        torch_values = torch_controller.evaluate(0.25, torch_state, torch_backend)
        for name in self.assembly.system.control_names:
            np.testing.assert_allclose(
                np.asarray(numpy_values[name]),
                torch_backend.to_numpy(torch_values[name]),
                rtol=1e-12,
                atol=1e-12,
            )
        torch_result = simulate_scanner_robot(
            self.assembly,
            duration=0.05,
            dt=0.05,
            num_samples=2,
            seed=7,
            backend="torch",
            device="cpu",
            dtype="float64",
        )
        np.testing.assert_allclose(
            np.asarray(self.result.samples),
            torch_backend.to_numpy(torch_result.samples),
            rtol=1e-8,
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
