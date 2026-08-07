from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from contraption.build import generate_build_instructions
from contraption.compiler import OnlineModelIR, compile_contraption
from contraption.controls import ControlProgram, ControllerRuntime
from contraption.scanner import (
    ScannerMission,
    ScannerSimulationCoverageError,
    make_scanner_aggregate_model,
    scanner_metrics,
    simulate_scanner_robot,
    validate_scanner_simulation_coverage,
)
from contraption.specs import ContraptionSpec
from contraption.validation import validate_contraption_structure
from contraption.visualization import generate_viewer


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "scanner_robot"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class ScannerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec_data = load(EXAMPLE / "contraption.json")
        cls.coverage_data = load(EXAMPLE / "simulation_coverage.json")
        cls.program_data = load(EXAMPLE / "controls" / "scanner.control.json")
        cls.online_data = load(EXAMPLE / "online_model.json")
        cls.program = ControlProgram.from_dict(cls.program_data)

    def test_manifest_structure_and_controller_are_strict(self):
        spec = ContraptionSpec.from_dict(self.spec_data)
        self.assertTrue(validate_contraption_structure(spec).valid)
        runtime = ControllerRuntime(self.program)
        frame = runtime.step({"armed": True}, dt=0.01)
        self.assertEqual(frame.state, "scanning")
        stopped = runtime.step({"armed": True, "emergency_stop": True}, dt=0.01)
        self.assertEqual(stopped.state, "emergency")

    def test_internal_estop_dominates_batch(self):
        # This deliberately exercises the labeled low-level aggregate API.  It
        # makes no claim that a physical PMDL network was assembled.
        robot = make_scanner_aggregate_model(self.program)
        robot.controller.emergency_stop = True
        state = np.asarray([[0.85, 0.0, math.pi / 2, 4.0, 4.0, 0.2, 0.1]])
        from contraption.backend import NumpyBackend

        command = robot.controller.evaluate(0.0, state, NumpyBackend())
        np.testing.assert_array_equal(command["left_voltage"], [0.0])
        np.testing.assert_array_equal(command["right_voltage"], [0.0])
        np.testing.assert_array_equal(command["arm_command"], state[:, 5])

    def test_uncertain_closed_loop_mission_meets_reference_acceptance(self):
        result = simulate_scanner_robot(
            self.program,
            ScannerMission(duration_s=10.0),
            physical_spec=self.spec_data,
            simulation_coverage=self.coverage_data,
            duration=10.0,
            dt=0.05,
            num_samples=32,
            seed=7,
        )
        self.assertEqual(result.samples.shape, (32, 201, 7))
        metrics = scanner_metrics(result)
        # The short unit-test batch verifies motion behavior, but is correctly
        # too small to certify a <0.001 collision probability at 95% confidence.
        self.assertTrue(metrics["acceptance"]["orbit_radius_rmse"], metrics)
        self.assertTrue(metrics["acceptance"]["camera_pointing_p95"], metrics)
        self.assertFalse(metrics["acceptance"]["collision_probability"], metrics)
        self.assertFalse(metrics["accepted"], metrics)
        self.assertEqual(metrics["collision_probability"], 0.0)
        self.assertGreater(metrics["collision_probability_upper_95_wilson"], 0.001)
        self.assertFalse(result.metadata["pmdl_network_composed"])
        self.assertEqual(
            result.metadata["simulation_coverage_topology_sha256"],
            self.coverage_data["topology_sha256"],
        )

    def test_simulation_coverage_contract_matches_reference_topology(self):
        digest = validate_scanner_simulation_coverage(
            self.spec_data, self.coverage_data
        )
        self.assertEqual(digest, self.coverage_data["topology_sha256"])
        self.assertEqual(
            {entry["id"] for entry in self.coverage_data["components"]},
            {entry["id"] for entry in self.spec_data["components"]},
        )
        self.assertEqual(
            {entry["id"] for entry in self.coverage_data["connections"]},
            {entry["id"] for entry in self.spec_data["connections"]},
        )

    def test_simulation_fails_loudly_for_added_component(self):
        changed = copy.deepcopy(self.spec_data)
        addition = copy.deepcopy(changed["components"][0])
        addition["id"] = "uncovered-component"
        changed["components"].append(addition)
        with self.assertRaisesRegex(
            ScannerSimulationCoverageError,
            r"missing component coverage=\['uncovered-component'\]",
        ):
            simulate_scanner_robot(
                self.program,
                physical_spec=changed,
                simulation_coverage=self.coverage_data,
                duration=0.05,
                num_samples=1,
            )

    def test_simulation_fails_loudly_for_component_model_drift(self):
        changed = copy.deepcopy(self.spec_data)
        changed["components"][0]["model"] = "electrical.incompatible-model"
        with self.assertRaisesRegex(
            ScannerSimulationCoverageError,
            r"coverage component 'battery-pack' model drift",
        ):
            simulate_scanner_robot(
                self.program,
                physical_spec=changed,
                simulation_coverage=self.coverage_data,
                duration=0.05,
                num_samples=1,
            )

    def test_simulation_fails_loudly_for_connection_endpoint_drift(self):
        changed = copy.deepcopy(self.spec_data)
        changed["connections"][0]["endpoints"][0] = "battery-pack.unreviewed_terminal"
        with self.assertRaisesRegex(
            ScannerSimulationCoverageError,
            r"coverage connection 'battery-positive-bus' endpoints drift",
        ):
            simulate_scanner_robot(
                self.program,
                physical_spec=changed,
                simulation_coverage=self.coverage_data,
                duration=0.05,
                num_samples=1,
            )

    def test_simulation_fails_loudly_when_coverage_is_omitted(self):
        with self.assertRaisesRegex(
            ScannerSimulationCoverageError,
            r"simulation_coverage is required.*may not silently omit",
        ):
            simulate_scanner_robot(
                self.program,
                physical_spec=self.spec_data,
                duration=0.05,
                num_samples=1,
            )

    def test_exclusion_requires_nonempty_rationale(self):
        invalid = copy.deepcopy(self.coverage_data)
        battery = next(entry for entry in invalid["components"] if entry["id"] == "battery-pack")
        battery["representations"][0]["limitation"] = "  "
        with self.assertRaisesRegex(
            ScannerSimulationCoverageError,
            r"exclusion requires a non-empty rationale/limitation",
        ):
            simulate_scanner_robot(
                self.program,
                physical_spec=self.spec_data,
                simulation_coverage=invalid,
                duration=0.05,
                num_samples=1,
            )

    def test_online_compiler_scopes_whole_contraption(self):
        ir = OnlineModelIR.from_dict(self.online_data)
        self.assertEqual(ir.A.shape, (6, 6))
        artifact = compile_contraption(
            self.spec_data,
            assembled_system=self.online_data,
            model_name="scanner_test",
        )
        self.assertIn("contraption_scope", artifact.manifest)
        scope = artifact.manifest["contraption_scope"]
        self.assertEqual(scope["validation_level"], "reviewed_abstraction")
        self.assertFalse(scope["models_registry_validated"])
        self.assertIn("not registry-validated", scope["admission"])
        self.assertEqual(
            set(scope["component_ids"]),
            {component["id"] for component in self.spec_data["components"]},
        )
        self.assertEqual(
            set(scope["connection_ids"]),
            {connection["id"] for connection in self.spec_data["connections"]},
        )
        self.assertNotIn("malloc(", artifact.source)
        self.assertIn("scanner_test_predict", artifact.source)
        with tempfile.TemporaryDirectory() as temporary:
            paths = artifact.write(temporary)
            self.assertTrue(paths["scanner_test.c"].is_file())
            manifest = json.loads(
                paths["scanner_test.manifest.json"].read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["contraption_scope"]["contraption_id"], "apartment-scanner-robot")

    def test_build_and_viewer_artifacts_are_offline(self):
        plan = generate_build_instructions(self.spec_data)
        self.assertGreaterEqual(len(plan.steps), 5)
        self.assertGreaterEqual(len(plan.wiring), 8)
        self.assertTrue(any("certification" in note for note in plan.safety_notes))
        trajectory = {
            "time": [0.0, 0.1],
            "state_names": ["x", "y", "yaw"],
            "mean": [[0.85, 0.0, math.pi / 2], [0.85, 0.01, math.pi / 2]],
        }
        viewer = generate_viewer(self.spec_data, trajectory, runtime_model=self.online_data)
        self.assertIn("Romi orbital 3D-scanning", viewer.html)
        self.assertNotIn("src=\"http", viewer.html)
        self.assertIn("contraption.viewer/v1", viewer.html)


if __name__ == "__main__":
    unittest.main()
