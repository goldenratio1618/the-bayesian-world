from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from contraption.compiler import (
    IRValidationError,
    OnlineCompiler,
    OnlineModelIR,
    _records,
    _topology_sha256,
    compile_contraption,
)
from contraption.controls import (
    ControlProgram,
    ControlValidationError,
    ControllerRuntime,
    evaluate_state_outputs,
    load_control_program,
)
from contraption.specs import ModelSpec


ROOT = Path(__file__).resolve().parents[1]


def controller_data() -> dict:
    return {
        "name": "arm_controller",
        "version": "1.0.0",
        "inputs": [
            {
                "name": "enable",
                "type": "boolean",
                "source": "external",
                "default": False,
            },
            {
                "name": "height",
                "type": "number",
                "source": "sensor",
                "default": 0.0,
                "min": -2.0,
                "max": 2.0,
                "unit": "m",
            },
        ],
        "outputs": [
            {
                "name": "arm_command",
                "type": "number",
                "source": "output",
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
            }
        ],
        "parameters": {"target": 0.8, "gain": 2.0},
        "registers": {
            "integral": {"initial": 0.0, "min": -0.25, "max": 0.25}
        },
        "states": [
            {
                "name": "idle",
                "outputs": {"arm_command": 0.0},
                "updates": {"integral": 0.0},
                "transitions": [
                    {
                        "target": "tracking",
                        "priority": 10,
                        "when": {"ref": "external.enable"},
                    }
                ],
            },
            {
                "name": "tracking",
                "outputs": {
                    "arm_command": {
                        "op": "clamp",
                        "args": [
                            {
                                "op": "mul",
                                "args": [
                                    {"ref": "parameter.gain"},
                                    {
                                        "op": "sub",
                                        "args": [
                                            {"ref": "parameter.target"},
                                            {"ref": "sensor.height"},
                                        ],
                                    },
                                ],
                            },
                            -1.0,
                            1.0,
                        ],
                    }
                },
                "updates": {
                    "integral": {
                        "op": "add",
                        "args": [
                            {"ref": "register.integral"},
                            {
                                "op": "mul",
                                "args": [
                                    {"op": "sub", "args": [{"ref": "parameter.target"}, {"ref": "sensor.height"}]},
                                    {"ref": "dt"},
                                ],
                            },
                        ],
                    }
                },
                "transitions": [
                    {
                        "target": "idle",
                        "priority": 10,
                        "when": {"op": "not", "args": [{"ref": "external.enable"}]},
                    }
                ],
            },
        ],
        "initial_state": "idle",
    }


def online_ir_data() -> dict:
    return {
        "kind": "linear",
        "state_names": ["position", "velocity"],
        "input_names": ["force"],
        "measurement_names": ["position_sensor"],
        "A": [[0.0, 1.0], [0.0, -0.2]],
        "B": [[0.0], [1.0]],
        "C": [[1.0, 0.0]],
        "D": [[0.0]],
        "dynamics_bias": [0.0, 0.0],
        "measurement_bias": [0.0],
        "Q": [[1e-5, 0.0], [0.0, 1e-4]],
        "R": [[1e-3]],
        "x0": [0.0, 0.0],
        "P0": [[0.1, 0.0], [0.0, 0.1]],
        "dt": 0.01,
        "max_dt": 0.05,
        "state_bounds": [[-10.0, 10.0], [-5.0, 5.0]],
    }


def covered_online_ir(spec: dict, *, reviewed: bool) -> dict:
    data = online_ir_data()
    components = _records(spec["components"], "components")
    connections = _records(spec.get("connections", []), "connections")
    coverage = {
        "schema": "contraption.online-assembly-coverage/v1",
        "component_ids": [component["id"] for component in components],
        "connection_ids": [connection["id"] for connection in connections],
        "component_models": {
            component["id"]: component["model"] for component in components
        },
        "topology_sha256": _topology_sha256(components, connections),
    }
    if reviewed:
        coverage["review"] = {
            "review_id": "unit-test-review",
            "reviewed_by": "compiler unit test",
            "basis": "Explicit fixture review of the tiny affine abstraction.",
            "component_contracts_reviewed": True,
            "ports_and_connections_reviewed": True,
            "assembled_ir_coverage_reviewed": True,
            "limitations": ["The matrix fixture is synthetic."],
        }
    data["metadata"] = {"assembly_coverage": coverage}
    return data


def admitted_fixture_model(relative_path: str) -> ModelSpec:
    data = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    data.setdefault("metadata", {})["online_admission"] = {
        "admitted": True,
        "kind": "linearizable",
        "mechanics": "rigid_body",
    }
    return ModelSpec.from_dict(data)


class ControlProgramTests(unittest.TestCase):
    def test_synchronous_state_machine_and_bounded_register(self) -> None:
        runtime = ControllerRuntime(controller_data())
        first = runtime.step(external={"enable": True}, sensors={"height": 0.0}, dt=0.1)
        self.assertEqual(first.outputs["arm_command"], 0.0)
        self.assertEqual(first.state, "tracking")
        self.assertEqual(first.transitioned_from, "idle")

        second = runtime.step(external={"enable": True}, sensors={"height": 0.0}, dt=0.1)
        self.assertEqual(second.outputs["arm_command"], 1.0)
        self.assertAlmostEqual(second.registers["integral"], 0.08)
        for _ in range(10):
            second = runtime.step(
                external={"enable": True}, sensors={"height": -2.0}, dt=0.1
            )
        self.assertEqual(second.registers["integral"], 0.25)

        final = runtime.step(external={"enable": False}, sensors={"height": 0.8}, dt=0.1)
        self.assertEqual(final.state, "idle")
        self.assertEqual(final.transitioned_from, "tracking")

    def test_program_round_trip_and_json_loader(self) -> None:
        program = ControlProgram.from_dict(controller_data())
        loaded = load_control_program(json.dumps(program.to_dict()))
        self.assertEqual(loaded.to_dict(), program.to_dict())

    def test_rejects_code_strings_unknown_ops_and_ambiguous_transitions(self) -> None:
        code = controller_data()
        code["states"][0]["outputs"]["arm_command"] = "__import__('os').system('bad')"
        with self.assertRaises(ControlValidationError):
            ControlProgram.from_dict(code)

        unknown = controller_data()
        unknown["states"][0]["outputs"]["arm_command"] = {
            "op": "python_eval",
            "args": [1.0],
        }
        with self.assertRaisesRegex(ControlValidationError, "unsupported"):
            ControlProgram.from_dict(unknown)

        ambiguous = controller_data()
        ambiguous["states"][0]["transitions"].append(
            {"target": "tracking", "priority": 10, "when": True}
        )
        with self.assertRaisesRegex(ControlValidationError, "ambiguous"):
            ControlProgram.from_dict(ambiguous)

    def test_numpy_batch_evaluation_preserves_samples(self) -> None:
        program = ControlProgram.from_dict(controller_data())
        heights = np.array([-0.2, 0.4, 1.4])
        outputs = evaluate_state_outputs(
            program,
            "tracking",
            {"height": heights, "enable": np.array([True, True, True]), "dt": 0.01},
            backend="numpy",
        )
        np.testing.assert_allclose(outputs["arm_command"], [1.0, 0.8, -1.0])


class CompilerTests(unittest.TestCase):
    def test_validates_and_emits_deterministic_fixed_allocation_c99(self) -> None:
        compiler = OnlineCompiler()
        first = compiler.compile(online_ir_data(), model_name="scanner_online")
        second = compiler.compile(online_ir_data(), model_name="scanner_online")
        self.assertEqual(first.header, second.header)
        self.assertEqual(first.source, second.source)
        self.assertEqual(first.manifest_json, second.manifest_json)
        self.assertIn("#define SCANNER_ONLINE_NX 2", first.header)
        self.assertIn("Joseph covariance update", first.source)
        self.assertNotIn("malloc", first.source)
        self.assertNotIn("calloc", first.source)
        self.assertEqual(first.manifest["dimensions"]["states"], 2)
        self.assertEqual(first.manifest["numeric_contract"]["allocation"], "fixed/stack-or-struct; no heap")

        result = first.syntax_check()
        if result.compiler is not None:
            self.assertTrue(result.ok, result.stderr)

    def test_disk_artifacts_are_complete(self) -> None:
        artifact = OnlineCompiler().compile(online_ir_data(), model_name="disk_model")
        with tempfile.TemporaryDirectory() as directory:
            paths = artifact.write(directory)
            self.assertEqual(
                set(paths),
                {"disk_model.c", "disk_model.h", "disk_model.manifest.json"},
            )
            manifest = json.loads(
                paths["disk_model.manifest.json"].read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["model_name"], "disk_model")

    def test_rejects_bad_covariance_and_dimensions(self) -> None:
        bad = online_ir_data()
        bad["R"] = [[0.0]]
        with self.assertRaisesRegex(IRValidationError, "positive definite"):
            OnlineModelIR.from_dict(bad)
        bad = online_ir_data()
        bad["B"] = [[1.0], [2.0], [3.0]]
        with self.assertRaisesRegex(IRValidationError, "B must"):
            OnlineModelIR.from_dict(bad)

    def test_contraption_facade_checks_model_admission(self) -> None:
        spec = {
            "format": "contraption-1",
            "id": "tiny_bot",
            "name": "Tiny bot",
            "version": "1.0.0",
            "components": [
                {
                    "id": "body",
                    "model": "rigid_body",
                    "online_admission": {"admitted": True, "kind": "linearizable", "mechanics": "rigid_body"},
                },
                {
                    "id": "motor",
                    "model": "dc_motor",
                    "online_admission": {"admitted": True, "kind": "linearized", "mechanics": "rigid_body"},
                },
            ],
            "connections": [
                {
                    "id": "mount",
                    "kind": "attachment",
                    "endpoints": [
                        {"component": "body", "port": "mount"},
                        {"component": "motor", "port": "case"},
                    ],
                }
            ],
        }
        artifact = compile_contraption(
            spec,
            assembled_system=covered_online_ir(spec, reviewed=True),
            model_name="tiny_bot",
        )
        self.assertEqual(artifact.manifest["contraption_scope"]["component_ids"], ["body", "motor"])
        self.assertEqual(
            artifact.manifest["contraption_scope"]["connection_ids"], ["mount"]
        )
        self.assertEqual(
            artifact.manifest["contraption_scope"]["validation_level"],
            "reviewed_abstraction",
        )
        self.assertFalse(
            artifact.manifest["contraption_scope"]["models_registry_validated"]
        )
        self.assertIn(
            "not registry-validated",
            artifact.manifest["contraption_scope"]["admission"],
        )

        legacy = online_ir_data()
        legacy["metadata"] = {
            "admitted_models": ["rigid_body", "dc_motor"]
        }
        with self.assertRaisesRegex(IRValidationError, "not an admission contract"):
            compile_contraption(spec, assembled_system=legacy)

    def test_assembly_coverage_rejects_component_connection_and_topology_drift(self) -> None:
        spec = {
            "format": "contraption-1",
            "id": "covered_bot",
            "name": "Covered bot",
            "version": "1.0.0",
            "components": [
                {"id": "body", "model": "rigid_body"},
                {"id": "motor", "model": "dc_motor"},
            ],
            "connections": [
                {
                    "id": "mount",
                    "kind": "attachment",
                    "endpoints": ["body.mount", "motor.case"],
                }
            ],
        }

        missing_component = covered_online_ir(spec, reviewed=True)
        coverage = missing_component["metadata"]["assembly_coverage"]
        coverage["component_ids"].remove("motor")
        coverage["component_models"].pop("motor")
        with self.assertRaisesRegex(IRValidationError, "component_ids.*missing"):
            compile_contraption(spec, assembled_system=missing_component)

        missing_connection = covered_online_ir(spec, reviewed=True)
        missing_connection["metadata"]["assembly_coverage"]["connection_ids"] = []
        with self.assertRaisesRegex(IRValidationError, "connection_ids.*missing"):
            compile_contraption(spec, assembled_system=missing_connection)

        extra_connection = covered_online_ir(spec, reviewed=True)
        extra_connection["metadata"]["assembly_coverage"]["connection_ids"].append(
            "not-represented"
        )
        with self.assertRaisesRegex(IRValidationError, "connection_ids.*extra"):
            compile_contraption(spec, assembled_system=extra_connection)

        stale_topology = covered_online_ir(spec, reviewed=True)
        spec["connections"][0]["endpoints"][1] = "motor.other_case"
        with self.assertRaisesRegex(IRValidationError, "topology_sha256"):
            compile_contraption(spec, assembled_system=stale_topology)

        unknown_component = covered_online_ir(spec, reviewed=True)
        spec["connections"][0]["endpoints"][1] = "ghost.case"
        with self.assertRaisesRegex(IRValidationError, "unknown component endpoint"):
            compile_contraption(spec, assembled_system=unknown_component)

    def test_full_registry_validates_models_ports_and_domains(self) -> None:
        resistor = admitted_fixture_model("models/electrical/resistor.pmdl")
        motor = admitted_fixture_model("models/electrical/dc_motor.pmdl")
        registry = {resistor.id: resistor, motor.id: motor}

        def spec_with_motor_port(port: str) -> dict:
            return {
                "format": "contraption-1",
                "id": "registry_bot",
                "name": "Registry bot",
                "version": "1.0.0",
                "components": [
                    {"id": "resistor", "model": resistor.id},
                    {"id": "motor", "model": motor.id},
                ],
                "connections": [
                    {
                        "id": "electrical-link",
                        "kind": "power",
                        "endpoints": ["resistor.p", f"motor.{port}"],
                    }
                ],
            }

        valid = spec_with_motor_port("p")
        artifact = compile_contraption(
            valid,
            model_registry=registry,
            assembled_system=covered_online_ir(valid, reviewed=False),
        )
        scope = artifact.manifest["contraption_scope"]
        self.assertEqual(scope["validation_level"], "validated_model_registry")
        self.assertTrue(scope["models_registry_validated"])
        self.assertEqual(
            scope["admission"],
            "all referenced models validated through explicit registry",
        )

        unknown_port = spec_with_motor_port("missing_port")
        with self.assertRaisesRegex(
            IRValidationError, "compatibility validation failed.*reference.port"
        ):
            compile_contraption(
                unknown_port,
                model_registry=registry,
                assembled_system=covered_online_ir(unknown_port, reviewed=False),
            )

        incompatible_domain = spec_with_motor_port("shaft")
        with self.assertRaisesRegex(
            IRValidationError, "compatibility validation failed.*connection.domain"
        ):
            compile_contraption(
                incompatible_domain,
                model_registry=registry,
                assembled_system=covered_online_ir(incompatible_domain, reviewed=False),
            )


if __name__ == "__main__":
    unittest.main()
