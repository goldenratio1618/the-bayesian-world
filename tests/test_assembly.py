"""Focused tests for fail-closed PMDL component-network assembly."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from contraption.physics.assembly import (
    AssemblyBalanceError,
    AssemblyError,
    NetworkInvariantError,
    UnsupportedAssemblySemanticsError,
    assemble_contraption as _assemble_contraption,
)
from contraption.physics.backend import NumpyBackend
from contraption.physics.dsl import load_model, parse_model
from contraption.physics.specs import ActuatorBindingSpec, PortRef
from contraption.physics.simulator import simulate


ROOT = Path(__file__).resolve().parents[1]


def assemble_contraption(specification, *args, **kwargs):
    if isinstance(specification, dict):
        components = tuple(
            SimpleNamespace(
                id=item["id"],
                model_id=item.get("model_id", item.get("model")),
                parameters=item.get("parameters", {}),
            )
            for item in specification["components"]
        )
        connections = tuple(
            SimpleNamespace(
                id=item["id"],
                kind=item["kind"],
                domain=item.get("domain"),
                endpoints=tuple(PortRef.from_dict(value) for value in item["endpoints"]),
                metadata=item.get("metadata", {}),
            )
            for item in specification.get("connections", [])
        )
        actuators = tuple(
            ActuatorBindingSpec.from_dict(item) for item in specification.get("actuators", [])
        )
        source = dict(specification)
        specification = SimpleNamespace(
            id=source["id"],
            name=source["name"],
            version=source["version"],
            components=components,
            connections=connections,
            actuators=actuators,
            controllers=(),
            verifications=(),
            environment=source.get("environment", {}),
            metadata=source.get("metadata", {}),
            to_dict=lambda: source,
        )
    return _assemble_contraption(specification, *args, **kwargs)


def _estimated_box() -> dict[str, object]:
    """Return explicit fallback geometry for PMDL-only unit-test components."""

    return {"kind": "box", "dimensions": [0.01, 0.01, 0.01]}


def _ground_model(*, constrained: bool = True):
    relations = (
        [{"name": "zero_potential", "expression": "ground_voltage"}]
        if constrained
        else []
    )
    return parse_model(
        {
            "format": "pmdl-1",
            "id": "electrical.reference.test",
            "name": "Test electrical reference",
            "version": "1.0.0",
            "domains": ["electrical"],
            "implements": "reference-boundary",
            "power_ports": [
                {
                    "name": "ground",
                    "domain": "electrical",
                    "effort": "ground_voltage",
                    "flow": "ground_current",
                    "effort_unit": "V",
                    "flow_unit": "A",
                    "orientation": "into_component",
                }
            ],
            "relations": relations,
            "initialization": {"strategy": "consistent", "constraints": []},
            "validity": {"assumptions": ["Ideal zero-potential reference"]},
        }
    )


def _circuit_models(*, constrained_ground: bool = True):
    models = [
        load_model(ROOT / "model_catalog" / "electrical" / "voltage_sources" / "voltage_source.pmdl"),
        load_model(ROOT / "model_catalog" / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl"),
        _ground_model(constrained=constrained_ground),
    ]
    return {model.id: model for model in models}


def _circuit_spec(*, include_return: bool = True):
    connections = [
        {
            "id": "positive",
            "kind": "power",
            "domain": "electrical",
            "endpoints": ["source.p", "load.p"],
        }
    ]
    if include_return:
        connections.append(
            {
                "id": "return",
                "kind": "power",
                "domain": "electrical",
                "endpoints": ["source.n", "load.n", "ground.ground"],
            }
        )
    return {
        "format": "resolved-assembly-test-1",
        "id": "test.resistor-circuit",
        "name": "PMDL assembled resistor circuit",
        "version": "1.0.0",
        "components": [
            {
                "id": "source",
                "model_id": "electrical.voltage_source.ideal",
            },
            {
                "id": "load",
                "model_id": "electrical.resistor.ideal",
                "parameters": {"resistance": 100.0},
            },
            {
                "id": "ground",
                "model_id": "electrical.reference.test",
            },
        ],
        "connections": connections,
        "actuators": [
            {
                "id": "source-command",
                "source": "external.voltage",
                "target": "source.voltage_command",
                "settings": {"default": 5.0},
                "external": True,
            }
        ],
    }


def _consistent_circuit_state(system) -> np.ndarray:
    values = {
        "source.v_p": 5.0,
        "source.i_p": -0.05,
        "source.v_n": 0.0,
        "source.i_n": 0.05,
        "source.voltage_command": 5.0,
        "load.v_p": 5.0,
        "load.i_p": 0.05,
        "load.v_n": 0.0,
        "load.i_n": -0.05,
        "ground.ground_voltage": 0.0,
        "ground.ground_current": 0.0,
    }
    return np.asarray([[values[name] for name in system.state_names]], dtype=float)


def _dynamic_signal_models():
    producer = parse_model(
        {
            "format": "pmdl-1",
            "id": "signal.producer.test",
            "name": "Signal producer",
            "version": "1.0.0",
            "domains": ["control"],
            "implements": "signal-source",
            "signal_ports": [
                {"name": "y", "direction": "output", "unit": "A"}
            ],
            "states": [
                {"name": "x", "unit": "A", "initial": 0.0, "derivative": "x_dot"}
            ],
            "relations": [
                {"name": "constant", "expression": "x_dot"},
                {"name": "observe", "expression": "y - x"},
            ],
            "validity": {"assumptions": ["Test producer"]},
        }
    )
    consumer = parse_model(
        {
            "format": "pmdl-1",
            "id": "signal.consumer.test",
            "name": "Signal consumer",
            "version": "1.0.0",
            "domains": ["control"],
            "implements": "signal-sink",
            "signal_ports": [
                {"name": "u", "direction": "input", "unit": "mA"}
            ],
            "states": [
                {
                    "name": "z",
                    "unit": "mA*s",
                    "initial": 0.0,
                    "derivative": "z_dot",
                }
            ],
            "relations": [
                {"name": "integrate", "expression": "z_dot - u"}
            ],
            "validity": {"assumptions": ["Test consumer"]},
        }
    )
    return producer, consumer


def _trivial_model(model_id: str):
    return parse_model(
        {
            "format": "pmdl-1",
            "id": model_id,
            "name": model_id,
            "version": "1.0.0",
            "domains": ["mechanical"],
            "implements": "rigid-part",
            "states": [
                {"name": "q", "unit": "1", "initial": 0.0, "derivative": "q_dot"}
            ],
            "relations": [{"name": "stationary", "expression": "q_dot"}],
            "validity": {"assumptions": ["Stationary test body"]},
        }
    )


def _mechanical_pair_models():
    source = parse_model(
        {
            "format": "pmdl-1",
            "id": "mechanical.torque-source.test",
            "name": "Torque source",
            "version": "1.0.0",
            "domains": ["mechanical"],
            "implements": "torque-source",
            "power_ports": [
                {
                    "name": "shaft",
                    "domain": "mechanical",
                    "effort": "torque",
                    "flow": "omega",
                    "effort_unit": "N*m",
                    "flow_unit": "rad/s",
                    "orientation": "into_component",
                }
            ],
            "parameters": [
                {"name": "source_torque", "unit": "N*m", "default": 1.0}
            ],
            "relations": [
                {"name": "imposed_torque", "expression": "torque - source_torque"}
            ],
            "validity": {"assumptions": ["Ideal torque source"]},
        }
    )
    load = parse_model(
        {
            "format": "pmdl-1",
            "id": "mechanical.rotational-load.test",
            "name": "Rotational viscous load",
            "version": "1.0.0",
            "domains": ["mechanical"],
            "implements": "rotational-load",
            "power_ports": [
                {
                    "name": "shaft",
                    "domain": "mechanical",
                    "effort": "torque",
                    "flow": "omega",
                    "effort_unit": "N*m",
                    "flow_unit": "rad/s",
                    "orientation": "into_component",
                }
            ],
            "parameters": [
                {"name": "damping", "unit": "N*m*s/rad", "default": 1.0}
            ],
            "relations": [
                {"name": "viscous_torque", "expression": "torque + damping * omega"}
            ],
            "validity": {"assumptions": ["Linear viscous load"]},
        }
    )
    return source, load


class PMDLAssemblyTests(unittest.TestCase):
    def test_bundled_acausal_models_are_namespaced_and_balanced(self) -> None:
        system = assemble_contraption(_circuit_spec(), _circuit_models())

        self.assertEqual(system.balance.unknown_count, 11)
        self.assertEqual(system.balance.equation_count, 11)
        self.assertTrue(system.balance.structurally_full_rank)
        self.assertIn("load.v_p", system.state_names)
        self.assertIn("load.ohms_law", system.residual_names)
        self.assertEqual(system.control_names, ("external.voltage",))
        self.assertRegex(system.assembly_sha256, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(system.assembly_sha256, system.pmdl_sha256)
        self.assertEqual(system.diagnostics["pmdl_sha256"], system.pmdl_sha256)

        state = _consistent_circuit_state(system)
        residual = system.residual(
            0.0,
            state,
            np.zeros_like(state),
            system.default_parameters,
            {},
            NumpyBackend(),
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    def test_canonical_physical_hash_overrides_artifact_identity_only(self) -> None:
        canonical = "sha256:" + "a" * 64
        system = assemble_contraption(
            _circuit_spec(),
            _circuit_models(),
            canonical_assembly_sha256=canonical,
        )

        self.assertEqual(system.assembly_sha256, canonical)
        self.assertRegex(system.pmdl_sha256, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(system.pmdl_sha256, canonical)
        self.assertEqual(system.diagnostics["assembly_sha256"], canonical)
        with self.assertRaisesRegex(AssemblyError, "canonical_assembly_sha256"):
            assemble_contraption(
                _circuit_spec(),
                _circuit_models(),
                canonical_assembly_sha256="sha256:ABC",
            )

    def test_component_models_are_resolved_by_instance_without_model_duplication(self) -> None:
        registry = _circuit_models()
        spec = _circuit_spec()
        for component in spec["components"]:
            component["model"] = "package.owned.placeholder"
        component_models = {
            "source": registry["electrical.voltage_source.ideal"],
            "load": registry["electrical.resistor.ideal"],
            "ground": registry["electrical.reference.test"],
        }

        system = assemble_contraption(spec, component_models=component_models)

        self.assertTrue(system.balance.structurally_full_rank)
        with self.assertRaisesRegex(AssemblyError, r"exactly match.*missing=\['ground'\]"):
            assemble_contraption(
                spec,
                component_models={key: value for key, value in component_models.items() if key != "ground"},
            )

    def test_network_invariant_check_names_the_failed_equation(self) -> None:
        system = assemble_contraption(_circuit_spec(), _circuit_models())
        backend = NumpyBackend()
        state = _consistent_circuit_state(system)
        system.require_network_invariants(state, None, backend)

        state[0, system.state_names.index("load.v_p")] = 4.9
        with self.assertRaisesRegex(
            NetworkInvariantError,
            r"connection\.positive\.effort\.load\.p.*absolute residual",
        ):
            system.require_network_invariants(state, None, backend, time=0.25)

    def test_control_binding_converts_declared_unit_scale(self) -> None:
        model = parse_model(
            {
                "format": "pmdl-1",
                "id": "signal.percent-command.test",
                "name": "Percent command sink",
                "version": "1.0.0",
                "domains": ["control"],
                "implements": "command-sink",
                "signal_ports": [
                    {"name": "command", "direction": "input", "unit": "1"}
                ],
                "validity": {"assumptions": ["Test command"]},
            }
        )
        spec = {
            "format": "resolved-assembly-test-1",
            "id": "test.percent-control",
            "name": "Unit-scaled control",
            "version": "1.0.0",
            "components": [
                {"id": "sink", "model_id": model.id, "geometry": _estimated_box()}
            ],
            "actuators": [
                {
                    "id": "command",
                    "source": "external.percent",
                    "target": "sink.command",
                    "settings": {"default": 50.0, "unit": "%"},
                    "external": True,
                }
            ],
        }
        system = assemble_contraption(spec, component_models={"sink": model})
        state = np.asarray([[0.5]])

        residuals = system.network_residuals(state, None, NumpyBackend())

        np.testing.assert_allclose(residuals["control.command"], 0.0)

    def test_control_limits_are_validated_and_never_silently_ignored(self) -> None:
        model = parse_model(
            {
                "format": "pmdl-1",
                "id": "signal.bounded-command.test",
                "name": "Bounded command sink",
                "version": "1.0.0",
                "domains": ["control"],
                "implements": "command-sink",
                "signal_ports": [
                    {"name": "command", "direction": "input", "unit": "V"}
                ],
                "validity": {"assumptions": ["Test command"]},
            }
        )
        settings = {
            "unit": "V",
            "default": 0.0,
            "minimum": -1.0,
            "maximum": 1.0,
            "slew_per_second": 2.0,
        }
        spec = {
            "format": "resolved-assembly-test-1",
            "id": "test.bounded-control",
            "name": "Bounded control",
            "version": "1.0.0",
            "components": [
                {"id": "sink", "model_id": model.id, "geometry": _estimated_box()}
            ],
            "actuators": [
                {
                    "id": "command",
                    "source": "external.command",
                    "target": "sink.command",
                    "settings": settings,
                    "external": True,
                }
            ],
        }
        system = assemble_contraption(spec, component_models={"sink": model})
        self.assertEqual(system.control_bounds["external.command"], (-1.0, 1.0))
        self.assertEqual(system.control_slew_rates["external.command"], 2.0)
        with self.assertRaisesRegex(ValueError, "violates declared bounds"):
            system.network_residuals(
                np.asarray([[0.0]]),
                {"external.command": 1.1},
                NumpyBackend(),
            )

        spec["actuators"][0]["settings"] = {"smoothing": "silent"}
        with self.assertRaisesRegex(
            UnsupportedAssemblySemanticsError, "unsupported setting.*smoothing"
        ):
            assemble_contraption(spec, component_models={"sink": model})

    def test_numpy_implicit_simulation_solves_acausal_network(self) -> None:
        system = assemble_contraption(_circuit_spec(), _circuit_models())

        result = simulate(
            system,
            duration=0.001,
            dt=0.001,
            num_samples=1,
            use_model_uncertainty=False,
            process_noise=False,
        )

        final = result.mean[-1]
        self.assertAlmostEqual(final[system.state_names.index("load.i_p")], 0.05, places=10)
        self.assertAlmostEqual(
            final[system.state_names.index("source.voltage_command")], 5.0, places=10
        )
        system.require_network_invariants(final[None, :], None, NumpyBackend())

    def test_descriptor_external_control_provider_cannot_read_plant_state(self) -> None:
        system = assemble_contraption(_circuit_spec(), _circuit_models())

        def feedback(t, state):
            return {"external.voltage": 1.0 + state[:, 0]}

        with self.assertRaisesRegex(TypeError, "open-loop.*plant state"):
            simulate(
                system,
                duration=0.001,
                dt=0.001,
                controls=feedback,
                num_samples=1,
                use_model_uncertainty=False,
                process_noise=False,
            )

    def test_torch_backend_matches_numpy_when_available(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        system = assemble_contraption(_circuit_spec(), _circuit_models())

        numpy_result = simulate(
            system,
            duration=0.001,
            dt=0.001,
            num_samples=1,
            use_model_uncertainty=False,
            process_noise=False,
        )
        torch_result = simulate(
            system,
            duration=0.001,
            dt=0.001,
            num_samples=1,
            backend="torch",
            device="cpu",
            use_model_uncertainty=False,
            process_noise=False,
        )

        np.testing.assert_allclose(torch_result.mean, numpy_result.mean, atol=1e-10)

    def test_signal_net_is_directed_and_converts_compatible_units(self) -> None:
        producer, consumer = _dynamic_signal_models()
        spec = {
            "format": "resolved-assembly-test-1",
            "id": "test.signal-network",
            "name": "Directed signal network",
            "version": "1.0.0",
            "components": [
                {
                    "id": "producer",
                    "model_id": producer.id,
                    "geometry": _estimated_box(),
                },
                {
                    "id": "consumer",
                    "model_id": consumer.id,
                    "geometry": _estimated_box(),
                },
            ],
            "connections": [
                {
                    "id": "measurement",
                    "kind": "signal",
                    "endpoints": ["producer.y", "consumer.u"],
                }
            ],
        }
        system = assemble_contraption(
            spec, component_models={"producer": producer, "consumer": consumer}
        )
        values = {
            "producer.x": 1.0,
            "producer.y": 1.0,
            "consumer.z": 0.0,
            "consumer.u": 1000.0,
        }
        derivatives = {"producer.x": 0.0, "consumer.z": 1000.0}
        state = np.asarray([[values[name] for name in system.state_names]])
        state_dot = np.asarray([[derivatives.get(name, 0.0) for name in system.state_names]])

        residual = system.residual(
            0.0, state, state_dot, {}, {}, NumpyBackend()
        )

        np.testing.assert_allclose(residual, 0.0, atol=1e-12)
        self.assertEqual(
            system.network_residual_names,
            ("connection.measurement.signal.consumer.u",),
        )

    def test_mechanical_net_shares_velocity_and_conserves_torque(self) -> None:
        source, load = _mechanical_pair_models()
        spec = {
            "format": "resolved-assembly-test-1",
            "id": "test.mechanical-network",
            "name": "Mechanical power network",
            "version": "1.0.0",
            "components": [
                {
                    "id": "source",
                    "model_id": source.id,
                    "geometry": _estimated_box(),
                },
                {
                    "id": "load",
                    "model_id": load.id,
                    "geometry": _estimated_box(),
                },
            ],
            "connections": [
                {
                    "id": "shaft",
                    "kind": "power",
                    "domain": "mechanical",
                    "endpoints": ["source.shaft", "load.shaft"],
                }
            ],
        }
        system = assemble_contraption(
            spec, component_models={"source": source, "load": load}
        )
        values = {
            "source.torque": 1.0,
            "source.omega": 1.0,
            "load.torque": -1.0,
            "load.omega": 1.0,
        }
        state = np.asarray([[values[name] for name in system.state_names]])

        residual = system.residual(
            0.0, state, np.zeros_like(state), system.default_parameters, {}, NumpyBackend()
        )

        np.testing.assert_allclose(residual, 0.0, atol=1e-12)
        self.assertIn("connection.shaft.flow.load.shaft", system.network_residual_names)
        self.assertIn("connection.shaft.effort_conservation", system.network_residual_names)

    def test_physical_only_attachment_requires_explicit_none_bindings(self) -> None:
        left = _trivial_model("mechanical.left.test")
        right = _trivial_model("mechanical.right.test")
        spec = {
            "format": "resolved-assembly-test-1",
            "id": "test.kinematic-attachment",
            "name": "Kinematic-only attachment",
            "version": "1.0.0",
            "components": [
                {
                    "id": "left",
                    "model_id": left.id,
                    "geometry": _estimated_box(),
                },
                {
                    "id": "right",
                    "model_id": right.id,
                    "geometry": _estimated_box(),
                },
            ],
            "connections": [
                {
                    "id": "mount",
                    "kind": "attachment",
                    "domain": "rigid_mechanical",
                    "endpoints": ["left.mount", "right.mount"],
                }
            ],
        }
        models = {"left": left, "right": right}

        with self.assertRaisesRegex(AssemblyError, "map it explicitly"):
            assemble_contraption(spec, component_models=models)
        system = assemble_contraption(
            spec,
            component_models=models,
            connector_bindings={"left.mount": None, "right.mount": None},
        )

        self.assertEqual(system.kinematic_connection_ids, ("mount",))
        self.assertEqual(dict(system.balance.connection_equations), {"mount": 0})

        mechanical, _ = _mechanical_pair_models()
        mixed_spec = {
            **spec,
            "components": [
                {
                    "id": "left",
                    "model_id": mechanical.id,
                    "geometry": _estimated_box(),
                },
                {
                    "id": "right",
                    "model_id": right.id,
                    "geometry": _estimated_box(),
                },
            ],
            "connections": [
                {
                    "id": "mount",
                    "kind": "attachment",
                    "domain": "rigid_mechanical",
                    "endpoints": ["left.shaft", "right.mount"],
                }
            ],
        }
        with self.assertRaisesRegex(AssemblyError, "mixes PMDL mechanical power endpoints"):
            assemble_contraption(
                mixed_spec,
                component_models={"left": mechanical, "right": right},
                connector_bindings={"right.mount": None},
            )

    def test_equation_balance_and_unconnected_ports_fail_loudly(self) -> None:
        with self.assertRaisesRegex(
            AssemblyBalanceError,
            r"underdetermined: equations=10, unknowns=11",
        ):
            assemble_contraption(
                _circuit_spec(), _circuit_models(constrained_ground=False)
            )

        with self.assertRaisesRegex(
            AssemblyError,
            r"unconnected ports=.*ground\.ground.*load\.n.*source\.n",
        ):
            assemble_contraption(
                _circuit_spec(include_return=False), _circuit_models()
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
