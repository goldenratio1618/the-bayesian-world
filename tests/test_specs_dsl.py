"""Contract tests for the Phase 1 declarative model boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from contraption.physics.dsl import (
    DSLParseError, ExpressionTypeError, ModelRegistry, evaluate_expression,
    load_model, parse_expression, parse_model,
)
from contraption.physics.specs import (
    ActuatorBindingSpec, CatalogLinkSpec, ComponentReferenceSpec, ConnectionSpec,
    ContraptionSpec, ControllerOutputBindingSpec, FrozenDict, ModelSpec, PortRef,
    SpecError, json_schema_for,
)
from contraption.catalog.instantiations import PartInstantiationRegistry
from contraption.catalog.interfaces import ModelInterfaceCatalog, load_interface_catalog
from contraption.physics.units import UnitError, parse_unit, require_compatible
from contraption.physics.validation import (
    model_symbol_table, validate_contraption_structure,
    validate_model,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "model_catalog"
RESISTOR = MODELS / "electrical" / "resistors" / "resistor.pmdl"
CAPACITOR = MODELS / "electrical" / "capacitors" / "capacitor.pmdl"
PLANAR_BODY = (
    ROOT
    / "assembled_contraptions"
    / "examples"
    / "test_systems"
    / "planar_rigid_body"
    / "catalog"
    / "mechanical"
    / "reference_systems"
    / "planar_rigid_body.pmdl"
)


class UnitTests(unittest.TestCase):
    def test_derived_units_are_compatible(self) -> None:
        require_compatible("V*A", "W")
        require_compatible("N*m", "J")
        require_compatible("C/s", "A")
        require_compatible("kg*m^2/s^3", "W")

    def test_unit_conversion_is_explicit(self) -> None:
        self.assertAlmostEqual(parse_unit("cm").convert_value_to(100.0, parse_unit("m")), 1.0)

    def test_unit_parser_rejects_code_and_unknown_units(self) -> None:
        for source in ("__import__('os')", "V;A", "furlong"):
            with self.subTest(source=source), self.assertRaises(UnitError):
                parse_unit(source)


class ExpressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resistor = load_model(RESISTOR)
        self.symbols = model_symbol_table(self.resistor)

    def test_allowed_expression_evaluates(self) -> None:
        expression = parse_expression("v_p - v_n - resistance * i_p")
        self.assertEqual(expression.variables(), frozenset({"v_p", "v_n", "resistance", "i_p"}))
        self.assertAlmostEqual(evaluate_expression(expression, {"v_p": 5.0, "v_n": 0.0, "resistance": 100.0, "i_p": 0.05}), 0.0)
        self.assertEqual(expression.infer_type(self.symbols).dimension, parse_unit("V").dimension)

    def test_dimension_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ExpressionTypeError):
            parse_expression("v_p + i_p").infer_type(self.symbols)

    def test_tanh_is_safe_dimensionless_math(self) -> None:
        expression = parse_expression("tanh(0.5)")
        inferred = expression.infer_type(self.symbols)
        self.assertTrue(inferred.dimension.is_dimensionless)
        self.assertAlmostEqual(evaluate_expression(expression, {}), np.tanh(0.5))

        values = np.array([-2.0, 0.0, 2.0])
        np.testing.assert_allclose(
            evaluate_expression("tanh(value)", {"value": values}),
            np.tanh(values),
        )

    def test_tanh_fails_loudly_for_units_or_wrong_arity(self) -> None:
        with self.assertRaisesRegex(
            ExpressionTypeError,
            "tanh requires a dimensionless argument",
        ):
            parse_expression("tanh(v_p)").infer_type(self.symbols)
        with self.assertRaisesRegex(ExpressionTypeError, r"tanh\(\) expects 1 argument"):
            parse_expression("tanh(0.1, 0.2)")

    def test_derivative_has_state_unit_per_second(self) -> None:
        capacitor = load_model(CAPACITOR)
        result = parse_expression("der(charge)").infer_type(model_symbol_table(capacitor))
        self.assertEqual(result.dimension, parse_unit("A").dimension)

    def test_escape_syntax_is_rejected(self) -> None:
        forbidden = (
            "__import__('os').system('echo unsafe')",
            "(1).__class__",
            "open('secret')",
            "x[0]",
            "lambda x: x",
            "[x for x in y]",
        )
        for source in forbidden:
            with self.subTest(source=source), self.assertRaises(DSLParseError):
                parse_expression(source)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaises(DSLParseError):
            parse_model('{"format":"pmdl-1","format":"pmdl-1"}')


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.interfaces = load_interface_catalog(MODELS)
        cls.paths = sorted(
            path for path in MODELS.rglob("*.pmdl") if path.name != "interface.pmdl"
        )

    def test_every_gold_model_is_strict_and_valid(self) -> None:
        gold_paths = [
            path for path in self.paths if bool(load_model(path).metadata.get("gold", False))
        ]
        self.assertGreaterEqual(len(gold_paths), 8)
        for path in gold_paths:
            with self.subTest(path=path):
                model = load_model(path)
                report = validate_model(model, self.interfaces)
                self.assertTrue(report.valid, "\n".join(str(issue) for issue in report.issues))
                self.assertEqual(model.metadata["descriptor_form"], "F(t,z,zdot,theta,u)=0")
                self.assertTrue(model.metadata["gold"])

    def test_every_bundled_model_is_structurally_valid(self) -> None:
        """Prototype device models are valid PMDL without claiming taxonomy gold status."""

        self.assertGreaterEqual(len(self.paths), 8)
        for path in self.paths:
            with self.subTest(path=path):
                model = load_model(path)
                report = validate_model(model)
                self.assertTrue(report.valid, "\n".join(str(issue) for issue in report.issues))
                self.assertEqual(model.metadata["descriptor_form"], "F(t,z,zdot,theta,u)=0")

    def test_hbridge_has_truthful_electrical_interface(self) -> None:
        model = load_model(MODELS / "electrical" / "power_converters" / "dual_hbridges" / "dual_hbridge.pmdl")
        self.assertEqual(model.domains, ("electrical",))
        report = validate_model(model, self.interfaces)
        self.assertTrue(report.valid, "\n".join(str(issue) for issue in report.issues))

    def test_serialization_is_deterministic_and_round_trips(self) -> None:
        model = load_model(CAPACITOR)
        canonical = model.to_json()
        self.assertEqual(canonical, model.to_json())
        self.assertEqual(ModelSpec.from_json(canonical), model)
        self.assertEqual(json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":")), canonical)

    def test_records_are_frozen_and_mappings_are_immutable(self) -> None:
        model = load_model(RESISTOR)
        with self.assertRaises(FrozenInstanceError):
            model.parameters[0].default = 7.0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            model.metadata["new"] = True  # type: ignore[index]

    def test_unknown_keys_are_rejected_at_every_level(self) -> None:
        source = json.loads(RESISTOR.read_text(encoding="utf-8"))
        source["physics_code"] = "print('unsafe')"
        with self.assertRaises(DSLParseError):
            parse_model(source)
        source.pop("physics_code")
        source["parameters"][0]["mystery"] = 1
        with self.assertRaises(DSLParseError):
            parse_model(source)

    def test_process_noise_is_strict_typed_and_canonical(self) -> None:
        planar = load_model(PLANAR_BODY)
        report = validate_model(planar)
        self.assertTrue(report.valid, report.to_dict())
        self.assertEqual(
            planar.process_noise.seed_policy,
            "simulation_seed",
        )
        self.assertEqual(
            planar.process_noise.reproducibility,
            "same_backend_device",
        )
        self.assertEqual(
            tuple(channel.name for channel in planar.process_noise.channels),
            ("roughness_x", "roughness_y", "roughness_yaw"),
        )

        resistor_source = json.loads(RESISTOR.read_text(encoding="utf-8"))
        absent = parse_model(resistor_source)
        resistor_source["process_noise"] = {}
        explicit_empty = parse_model(resistor_source)
        self.assertEqual(absent, explicit_empty)
        self.assertNotIn("process_noise", explicit_empty.to_dict())

        resistor_source["process_noise"] = {
            "seed_policy": "simulation_seed",
        }
        with self.assertRaisesRegex(DSLParseError, "missing process_noise key"):
            parse_model(resistor_source)

    def test_process_noise_rejects_nonstate_targets_and_unit_errors(self) -> None:
        source = json.loads(PLANAR_BODY.read_text(encoding="utf-8"))
        source["process_noise"]["increments"][0]["target"] = "y_observation"
        report = validate_model(parse_model(source))
        self.assertIn("process_noise.target", {issue.code for issue in report.errors})

        source = json.loads(PLANAR_BODY.read_text(encoding="utf-8"))
        source["process_noise"]["increments"][0]["expression"] = (
            "roughness_reference_length * sqrt(dt) * roughness_x"
        )
        report = validate_model(parse_model(source))
        self.assertIn("process_noise.unit", {issue.code for issue in report.errors})

        source = json.loads(PLANAR_BODY.read_text(encoding="utf-8"))
        source["process_noise"]["increments"][0]["expression"] = (
            "roughness_std * roughness_x * sqrt(t)"
        )
        report = validate_model(parse_model(source))
        codes = {issue.code for issue in report.errors}
        self.assertIn("process_noise.dt", codes)

        source = json.loads(PLANAR_BODY.read_text(encoding="utf-8"))
        source["process_noise"]["channels"][0]["name"] = "x_dot"
        source["process_noise"]["increments"][0]["expression"] = (
            "roughness_std * sqrt(dt) * x_dot"
        )
        report = validate_model(parse_model(source))
        self.assertIn(
            "process_noise.channel_collision",
            {issue.code for issue in report.errors},
        )

    def test_universal_residual_evaluation(self) -> None:
        model = load_model(RESISTOR)
        residual = model.evaluate_residual(
            0.0, [], [], {"resistance": 100.0},
            {"v_p": 5.0, "v_n": 0.0, "i_p": 0.05, "i_n": -0.05},
        )
        np.testing.assert_allclose(residual, np.zeros(2), atol=1e-14)

    def test_nonsmooth_residual_is_rejected(self) -> None:
        source = json.loads(RESISTOR.read_text(encoding="utf-8"))
        source["relations"][0]["expression"] = "v_p - v_n - resistance * abs(i_p)"
        report = validate_model(parse_model(source), self.interfaces)
        self.assertFalse(report.valid)
        self.assertIn("differentiability.nonsmooth", {issue.code for issue in report.errors})

    def test_registry_rejects_duplicate_ids(self) -> None:
        model = load_model(RESISTOR)
        registry = ModelRegistry([model])
        with self.assertRaises(SpecError):
            registry.register(model)


class ContraptionContractTests(unittest.TestCase):
    def _spec(self) -> ContraptionSpec:
        return ContraptionSpec(
            format="contraption-4", id="bench-circuit", name="Bench circuit", version="1.0.0",
            catalogs=(CatalogLinkSpec("model_catalog"),),
            physical_root=FrozenDict({"component": "source"}),
            components=(
                ComponentReferenceSpec(id="source", instantiation="bench.source.v1"),
                ComponentReferenceSpec(id="load", instantiation="generic-100ohm-resistor.v1"),
            ),
            connections=(
                ConnectionSpec(id="positive-net", kind="power", endpoints=(PortRef("source", "p"), PortRef("load", "p")), domain="electrical"),
                ConnectionSpec(id="return-net", kind="power", endpoints=(PortRef("source", "n"), PortRef("load", "n")), domain="electrical"),
            ),
            actuators=(ActuatorBindingSpec(id="source-command", source="voltage", target=PortRef("source", "voltage_command"), external=True),),
            environment={"ambient_temperature_K": 293.15}, metadata={"fixture": True},
        )

    def test_contraption_round_trip_and_validation(self) -> None:
        spec = self._spec()
        report = validate_contraption_structure(spec)
        self.assertTrue(report.valid, report.issues)
        self.assertEqual(ContraptionSpec.from_json(spec.to_json()), spec)

    def test_component_parameters_and_models_are_rejected(self) -> None:
        data = self._spec().to_dict()
        data["components"][1]["parameters"] = {"resistance": 10.0}
        with self.assertRaises(SpecError):
            ContraptionSpec.from_dict(data)
        data = self._spec().to_dict()
        data["components"][1]["model"] = "electrical.resistor.ideal"
        with self.assertRaises(SpecError):
            ContraptionSpec.from_dict(data)

    def test_structural_validation_rejects_unknown_component(self) -> None:
        data = self._spec().to_dict()
        data["connections"][0]["endpoints"][1]["component"] = "missing"
        report = validate_contraption_structure(ContraptionSpec.from_dict(data))
        self.assertFalse(report.valid)
        self.assertIn("reference.component", {issue.code for issue in report.errors})

    def test_structural_validation_accepts_external_controller_output(self) -> None:
        data = self._spec().to_dict()
        data["controllers"] = [
            {
                "id": "camera-controller",
                "program": {
                    "path": "controls/camera.control",
                    "sha256": "sha256:" + "0" * 64,
                },
                "explicit_inputs": {},
                "implicit_inputs": {},
                "outputs": {"record_video": {"external": "record_video"}},
            }
        ]
        report = validate_contraption_structure(ContraptionSpec.from_dict(data))
        self.assertTrue(report.valid, report.issues)

    def test_unknown_contraption_key_is_rejected(self) -> None:
        data = self._spec().to_dict()
        data["callback"] = "arbitrary_python"
        with self.assertRaises(SpecError):
            ContraptionSpec.from_dict(data)

    def test_contraption_metadata_is_required(self) -> None:
        data = self._spec().to_dict()
        del data["metadata"]
        with self.assertRaisesRegex(SpecError, "missing.*metadata"):
            ContraptionSpec.from_dict(data)

    def test_controller_output_bindings_are_strict_typed_unions(self) -> None:
        self.assertEqual(
            ControllerOutputBindingSpec.from_dict(
                {"signal": "controller.command"}
            ).signal,
            "controller.command",
        )
        self.assertEqual(
            ControllerOutputBindingSpec.from_dict(
                {"external": "record_video"}
            ).external,
            "record_video",
        )
        for invalid in (
            "controller.command",
            {},
            {"signal": "controller.command", "external": "record_video"},
            {"telemetry": "record_video"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(SpecError):
                ControllerOutputBindingSpec.from_dict(invalid)  # type: ignore[arg-type]


class CatalogContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.interfaces = load_interface_catalog(MODELS)
        cls.models = ModelRegistry()
        cls.models.load_directory(MODELS, interfaces=cls.interfaces)

    def test_interface_round_trip_and_hierarchy(self) -> None:
        self.assertEqual(
            ModelInterfaceCatalog.from_json(self.interfaces.to_json()).to_dict(),
            self.interfaces.to_dict(),
        )
        self.assertEqual(self.interfaces.ancestry("camera-mass"), ("inert-object", "camera-mass"))
        self.assertEqual(self.interfaces.category_for("brushed-dc-motor").id, "motor")

    def test_generic_ideal_models_are_colocated_with_category_interfaces(self) -> None:
        resistor = MODELS / "electrical" / "resistors" / "resistor.pmdl"
        capacitor = MODELS / "electrical" / "capacitors" / "capacitor.pmdl"
        self.assertTrue(resistor.is_file())
        self.assertTrue(capacitor.is_file())
        self.assertEqual(self.models["electrical.resistor.ideal"].implements, "resistor")
        self.assertEqual(self.models["electrical.capacitor.ideal"].implements, "capacitor")

    def test_instantiations_have_static_parts_and_initialized_models(self) -> None:
        registry = PartInstantiationRegistry.load_catalog(MODELS, models=self.models)
        self.assertIn("generic-100ohm-resistor.v1", registry)
        self.assertIn("C1210C476K8RAC.v1", registry)
        self.assertIn("scanner.position_servo.v2", registry)
        self.assertEqual(registry["generic-100ohm-resistor.v1"].parameters["resistance"]["value"], 100.0)

    def test_interface_requires_physical_specificity(self) -> None:
        data = self.interfaces.to_dict()
        data["devices"][0]["model_specificity"] = "synonym"
        with self.assertRaises(SpecError):
            ModelInterfaceCatalog.from_dict(data)

    def test_machine_readable_schemas_are_strict(self) -> None:
        for record in (ModelSpec, ContraptionSpec):
            schema = json_schema_for(record)
            root = schema["$defs"][record.__name__]
            self.assertFalse(root["additionalProperties"])
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(
            json_schema_for(ContraptionSpec)["$defs"]["ContraptionSpec"]["properties"]["format"],
            {"const": "contraption-4"},
        )


if __name__ == "__main__":
    unittest.main()
