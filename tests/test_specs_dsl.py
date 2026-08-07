"""Contract tests for the Phase 1 declarative model boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from contraption.dsl import (
    DSLParseError, ExpressionTypeError, ModelRegistry, evaluate_expression,
    load_model, parse_expression, parse_model,
)
from contraption.specs import (
    ComponentInstanceSpec, ConnectionSpec, ContraptionSpec, ControlBindingSpec,
    GeometrySpec, ModelSpec, PortRef, SpecError, json_schema_for,
)
from contraption.taxonomy import Taxonomy, load_default_taxonomy
from contraption.units import UnitError, parse_unit, require_compatible
from contraption.validation import (
    model_symbol_table, validate_contraption, validate_contraption_structure,
    validate_model,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


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
        self.resistor = load_model(MODELS / "electrical" / "resistor.pmdl")
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
        capacitor = load_model(MODELS / "electrical" / "capacitor.pmdl")
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
        cls.taxonomy = load_default_taxonomy()
        cls.paths = sorted(MODELS.rglob("*.pmdl"))

    def test_every_gold_model_is_strict_and_valid(self) -> None:
        self.assertGreaterEqual(len(self.paths), 8)
        for path in self.paths:
            with self.subTest(path=path):
                model = load_model(path)
                report = validate_model(model, self.taxonomy)
                self.assertTrue(report.valid, "\n".join(str(issue) for issue in report.issues))
                self.assertEqual(model.metadata["descriptor_form"], "F(t,z,zdot,theta,u)=0")
                self.assertTrue(model.metadata["gold"])

    def test_serialization_is_deterministic_and_round_trips(self) -> None:
        model = load_model(MODELS / "electrical" / "capacitor.pmdl")
        canonical = model.to_json()
        self.assertEqual(canonical, model.to_json())
        self.assertEqual(ModelSpec.from_json(canonical), model)
        self.assertEqual(json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":")), canonical)

    def test_records_are_frozen_and_mappings_are_immutable(self) -> None:
        model = load_model(MODELS / "electrical" / "resistor.pmdl")
        with self.assertRaises(FrozenInstanceError):
            model.parameters[0].default = 7.0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            model.metadata["new"] = True  # type: ignore[index]

    def test_unknown_keys_are_rejected_at_every_level(self) -> None:
        source = json.loads((MODELS / "electrical" / "resistor.pmdl").read_text(encoding="utf-8"))
        source["physics_code"] = "print('unsafe')"
        with self.assertRaises(DSLParseError):
            parse_model(source)
        source.pop("physics_code")
        source["parameters"][0]["mystery"] = 1
        with self.assertRaises(DSLParseError):
            parse_model(source)

    def test_universal_residual_evaluation(self) -> None:
        model = load_model(MODELS / "electrical" / "resistor.pmdl")
        residual = model.evaluate_residual(
            0.0, [], [], {"resistance": 100.0},
            {"v_p": 5.0, "v_n": 0.0, "i_p": 0.05, "i_n": -0.05},
        )
        np.testing.assert_allclose(residual, np.zeros(2), atol=1e-14)

    def test_nonsmooth_residual_is_rejected(self) -> None:
        source = json.loads((MODELS / "electrical" / "resistor.pmdl").read_text(encoding="utf-8"))
        source["relations"][0]["expression"] = "v_p - v_n - resistance * abs(i_p)"
        report = validate_model(parse_model(source), self.taxonomy)
        self.assertFalse(report.valid)
        self.assertIn("differentiability.nonsmooth", {issue.code for issue in report.errors})

    def test_registry_rejects_duplicate_ids(self) -> None:
        model = load_model(MODELS / "electrical" / "resistor.pmdl")
        registry = ModelRegistry([model])
        with self.assertRaises(SpecError):
            registry.register(model)


class ContraptionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = ModelRegistry()
        cls.registry.load_directory(MODELS)

    def _spec(self) -> ContraptionSpec:
        return ContraptionSpec(
            format="contraption-1", id="bench-circuit", name="Bench circuit", version="1.0.0",
            components=(
                ComponentInstanceSpec(id="source", model="electrical.voltage_source.ideal"),
                ComponentInstanceSpec(id="load", model="electrical.resistor.ideal", parameters={"resistance": 100.0}),
            ),
            connections=(
                ConnectionSpec(id="positive-net", kind="power", endpoints=(PortRef("source", "p"), PortRef("load", "p")), domain="electrical"),
                ConnectionSpec(id="return-net", kind="power", endpoints=(PortRef("source", "n"), PortRef("load", "n")), domain="electrical"),
            ),
            controls=(ControlBindingSpec(id="source-command", source="external.voltage", target=PortRef("source", "voltage_command"), external=True),),
            environment={"ambient_temperature_K": 293.15}, metadata={"fixture": True},
        )

    def test_contraption_round_trip_and_validation(self) -> None:
        spec = self._spec()
        report = validate_contraption(spec, self.registry)
        self.assertTrue(report.valid, report.issues)
        self.assertEqual(ContraptionSpec.from_json(spec.to_json()), spec)

    def test_bad_port_and_parameter_are_rejected(self) -> None:
        data = self._spec().to_dict()
        data["components"][1]["parameters"]["not_a_parameter"] = 1.0
        data["connections"][0]["endpoints"][1]["port"] = "missing"
        report = validate_contraption(ContraptionSpec.from_dict(data), self.registry)
        self.assertFalse(report.valid)
        codes = {issue.code for issue in report.errors}
        self.assertIn("component.parameter_unknown", codes)
        self.assertIn("reference.port", codes)

    def test_full_validation_requires_registry(self) -> None:
        report = validate_contraption(self._spec())
        self.assertFalse(report.valid)
        self.assertIn("registry.required", {issue.code for issue in report.errors})
        self.assertTrue(validate_contraption_structure(self._spec()).valid)

    def test_attachment_rejects_electrical_port_on_motor_shaft(self) -> None:
        spec = ContraptionSpec(
            format="contraption-1", id="bad-attachment", name="Bad attachment", version="1.0.0",
            components=(
                ComponentInstanceSpec(id="resistor", model="electrical.resistor.ideal"),
                ComponentInstanceSpec(id="motor", model="electromechanical.dc_motor.ideal"),
            ),
            connections=(
                ConnectionSpec(
                    id="electrical-to-shaft", kind="attachment", domain="rigid_mechanical",
                    endpoints=(PortRef("resistor", "p"), PortRef("motor", "shaft")),
                ),
            ),
        )
        report = validate_contraption(spec, self.registry)
        self.assertFalse(report.valid)
        codes = {issue.code for issue in report.errors}
        self.assertIn("connection.attachment_domain", codes)
        self.assertIn("connection.attachment_units", codes)

    def test_unknown_contraption_key_is_rejected(self) -> None:
        data = self._spec().to_dict()
        data["callback"] = "arbitrary_python"
        with self.assertRaises(SpecError):
            ContraptionSpec.from_dict(data)


class TaxonomyTests(unittest.TestCase):
    def test_taxonomy_round_trip_and_hierarchy(self) -> None:
        taxonomy = load_default_taxonomy()
        self.assertEqual(Taxonomy.from_json(taxonomy.to_json()), taxonomy)
        self.assertEqual(taxonomy.ancestry("camera-mass"), ("inert-object", "planar-rigid-body", "camera-mass"))
        self.assertEqual(taxonomy.category_for("brushed-dc-motor").id, "motor")

    def test_instantiation_defaults_are_safe_and_explicit(self) -> None:
        taxonomy = load_default_taxonomy()
        instance = taxonomy.instantiate(id="camera-serial-001", name="Camera 001", taxonomy_node="camera-mass", model="mechanical.camera_mass.planar")
        self.assertEqual(instance.condition, "unverified")
        self.assertEqual(instance.geometry.kind, "box")
        self.assertEqual(instance.geometry.dimensions, (0.01, 0.01, 0.01))

    def test_subcategory_requires_physical_specificity(self) -> None:
        data = load_default_taxonomy().to_dict()
        data["subcategories"][0]["model_specificity"] = "synonym"
        with self.assertRaises(SpecError):
            Taxonomy.from_dict(data)

    def test_machine_readable_schemas_are_strict(self) -> None:
        for record in (ModelSpec, ContraptionSpec, Taxonomy):
            schema = json_schema_for(record)
            root = schema["$defs"][record.__name__]
            self.assertFalse(root["additionalProperties"])
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")


if __name__ == "__main__":
    unittest.main()
