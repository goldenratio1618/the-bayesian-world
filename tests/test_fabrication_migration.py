from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from contraption.fabrication import (
    ConductorSpec,
    ConnectorFabricationSpec,
    FabricationSpecError,
)
from contraption.physics.physical import PhysicalSpecError
from scripts.migrate_static_parts_v2 import _migrate_model, _migrate_static, migrate


REPOSITORY = Path(__file__).resolve().parents[1]
SERVO_STATIC = REPOSITORY / (
    "model_catalog/electromechanical/servos/position_servos/instantiations/"
    "generic_position_servo/static.part"
)
ROMI_INPUT = REPOSITORY / (
    "outputs/scanner-part-import/component_inputs/romi_arm.json"
)
EXPECTED_INPUT_SHA256 = (
    "sha256:71181f6affed7b36e53bc77d637de781c498f41047a667d261a0eaa5947b0a8c"
)
EXPECTED_MISSING = [
    "conductor.standard",
    "conductor.material",
    "conductor.cross_section_m2",
    "conductor.insulation_standard",
    "conductor.voltage_rating_v",
    "conductor.temperature_rating_k",
    "termination",
]


def _connector(path: Path, connector_id: str) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return next(item for item in data["connectors"] if item["id"] == connector_id)


class PartialConductorTests(unittest.TestCase):
    def test_known_count_keeps_every_other_conductor_field_missing(self) -> None:
        conductor = ConductorSpec.from_dict({"conductor_count": 1})
        self.assertEqual(
            conductor.missing_fields,
            (
                "standard",
                "material",
                "cross_section_m2",
                "insulation_standard",
                "voltage_rating_v",
                "temperature_rating_k",
            ),
        )
        fabrication = ConnectorFabricationSpec.from_dict(
            {
                "kind": "electrical_termination",
                "status": "partial",
                "missing": EXPECTED_MISSING,
                "conductor": {"conductor_count": 1},
                "evidence": [
                    {"kind": "component_input", "source": "input.json"}
                ],
            }
        )
        self.assertEqual(
            list(fabrication.connector_missing_fields()), EXPECTED_MISSING
        )

    def test_empty_conductor_and_hidden_unknowns_are_rejected(self) -> None:
        with self.assertRaisesRegex(FabricationSpecError, "at least one known"):
            ConductorSpec.from_dict({})
        with self.assertRaisesRegex(FabricationSpecError, "every absent required"):
            ConnectorFabricationSpec.from_dict(
                {
                    "kind": "electrical_termination",
                    "status": "partial",
                    "missing": ["termination"],
                    "conductor": {"conductor_count": 1},
                    "evidence": [
                        {"kind": "component_input", "source": "input.json"}
                    ],
                }
            )


class FabricationEvidenceMigrationTests(unittest.TestCase):
    def test_servo_feedback_wire_has_exact_evidence_and_only_known_count(self) -> None:
        connector = _connector(SERVO_STATIC, "position_measurement")
        fabrication = connector["fabrication"]
        self.assertEqual(fabrication["status"], "partial")
        self.assertEqual(fabrication["conductor"], {"conductor_count": 1})
        self.assertEqual(fabrication["missing"], EXPECTED_MISSING)
        self.assertEqual(
            fabrication["evidence"],
            [
                {
                    "kind": "component_input",
                    "source": "outputs/scanner-part-import/component_inputs/romi_arm.json",
                    "locator": "$.features.feedback",
                    "sha256": EXPECTED_INPUT_SHA256,
                }
            ],
        )
        self.assertEqual(
            "sha256:" + hashlib.sha256(ROMI_INPUT.read_bytes()).hexdigest(),
            EXPECTED_INPUT_SHA256,
        )
        self.assertEqual(
            json.loads(ROMI_INPUT.read_text(encoding="utf-8"))["features"]["feedback"],
            "separate potentiometer feedback wire on each servo",
        )

    def test_extractor_rejects_changed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            target = repository / ROMI_INPUT.relative_to(REPOSITORY)
            target.parent.mkdir(parents=True)
            data = json.loads(ROMI_INPUT.read_text(encoding="utf-8"))
            data["features"]["feedback"] = "feedback lead"
            target.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to infer"):
                _migrate_static(SERVO_STATIC, repository=repository)

    def test_migration_is_idempotent_for_evidence_backed_part(self) -> None:
        report = migrate(
            (SERVO_STATIC.parent,), check=True, repository=REPOSITORY
        )
        self.assertEqual(report["changed"], [])

    def test_nominal_geometry_and_ambiguous_cable_terms_stay_missing(self) -> None:
        candidates = (
            (
                REPOSITORY
                / "model_catalog/electrical/resistors/fixed_resistors/instantiations/yageo_rc0603_100r/static.part",
                "p",
            ),
            (
                REPOSITORY
                / "model_catalog/electrical/capacitors/ceramic_capacitors/instantiations/C1210C476K8RAC/static.part",
                "p",
            ),
            (
                REPOSITORY
                / "model_catalog/optical/cameras/powered_rotational_cameras/instantiations/scanner_camera/static.part",
                "supply_p",
            ),
        )
        for path, connector_id in candidates:
            with self.subTest(path=path, connector=connector_id):
                fabrication = _connector(path, connector_id)["fabrication"]
                self.assertEqual(fabrication["status"], "missing")
                self.assertEqual(
                    fabrication["missing"], ["conductor", "termination"]
                )
                self.assertNotIn("conductor", fabrication)
                self.assertNotIn("termination", fabrication)
                self.assertNotIn("standards", fabrication)

        servo_shaft = _connector(SERVO_STATIC, "shaft")["fabrication"]
        self.assertEqual(servo_shaft["status"], "missing")
        self.assertEqual(
            servo_shaft["missing"], ["retention", "bearing", "travel"]
        )
        self.assertNotIn("bearing", servo_shaft)
        self.assertNotIn("retention", servo_shaft)
        self.assertNotIn("travel", servo_shaft)


class ModelMigrationChurnTests(unittest.TestCase):
    def test_model_without_reserved_identity_preserves_exact_bytes(self) -> None:
        source_path = REPOSITORY / (
            "model_catalog/electrical/resistors/fixed_resistors/instantiations/"
            "generic-100ohm-resistor/v1.model"
        )
        data = json.loads(source_path.read_text(encoding="utf-8"))
        data["metadata"] = {}
        raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False) + "  \n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.model"
            path.write_text(raw, encoding="utf-8", newline="")
            rendered, report = _migrate_model(path)
        self.assertEqual(rendered, raw)
        self.assertEqual(report["removed_identity"], {})

    def test_model_migration_removes_only_top_level_legacy_identity(self) -> None:
        source_path = REPOSITORY / (
            "model_catalog/electrical/resistors/fixed_resistors/instantiations/"
            "generic-100ohm-resistor/v1.model"
        )
        data = json.loads(source_path.read_text(encoding="utf-8"))
        data["metadata"] = {"manufacturer": "Acme", "note": "preserved"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.model"
            path.write_text(json.dumps(data), encoding="utf-8")
            rendered, report = _migrate_model(path)
        migrated = json.loads(rendered)
        self.assertEqual(report["removed_identity"], {"manufacturer": "Acme"})
        self.assertEqual(migrated["metadata"], {"note": "preserved"})

        data["metadata"] = {"procurement": {"manufacturer": "Acme"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.model"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(
                PhysicalSpecError,
                r"model_instance\.metadata\.procurement\.manufacturer",
            ):
                _migrate_model(path)


if __name__ == "__main__":
    unittest.main()
