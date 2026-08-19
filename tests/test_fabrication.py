from __future__ import annotations

import unittest

from contraption.fabrication import (
    ConnectorFabricationSpec,
    FabricationEvidenceSpec,
    FabricationSpecError,
    RetentionSpec,
    StandardReferenceSpec,
    fabrication_compatibility_issues,
    missing_fabrication,
)
from contraption.physics.physical import PhysicalConnectorSpec, PhysicalSpecError
from contraption.physics.specs import ConnectionSpec, SpecError


class FabricationSchemaTests(unittest.TestCase):
    def test_common_metric_thread_is_dimension_checked(self) -> None:
        value = StandardReferenceSpec(
            "iso_metric_thread",
            "ISO",
            "ISO 261",
            "M3x0.5",
            nominal_diameter_m=0.003,
            pitch_m=0.0005,
        )
        self.assertEqual(value.designation, "M3x0.5")
        with self.assertRaisesRegex(FabricationSpecError, "disagrees"):
            StandardReferenceSpec(
                "iso_metric_thread",
                "ISO",
                "ISO 261",
                "M3x0.5",
                nominal_diameter_m=0.004,
                pitch_m=0.0005,
            )

    def test_rare_standard_requires_an_authoritative_extension_uri(self) -> None:
        with self.assertRaisesRegex(FabricationSpecError, "authoritative uri"):
            StandardReferenceSpec("extension", "Acme", "AC-7", "size-q")
        value = StandardReferenceSpec(
            "extension",
            "Acme",
            "AC-7",
            "size-q",
            uri="https://example.test/standards/ac-7",
        )
        self.assertEqual(value.family, "extension")

    def test_missing_records_cannot_hide_required_fields(self) -> None:
        with self.assertRaisesRegex(FabricationSpecError, "every absent required"):
            ConnectorFabricationSpec("rotary_support", "missing", ("bearing",))
        value = missing_fabrication(
            "rotary_support", ("retention", "bearing", "travel")
        )
        self.assertEqual(
            value.connector_missing_fields(), ("retention", "bearing", "travel")
        )

    def test_partial_conductor_compatibility_ignores_unknown_standards(self) -> None:
        raw = {
            "kind": "electrical_termination",
            "status": "partial",
            "missing": [
                "conductor.standard",
                "conductor.material",
                "conductor.cross_section_m2",
                "conductor.insulation_standard",
                "conductor.voltage_rating_v",
                "conductor.temperature_rating_k",
                "termination",
            ],
            "conductor": {"conductor_count": 1},
            "evidence": [{"kind": "component_input", "source": "input.json"}],
        }
        partial = ConnectorFabricationSpec.from_dict(raw)
        self.assertEqual(fabrication_compatibility_issues(partial, partial), ())

    def test_part_connector_round_trips_typed_fabrication(self) -> None:
        raw = {
            "id": "mount",
            "model_port": None,
            "body": "body",
            "domain": "mechanical",
            "interface": "fixed-mount",
            "local_pose": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "provenance": {"kind": "vendor", "source": "drawing", "reference": None},
            "fabrication": {
                "kind": "fixed_mount",
                "status": "specified",
                "missing": [],
                "retention": {"method": "integral"},
                "evidence": [{"kind": "drawing", "source": "drawing.pdf", "page": 2}],
            },
        }
        connector = PhysicalConnectorSpec.from_dict(raw)
        self.assertEqual(connector.to_dict()["fabrication"], raw["fabrication"])
        raw["fabrication"] = {
            "kind": "electrical_termination",
            "status": "missing",
            "missing": ["conductor", "termination"],
        }
        with self.assertRaisesRegex(PhysicalSpecError, "requires fabrication.kind"):
            PhysicalConnectorSpec.from_dict(raw)

    def test_connection_implementation_is_optional_but_strict(self) -> None:
        base = {
            "id": "net",
            "kind": "signal",
            "domain": "signal",
            "endpoints": [
                {"component": "source", "port": "out"},
                {"component": "sink", "port": "in"},
            ],
        }
        self.assertIsNone(ConnectionSpec.from_dict(base).implementation)
        base["implementation"] = {
            "kind": "electrical_termination",
            "status": "missing",
            "missing": ["conductor", "termination"],
        }
        with self.assertRaisesRegex(SpecError, "every missing construction field"):
            ConnectionSpec.from_dict(base)
        base["implementation"]["missing"] = [
            "conductor",
            "termination",
            "protection",
            "route",
        ]
        self.assertEqual(
            ConnectionSpec.from_dict(base).implementation.status, "missing"
        )

    def test_complete_wiring_implementation_round_trips_construction_details(self) -> None:
        awg = {
            "family": "awg",
            "authority": "ASTM",
            "document": "ASTM B258",
            "designation": "24 AWG",
            "gauge_awg": 24,
        }
        insulation = {
            "family": "manufacturer_cable",
            "authority": "Acme",
            "document": "WIRE-24",
            "designation": "PVC-300V",
        }
        raw = {
            "id": "net",
            "kind": "power",
            "domain": "electrical",
            "endpoints": [
                {"component": "source", "port": "p"},
                {"component": "load", "port": "p"},
            ],
            "implementation": {
                "kind": "electrical_termination",
                "status": "specified",
                "missing": [],
                "conductor": {
                    "standard": awg,
                    "conductor_count": 1,
                    "material": "copper",
                    "cross_section_m2": 2.05e-7,
                    "insulation_standard": insulation,
                    "voltage_rating_v": 300.0,
                    "temperature_rating_k": 378.15,
                },
                "termination": {
                    "method": "solder",
                    "installation_process": "IPC J-STD-001 qualified solder joint",
                },
                "protection": {"kind": "none"},
                "route": {
                    "topology": "point_to_point",
                    "routed_length_m": 0.25,
                    "minimum_bend_radius_m": 0.01,
                    "service_loop_m": 0.02,
                    "strain_relief": "clamp",
                    "waypoints": ["frame.clip-1"],
                },
                "evidence": [
                    {
                        "kind": "manual",
                        "source": "approved-wiring-record.json",
                        "sha256": "sha256:" + "a" * 64,
                    }
                ],
            },
        }
        parsed = ConnectionSpec.from_dict(raw)
        self.assertEqual(ConnectionSpec.from_dict(parsed.to_dict()), parsed)
        self.assertEqual(parsed.implementation.route.routed_length_m, 0.25)
        self.assertEqual(parsed.implementation.conductor.standard.gauge_awg, 24)
        self.assertEqual(parsed.implementation.implementation_missing_fields(), ())

    def test_known_standard_conflicts_fail_closed(self) -> None:
        evidence = (FabricationEvidenceSpec("drawing", "drawing.pdf"),)
        left = ConnectorFabricationSpec(
            "fixed_mount",
            "specified",
            (),
            standards=(
                StandardReferenceSpec(
                    "iso_metric_thread",
                    "ISO",
                    "ISO 261",
                    "M3x0.5",
                    role="internal",
                    nominal_diameter_m=0.003,
                    pitch_m=0.0005,
                ),
            ),
            retention=RetentionSpec("integral"),
            evidence=evidence,
        )
        right = ConnectorFabricationSpec(
            "fixed_mount",
            "specified",
            (),
            standards=(
                StandardReferenceSpec(
                    "iso_metric_thread",
                    "ISO",
                    "ISO 261",
                    "M4x0.7",
                    role="external",
                    nominal_diameter_m=0.004,
                    pitch_m=0.0007,
                ),
            ),
            retention=RetentionSpec("integral"),
            evidence=evidence,
        )
        self.assertTrue(fabrication_compatibility_issues(left, right))


if __name__ == "__main__":
    unittest.main()
