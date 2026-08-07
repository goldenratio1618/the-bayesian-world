from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from contraption.build import BuildInstructionError, generate_build_instructions
from contraption.visualization import VisualizationError, generate_viewer


def scanner_fragment() -> dict:
    return {
        "id": "apartment_scanner",
        "metadata": {
            "external_controls": [
                {
                    "name": "drive_speed",
                    "label": "Drive speed",
                    "min": -0.5,
                    "max": 0.5,
                    "default": 0.1,
                    "unit": "m/s",
                    "visual_effects": [
                        {
                            "component": "controller",
                            "property": "position.x",
                            "scale": 0.2,
                        }
                    ],
                }
            ]
        },
        "components": [
            {
                "id": "chassis",
                "model": "rigid_chassis",
                "geometry": {
                    "kind": "box",
                    "dimensions": [0.35, 0.28, 0.06],
                    "position": [0.0, 0.0, 0.08],
                },
                "metadata": {
                    "display_name": "Mobile chassis",
                    "fabrication_method": "FDM 3D print",
                },
            },
            {
                "id": "battery",
                "model": "lithium_battery",
                "geometry": {
                    "kind": "box",
                    "dimensions": [0.14, 0.05, 0.04],
                    "position": [-0.15, 0.0, 0.11],
                },
                "parameters": {"rated_voltage_v": 12.0},
                "purchasing": {
                    "name": "Protected 12 V battery pack",
                    "manufacturer": "Example",
                    "part_number": "BAT-12",
                },
                "condition": "unverified",
            },
            {
                "id": "controller",
                "model": "motor_controller",
                "geometry": {
                    "kind": "box",
                    "dimensions": [0.08, 0.06, 0.02],
                    "position": [0.15, 0.0, 0.11],
                },
                "purchasing": {
                    "name": "Dual motor controller",
                    "part_number": "CTRL-2",
                },
            },
            {
                "id": "unsafe_name_</script><script>alert(1)</script>",
                "model": "inert_payload",
                "geometry": {"kind": "sphere", "dimensions": [0.04, 0.04, 0.04]},
                "metadata": {"fixed": True},
            },
        ],
        "connections": [
            {
                "id": "controller_mount",
                "kind": "attachment",
                "endpoints": ["chassis.electronics_bay", "controller.mount"],
                "metadata": {
                    "fastener": {
                        "type": "socket-head screw",
                        "size": "M3x8",
                        "quantity": 4,
                        "material": "stainless steel",
                        "torque_nm": 0.5,
                        "locking_method": "medium threadlocker",
                    }
                },
            },
            {
                "id": "battery_power",
                "kind": "power",
                "endpoints": ["battery.positive", "controller.supply_positive"],
                "metadata": {
                    "wire_gauge": "18 AWG",
                    "color": "red",
                    "signal": "power",
                    "protection": "5 A fuse at battery",
                },
            },
        ],
        "controls": [{"source": "external.drive_speed", "target": "controller.command"}],
    }


class BuildInstructionTests(unittest.TestCase):
    def test_deterministic_bom_wiring_fasteners_and_safety(self) -> None:
        first = generate_build_instructions(scanner_fragment())
        second = generate_build_instructions(scanner_fragment())
        self.assertEqual(first, second)
        self.assertEqual(first.to_markdown(), second.to_markdown())
        self.assertEqual(first.wiring[0].length_m, 0.5)
        self.assertEqual(first.fasteners[0].quantity, 4)
        self.assertEqual(first.fasteners[0].torque_nm, 0.5)
        self.assertTrue(any(item.part_number == "BAT-12" for item in first.bill_of_materials))
        self.assertTrue(any("battery" in note.lower() for note in first.safety_notes))
        self.assertIn("Continuity/polarity", first.wiring[0].verification)
        self.assertIn("Assembly procedure", first.to_markdown())

    def test_missing_ratings_become_unresolved_not_invented(self) -> None:
        spec = scanner_fragment()
        del spec["connections"][1]["metadata"]["wire_gauge"]
        for component in spec["components"]:
            component.get("geometry", {}).pop("position", None)
        result = generate_build_instructions(spec)
        self.assertTrue(any("wire gauge" in item for item in result.unresolved))
        self.assertTrue(any("wire_length_m" in item for item in result.unresolved))

    def test_rejects_duplicate_component_identifiers(self) -> None:
        spec = scanner_fragment()
        spec["components"][1]["id"] = "chassis"
        with self.assertRaisesRegex(BuildInstructionError, "unique"):
            generate_build_instructions(spec)

    def test_unknown_connection_component_fails_loudly(self) -> None:
        spec = scanner_fragment()
        spec["connections"][0]["endpoints"][1] = "missing-controller.mount"
        with self.assertRaisesRegex(
            BuildInstructionError, "controller_mount.*missing-controller"
        ):
            generate_build_instructions(spec)

    def test_unsupported_connection_kind_fails_loudly(self) -> None:
        spec = scanner_fragment()
        spec["connections"][0]["kind"] = "telepathy"
        with self.assertRaisesRegex(BuildInstructionError, "unsupported.*telepathy"):
            generate_build_instructions(spec)


class VisualizationTests(unittest.TestCase):
    def test_standalone_viewer_embeds_geometry_trajectory_and_controls(self) -> None:
        trajectory = {
            "time": [0.0, 0.1, 0.2],
            "state_names": ["x", "y", "yaw", "arm_elevation", "camera_pitch"],
            "mean": [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.02, 0.0, 0.1, 0.02, -0.1],
                [0.04, 0.01, 0.2, 0.04, -0.2],
            ],
        }
        artifact = generate_viewer(scanner_fragment(), trajectory, title="Scanner <offline>")
        self.assertIn("<canvas id=\"scene\"", artifact.html)
        self.assertIn("electrical-diagram", artifact.html)
        self.assertIn("global-alpha", artifact.html)
        self.assertIn("drive_speed", artifact.html)
        self.assertIn("\\u003c/script\\u003e", artifact.html)
        self.assertNotIn("<script src=", artifact.html)
        self.assertNotIn("<link rel=\"stylesheet\"", artifact.html)
        self.assertIn("wheel", artifact.javascript)
        self.assertIn("pointermove", artifact.javascript)
        self.assertIn("requestAnimationFrame", artifact.javascript)
        self.assertEqual(artifact.data["schema"], "contraption.viewer/v1")

    def test_writes_single_page_and_inspectable_bundle(self) -> None:
        artifact = generate_viewer(scanner_fragment())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single = artifact.write(root / "scanner.html")
            self.assertEqual(set(single), {"scanner.html"})
            self.assertIn(
                "Self-contained offline viewer",
                (root / "scanner.html").read_text(encoding="utf-8"),
            )
            bundle = artifact.write(root / "bundle")
            self.assertEqual(
                set(bundle), {"index.html", "viewer.js", "style.css", "viewer-data.json"}
            )
            data = json.loads(
                (root / "bundle" / "viewer-data.json").read_text(encoding="utf-8")
            )
            self.assertEqual(data["specification"]["id"], "apartment_scanner")

    def test_rejects_non_finite_browser_data(self) -> None:
        with self.assertRaisesRegex(VisualizationError, "NaN"):
            generate_viewer(scanner_fragment(), {"time": [float("nan")]})


if __name__ == "__main__":
    unittest.main()
