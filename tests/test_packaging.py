from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from contraption.catalog.interfaces import load_interface_catalog
import contraption.paths as runtime_paths
from contraption.paths import asset_root


class RuntimeAssetTests(unittest.TestCase):
    def test_complete_asset_root_is_discoverable(self) -> None:
        root = asset_root()
        catalog = root / "model_catalog"
        self.assertTrue((catalog / "electrical" / "resistors" / "fixed_resistors" / "resistor.pmdl").is_file())
        self.assertTrue((catalog / "mechanical" / "chassis" / "differential_drive_chassis" / "differential_chassis.pmdl").is_file())
        self.assertTrue(
            (catalog / "electrical" / "resistors" / "fixed_resistors" / "instantiations" / "generic-100ohm-resistor" / "static.part").is_file()
        )
        self.assertTrue(
            (
                root
                / "examples"
                / "scanner_robot"
                / "controls"
                / "scanner.control.json"
            ).is_file()
        )
        self.assertTrue((root / "web" / "viewer.js").is_file())
        self.assertGreater(len(load_interface_catalog(catalog).categories), 1)

    def test_target_install_finds_adjacent_share_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            package = target / "contraption"
            package.mkdir(parents=True)
            assets = target / "share" / "contraption"
            for relative in (
                "model_catalog/electrical/interface.pmdl",
                "model_catalog/electrical/resistors/fixed_resistors/resistor.pmdl",
                "examples/scanner_robot/contraption.json",
                "examples/scanner_robot/controls/scanner.control.json",
            ):
                path = assets / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")

            with mock.patch.object(
                runtime_paths, "__file__", str(package / "paths.py")
            ), mock.patch.dict("os.environ", {"CONTRAPTION_DATA_ROOT": ""}):
                self.assertEqual(runtime_paths.asset_root(), assets)


if __name__ == "__main__":
    unittest.main()
