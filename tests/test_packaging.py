from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import contraption.paths as runtime_paths
from contraption.paths import asset_root
from contraption.taxonomy import default_taxonomy_path, load_default_taxonomy


class RuntimeAssetTests(unittest.TestCase):
    def test_complete_asset_root_is_discoverable(self) -> None:
        root = asset_root()
        self.assertTrue((root / "models" / "electrical" / "resistor.pmdl").is_file())
        self.assertTrue((root / "models" / "scanner" / "differential_chassis.pmdl").is_file())
        self.assertTrue(
            (root / "examples" / "scanner_robot" / "component_packages.json").is_file()
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
        self.assertEqual(default_taxonomy_path(), root / "data" / "taxonomy.json")
        self.assertGreater(len(load_default_taxonomy().categories), 1)

    def test_target_install_finds_adjacent_share_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            package = target / "contraption"
            package.mkdir(parents=True)
            assets = target / "share" / "contraption"
            for relative in (
                "data/taxonomy.json",
                "models/electrical/resistor.pmdl",
                "models/scanner/differential_chassis.pmdl",
                "examples/scanner_robot/contraption.json",
                "examples/scanner_robot/component_packages.json",
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
