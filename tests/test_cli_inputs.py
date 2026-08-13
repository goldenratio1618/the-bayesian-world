from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from contraption.cli import (
    _controller_inputs_from_args,
    _simulation_dt,
    build_parser,
    command_doctor,
    command_serve,
    command_validate,
)
from contraption.physics.specs import FrozenDict


ROOT = Path(__file__).resolve().parents[1]
SCANNER_SPEC = ROOT / "assembled_contraptions" / "scanner" / "contraption.json"


class ControllerInputCliTests(unittest.TestCase):
    def test_contraption_commands_require_an_explicit_spec(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["simulate"])
        arguments = build_parser().parse_args(
            ["simulate", "--spec", str(SCANNER_SPEC)]
        )
        self.assertEqual(Path(arguments.spec), SCANNER_SPEC)

    def test_repeatable_json_scalars_and_strict_input_file_are_decoded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "inputs.json"
            source.write_text('{"target_speed":0.2}\n', encoding="utf-8")
            arguments = build_parser().parse_args(
                [
                    "demo",
                    "--spec",
                    str(SCANNER_SPEC),
                    "--controller-input-file",
                    str(source),
                    "--controller-input",
                    "armed=true",
                    "--controller-input",
                    "orbit_radius=0.35",
                ]
            )
            self.assertEqual(
                _controller_inputs_from_args(arguments),
                {"target_speed": 0.2, "armed": True, "orbit_radius": 0.35},
            )

    def test_controller_input_duplicates_and_non_scalar_values_fail_closed(self) -> None:
        duplicate = build_parser().parse_args(
            [
                "simulate",
                "--spec",
                str(SCANNER_SPEC),
                "--controller-input",
                "armed=true",
                "--controller-input",
                "armed=false",
            ]
        )
        with self.assertRaisesRegex(ValueError, "supplied more than once"):
            _controller_inputs_from_args(duplicate)
        nonscalar = build_parser().parse_args(
            [
                "simulate",
                "--spec",
                str(SCANNER_SPEC),
                "--controller-input",
                'armed={"forged":true}',
            ]
        )
        with self.assertRaisesRegex(ValueError, "finite JSON number or boolean"):
            _controller_inputs_from_args(nonscalar)

    def test_default_and_requested_steps_support_distinct_controller_periods(self) -> None:
        assembly = SimpleNamespace(
            controllers=FrozenDict(
                {
                    "fast": SimpleNamespace(spec=SimpleNamespace(period_s=0.01)),
                    "slow": SimpleNamespace(spec=SimpleNamespace(period_s=0.015)),
                }
            )
        )
        self.assertEqual(_simulation_dt(assembly, None), 0.005)
        self.assertEqual(_simulation_dt(assembly, 0.005), 0.005)
        with self.assertRaisesRegex(ValueError, "not commensurate"):
            _simulation_dt(assembly, 0.01)

    def test_serve_passes_initial_controller_inputs_to_live_application(self) -> None:
        arguments = build_parser().parse_args(
            [
                "serve",
                "--spec",
                str(SCANNER_SPEC),
                "--controller-input",
                "armed=true",
            ]
        )
        assembly = SimpleNamespace(
            assembly_sha256="sha256:" + "a" * 64,
            controllers=FrozenDict(),
        )
        with mock.patch(
            "contraption.cli._assembly_from_args", return_value=assembly
        ), mock.patch("contraption.cli.LiveApplication") as application, mock.patch(
            "contraption.cli.serve_live"
        ) as serve:
            with redirect_stdout(StringIO()):
                self.assertEqual(command_serve(arguments), 0)
        self.assertEqual(application.call_args.kwargs["initial_inputs"], {"armed": True})
        serve.assert_called_once_with(
            application.return_value, host="127.0.0.1", port=8000
        )

    def test_doctor_reports_available_hdl_tool_paths(self) -> None:
        paths = {
            "cc": "/usr/bin/cc",
            "iverilog": "/usr/bin/iverilog",
            "verilator": "/usr/bin/verilator",
            "yosys": "/usr/bin/yosys",
        }
        stream = StringIO()
        with mock.patch(
            "contraption.cli.shutil.which", side_effect=lambda name: paths.get(name)
        ), mock.patch(
            "contraption.cli._torch_diagnostics", return_value={"installed": False}
        ), mock.patch(
            "contraption.cli.importlib.util.find_spec", return_value=None
        ), mock.patch(
            "contraption.cli.load_dotenv_key", return_value=None
        ), redirect_stdout(stream):
            self.assertEqual(command_doctor(SimpleNamespace()), 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["iverilog"], "/usr/bin/iverilog")
        self.assertEqual(report["verilator"], "/usr/bin/verilator")
        self.assertEqual(report["yosys"], "/usr/bin/yosys")

    def test_scanner_validation_report_is_recursive_plain_json(self) -> None:
        arguments = build_parser().parse_args(
            ["validate", "--spec", str(SCANNER_SPEC)]
        )
        stream = StringIO()
        with redirect_stdout(stream):
            self.assertEqual(command_validate(arguments), 0)
        report = json.loads(stream.getvalue())
        self.assertTrue(report["valid"])
        compilation = report["controller_compilation"]["scanner_orbit_controller"]
        self.assertEqual(
            compilation["closure"]["assembly_sha256"],
            report["assembly"]["assembly_sha256"],
        )
        self.assertIn("observer_derivation", compilation["closure"])


if __name__ == "__main__":
    unittest.main()
