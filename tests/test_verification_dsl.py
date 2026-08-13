from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from contraption.verification import (
    DifferentiabilityClass,
    VerificationRuntimeError,
    VerificationSpecError,
    dump_verification,
    evaluate_verification,
    load_verification,
    parse_verification,
)


def verification_data() -> dict:
    return {
        "format": "verification-1",
        "id": "test.trajectory",
        "name": "Trajectory reducer verification",
        "version": "1.0.0",
        "description": "Exercises every verification-1 time reducer.",
        "inputs": [
            {"name": "x", "unit": "m", "description": "Scalar position."},
        ],
        "parameters": [
            {"name": "target", "unit": "m", "value": 0.0},
            {"name": "closure_limit", "unit": "m", "value": 1.5},
            {"name": "final_limit", "unit": "m", "value": 1.0},
        ],
        "metrics": [
            {"name": "initial_x", "expression": "x", "reducer": "initial", "unit": "m"},
            {"name": "final_x", "expression": "x", "reducer": "final", "unit": "m"},
            {"name": "mean_x", "expression": "x", "reducer": "mean", "unit": "m"},
            {"name": "min_x", "expression": "x", "reducer": "min", "unit": "m"},
            {"name": "max_x", "expression": "x", "reducer": "max", "unit": "m"},
            {
                "name": "rmse_x",
                "expression": "x - target",
                "reducer": "rmse",
                "unit": "m",
            },
        ],
        "criteria": [
            {
                "name": "closure",
                "expression": "abs(final_x - initial_x) <= closure_limit",
                "minimum_probability": 0.3,
                "confidence_level": 0.95,
            },
            {
                "name": "strict_final",
                "expression": "final_x <= final_limit",
                "minimum_probability": 0.3,
                "confidence_level": 0.95,
            },
        ],
    }


class VerificationParsingTests(unittest.TestCase):
    def test_parse_is_strict_typed_and_canonically_hashable(self) -> None:
        program = parse_verification(verification_data())
        self.assertEqual(program.format, "verification-1")
        self.assertEqual(program.id, "test.trajectory")
        self.assertEqual(len(program.metrics), 6)
        self.assertRegex(program.sha256, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(program.sha256, parse_verification(program.canonical_json()).sha256)
        self.assertEqual(
            program.metrics[0].differentiability,
            DifferentiabilityClass.SMOOTH,
        )
        self.assertEqual(
            program.criteria[0].differentiability,
            DifferentiabilityClass.DISCRETE,
        )
        self.assertEqual(program.differentiability["admission"], "discrete")

        reordered = json.dumps(verification_data(), indent=4, sort_keys=True)
        self.assertEqual(program.sha256, parse_verification(reordered).sha256)
        self.assertEqual(dump_verification(program, indent=None), program.canonical_json())

    def test_loader_reads_a_normalized_program(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.verify"
            path.write_text(json.dumps(verification_data(), indent=2), encoding="utf-8")
            loaded = load_verification(path)
            normalized_path = Path(directory) / "normalized.verify"
            dump_verification(loaded, normalized_path)
            self.assertEqual(load_verification(normalized_path).sha256, loaded.sha256)

    def test_unknown_duplicate_and_nonfinite_json_are_rejected(self) -> None:
        unknown = verification_data()
        unknown["escape_hatch"] = True
        with self.assertRaisesRegex(VerificationSpecError, "unknown.*escape_hatch"):
            parse_verification(unknown)
        with self.assertRaisesRegex(VerificationSpecError, "duplicate JSON field"):
            parse_verification('{"format":"verification-1","format":"verification-1"}')
        with self.assertRaisesRegex(VerificationSpecError, "non-finite"):
            parse_verification('{"format": NaN}')

    def test_expressions_are_allow_listed_and_dimension_checked(self) -> None:
        unsafe = copy.deepcopy(verification_data())
        unsafe["metrics"][0]["expression"] = "__import__('os').system('id')"
        with self.assertRaisesRegex(VerificationSpecError, "not allow-listed|direct allow-listed"):
            parse_verification(unsafe)

        graph_break = copy.deepcopy(verification_data())
        graph_break["metrics"][0]["expression"] = "detach(x)"
        with self.assertRaisesRegex(
            VerificationSpecError, "not allow-listed|backend-native differentiability"
        ):
            parse_verification(graph_break)

        derivative = copy.deepcopy(verification_data())
        derivative["metrics"][0]["expression"] = "der(x)"
        derivative["metrics"][0]["unit"] = "m/s"
        with self.assertRaisesRegex(VerificationSpecError, "may not use der"):
            parse_verification(derivative)

        unknown = copy.deepcopy(verification_data())
        unknown["metrics"][0]["expression"] = "hidden_state"
        with self.assertRaisesRegex(VerificationSpecError, "unknown symbol"):
            parse_verification(unknown)

        wrong_unit = copy.deepcopy(verification_data())
        wrong_unit["metrics"][0]["unit"] = "s"
        with self.assertRaisesRegex(VerificationSpecError, "expected real/time"):
            parse_verification(wrong_unit)

    def test_criteria_must_be_boolean_and_use_only_metrics_or_parameters(self) -> None:
        numeric = copy.deepcopy(verification_data())
        numeric["criteria"][0]["expression"] = "final_x"
        with self.assertRaisesRegex(VerificationSpecError, "must be boolean"):
            parse_verification(numeric)

        raw_input = copy.deepcopy(verification_data())
        raw_input["criteria"][0]["expression"] = "x <= closure_limit"
        with self.assertRaisesRegex(VerificationSpecError, "unknown symbol 'x'"):
            parse_verification(raw_input)

        bad_probability = copy.deepcopy(verification_data())
        bad_probability["criteria"][0]["minimum_probability"] = 1.01
        with self.assertRaisesRegex(VerificationSpecError, "within \\[0, 1\\]"):
            parse_verification(bad_probability)

        missing_confidence = copy.deepcopy(verification_data())
        del missing_confidence["criteria"][0]["confidence_level"]
        with self.assertRaisesRegex(VerificationSpecError, "missing.*confidence_level"):
            parse_verification(missing_confidence)

        bad_confidence = copy.deepcopy(verification_data())
        bad_confidence["criteria"][0]["confidence_level"] = 1.0
        with self.assertRaisesRegex(VerificationSpecError, "within \\(0.5, 1\\)"):
            parse_verification(bad_confidence)


class VerificationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.program = parse_verification(verification_data())
        self.x = np.asarray(
            [
                [0.0, 1.0, 1.0],
                [0.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.5, 1.5],
            ],
            dtype=np.float64,
        )
        self.time = np.asarray([0.0, 0.25, 2.0], dtype=np.float64)

    def test_all_reducers_use_exact_irregular_time_and_confidence_gates(self) -> None:
        report = evaluate_verification(self.program, {"x": self.x}, time=self.time)
        self.assertEqual(report.sample_count, 4)
        self.assertEqual(report.time_count, 3)
        np.testing.assert_allclose(report.metrics["initial_x"].values, [0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(report.metrics["final_x"].values, [1.0, 2.0, 1.0, 1.5])
        widths = np.diff(self.time)
        expected_mean = np.sum(
            0.5 * (self.x[:, :-1] + self.x[:, 1:]) * widths,
            axis=1,
        ) / (self.time[-1] - self.time[0])
        np.testing.assert_allclose(report.metrics["mean_x"].values, expected_mean)
        np.testing.assert_allclose(report.metrics["min_x"].values, [0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(report.metrics["max_x"].values, [1.0, 2.0, 1.0, 1.5])
        expected_mean_square = np.sum(
            0.5 * (self.x[:, :-1] ** 2 + self.x[:, 1:] ** 2) * widths,
            axis=1,
        ) / (self.time[-1] - self.time[0])
        np.testing.assert_allclose(
            report.metrics["rmse_x"].values,
            np.sqrt(expected_mean_square),
        )

        closure = report.criteria["closure"]
        self.assertEqual(closure.pass_count, 3)
        self.assertEqual(closure.probability_estimate, 0.75)
        self.assertEqual(closure.effective_sample_count, 4.0)
        self.assertLess(closure.probability_lower_bound, closure.probability_estimate)
        self.assertGreater(closure.probability_upper_bound, closure.probability_estimate)
        self.assertTrue(closure.accepted)
        strict = report.criteria["strict_final"]
        self.assertEqual(strict.pass_count, 2)
        self.assertEqual(strict.probability_estimate, 0.5)
        self.assertFalse(strict.accepted)
        self.assertFalse(report.accepted)
        payload = report.to_dict()
        self.assertEqual(payload["program"]["sha256"], self.program.sha256)
        self.assertEqual(payload["schema"], "contraption.verification-report/v2")
        self.assertIn("probability_lower_bound", payload["criteria"]["closure"])

    def test_inputs_are_exact_finite_scalar_trajectory_ensembles(self) -> None:
        with self.assertRaisesRegex(VerificationRuntimeError, "coverage mismatch"):
            evaluate_verification(self.program, {}, time=self.time)
        with self.assertRaisesRegex(VerificationRuntimeError, "unknown=.*extra"):
            evaluate_verification(
                self.program, {"x": self.x, "extra": self.x}, time=self.time
            )
        with self.assertRaisesRegex(VerificationRuntimeError, "shape \\[sample,time\\]"):
            evaluate_verification(
                self.program, {"x": self.x[:, :, None]}, time=self.time
            )
        with self.assertRaisesRegex(VerificationRuntimeError, "at least two time points"):
            evaluate_verification(
                self.program, {"x": self.x[:, :1]}, time=self.time[:1]
            )
        nonfinite = self.x.copy()
        nonfinite[1, 1] = np.nan
        with self.assertRaisesRegex(VerificationRuntimeError, "non-finite"):
            evaluate_verification(self.program, {"x": nonfinite}, time=self.time)

        with self.assertRaisesRegex(VerificationRuntimeError, "matching the trajectory axis"):
            evaluate_verification(self.program, {"x": self.x}, time=self.time[:-1])
        with self.assertRaisesRegex(VerificationRuntimeError, "strictly increasing"):
            evaluate_verification(
                self.program, {"x": self.x}, time=np.asarray([0.0, 1.0, 1.0])
            )
        bad_time = self.time.copy()
        bad_time[1] = np.nan
        with self.assertRaisesRegex(VerificationRuntimeError, "non-finite"):
            evaluate_verification(self.program, {"x": self.x}, time=bad_time)

    def test_runtime_rejects_nonfinite_expression_results(self) -> None:
        data = verification_data()
        data["parameters"].append({"name": "zero", "unit": "1", "value": 0.0})
        data["metrics"][0] = {
            "name": "initial_x",
            "expression": "x / zero",
            "reducer": "initial",
            "unit": "m",
        }
        program = parse_verification(data)
        with np.errstate(divide="ignore", invalid="ignore"):
            with self.assertRaisesRegex(VerificationRuntimeError, "non-finite"):
                evaluate_verification(program, {"x": self.x}, time=self.time)

    def test_sample_independent_criterion_is_broadcast_to_the_posterior(self) -> None:
        data = verification_data()
        data["criteria"] = [
            {
                "name": "configuration",
                "expression": "closure_limit >= final_limit",
                "minimum_probability": 0.5,
                "confidence_level": 0.95,
            }
        ]
        report = evaluate_verification(
            parse_verification(data), {"x": self.x}, time=self.time
        )
        criterion = report.criteria["configuration"]
        self.assertEqual(criterion.pass_count, 4)
        self.assertEqual(criterion.probability_estimate, 1.0)
        self.assertTrue(criterion.accepted)

    def test_one_pass_cannot_admit_a_high_probability_claim(self) -> None:
        data = verification_data()
        for criterion in data["criteria"]:
            criterion["minimum_probability"] = 0.999
        report = evaluate_verification(
            parse_verification(data), {"x": self.x[:1]}, time=self.time
        )
        criterion = report.criteria["closure"]
        self.assertEqual(criterion.pass_count, 1)
        self.assertEqual(criterion.probability_estimate, 1.0)
        self.assertLess(criterion.probability_lower_bound, 0.999)
        self.assertFalse(criterion.accepted)

    @staticmethod
    def _lazy_domain_program():
        data = {
            "format": "verification-1",
            "id": "test.lazy_domains",
            "name": "Lazy masked domain verification",
            "version": "1.0.0",
            "inputs": [{"name": "x", "unit": "1"}],
            "metrics": [
                {
                    "name": "safe_where",
                    "expression": "where(x > 0, sqrt(x), 0)",
                    "reducer": "mean",
                    "unit": "1",
                },
                {
                    "name": "safe_conditional",
                    "expression": "sqrt(x) if x > 0 else 0",
                    "reducer": "mean",
                    "unit": "1",
                },
                {
                    "name": "final_x",
                    "expression": "x",
                    "reducer": "final",
                    "unit": "1",
                },
            ],
            "criteria": [
                {
                    "name": "lazy_or",
                    "expression": "final_x <= 0 or sqrt(final_x) >= 0",
                    "minimum_probability": 0.0,
                    "confidence_level": 0.95,
                },
                {
                    "name": "lazy_and",
                    "expression": "final_x > 0 and log(final_x) >= -100",
                    "minimum_probability": 0.0,
                    "confidence_level": 0.95,
                },
            ],
        }
        return parse_verification(data)

    def test_numpy_where_conditional_and_boolean_short_circuit_are_masked(self) -> None:
        values = np.asarray([[-4.0, -1.0], [4.0, 9.0]])
        program = self._lazy_domain_program()
        self.assertEqual(
            program.metrics[0].differentiability,
            DifferentiabilityClass.PIECEWISE_SMOOTH,
        )
        self.assertEqual(program.differentiability["criteria"]["lazy_or"], "discrete")
        with np.errstate(invalid="raise", divide="raise"):
            report = evaluate_verification(
                program, {"x": values}, time=[0.0, 1.0]
            )
        np.testing.assert_allclose(report.metrics["safe_where"].values, [0.0, 2.5])
        np.testing.assert_allclose(
            report.metrics["safe_conditional"].values, [0.0, 2.5]
        )
        np.testing.assert_array_equal(report.criteria["lazy_or"].passes, [True, True])
        np.testing.assert_array_equal(report.criteria["lazy_and"].passes, [False, True])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is optional")
    def test_torch_metric_path_retains_autograd(self) -> None:
        import torch

        values = torch.tensor(self.x, dtype=torch.float64, requires_grad=True)
        report = evaluate_verification(self.program, {"x": values}, time=self.time)
        final_values = report.metrics["final_x"].values
        rmse_values = report.metrics["rmse_x"].values
        self.assertTrue(final_values.requires_grad)
        self.assertTrue(rmse_values.requires_grad)
        self.assertFalse(report.criteria["closure"].passes.requires_grad)
        self.assertIsInstance(report.criteria["closure"].accepted, bool)
        (final_values.sum() + rmse_values.sum()).backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is optional")
    def test_torch_negative_domains_are_masked_without_gradient_poisoning(self) -> None:
        import torch

        values = torch.tensor(
            [[-4.0, -1.0], [4.0, 9.0]], dtype=torch.float64, requires_grad=True
        )
        report = evaluate_verification(
            self._lazy_domain_program(), {"x": values}, time=[0.0, 1.0]
        )
        where_values = report.metrics["safe_where"].values
        conditional_values = report.metrics["safe_conditional"].values
        expected = torch.tensor([0.0, 2.5], dtype=torch.float64)
        torch.testing.assert_close(where_values, expected)
        torch.testing.assert_close(conditional_values, expected)
        (where_values.sum() + conditional_values.sum()).backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        torch.testing.assert_close(values.grad[0], torch.zeros(2, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
