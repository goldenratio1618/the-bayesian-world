"""Inference and explicitly approximate model-selection tests."""

from __future__ import annotations

import unittest

import numpy as np

from contraption.fitting import (
    CandidateModel,
    ExperimentalData,
    FitOptions,
    fit_parameters,
    predict_experiment,
    select_models,
)
from contraption.simulator import RCCircuit, simulate


def synthetic_rc_data(
    resistance: float = 2.5,
    capacitance: float = 0.4,
    *,
    noise_std: float = 0.002,
) -> ExperimentalData:
    times = np.linspace(0.0, 2.0, 101)
    truth = simulate(
        RCCircuit(resistance, capacitance),
        times=times,
        controls={"voltage": 4.0},
        num_samples=1,
        use_model_uncertainty=False,
        process_noise=False,
    )
    rng = np.random.default_rng(17)
    voltage = truth.series("capacitor_voltage")[0] + rng.normal(0.0, noise_std, len(times))
    return ExperimentalData(
        time=times,
        observations={"capacitor_voltage": voltage},
        controls={"voltage": 4.0},
        initial_state=[0.0],
        observation_std=noise_std,
    )


class ParameterFittingTests(unittest.TestCase):
    def test_rc_parameter_fit_recovers_time_constant(self) -> None:
        data = synthetic_rc_data()
        model = RCCircuit(resistance=4.0, capacitance=0.4)
        result = fit_parameters(
            model,
            data,
            {"resistance": 4.0},
            parameter_names=("resistance",),
            fixed_parameters={"capacitance": 0.4},
            options=FitOptions(max_iterations=80, tolerance=1e-10),
        )
        self.assertTrue(result.converged, result.message)
        self.assertAlmostEqual(float(result.parameters["resistance"]), 2.5, delta=0.015)
        self.assertEqual(result.covariance.shape, (1, 1))
        self.assertGreaterEqual(float(result.covariance[0, 0]), 0.0)
        self.assertLess(result.loss_history[-1], result.loss_history[0] * 0.01)

    def test_two_outputs_identify_resistance_and_capacitance(self) -> None:
        times = np.linspace(0.0, 1.5, 121)
        truth = simulate(
            RCCircuit(3.0, 0.2),
            times=times,
            controls={"voltage": 5.0},
            num_samples=1,
            use_model_uncertainty=False,
            process_noise=False,
        )
        data = ExperimentalData(
            times,
            {
                "capacitor_voltage": truth.series("capacitor_voltage")[0],
                "resistor_current": truth.series("resistor_current", outputs_first=True)[0],
            },
            controls={"voltage": 5.0},
            initial_state=[0.0],
            observation_std={"capacitor_voltage": 0.01, "resistor_current": 0.01},
        )
        result = fit_parameters(
            RCCircuit(5.0, 0.4),
            data,
            {"resistance": 5.0, "capacitance": 0.4},
            options=FitOptions(max_iterations=100, tolerance=1e-11),
        )
        self.assertAlmostEqual(float(result.parameters["resistance"]), 3.0, delta=2e-3)
        self.assertAlmostEqual(float(result.parameters["capacitance"]), 0.2, delta=2e-3)

    def test_fit_honors_parameter_bounds(self) -> None:
        data = synthetic_rc_data()
        result = fit_parameters(
            RCCircuit(4.0, 0.4),
            data,
            {"resistance": 4.0},
            parameter_names=("resistance",),
            bounds={"resistance": (3.0, 8.0)},
            options=FitOptions(max_iterations=40),
        )
        self.assertAlmostEqual(float(result.parameters["resistance"]), 3.0, places=10)

    def test_prediction_selects_named_derived_output(self) -> None:
        times = np.linspace(0.0, 0.3, 10)
        model = RCCircuit(2.0, 0.5)
        truth = simulate(
            model,
            times=times,
            controls={"voltage": 3.0},
            num_samples=1,
            use_model_uncertainty=False,
            process_noise=False,
        )
        observations = truth.series("resistor_current", outputs_first=True)[0]
        data = ExperimentalData(
            times,
            {"resistor_current": observations},
            controls={"voltage": 3.0},
        )
        _, predicted, observed, residual = predict_experiment(
            model, data, model.default_parameters
        )
        np.testing.assert_allclose(predicted, observed, atol=1e-14)
        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_nan_observations_are_ignored(self) -> None:
        data = synthetic_rc_data()
        observations = dict(data.observations)
        observations["capacitor_voltage"] = observations["capacitor_voltage"].copy()
        observations["capacitor_voltage"][10:15] = np.nan
        with_missing = ExperimentalData(
            data.time,
            observations,
            controls=data.controls,
            initial_state=data.initial_state,
            observation_std=data.observation_std,
        )
        result = fit_parameters(
            RCCircuit(3.5, 0.4),
            with_missing,
            {"resistance": 3.5},
            parameter_names=("resistance",),
            options=FitOptions(max_iterations=60),
        )
        self.assertEqual(result.observation_count, len(data.time) - 5)
        self.assertAlmostEqual(float(result.parameters["resistance"]), 2.5, delta=0.02)


class ModelSelectionTests(unittest.TestCase):
    def test_bic_prefers_data_generating_candidate(self) -> None:
        data = synthetic_rc_data(noise_std=0.004)
        candidates = {
            "correct": RCCircuit(2.5, 0.4),
            "too_slow": RCCircuit(6.0, 0.4),
            "too_fast": RCCircuit(0.7, 0.4),
        }
        result = select_models(candidates, data, criterion="bic")
        self.assertEqual(result.best_model, "correct")
        self.assertIn("BIC approximation", result.approximation_label)
        self.assertGreater(result.posterior_probabilities["correct"], 0.999)
        self.assertAlmostEqual(sum(result.posterior_probabilities.values()), 1.0, places=12)

    def test_laplace_selection_is_clearly_labeled(self) -> None:
        data = synthetic_rc_data(noise_std=0.005)
        candidates = [
            CandidateModel(
                "adjustable",
                RCCircuit(3.0, 0.4),
                {"resistance": 3.0},
                ("resistance",),
                prior_std={"resistance": 2.0},
            ),
            CandidateModel("fixed_wrong", RCCircuit(7.0, 0.4), {}, ()),
        ]
        result = select_models(
            candidates,
            data,
            criterion="laplace",
            fit_options=FitOptions(max_iterations=70),
        )
        self.assertEqual(result.best_model, "adjustable")
        self.assertEqual(result.criterion, "laplace")
        self.assertIn("Laplace approximation", result.approximation_label)
        payload = result.to_dict()
        self.assertEqual(payload["models"]["adjustable"]["approximation_label"], result.approximation_label)


if __name__ == "__main__":
    unittest.main()
