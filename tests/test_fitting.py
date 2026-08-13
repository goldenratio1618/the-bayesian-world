"""Inference and explicitly approximate model-selection tests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import unittest

import numpy as np

from contraption import load_contraption
from contraption.physics.fitting import (
    CandidateModel,
    ExperimentalData,
    FitOptions,
    fit_parameters,
    predict_experiment,
    select_models,
)
from contraption.physics.simulator import simulate


ROOT = Path(__file__).resolve().parents[1]
RC_BUNDLE = (
    ROOT
    / "assembled_contraptions"
    / "examples"
    / "test_systems"
    / "rc_circuit"
    / "contraption.json"
)


@lru_cache(maxsize=1)
def rc_system():
    """Load the real declarative RC fixture through the public bundle loader."""

    return load_contraption(RC_BUNDLE).system


def synthetic_rc_data(
    resistance: float = 2.5,
    capacitance: float = 0.4,
    *,
    noise_std: float = 0.002,
) -> ExperimentalData:
    times = np.linspace(0.0, 2.0, 101)
    truth = simulate(
        rc_system(),
        times=times,
        controls={"voltage": 4.0},
        parameters={"rc.resistance": resistance, "rc.capacitance": capacitance},
        num_samples=1,
        use_model_uncertainty=False,
        process_noise=False,
    )
    rng = np.random.default_rng(17)
    voltage = truth.series("rc.capacitor_voltage")[0] + rng.normal(
        0.0, noise_std, len(times)
    )
    return ExperimentalData(
        time=times,
        observations={"rc.capacitor_voltage": voltage},
        controls={"voltage": 4.0},
        observation_std=noise_std,
    )


class ParameterFittingTests(unittest.TestCase):
    def test_rc_parameter_fit_recovers_time_constant(self) -> None:
        data = synthetic_rc_data()
        model = rc_system()
        result = fit_parameters(
            model,
            data,
            {"rc.resistance": 4.0},
            parameter_names=("rc.resistance",),
            fixed_parameters={"rc.capacitance": 0.4},
            options=FitOptions(max_iterations=80, tolerance=1e-10),
        )
        self.assertTrue(result.converged, result.message)
        self.assertAlmostEqual(
            float(result.parameters["rc.resistance"]), 2.5, delta=0.015
        )
        self.assertEqual(result.covariance.shape, (1, 1))
        self.assertGreaterEqual(float(result.covariance[0, 0]), 0.0)
        self.assertLess(result.loss_history[-1], result.loss_history[0] * 0.01)

    def test_two_outputs_identify_resistance_and_capacitance(self) -> None:
        times = np.linspace(0.0, 1.5, 121)
        truth = simulate(
            rc_system(),
            times=times,
            controls={"voltage": 5.0},
            parameters={"rc.resistance": 3.0, "rc.capacitance": 0.2},
            num_samples=1,
            use_model_uncertainty=False,
            process_noise=False,
        )
        data = ExperimentalData(
            times,
            {
                "rc.capacitor_voltage": truth.series("rc.capacitor_voltage")[0],
                "rc.resistor_current": truth.series("rc.resistor_current")[0],
            },
            controls={"voltage": 5.0},
            observation_std={
                "rc.capacitor_voltage": 0.01,
                "rc.resistor_current": 0.01,
            },
        )
        result = fit_parameters(
            rc_system(),
            data,
            {"rc.resistance": 5.0, "rc.capacitance": 0.4},
            bounds={
                "rc.resistance": (0.1, 20.0),
                "rc.capacitance": (0.01, 2.0),
            },
            options=FitOptions(max_iterations=100, tolerance=1e-11),
        )
        self.assertAlmostEqual(
            float(result.parameters["rc.resistance"]), 3.0, delta=2e-3
        )
        self.assertAlmostEqual(
            float(result.parameters["rc.capacitance"]), 0.2, delta=2e-3
        )

    def test_fit_honors_parameter_bounds(self) -> None:
        data = synthetic_rc_data()
        result = fit_parameters(
            rc_system(),
            data,
            {"rc.resistance": 4.0},
            parameter_names=("rc.resistance",),
            fixed_parameters={"rc.capacitance": 0.4},
            bounds={"rc.resistance": (3.0, 8.0)},
            options=FitOptions(max_iterations=40),
        )
        self.assertAlmostEqual(
            float(result.parameters["rc.resistance"]), 3.0, places=10
        )

    def test_prediction_selects_named_derived_output(self) -> None:
        times = np.linspace(0.0, 0.3, 10)
        model = rc_system()
        truth = simulate(
            model,
            times=times,
            controls={"voltage": 3.0},
            parameters={"rc.resistance": 2.0, "rc.capacitance": 0.5},
            num_samples=1,
            use_model_uncertainty=False,
            process_noise=False,
        )
        observations = truth.series("rc.resistor_current")[0]
        data = ExperimentalData(
            times,
            {"rc.resistor_current": observations},
            controls={"voltage": 3.0},
        )
        _, predicted, observed, residual = predict_experiment(
            model,
            data,
            {"rc.resistance": 2.0, "rc.capacitance": 0.5},
        )
        np.testing.assert_allclose(predicted, observed, atol=1e-14)
        np.testing.assert_allclose(residual, 0.0, atol=1e-14)

    def test_nan_observations_are_ignored(self) -> None:
        data = synthetic_rc_data()
        observations = dict(data.observations)
        observations["rc.capacitor_voltage"] = observations[
            "rc.capacitor_voltage"
        ].copy()
        observations["rc.capacitor_voltage"][10:15] = np.nan
        with_missing = ExperimentalData(
            data.time,
            observations,
            controls=data.controls,
            initial_state=data.initial_state,
            observation_std=data.observation_std,
        )
        result = fit_parameters(
            rc_system(),
            with_missing,
            {"rc.resistance": 3.5},
            parameter_names=("rc.resistance",),
            fixed_parameters={"rc.capacitance": 0.4},
            options=FitOptions(max_iterations=60),
        )
        self.assertEqual(result.observation_count, len(data.time) - 5)
        self.assertAlmostEqual(
            float(result.parameters["rc.resistance"]), 2.5, delta=0.02
        )


class ModelSelectionTests(unittest.TestCase):
    def test_bic_prefers_data_generating_candidate(self) -> None:
        data = synthetic_rc_data(noise_std=0.004)
        candidates = [
            CandidateModel(
                "correct",
                rc_system(),
                {"rc.resistance": 2.5, "rc.capacitance": 0.4},
                (),
            ),
            CandidateModel(
                "too_slow",
                rc_system(),
                {"rc.resistance": 6.0, "rc.capacitance": 0.4},
                (),
            ),
            CandidateModel(
                "too_fast",
                rc_system(),
                {"rc.resistance": 0.7, "rc.capacitance": 0.4},
                (),
            ),
        ]
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
                rc_system(),
                {"rc.resistance": 3.0, "rc.capacitance": 0.4},
                ("rc.resistance",),
                prior_std={"rc.resistance": 2.0},
            ),
            CandidateModel(
                "fixed_wrong",
                rc_system(),
                {"rc.resistance": 7.0, "rc.capacitance": 0.4},
                (),
            ),
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
