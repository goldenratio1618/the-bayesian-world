"""Analytic, probabilistic, descriptor, and differentiability baselines."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from contraption import load_contraption
from contraption.physics.backend import NumpyBackend, TorchBackend, get_backend
from contraption.physics.dsl import parse_model
from contraption.physics.simulator import (
    OfflineSimulator,
    ResidualSystem,
    SimulationConfig,
    simulate,
)
from contraption.physics.uq import (
    GaussianParameterDistribution,
    Normal,
    ekf_predict,
    ekf_update,
    sample_parameters,
    summarize_samples,
)
from contraption.verification import evaluate_verification, load_verification


ROOT = Path(__file__).resolve().parents[1]
TEST_SYSTEMS = ROOT / "assembled_contraptions" / "examples" / "test_systems"


@lru_cache(maxsize=None)
def load_test_system(name: str):
    """Load one real declarative fixture through the public bundle loader."""

    return load_contraption(TEST_SYSTEMS / name / "contraption.json").system


class DeclarativeBundleTests(unittest.TestCase):
    def test_reference_system_verifications_accept_their_physics(self) -> None:
        cases = {
            "rc_circuit": (
                {"voltage": 5.0},
                {
                    "voltage": "rc.applied_voltage",
                    "capacitor_voltage": "rc.capacitor_voltage_observation",
                    "resistor_current": "rc.resistor_current",
                },
            ),
            "rl_circuit": (
                {"voltage": 8.0},
                {
                    "voltage": "rl.applied_voltage",
                    "inductor_current": "rl.inductor_current_observation",
                    "resistor_voltage": "rl.resistor_voltage",
                    "inductor_voltage": "rl.inductor_voltage",
                },
            ),
            "dc_motor": (
                {"voltage": 6.0},
                {
                    "armature_current": "motor.armature_current_observation",
                    "electromagnetic_torque": "motor.electromagnetic_torque",
                },
            ),
            "planar_rigid_body": (
                {"force_x": 4.0},
                {"y": "body.y_observation", "yaw": "body.yaw_observation"},
            ),
        }
        for name, (controls, bindings) in cases.items():
            with self.subTest(bundle=name):
                result = simulate(
                    load_test_system(name),
                    duration=0.02,
                    dt=0.01,
                    controls=controls,
                    num_samples=64,
                    use_model_uncertainty=False,
                    process_noise=False,
                )
                program = load_verification(TEST_SYSTEMS / name / "verification.verify")
                report = evaluate_verification(
                    program,
                    {
                        input_name: result.series(system_name)
                        for input_name, system_name in bindings.items()
                    },
                    time=result.time,
                )
                self.assertTrue(report.accepted, report.to_dict())


class ElectricalBaselineTests(unittest.TestCase):
    def test_rc_step_matches_closed_form(self) -> None:
        model = load_test_system("rc_circuit")
        result = simulate(
            model,
            duration=2.0,
            dt=0.001,
            controls={"voltage": 5.0},
            parameters={"rc.resistance": 2.0, "rc.capacitance": 0.5},
            num_samples=1,
            use_model_uncertainty=False,
            process_noise=False,
        )
        expected = 5.0 * (1.0 - math.exp(-2.0))
        voltage = result.series("rc.capacitor_voltage")[0, -1]
        self.assertAlmostEqual(float(voltage), expected, delta=7e-4)
        current = result.series("rc.resistor_current")[0, -1]
        self.assertAlmostEqual(float(current), (5.0 - float(voltage)) / 2.0, places=10)

    def test_rl_step_matches_closed_form(self) -> None:
        model = load_test_system("rl_circuit")
        result = simulate(
            model,
            duration=1.5,
            dt=0.0005,
            controls={"voltage": 8.0},
            parameters={"rl.resistance": 4.0, "rl.inductance": 2.0},
            num_samples=1,
            use_model_uncertainty=False,
            process_noise=False,
        )
        expected = 2.0 * (1.0 - math.exp(-3.0))
        current = result.series("rl.inductor_current")[0, -1]
        self.assertAlmostEqual(float(current), expected, delta=8e-4)

    def test_dc_motor_approaches_analytic_steady_state(self) -> None:
        model = load_test_system("dc_motor")
        result = simulate(
            model,
            duration=8.0,
            dt=0.002,
            controls={"voltage": 6.0},
            parameters={
                "motor.resistance": 2.0,
                "motor.inductance": 0.05,
                "motor.torque_constant": 0.1,
                "motor.back_emf_constant": 0.1,
                "motor.inertia": 0.02,
                "motor.viscous_friction": 0.02,
            },
            num_samples=1,
            use_model_uncertainty=False,
            process_noise=False,
        )
        expected_speed = 0.1 * 6.0 / (2.0 * 0.02 + 0.1 * 0.1)
        expected_current = 0.02 * expected_speed / 0.1
        current = result.series("motor.armature_current")[0, -1]
        speed = result.series("motor.angular_velocity")[0, -1]
        self.assertAlmostEqual(float(current), expected_current, delta=2e-3)
        self.assertAlmostEqual(float(speed), expected_speed, delta=1e-2)


class MechanicalBaselineTests(unittest.TestCase):
    def test_planar_rigid_body_constant_force(self) -> None:
        model = load_test_system("planar_rigid_body")
        result = simulate(
            model,
            duration=2.0,
            dt=0.001,
            controls={"force_x": 4.0, "force_y": 0.0, "torque": 0.0},
            parameters={
                "body.mass": 2.0,
                "body.moment_of_inertia": 0.5,
                "body.linear_drag": 0.0,
                "body.angular_drag": 0.0,
            },
            num_samples=1,
            use_model_uncertainty=False,
            process_noise=False,
        )
        self.assertAlmostEqual(float(result.series("body.x")[0, -1]), 4.0, delta=0.003)
        self.assertAlmostEqual(float(result.series("body.velocity_x")[0, -1]), 4.0, places=9)
        self.assertAlmostEqual(float(result.series("body.y")[0, -1]), 0.0, places=12)

    def test_planar_process_noise_is_seeded_and_reconciles_outputs(self) -> None:
        model = load_test_system("planar_rigid_body")
        common = dict(
            duration=0.03,
            dt=0.01,
            parameters={
                "body.roughness_std": 0.4,
                "body.roughness_reference_length": 2.0,
            },
            num_samples=128,
            use_model_uncertainty=False,
        )
        first = simulate(model, seed=91, **common)
        repeat = simulate(model, seed=91, **common)
        changed = simulate(model, seed=92, **common)
        np.testing.assert_array_equal(first.samples, repeat.samples)
        self.assertFalse(np.array_equal(first.samples, changed.samples))
        np.testing.assert_allclose(
            first.series("body.y_observation"),
            first.series("body.y"),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            first.series("body.yaw_observation"),
            first.series("body.yaw"),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertTrue(first.metadata["process_noise_declared"])
        self.assertEqual(
            first.metadata["process_noise_seed_policy"], "simulation_seed"
        )
        self.assertEqual(
            first.metadata["process_noise_reproducibility"],
            "same_backend_device",
        )
        self.assertEqual(
            model.process_noise_channel_names,
            tuple(sorted(model.process_noise_channel_names)),
        )

    def test_planar_process_noise_has_declared_diffusion_variance(self) -> None:
        translation_scale = 0.25
        reference_length = 2.5
        dt = 0.01
        result = simulate(
            load_test_system("planar_rigid_body"),
            duration=dt,
            dt=dt,
            parameters={
                "body.roughness_std": translation_scale,
                "body.roughness_reference_length": reference_length,
            },
            num_samples=4096,
            seed=27,
            use_model_uncertainty=False,
        )
        expected = {
            "body.x": translation_scale**2 * dt,
            "body.y": translation_scale**2 * dt,
            "body.yaw": (translation_scale / reference_length) ** 2 * dt,
        }
        for state_name, expected_variance in expected.items():
            with self.subTest(state=state_name):
                empirical = float(np.var(result.series(state_name)[:, -1], ddof=1))
                self.assertAlmostEqual(
                    empirical,
                    expected_variance,
                    delta=expected_variance * 0.08,
                )

    def test_planar_process_noise_can_be_zero_or_disabled(self) -> None:
        model = load_test_system("planar_rigid_body")
        zero_scale = simulate(
            model,
            duration=0.03,
            dt=0.01,
            num_samples=32,
            seed=8,
            use_model_uncertainty=False,
        )
        disabled = simulate(
            model,
            duration=0.03,
            dt=0.01,
            parameters={
                "body.roughness_std": 1.0,
                "body.roughness_reference_length": 1.0,
            },
            num_samples=32,
            seed=8,
            use_model_uncertainty=False,
            process_noise=False,
        )
        for result in (zero_scale, disabled):
            for state_name in ("body.x", "body.y", "body.yaw"):
                np.testing.assert_array_equal(
                    result.series(state_name),
                    np.zeros((32, 4)),
                )

class DescriptorAndControlTests(unittest.TestCase):
    @staticmethod
    def _descriptor(residual, *, initial_state=(1.0,)) -> ResidualSystem:
        return ResidualSystem(
            state_names=("x",),
            residual_function=residual,
            initial_state=initial_state,
        )

    @staticmethod
    def _declarative_model(
        *,
        expression="der(x)",
        initial=0.0,
        validity=None,
        modes=(),
        initialization=None,
        fidelity_levels=(),
        properties=(),
        parameters=(),
    ):
        return SimpleNamespace(
            state_names=("x",),
            algebraic_names=(),
            input_names=(),
            states=(SimpleNamespace(name="x", initial=initial, derivative=None),),
            algebraics=(),
            parameters=parameters,
            relations=(SimpleNamespace(name="dynamics", expression=expression),),
            validity=validity,
            modes=modes,
            initialization=initialization,
            fidelity_levels=fidelity_levels,
            properties=properties,
            evaluate_residual=lambda *args: None,
        )

    @staticmethod
    def _direct_process_noise_model(*, algebraic: bool = False):
        algebraics = (
            [{"name": "copy", "unit": "m", "initial": 0.0}]
            if algebraic
            else []
        )
        relations = [{"name": "dynamics", "expression": "x_dot"}]
        if algebraic:
            relations.append(
                {"name": "copy_relation", "expression": "copy - x"}
            )
        return parse_model(
            {
                "format": "pmdl-1",
                "id": "test.direct_process_noise",
                "name": "Direct process-noise adapter fixture",
                "version": "1.0.0",
                "domains": ["mechanical"],
                "implements": "test-direct-process-noise",
                "states": [
                    {
                        "name": "x",
                        "unit": "m",
                        "initial": 0.0,
                        "derivative": "x_dot",
                    }
                ],
                "algebraics": algebraics,
                "parameters": [
                    {
                        "name": "diffusion",
                        "unit": "m/s^1/2",
                        "default": 0.2,
                        "bounds": {"lower": 0.0, "upper": 10.0},
                    }
                ],
                "relations": relations,
                "process_noise": {
                    "seed_policy": "simulation_seed",
                    "reproducibility": "same_backend_device",
                    "application": "accepted_step_increment",
                    "channels": [
                        {
                            "name": "roughness",
                            "distribution": "standard_normal",
                        }
                    ],
                    "increments": [
                        {
                            "target": "x",
                            "expression": "diffusion * sqrt(dt) * roughness",
                        }
                    ],
                },
                "validity": {
                    "assumptions": ["Synthetic direct-adapter fixture"],
                    "max_timestep": 0.01,
                },
            }
        )

    def test_direct_modelspec_adapter_executes_seeded_process_noise(self) -> None:
        model = self._direct_process_noise_model()
        first = simulate(
            model,
            duration=0.01,
            dt=0.01,
            num_samples=512,
            seed=44,
            use_model_uncertainty=False,
        )
        repeat = simulate(
            model,
            duration=0.01,
            dt=0.01,
            num_samples=512,
            seed=44,
            use_model_uncertainty=False,
        )
        np.testing.assert_array_equal(first.samples, repeat.samples)
        self.assertGreater(float(np.var(first.series("x")[:, -1])), 0.0)

    def test_direct_modelspec_noise_fails_without_algebraic_reconciler(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            "no consistent-state reconciliation solver",
        ):
            simulate(
                self._direct_process_noise_model(algebraic=True),
                duration=0.01,
                dt=0.01,
                num_samples=2,
                seed=2,
                use_model_uncertainty=False,
            )

    def test_unknown_control_channel_fails_loudly(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown control source.*voltgae"):
            simulate(
                load_test_system("rc_circuit"),
                duration=0.02,
                dt=0.01,
                controls={"voltgae": 5.0},
                num_samples=1,
                process_noise=False,
            )

    def test_unknown_parameter_override_fails_loudly(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown physical parameter.*resistence"):
            simulate(
                load_test_system("rc_circuit"),
                duration=0.02,
                dt=0.01,
                parameters={"resistence": 2.0},
                num_samples=1,
                process_noise=False,
            )

    def test_implicit_descriptor_matches_backward_euler(self) -> None:
        def residual(t, state, state_dot, parameters, controls):
            return state_dot + parameters["decay"][:, None] * state

        model = ResidualSystem(
            state_names=("amount",),
            residual_function=residual,
            initial_state=[1.0],
            default_parameters={"decay": 2.0},
        )
        result = simulate(
            model,
            duration=1.0,
            dt=0.01,
            num_samples=1,
            process_noise=False,
        )
        expected = (1.0 / 1.02) ** 100
        self.assertAlmostEqual(float(result.mean[-1, 0]), expected, places=7)

    def test_consistent_initializer_defines_the_published_first_frame(self) -> None:
        class ConsistentlyInitializedSystem:
            state_names = ("x", "algebraic_copy")
            initial_state = (2.0, 0.0)
            default_parameters = {}
            control_names = ()

            @staticmethod
            def consistent_initial_state(t, state, parameters, controls, backend):
                return backend.stack((state[:, 0], state[:, 0]), axis=-1)

            @staticmethod
            def residual(t, state, state_dot, parameters, controls, backend):
                return backend.stack(
                    (state_dot[:, 0], state[:, 1] - state[:, 0]), axis=-1
                )

        result = simulate(
            ConsistentlyInitializedSystem(),
            duration=0.1,
            dt=0.1,
            num_samples=1,
            process_noise=False,
        )

        np.testing.assert_allclose(result.samples[0, 0], [2.0, 2.0])

    def test_resolved_wrapper_validates_complete_result_before_return(self) -> None:
        class ExplicitSystem:
            state_names = ("x",)
            initial_state = (0.0,)
            default_parameters = {}

            @staticmethod
            def derivative(t, state, parameters, controls, backend):
                return state * 0.0

        class ValidatedContraption:
            system = ExplicitSystem()
            controller = None

            def __init__(self) -> None:
                self.validated = False

            def validate_simulation_result(self, result) -> None:
                self.validated = True
                self.asserted_shape = tuple(result.samples.shape)

        contraption = ValidatedContraption()
        result = simulate(
            contraption,
            duration=0.1,
            dt=0.1,
            num_samples=2,
            process_noise=False,
        )
        self.assertTrue(contraption.validated)
        self.assertEqual(contraption.asserted_shape, (2, 2, 1))
        self.assertEqual(result.samples.shape, (2, 2, 1))

    def test_singular_descriptor_jacobian_fails_with_context(self) -> None:
        model = self._descriptor(lambda t, state, state_dot, parameters, controls: state * 0.0)
        with self.assertRaisesRegex(
            RuntimeError, r"Jacobian is singular.*timestep=1.*sample=0.*rank=0/1"
        ):
            simulate(model, duration=0.01, dt=0.01, num_samples=2, process_noise=False)

    def test_nonconvergent_descriptor_fails_with_residual_context(self) -> None:
        model = self._descriptor(
            lambda t, state, state_dot, parameters, controls: state * state - 2.0
        )
        with self.assertRaisesRegex(
            RuntimeError, r"did not converge.*timestep=1.*sample=0.*residual_max="
        ):
            simulate(
                model,
                duration=0.01,
                dt=0.01,
                num_samples=1,
                process_noise=False,
                newton_max_iterations=1,
            )

    def test_nonfinite_descriptor_residual_fails_with_context(self) -> None:
        model = self._descriptor(
            lambda t, state, state_dot, parameters, controls: state * float("nan")
        )
        with self.assertRaisesRegex(
            FloatingPointError, r"residual is non-finite.*timestep=1.*sample=0"
        ):
            simulate(model, duration=0.01, dt=0.01, num_samples=1, process_noise=False)

    def test_nonfinite_descriptor_state_fails_with_context(self) -> None:
        model = self._descriptor(
            lambda t, state, state_dot, parameters, controls: state_dot + state
        )
        with self.assertRaisesRegex(
            FloatingPointError, r"state is non-finite.*timestep=1.*sample=0"
        ):
            simulate(
                model,
                duration=0.01,
                dt=0.01,
                initial_state=[float("nan")],
                num_samples=1,
                process_noise=False,
            )

    def test_data_only_modelspec_layout_is_adapted_to_batched_engine(self) -> None:
        class DataOnlyModel:
            state_names = ("x", "y")
            algebraic_names = ()
            input_names = ()
            states = (
                SimpleNamespace(name="x", initial=1.0),
                SimpleNamespace(name="y", initial=2.0),
            )
            algebraics = ()
            parameters = ()

            @staticmethod
            def evaluate_residual(t, z, zdot, theta, u):
                # This is the variable-major contract used by safe ModelSpec.
                self_assertions = (len(z) == 2, len(zdot) == 2)
                if not all(self_assertions):
                    raise AssertionError("ModelSpec adapter passed sample-major values")
                return np.stack((zdot[0] + z[0], zdot[1] + 2.0 * z[1]), axis=-1)

        result = simulate(
            DataOnlyModel(), duration=0.2, dt=0.01, num_samples=3, process_noise=False
        )
        self.assertEqual(result.samples.shape, (3, 21, 2))
        self.assertAlmostEqual(float(result.mean[-1, 0]), (1.0 / 1.01) ** 20, places=7)
        self.assertAlmostEqual(float(result.mean[-1, 1]), 2.0 * (1.0 / 1.02) ** 20, places=7)

    def test_pmdl_max_timestep_is_enforced_on_actual_time_grid(self) -> None:
        model = self._declarative_model(
            validity=SimpleNamespace(ranges={}, max_timestep=0.01)
        )
        with self.assertRaisesRegex(
            ValueError,
            r"validity\.max_timestep exceeded.*timestep=2.*requested=0\.014999.*declared_max=0\.01",
        ):
            simulate(
                model,
                times=[0.0, 0.005, 0.02],
                num_samples=1,
                process_noise=False,
            )

    def test_pmdl_initial_runtime_validity_violation_fails_with_context(self) -> None:
        model = self._declarative_model(
            initial=1.5,
            validity=SimpleNamespace(
                ranges={"x": SimpleNamespace(lower=0.0, upper=1.0)},
                max_timestep=None,
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            r"validity range violation during initialization.*symbol='x'.*sample=0.*value=1\.5.*allowed=\[0, 1\]",
        ):
            simulate(
                model,
                duration=0.01,
                dt=0.01,
                num_samples=1,
                process_noise=False,
            )

    def test_pmdl_runtime_validity_is_checked_after_each_accepted_step(self) -> None:
        class ExplicitValidityModel:
            state_names = ("x",)
            initial_state = (0.9,)
            default_parameters = {}
            parameter_bounds = {}
            control_names = ()
            validity = SimpleNamespace(
                ranges={"x": SimpleNamespace(lower=0.0, upper=1.0)},
                max_timestep=None,
            )

            @staticmethod
            def derivative(t, state, parameters, controls, backend):
                return state * 0.0 + 1.0

        with self.assertRaisesRegex(
            ValueError,
            r"validity range violation during accepted timestep 1.*symbol='x'.*value=1\.1",
        ):
            simulate(
                ExplicitValidityModel(),
                duration=0.2,
                dt=0.2,
                integrator="euler",
                num_samples=1,
                process_noise=False,
            )

    def test_state_dependent_external_control_provider_is_rejected(self) -> None:
        consumed = []

        class ControlDrivenDescriptor:
            state_names = ("x",)
            initial_state = (0.0,)
            default_parameters = {}
            parameter_bounds = {}
            control_names = ("command",)
            validity = SimpleNamespace(
                ranges={
                    "command": SimpleNamespace(lower=0.0, upper=1.0),
                },
                max_timestep=None,
            )

            @staticmethod
            def residual(t, state, state_dot, parameters, controls, backend):
                consumed.append(float(np.asarray(controls["command"]).reshape(-1)[0]))
                return state_dot - controls["command"]

        def state_dependent_command(t, state):
            if t == 0.0:
                return 0.5
            return 2.0 if float(state[0, 0]) < 1.0 else 0.5

        with self.assertRaisesRegex(TypeError, "open-loop.*plant state"):
            simulate(
                ControlDrivenDescriptor(),
                duration=1.0,
                dt=1.0,
                controls={"command": state_dependent_command},
                num_samples=1,
                process_noise=False,
            )
        self.assertEqual(consumed, [])

    def test_pmdl_unavailable_validity_symbol_is_rejected_at_admission(self) -> None:
        model = self._declarative_model(
            validity=SimpleNamespace(
                ranges={"terminal_current": SimpleNamespace(lower=-1.0, upper=1.0)},
                max_timestep=None,
            )
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            r"validity ranges reference symbol.*terminal_current.*will not be ignored",
        ):
            simulate(model, duration=0.01, dt=0.01, num_samples=1)

    def test_pmdl_modes_are_rejected_instead_of_being_always_active(self) -> None:
        model = self._declarative_model(modes=(SimpleNamespace(name="stalled"),))
        with self.assertRaisesRegex(
            NotImplementedError,
            r"discrete modes are declared.*not implemented.*always-active approximation",
        ):
            simulate(model, duration=0.01, dt=0.01, num_samples=1)

    def test_pmdl_initialization_constraints_are_rejected(self) -> None:
        model = self._declarative_model(
            initialization=SimpleNamespace(
                strategy="consistent",
                constraints=(SimpleNamespace(expression="x"),),
                required=("x",),
            )
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            r"initialization constraints are declared.*refusing to ignore 1 constraint",
        ):
            simulate(model, duration=0.01, dt=0.01, num_samples=1)

    def test_pmdl_multiple_fidelity_levels_require_an_explicit_selector(self) -> None:
        levels = tuple(
            SimpleNamespace(
                name=name,
                active_relations=("dynamics",),
                parameter_overrides={},
            )
            for name in ("fast", "detailed")
        )
        model = self._declarative_model(fidelity_levels=levels)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"multiple fidelity levels.*no fidelity selector.*refusing to choose silently",
        ):
            simulate(model, duration=0.01, dt=0.01, num_samples=1)

    def test_pmdl_nontrivial_single_fidelity_level_is_rejected(self) -> None:
        model = self._declarative_model(
            fidelity_levels=(
                SimpleNamespace(
                    name="altered",
                    active_relations=("dynamics",),
                    parameter_overrides={"gain": 2.0},
                ),
            )
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            r"fidelity level 'altered' changes active relations or parameter values",
        ):
            simulate(model, duration=0.01, dt=0.01, num_samples=1)

    def test_pmdl_property_tests_cannot_be_mistaken_for_executed_tests(self) -> None:
        model = self._declarative_model(
            properties=(SimpleNamespace(name="nonnegative_energy"),)
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            r"property tests are declared.*no property-test executor.*refusing to treat type-checking as a pass",
        ):
            simulate(model, duration=0.01, dt=0.01, num_samples=1)

    def test_pmdl_single_identity_fidelity_level_remains_admissible(self) -> None:
        model = self._declarative_model(
            fidelity_levels=(
                SimpleNamespace(
                    name="base",
                    active_relations=("dynamics",),
                    parameter_overrides={},
                ),
            )
        )
        result = simulate(
            model,
            duration=0.01,
            dt=0.01,
            num_samples=1,
            process_noise=False,
        )
        self.assertEqual(result.samples.shape, (1, 2, 1))

    def test_modelspec_parameter_uncertainty_is_used_by_default(self) -> None:
        class DeclarativeModel:
            state_names = ("x",)
            algebraic_names = ()
            input_names = ()
            states = (SimpleNamespace(name="x", initial=1.0, derivative=None),)
            algebraics = ()
            parameters = (
                SimpleNamespace(
                    name="rate",
                    default=1.0,
                    bounds=SimpleNamespace(lower=0.01, upper=5.0),
                    uncertainty=SimpleNamespace(
                        distribution="lognormal",
                        parameters={"std": 0.2},
                        correlation_group=None,
                    ),
                ),
            )
            relations = (SimpleNamespace(expression="der(x) + rate * x"),)

            @staticmethod
            def evaluate_residual(*args):
                raise AssertionError("backend-native PMDL evaluator was bypassed")

        uncertain = simulate(
            DeclarativeModel(),
            duration=0.1,
            dt=0.02,
            num_samples=256,
            seed=17,
            process_noise=False,
        )
        nominal = simulate(
            DeclarativeModel(),
            duration=0.1,
            dt=0.02,
            num_samples=256,
            seed=17,
            use_model_uncertainty=False,
            process_noise=False,
        )
        self.assertGreater(float(np.std(uncertain.samples[:, -1, 0])), 1e-3)
        self.assertAlmostEqual(float(np.std(nominal.samples[:, -1, 0])), 0.0, places=14)

class UncertaintyAndLinearizationTests(unittest.TestCase):
    def test_seeded_monte_carlo_is_reproducible_and_sensitive(self) -> None:
        model = load_test_system("rc_circuit")
        common = dict(
            duration=1.0,
            dt=0.01,
            controls={"voltage": 5.0},
            parameters={"rc.resistance": 2.0, "rc.capacitance": 0.5},
            num_samples=512,
            use_model_uncertainty=False,
            process_noise=False,
        )
        low = simulate(
            model,
            seed=41,
            parameter_distribution={"rc.resistance": (2.0, 0.02)},
            **common,
        )
        repeat = simulate(
            model,
            seed=41,
            parameter_distribution={"rc.resistance": (2.0, 0.02)},
            **common,
        )
        high = simulate(
            model,
            seed=41,
            parameter_distribution={"rc.resistance": (2.0, 0.4)},
            **common,
        )
        np.testing.assert_array_equal(low.samples, repeat.samples)
        self.assertGreater(float(high.covariance[-1, 0, 0]), 50.0 * float(low.covariance[-1, 0, 0]))
        lower, upper = high.confidence_interval
        final = high.samples[:, -1, 0]
        inside = np.mean((final >= lower[-1, 0]) & (final <= upper[-1, 0]))
        self.assertAlmostEqual(float(inside), 0.95, delta=0.015)

    def test_correlated_parameter_sampler(self) -> None:
        distribution = GaussianParameterDistribution(
            ("a", "b"), [1.0, 2.0], [[0.04, 0.018], [0.018, 0.09]]
        )
        backend = NumpyBackend()
        draws = sample_parameters(
            {"a": 1.0, "b": 2.0}, distribution, 20_000, backend=backend, seed=5
        )
        empirical = np.cov(np.stack((draws["a"], draws["b"])), ddof=1)
        np.testing.assert_allclose(empirical, distribution.covariance, atol=0.004)

    def test_declarative_parameter_distribution_kinds_are_sampleable(self) -> None:
        specifications = {
            "normal": {"distribution": "normal", "parameters": {"std": 0.1}},
            "lognormal": {"distribution": "lognormal", "parameters": {"std": 0.1}},
            "uniform": {
                "distribution": "uniform",
                "parameters": {"lower": 0.5, "upper": 1.5},
            },
            "triangular": {
                "distribution": "triangular",
                "parameters": {"lower": 0.5, "mode": 1.0, "upper": 1.5},
            },
            "empirical": {
                "distribution": "empirical",
                "parameters": {"values": [0.75, 1.25], "probabilities": [0.25, 0.75]},
            },
        }
        for name, specification in specifications.items():
            with self.subTest(distribution=name):
                draws = sample_parameters(
                    {"value": 1.0},
                    {"value": specification},
                    128,
                    backend=NumpyBackend(),
                    seed=3,
                    bounds={"value": (0.1, 2.0)},
                )["value"]
                self.assertEqual(draws.shape, (128,))
                self.assertTrue(np.all(np.isfinite(draws)))
                self.assertGreater(float(np.std(draws)), 0.0)

    def test_dsl_rc_preserves_canonical_contract_names(self) -> None:
        model = load_test_system("rc_circuit")
        self.assertIn("rc.capacitor_voltage", model.state_names)
        self.assertIn("rc.resistor_current", model.state_names)
        self.assertEqual(model.control_names, ("voltage",))
        self.assertIn("rc.resistance", model.default_parameters)
        self.assertIn("rc.capacitance", model.default_parameters)

    def test_ekf_helpers(self) -> None:
        mean, covariance = ekf_predict(
            np.array([1.0]),
            np.array([[0.2]]),
            lambda state: 0.9 * state,
            np.array([[0.01]]),
        )
        self.assertAlmostEqual(float(mean[0]), 0.9)
        self.assertAlmostEqual(float(covariance[0, 0]), 0.172, places=6)
        updated_mean, updated_covariance = ekf_update(
            mean,
            covariance,
            np.array([1.0]),
            lambda state: state,
            np.array([[0.04]]),
        )
        self.assertGreater(float(updated_mean[0]), float(mean[0]))
        self.assertLess(float(updated_covariance[0, 0]), float(covariance[0, 0]))

    def test_result_is_json_serializable(self) -> None:
        result = simulate(
            load_test_system("rc_circuit"),
            duration=0.02,
            dt=0.01,
            controls={"voltage": 1.0},
            num_samples=3,
        )
        json.dumps(result.to_dict())
        self.assertEqual(result.metadata["interval_kind"], "pointwise predictive interval")


class BackendSelectionTests(unittest.TestCase):
    def test_explicit_cuda_with_numpy_never_silently_falls_back(self) -> None:
        for backend in (None, "numpy", "np", "cpu", NumpyBackend()):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, r"cuda.*(NumPy|torch backend)"):
                    get_backend(backend, device="cuda")

    def test_auto_cuda_propagates_unavailable_cuda_error(self) -> None:
        with patch(
            "contraption.physics.backend.TorchBackend",
            side_effect=RuntimeError("mock CUDA unavailable"),
        ) as constructor:
            with self.assertRaisesRegex(RuntimeError, "mock CUDA unavailable"):
                get_backend("auto", device="cuda")
        constructor.assert_called_once()

    def test_cuda_backend_alias_rejects_a_conflicting_cpu_device(self) -> None:
        with self.assertRaisesRegex(ValueError, r"backend='cuda'.*device='cpu'"):
            get_backend("cuda", device="cpu")


class OptionalTorchTests(unittest.TestCase):
    @staticmethod
    def _torch_or_skip(test_case):
        try:
            import torch
        except ImportError:
            test_case.skipTest("optional PyTorch is not installed")
        return torch

    def test_torch_trajectory_keeps_parameter_gradient(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch is not installed")
        resistance = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        model = load_test_system("rc_circuit")
        result = simulate(
            model,
            duration=0.5,
            dt=0.01,
            controls={"voltage": 5.0},
            parameters={"rc.resistance": resistance, "rc.capacitance": 0.5},
            num_samples=1,
            backend="torch",
            device="cpu",
            use_model_uncertainty=False,
            process_noise=False,
        )
        loss = result.series("rc.capacitor_voltage")[0, -1]
        loss.backward()
        self.assertIsNotNone(resistance.grad)
        self.assertTrue(torch.isfinite(resistance.grad))
        self.assertGreater(abs(float(resistance.grad)), 1e-4)

    def test_torch_process_noise_replays_and_differentiates_scale(self) -> None:
        torch = self._torch_or_skip(self)
        scale = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        common = dict(
            duration=0.01,
            dt=0.01,
            parameters={
                "body.roughness_std": scale,
                "body.roughness_reference_length": 1.0,
            },
            num_samples=128,
            seed=73,
            backend="torch",
            device="cpu",
            use_model_uncertainty=False,
        )
        first = simulate(load_test_system("planar_rigid_body"), **common)
        repeat = simulate(load_test_system("planar_rigid_body"), **common)
        self.assertTrue(torch.equal(first.samples, repeat.samples))
        loss = torch.mean(first.series("body.x")[:, -1] ** 2)
        loss.backward()
        self.assertIsNotNone(scale.grad)
        self.assertTrue(torch.isfinite(scale.grad))
        self.assertGreater(float(scale.grad), 0.0)

    def test_torch_descriptor_failures_are_not_regularized_or_returned(self) -> None:
        self._torch_or_skip(self)
        cases = (
            (
                lambda t, state, state_dot, parameters, controls: state * 0.0,
                RuntimeError,
                r"Jacobian is singular.*timestep=1.*sample=0",
                {},
            ),
            (
                lambda t, state, state_dot, parameters, controls: state * state - 2.0,
                RuntimeError,
                r"did not converge.*timestep=1.*sample=0.*residual_max=",
                {"newton_max_iterations": 1},
            ),
            (
                lambda t, state, state_dot, parameters, controls: state * float("nan"),
                FloatingPointError,
                r"residual is non-finite.*timestep=1.*sample=0",
                {},
            ),
        )
        for residual, exception, pattern, options in cases:
            with self.subTest(pattern=pattern):
                model = ResidualSystem(("x",), residual, [1.0])
                with self.assertRaisesRegex(exception, pattern):
                    simulate(
                        model,
                        duration=0.01,
                        dt=0.01,
                        num_samples=1,
                        backend="torch",
                        device="cpu",
                        process_noise=False,
                        **options,
                    )

    def test_backend_native_pmdl_math_preserves_torch_autograd_and_device(self) -> None:
        torch = self._torch_or_skip(self)

        class DeclarativeModel:
            state_names = ("angle",)
            algebraic_names = ()
            input_names = ()
            states = (SimpleNamespace(name="angle", initial=0.4, derivative=None),)
            algebraics = ()
            parameters = (
                SimpleNamespace(
                    name="gain",
                    default=1.0,
                    bounds=SimpleNamespace(lower=0.01, upper=10.0),
                    uncertainty=SimpleNamespace(
                        distribution="fixed", parameters={}, correlation_group=None
                    ),
                ),
            )
            relations = (
                SimpleNamespace(
                    expression=(
                        "der(angle) + gain * (sin(angle) + cos(angle) + tan(0.1 * angle) "
                        "+ tanh(angle) "
                        "+ asin(0.1 * angle) + acos(0.1 * angle) + atan(angle) "
                        "+ atan2(angle, 1) + exp(0.1 * angle) + log(angle + 2) "
                        "+ log10(angle + 2) + sqrt(angle + 1) + abs(angle) "
                        "+ min(angle, 1) + max(angle, 0) + clip(angle, 0, 1) "
                        "+ sign(angle) + where(angle > 0, angle, -angle) "
                        "+ smooth_abs(angle))"
                    )
                ),
            )

            @staticmethod
            def evaluate_residual(*args):
                raise AssertionError("NumPy PMDL evaluator was called")

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
            automatic = get_backend("auto", device="cuda")
            self.assertTrue(automatic.is_torch)
            self.assertEqual(automatic.device.type, "cuda")
        for device in devices:
            with self.subTest(device=device):
                gain = torch.tensor(1.3, dtype=torch.float64, device=device, requires_grad=True)
                result = simulate(
                    DeclarativeModel(),
                    duration=0.002,
                    dt=0.001,
                    parameters={"gain": gain},
                    num_samples=1,
                    backend="torch",
                    device=device,
                    use_model_uncertainty=False,
                    process_noise=False,
                )
                self.assertEqual(result.samples.device.type, device)
                loss = result.mean[-1, 0]
                loss.backward()
                self.assertIsNotNone(gain.grad)
                self.assertTrue(torch.isfinite(gain.grad))
                self.assertGreater(abs(float(gain.grad)), 1e-6)


if __name__ == "__main__":
    unittest.main()
