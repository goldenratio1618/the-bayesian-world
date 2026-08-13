from __future__ import annotations

from pathlib import Path
import importlib.util
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from contraption.control import parse_control
from contraption import load_contraption
from contraption.physics.backend import NumpyBackend
from contraption.physics.resolved import (
    ResolvedController,
    ResolvedControllerOutputBinding,
    ResolvedExplicitInputBinding,
)
from contraption.physics.simulator import simulate
from contraption.physics.specs import FrozenDict


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "assembled_contraptions" / "scanner" / "contraption.json"
CONTROLLER_DIGEST = "sha256:" + "0" * 64


def _feedback_controller(identifier: str, period_s: float) -> ResolvedController:
    program = parse_control(
        {
            "format": "control-1",
            "id": identifier,
            "name": identifier,
            "version": "1.0.0",
            "period_s": period_s,
            "explicit_inputs": [
                {
                    "name": "visible",
                    "source": "sensor",
                    "unit": "1",
                    "measurement_variance": 0.01,
                    "bounds": {"lower": -100.0, "upper": 100.0},
                }
            ],
            "outputs": [
                {
                    "name": "drive",
                    "unit": "1",
                    "bounds": {"lower": -100.0, "upper": 100.0},
                }
            ],
            "modes": [
                {"name": "active", "outputs": {"drive": "input.visible"}}
            ],
            "initial_mode": "active",
        }
    )
    return ResolvedController(
        id=identifier,
        spec=program,
        explicit_input_bindings=FrozenDict(
            {
                "visible": ResolvedExplicitInputBinding(
                    "sensor", "visible_sensor", "visible", 0
                )
            }
        ),
        implicit_input_bindings=FrozenDict(),
        output_bindings=FrozenDict(
            {
                "drive": ResolvedControllerOutputBinding(
                    "signal", f"{identifier}.drive", "visible"
                )
            }
        ),
        controller_link_digest=CONTROLLER_DIGEST,
    )


def _external_controller(
    identifier: str, period_s: float, pin: str
) -> ResolvedController:
    program = parse_control(
        {
            "format": "control-1",
            "id": identifier,
            "name": identifier,
            "version": "1.0.0",
            "period_s": period_s,
            "explicit_inputs": [
                {
                    "name": "command",
                    "source": "external",
                    "unit": "1",
                    "bounds": {"lower": -100.0, "upper": 100.0},
                }
            ],
            "outputs": [
                {
                    "name": "drive",
                    "unit": "1",
                    "bounds": {"lower": -100.0, "upper": 100.0},
                }
            ],
            "modes": [
                {"name": "active", "outputs": {"drive": "input.command"}}
            ],
            "initial_mode": "active",
        }
    )
    return ResolvedController(
        id=identifier,
        spec=program,
        explicit_input_bindings=FrozenDict(
            {"command": ResolvedExplicitInputBinding("external", pin)}
        ),
        implicit_input_bindings=FrozenDict(),
        output_bindings=FrozenDict(
            {
                "drive": ResolvedControllerOutputBinding(
                    "signal", f"{identifier}.drive", "visible"
                )
            }
        ),
        controller_link_digest=CONTROLLER_DIGEST,
    )


class ResolvedControllerSimulationTests(unittest.TestCase):
    def test_observability_warning_is_emitted_once_per_controller(self) -> None:
        import contraption.physics.simulator as simulator_module

        class FixtureWarning(UserWarning):
            pass

        def construct_runtime(*args: object, **kwargs: object) -> mock.Mock:
            if kwargs["emit_observability_warnings"]:
                warnings.warn("unobservable fixture", FixtureWarning, stacklevel=2)
            return mock.Mock()

        controller = _feedback_controller("warning_controller", 0.01)
        with mock.patch.object(
            simulator_module,
            "ControlRuntime",
            side_effect=construct_runtime,
        ) as runtime_constructor, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FixtureWarning)
            simulator_module._ResolvedControllerExecutor(
                controller,
                count=8,
                backend=NumpyBackend(),
            )

        self.assertEqual(len(caught), 1)
        self.assertEqual(
            [call.kwargs["emit_observability_warnings"] for call in runtime_constructor.call_args_list],
            [True, False, False, False, False, False, False, False],
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls.assembly = load_contraption(SCANNER)

    def test_explicit_pins_and_implicit_posterior_state_are_structurally_distinct(self) -> None:
        controller = self.assembly.controllers["scanner_orbit_controller"]
        explicit = {item.name: item for item in controller.spec.explicit_inputs}
        self.assertEqual(
            {name for name, item in explicit.items() if item.source == "external"},
            {"armed", "reset", "emergency_stop", "target_speed", "orbit_radius"},
        )
        self.assertEqual(
            {name for name, item in explicit.items() if item.source == "sensor"},
            {
                "position_x",
                "position_y",
                "heading_x",
                "heading_y",
                "left_wheel_rate",
                "right_wheel_rate",
                "lift_feedback",
                "tilt_feedback",
            },
        )
        for name, binding in controller.explicit_input_bindings.items():
            if explicit[name].source == "sensor":
                self.assertEqual(binding.kind, "sensor")
                self.assertIsNotNone(binding.state_name)
                self.assertIsNotNone(binding.state_index)
            else:
                self.assertEqual(binding.kind, "external")
                self.assertIsNone(binding.state_name)
                self.assertIsNone(binding.state_index)
        self.assertEqual(
            {item.name for item in controller.spec.implicit_inputs},
            {"forward_speed"},
        )
        self.assertEqual(set(controller.implicit_input_bindings), {"forward_speed"})
        self.assertIsNotNone(controller.observer)

    def test_controller_runs_inside_offline_simulation_and_verification_is_automatic(self) -> None:
        result = simulate(
            self.assembly,
            duration=0.02,
            dt=0.01,
            num_samples=2,
            use_model_uncertainty=False,
            process_noise=False,
            controller_inputs={"armed": True},
        )
        self.assertEqual(tuple(result.controller_traces), ("scanner_orbit_controller",))
        trace = result.controller_traces["scanner_orbit_controller"]
        self.assertEqual(
            trace.output_names,
            (
                "left_voltage",
                "right_voltage",
                "lift_target",
                "tilt_target",
                "record_video",
            ),
        )
        self.assertEqual(tuple(trace.output_samples.shape), (2, 3, 5))
        np.testing.assert_array_equal(
            trace.output_samples[:, :, trace.output_names.index("record_video")],
            np.asarray([[False, False, True], [False, False, True]]),
        )
        self.assertEqual(tuple(trace.implicit_means.shape), (2, 3, 1))
        self.assertEqual(tuple(trace.implicit_variances.shape), (2, 3, 1))
        self.assertEqual(trace.active_modes[0], ("idle", "idle"))
        self.assertEqual(trace.active_modes[1], ("idle", "idle"))
        self.assertEqual(trace.tick_mask, (False, True, True))
        self.assertEqual(trace.frame_times_s, (0.0, 0.01, 0.02))
        self.assertEqual(tuple(result.verification_reports), ("scanner.orbit_acceptance",))
        report = result.verification_reports["scanner.orbit_acceptance"]
        self.assertEqual(report.sample_count, 2)
        self.assertEqual(report.time_count, 3)

    def test_physics_step_must_be_a_controller_period_subdivision(self) -> None:
        with self.assertRaisesRegex(ValueError, "not commensurate"):
            simulate(
                self.assembly,
                duration=0.02,
                dt=0.02,
                num_samples=1,
                use_model_uncertainty=False,
                process_noise=False,
            )

    def test_heterogeneous_controller_periods_tick_and_hold_independently(self) -> None:
        class Plant:
            state_names = ("visible",)
            control_names = ("fast.drive", "slow.drive")
            initial_state = (1.0,)
            default_parameters: dict[str, float] = {}

            @staticmethod
            def derivative(t, state, parameters, controls, backend):
                del t, state, parameters
                return backend.stack(
                    [controls["fast.drive"] + controls["slow.drive"]], axis=-1
                )

        fast = _feedback_controller("fast", 0.01)
        slow = _feedback_controller("slow", 0.02)
        result = simulate(
            SimpleNamespace(
                system=Plant(),
                controllers=FrozenDict({fast.id: fast, slow.id: slow}),
                verifications=FrozenDict(),
            ),
            duration=0.04,
            dt=0.01,
            num_samples=1,
            integrator="euler",
            process_noise=False,
        )
        fast_trace = result.controller_traces["fast"]
        slow_trace = result.controller_traces["slow"]
        self.assertEqual(fast_trace.tick_mask, (False, True, True, True, True))
        self.assertEqual(slow_trace.tick_mask, (False, False, True, False, True))
        self.assertEqual(fast_trace.frame_times_s, (0.0, 0.01, 0.02, 0.03, 0.04))
        self.assertEqual(slow_trace.frame_times_s, (0.0, 0.01, 0.02, 0.03, 0.04))
        self.assertEqual(tuple(fast_trace.output_samples.shape), (1, 5, 1))
        self.assertEqual(tuple(slow_trace.output_samples.shape), (1, 5, 1))
        self.assertEqual(
            float(slow_trace.output_samples[0, 0, 0]),
            float(slow_trace.output_samples[0, 1, 0]),
        )
        self.assertNotEqual(
            float(slow_trace.output_samples[0, 1, 0]),
            float(slow_trace.output_samples[0, 2, 0]),
        )
        self.assertEqual(result.metadata["physics_dt_s"], 0.01)
        self.assertEqual(
            result.metadata["controller_periods_s"], {"fast": 0.01, "slow": 0.02}
        )

    def test_long_horizon_ticks_sample_only_due_controller_inputs(self) -> None:
        class Plant:
            state_names = ("x",)
            control_names = ("fast.drive", "slow.drive")
            initial_state = (0.0,)
            default_parameters: dict[str, float] = {}

            @staticmethod
            def derivative(t, state, parameters, controls, backend):
                del t, parameters, controls
                return backend.stack([0.0 * state[:, 0]], axis=-1)

        fast = _external_controller("fast", 0.01, "fast_pin")
        slow = _external_controller("slow", 0.02, "slow_pin")
        calls = {"fast": 0, "slow": 0}

        def fast_input(t):
            del t
            calls["fast"] += 1
            return 1.0

        def slow_input(t):
            del t
            calls["slow"] += 1
            return 2.0

        result = simulate(
            SimpleNamespace(
                system=Plant(),
                controllers=FrozenDict({fast.id: fast, slow.id: slow}),
                verifications=FrozenDict(),
            ),
            duration=1.0,
            dt=0.01,
            num_samples=1,
            integrator="euler",
            process_noise=False,
            controller_inputs={"fast_pin": fast_input, "slow_pin": slow_input},
        )
        self.assertEqual(sum(result.controller_traces["fast"].tick_mask), 100)
        self.assertEqual(sum(result.controller_traces["slow"].tick_mask), 50)
        self.assertEqual(calls, {"fast": 100, "slow": 50})

    def test_partial_final_physics_step_does_not_create_a_phantom_tick(self) -> None:
        class Plant:
            state_names = ("x",)
            control_names = ("slow.drive",)
            initial_state = (0.0,)
            default_parameters: dict[str, float] = {}

            @staticmethod
            def derivative(t, state, parameters, controls, backend):
                del t, parameters, controls
                return backend.stack([0.0 * state[:, 0]], axis=-1)

        slow = _external_controller("slow", 0.02, "slow_pin")
        result = simulate(
            SimpleNamespace(
                system=Plant(),
                controllers=FrozenDict({slow.id: slow}),
                verifications=FrozenDict(),
            ),
            duration=0.015,
            dt=0.01,
            num_samples=1,
            integrator="euler",
            process_noise=False,
            controller_inputs={"slow_pin": 1.0},
        )
        self.assertEqual(result.controller_traces["slow"].tick_mask, (False, False, False))

    def test_descriptor_sensors_are_sampled_only_after_consistent_initialization(self) -> None:
        class DescriptorPlant:
            state_names = ("visible",)
            control_names = ("feedback.drive",)
            initial_state = (0.0,)
            default_parameters: dict[str, float] = {}

            def __init__(self) -> None:
                self.initialization_commands: list[float] = []

            def consistent_initial_state(
                self, t, state, parameters, controls, backend
            ):
                del t, state, parameters
                command = controls["feedback.drive"]
                self.initialization_commands.append(float(np.asarray(command)[0]))
                return backend.stack([2.0 + command], axis=-1)

            @staticmethod
            def residual(t, state, state_derivative, parameters, controls, backend):
                del t, state_derivative, parameters
                return backend.stack(
                    [state[:, 0] - (2.0 + controls["feedback.drive"])], axis=-1
                )

        plant = DescriptorPlant()
        controller = _feedback_controller("feedback", 0.1)
        result = simulate(
            SimpleNamespace(
                system=plant,
                controllers=FrozenDict({controller.id: controller}),
                verifications=FrozenDict(),
            ),
            duration=0.1,
            dt=0.1,
            num_samples=1,
            process_noise=False,
        )
        self.assertEqual(plant.initialization_commands, [0.0])
        self.assertAlmostEqual(
            float(result.controller_traces["feedback"].output_samples[0, 0, 0]),
            0.0,
        )
        self.assertAlmostEqual(float(result.samples[0, 0, 0]), 2.0)
        self.assertAlmostEqual(
            float(result.controller_traces["feedback"].output_samples[0, 1, 0]),
            2.0,
        )

    def test_time_zero_trace_is_reset_state_and_does_not_sample_external_pin(self) -> None:
        class Plant:
            state_names = ("x",)
            control_names = ("external.drive",)
            initial_state = (0.0,)
            default_parameters: dict[str, float] = {}

            @staticmethod
            def derivative(t, state, parameters, controls, backend):
                del t, parameters
                return backend.stack(
                    [state[:, 0] * 0.0 + controls["external.drive"]], axis=-1
                )

        controller = _external_controller("external", 0.1, "command_pin")
        calls: list[float] = []

        def command(time_s):
            calls.append(float(time_s))
            return 3.0

        result = simulate(
            SimpleNamespace(
                system=Plant(),
                controllers=FrozenDict({controller.id: controller}),
                verifications=FrozenDict(),
            ),
            duration=0.1,
            dt=0.1,
            num_samples=1,
            integrator="euler",
            controller_inputs={"command_pin": command},
            process_noise=False,
        )
        trace = result.controller_traces[controller.id]
        self.assertEqual(trace.frame_times_s, (0.0, 0.1))
        self.assertEqual(trace.tick_mask, (False, True))
        self.assertEqual(calls, [0.1])
        self.assertEqual(float(trace.output_samples[0, 0, 0]), 0.0)
        self.assertEqual(float(trace.output_samples[0, 1, 0]), 3.0)
        self.assertEqual(trace.active_modes, (("active",), ("active",)))

    def test_multiple_controllers_execute_and_merge_distinct_actuator_sources(self) -> None:
        class Plant:
            state_names = ("visible",)
            control_names = ("controller_a.drive", "controller_b.drive")
            initial_state = (1.0,)
            default_parameters: dict[str, float] = {}

            @staticmethod
            def derivative(t, state, parameters, controls, backend):
                del t, state, parameters
                return backend.stack(
                    [controls["controller_a.drive"] + controls["controller_b.drive"]],
                    axis=-1,
                )

        def controller(identifier: str, expression: str) -> ResolvedController:
            program = parse_control(
                {
                    "format": "control-1",
                    "id": identifier,
                    "name": identifier,
                    "version": "1.0.0",
                    "period_s": 0.1,
                    "explicit_inputs": [
                        {
                            "name": "visible",
                            "source": "sensor",
                            "unit": "1",
                            "measurement_variance": 0.01,
                        }
                    ],
                    "outputs": [{"name": "drive", "unit": "1"}],
                    "modes": [
                        {"name": "active", "outputs": {"drive": expression}}
                    ],
                    "initial_mode": "active",
                }
            )
            return ResolvedController(
                id=identifier,
                spec=program,
                explicit_input_bindings=FrozenDict(
                    {
                        "visible": ResolvedExplicitInputBinding(
                            "sensor", "visible_sensor", "visible", 0
                        )
                    }
                ),
                implicit_input_bindings=FrozenDict(),
                output_bindings=FrozenDict(
                    {
                        "drive": ResolvedControllerOutputBinding(
                            "signal", f"{identifier}.drive", "visible"
                        )
                    }
                ),
                controller_link_digest=CONTROLLER_DIGEST,
            )

        first = controller("controller_a", "-input.visible")
        second = controller("controller_b", "0.5 * input.visible")
        contraption = SimpleNamespace(
            system=Plant(),
            controllers=FrozenDict({first.id: first, second.id: second}),
            verifications=FrozenDict(),
        )
        result = simulate(
            contraption,
            duration=0.2,
            dt=0.1,
            num_samples=1,
            integrator="euler",
            process_noise=False,
        )
        self.assertEqual(
            tuple(result.controller_traces), ("controller_a", "controller_b")
        )
        self.assertAlmostEqual(float(result.samples[0, -1, 0]), 0.95)
        for trace in result.controller_traces.values():
            self.assertEqual(trace.tick_mask, (False, True, True))
            self.assertEqual(trace.frame_times_s, (0.0, 0.1, 0.2))

    def test_unknown_external_controller_pin_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown external controller input"):
            simulate(
                self.assembly,
                duration=0.01,
                dt=0.01,
                num_samples=1,
                use_model_uncertainty=False,
                process_noise=False,
                controller_inputs={"hidden_truth": 1.0},
            )

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is optional")
    def test_multirate_holds_preserve_torch_autograd(self) -> None:
        import torch

        class Plant:
            state_names = ("visible",)
            control_names = ("fast.drive", "slow.drive")
            initial_state = (1.0,)
            default_parameters: dict[str, float] = {}

            @staticmethod
            def derivative(t, state, parameters, controls, backend):
                del t, state, parameters
                return backend.stack(
                    [controls["fast.drive"] + controls["slow.drive"]], axis=-1
                )

        fast = _feedback_controller("fast", 0.01)
        slow = _feedback_controller("slow", 0.02)
        initial = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        result = simulate(
            SimpleNamespace(
                system=Plant(),
                controllers=FrozenDict({fast.id: fast, slow.id: slow}),
                verifications=FrozenDict(),
            ),
            duration=0.02,
            dt=0.01,
            num_samples=1,
            initial_state=initial,
            backend="torch",
            integrator="euler",
            process_noise=False,
        )
        self.assertEqual(
            result.controller_traces["slow"].tick_mask, (False, False, True)
        )
        self.assertTrue(result.controller_traces["slow"].output_samples.requires_grad)
        loss = result.samples.sum() + sum(
            trace.output_samples.sum()
            for trace in result.controller_traces.values()
        )
        loss.backward()
        self.assertIsNotNone(initial.grad)
        self.assertTrue(torch.isfinite(initial.grad).all())

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is optional")
    def test_sensor_extraction_and_controller_output_mapping_retain_autograd(self) -> None:
        import torch

        class Plant:
            state_names = ("visible", "hidden")
            control_names = ("gradient_controller.drive",)
            initial_state = (1.0, 2.0)
            default_parameters: dict[str, float] = {}

            @staticmethod
            def derivative(t, state, parameters, controls, backend):
                del t, parameters
                return backend.stack(
                    [controls["gradient_controller.drive"], 0.0 * state[:, 1]],
                    axis=-1,
                )

        program = parse_control(
            {
                "format": "control-1",
                "id": "gradient_controller",
                "name": "Gradient controller",
                "version": "1.0.0",
                "period_s": 0.1,
                "explicit_inputs": [
                    {
                        "name": "visible",
                        "source": "sensor",
                        "unit": "1",
                        "measurement_variance": 0.01,
                        "bounds": {"lower": -100.0, "upper": 100.0},
                    }
                ],
                "outputs": [
                    {
                        "name": "drive",
                        "unit": "1",
                        "bounds": {"lower": -100.0, "upper": 100.0},
                    }
                ],
                "modes": [
                    {"name": "active", "outputs": {"drive": "-input.visible"}}
                ],
                "initial_mode": "active",
            }
        )
        controller = ResolvedController(
            id="gradient_controller",
            spec=program,
            explicit_input_bindings=FrozenDict(
                {
                    "visible": ResolvedExplicitInputBinding(
                        "sensor", "visible_sensor", "visible", 0
                    )
                }
            ),
            implicit_input_bindings=FrozenDict(),
            output_bindings=FrozenDict(
                {
                    "drive": ResolvedControllerOutputBinding(
                        "signal", "gradient_controller.drive", "visible"
                    )
                }
            ),
            controller_link_digest=CONTROLLER_DIGEST,
        )
        contraption = SimpleNamespace(
            system=Plant(),
            controllers=FrozenDict({controller.id: controller}),
            verifications=FrozenDict(),
        )
        initial = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        result = simulate(
            contraption,
            duration=0.1,
            dt=0.1,
            num_samples=1,
            initial_state=initial,
            backend="torch",
            integrator="euler",
            process_noise=False,
        )
        trace = result.controller_traces[controller.id]
        self.assertTrue(trace.output_samples.requires_grad)
        (result.samples.sum() + trace.output_samples.sum()).backward()
        self.assertIsNotNone(initial.grad)
        self.assertTrue(torch.isfinite(initial.grad).all())


if __name__ == "__main__":
    unittest.main()
