from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from contraption import load_contraption
from contraption.control import (
    ControlRuntime,
    ControlRuntimeError,
    ControlSpecError,
    ControlValidationError,
    control_digest,
    dump_control,
    observability_diagnostics,
    parse_control,
)


ROOT = Path(__file__).resolve().parents[1]


def controller_document() -> dict[str, object]:
    return {
        "format": "control-1",
        "id": "test.tracking_controller",
        "name": "Tracking controller",
        "version": "2.0.0",
        "period_s": 0.1,
        "explicit_inputs": [
            {
                "name": "measurement",
                "source": "sensor",
                "unit": "1",
                "default": 0.0,
                "bounds": [-10.0, 10.0],
            },
            {"name": "enable", "source": "external", "dtype": "bool", "default": False},
            {"name": "emergency", "source": "external", "dtype": "bool", "default": False},
        ],
        "outputs": [
            {
                "name": "command",
                "unit": "1",
                "default": 0.0,
                "bounds": [-1.0, 1.0],
                "slew_rate": 2.0,
                "emergency_value": 0.0,
            }
        ],
        "parameters": [
            {"name": "target", "unit": "1", "default": 1.0},
            {"name": "gain", "unit": "1", "default": 1.0},
            {"name": "integral_gain", "unit": "1", "default": 0.05},
        ],
        "registers": [
            {"name": "integral", "unit": "1", "initial": 0.0, "bounds": [-0.5, 0.5]}
        ],
        "derived": [
            {"name": "error", "unit": "1", "expression": "parameter.target - input.measurement"}
        ],
        "modes": [
            {
                "name": "idle",
                "outputs": {"command": "0"},
                "transitions": [{"target": "tracking", "guard": "input.enable", "priority": 10}],
            },
            {
                "name": "tracking",
                "outputs": {"command": "clip(parameter.gain * derived.error + register.integral, -1, 1)"},
                "updates": {"integral": "register.integral + parameter.integral_gain * derived.error"},
                "transitions": [{"target": "idle", "guard": "not input.enable", "priority": 10}],
            },
        ],
        "initial_mode": "idle",
        "emergency_when": "input.emergency",
        "metadata": {"owner": "tests"},
    }


def implicit_document() -> dict[str, object]:
    document = controller_document()
    document["explicit_inputs"][0]["measurement_variance"] = 0.25  # type: ignore[index]
    document["implicit_inputs"] = [
        {
            "name": "position",
            "unit": "1",
            "initial_variance": 1.0,
            "process_variance_per_s": 0.01,
            "bounds": [-10.0, 10.0],
        }
    ]
    document["observer"] = {
        "kind": "local_affine",
        "nonlinear_approximation": "approved",
        "acknowledged_open_gates": [],
        "sample_radius_relative": 0.0001,
        "maximum_sampled_remainder": 0.01,
    }
    return document


def test_control_1_is_strict_canonical_and_has_no_scalar_estimator_shape() -> None:
    document = controller_document()
    spec = parse_control(document)
    assert spec.explicit_inputs[0].name == "measurement"
    assert spec.implicit_inputs == ()
    assert spec.observer is None
    assert spec.to_dict()["format"] == "control-1"
    assert parse_control(dump_control(spec)) == spec
    assert control_digest(spec).startswith("sha256:")
    assert control_digest(parse_control(json.dumps(document))) == control_digest(spec)

    unknown = deepcopy(document)
    unknown["python_callback"] = "dangerous.module:function"
    with pytest.raises(ControlSpecError, match="unknown field"):
        parse_control(unknown)

    legacy = implicit_document()
    legacy["implicit_inputs"][0]["measurement"] = "measurement"  # type: ignore[index]
    with pytest.raises(ControlSpecError, match="unknown field"):
        parse_control(legacy)

    duplicated_truth = implicit_document()
    duplicated_truth["implicit_inputs"][0]["initial_mean"] = 0.0  # type: ignore[index]
    with pytest.raises(ControlSpecError, match="unknown field"):
        parse_control(duplicated_truth)


def test_sensor_variance_is_observer_scoped_and_boolean_sensor_is_valid() -> None:
    assert parse_control(controller_document()).explicit_inputs[0].measurement_variance is None

    missing = implicit_document()
    del missing["explicit_inputs"][0]["measurement_variance"]  # type: ignore[index]
    with pytest.raises(ControlValidationError, match="measurement_variance"):
        parse_control(missing)

    boolean_sensor = controller_document()
    boolean_sensor["explicit_inputs"][1]["source"] = "sensor"  # type: ignore[index]
    assert parse_control(boolean_sensor).explicit_inputs[1].source == "sensor"

    invalid_boolean_sensor = deepcopy(boolean_sensor)
    invalid_boolean_sensor["explicit_inputs"][1]["measurement_variance"] = 0.1  # type: ignore[index]
    with pytest.raises(ControlSpecError, match="boolean sensor"):
        parse_control(invalid_boolean_sensor)


def test_expression_admission_enforces_units_types_and_common_lowering() -> None:
    document = controller_document()
    document["explicit_inputs"][0]["unit"] = "m"  # type: ignore[index]
    with pytest.raises(ControlValidationError, match="same dimension|dimension mismatch"):
        parse_control(document)

    unknown = controller_document()
    unknown["derived"][0]["expression"] = "estimate.position.mean"  # type: ignore[index]
    with pytest.raises(ControlValidationError, match="unknown symbol"):
        parse_control(unknown)

    graph_breaking = controller_document()
    graph_breaking["derived"][0]["expression"] = "sign(input.measurement)"  # type: ignore[index]
    with pytest.raises(ControlValidationError, match="discontinuous numeric graph"):
        parse_control(graph_breaking)

    target_specific = controller_document()
    target_specific["derived"][0]["expression"] = "tanh(input.measurement)"  # type: ignore[index]
    assert parse_control(target_specific).derived[0].expression.startswith("tanh")

    boolean_as_real = controller_document()
    boolean_as_real["modes"][0]["outputs"]["command"] = "input.enable"  # type: ignore[index]
    with pytest.raises(ControlValidationError, match="expression type"):
        parse_control(boolean_as_real)

    unsafe_sqrt = controller_document()
    unsafe_sqrt["derived"][0]["expression"] = "sqrt(input.measurement)"  # type: ignore[index]
    with pytest.raises(ControlValidationError, match="sqrt argument interval"):
        parse_control(unsafe_sqrt)

    unsafe_divisor = controller_document()
    unsafe_divisor["derived"][0]["expression"] = "1.0 / input.measurement"  # type: ignore[index]
    with pytest.raises(ControlValidationError, match="does not exclude zero"):
        parse_control(unsafe_divisor)

    duplicate = '{"format":"control-1","format":"control-1"}'
    with pytest.raises(ControlSpecError, match="duplicate JSON key"):
        parse_control(duplicate)


def test_runtime_modes_registers_slew_emergency_and_fixed_period() -> None:
    runtime = ControlRuntime(parse_control(controller_document()), backend="numpy")
    first = runtime.step({"measurement": 0.0, "enable": True, "emergency": False})
    assert first.active_mode == "idle"
    assert first.next_mode == "tracking"
    assert np.asarray(first.outputs["command"]).item() == pytest.approx(0.0)

    second = runtime.step({"measurement": 0.0, "enable": True, "emergency": False})
    assert second.active_mode == "tracking"
    assert np.asarray(second.outputs["command"]).item() == pytest.approx(0.2)
    assert np.asarray(second.registers["integral"]).item() == pytest.approx(0.05)

    emergency = runtime.step({"measurement": 0.0, "enable": True, "emergency": True})
    assert emergency.emergency is True
    assert np.asarray(emergency.outputs["command"]).item() == pytest.approx(0.0)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        runtime.step({}, dt=0.2)  # type: ignore[call-arg]


def test_implicit_controller_requires_the_resolved_plant_observer() -> None:
    with pytest.raises(ControlRuntimeError, match="resolved affine observer"):
        ControlRuntime(parse_control(implicit_document()))


def test_resolved_scanner_observer_is_coupled_observable_and_differentiable() -> None:
    torch = pytest.importorskip("torch")
    assembly = load_contraption(ROOT / "assembled_contraptions/scanner/contraption.json")
    resolved = assembly.controllers["scanner_orbit_controller"]
    observer = resolved.observer
    assert observer is not None
    assert len(observer.state_names) > 1
    assert observer.C.shape[0] == 8
    diagnostic = observability_diagnostics(observer)[0]
    assert diagnostic.implicit_input == "forward_speed"
    assert diagnostic.variable == "chassis.forward_speed"
    assert diagnostic.observable

    runtime = ControlRuntime(resolved.spec, observer=observer, backend="torch")
    supplied: dict[str, object] = {item.name: item.default for item in resolved.spec.explicit_inputs}
    wheel_rate = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    supplied["left_wheel_rate"] = wheel_rate
    frame = runtime.step(supplied)
    mean = frame.implicit_inputs["forward_speed"].mean
    mean.backward()
    assert wheel_rate.grad is not None
    assert np.isfinite(wheel_rate.grad.item())
    assert wheel_rate.grad.item() != 0.0


def test_lazy_where_does_not_evaluate_inactive_negative_sqrt_branch() -> None:
    torch = pytest.importorskip("torch")
    document = controller_document()
    document["parameters"] = []
    document["registers"] = []
    document["explicit_inputs"] = [document["explicit_inputs"][0]]  # type: ignore[index]
    document["outputs"][0].pop("slew_rate")  # type: ignore[index]
    document["outputs"][0]["bounds"] = [-10.0, 10.0]  # type: ignore[index]
    document["derived"] = [
        {
            "name": "safe",
            "expression": "where(input.measurement > 0, sqrt(input.measurement), 0.0 * input.measurement)",
        }
    ]
    document["modes"] = [{"name": "idle", "outputs": {"command": "derived.safe"}}]
    document["initial_mode"] = "idle"
    document.pop("emergency_when")

    runtime = ControlRuntime(parse_control(document), backend="torch")
    value = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    frame = runtime.step({"measurement": value})
    output = frame.outputs["command"]
    output.backward()
    assert output.item() == pytest.approx(0.0)
    assert value.grad is not None
    assert np.isfinite(value.grad.item())
    assert value.grad.item() == pytest.approx(0.0)
