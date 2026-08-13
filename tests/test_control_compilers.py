from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from contraption import load_contraption
from contraption.control import (
    AffineObserverModel,
    ControlCompilerError,
    ControlIR,
    ControlRuntime,
    FixedPointFormat,
    compile_control,
    compile_resolved_controller,
    control_digest,
    generate_c99,
    generate_verilog,
    parse_control,
)
from contraption.control.compiler.ir import (
    _controller_expressions,
    _issue_ir,
    target_identifier,
)


ROOT = Path(__file__).resolve().parents[1]
HASH = "sha256:" + "1" * 64
LINK_HASH = "sha256:" + "2" * 64
ASSEMBLY_HASH = "sha256:" + "3" * 64
PMDL_HASH = "sha256:" + "4" * 64


def compiler_document() -> dict[str, object]:
    return {
        "format": "control-1",
        "id": "test.compiled_controller",
        "name": "Compiled controller",
        "version": "2.0.0",
        "period_s": 0.1,
        "explicit_inputs": [
            {"name": "measurement", "source": "external", "bounds": [-10.0, 10.0]},
            {"name": "enable", "source": "external", "dtype": "bool"},
            {"name": "emergency", "source": "external", "dtype": "bool"},
        ],
        "outputs": [
            {
                "name": "command",
                "bounds": [-1.0, 1.0],
                "slew_rate": 2.0,
                "emergency_value": 0.0,
            }
        ],
        "registers": [{"name": "memory", "initial": 0.0, "bounds": [-1.0, 1.0]}],
        "derived": [{"name": "error", "expression": "1.0 - input.measurement"}],
        "modes": [
            {
                "name": "idle",
                "outputs": {"command": "0"},
                "transitions": [{"target": "tracking", "guard": "input.enable", "priority": 1}],
            },
            {
                "name": "tracking",
                "outputs": {"command": "clip(derived.error + register.memory, -1, 1)"},
                "updates": {"memory": "register.memory + 0.01 * derived.error"},
                "transitions": [{"target": "idle", "guard": "not input.enable", "priority": 1}],
            },
        ],
        "initial_mode": "idle",
        "emergency_when": "input.emergency",
    }


@pytest.fixture(scope="module")
def scanner_assembly():
    return load_contraption(ROOT / "assembled_contraptions/scanner/contraption.json")


def test_standalone_bundle_is_deterministic_and_target_complete(tmp_path: Path) -> None:
    spec = parse_control(compiler_document())
    first = compile_control(spec)
    second = compile_control(spec)
    assert first.files == second.files
    assert first.manifest == second.manifest
    assert tuple(first.files) == (
        "test_compiled_controller.h",
        "test_compiled_controller.c",
        "test_compiled_controller.v",
    )
    assert first.manifest["schema"] == "contraption-control-compiler/v2"
    assert first.manifest["closure"] is None
    assert {path.name for path in first.write(tmp_path)} == {
        "test_compiled_controller.h",
        "test_compiled_controller.c",
        "test_compiled_controller.v",
        "manifest.json",
    }


def test_resolved_bundle_carries_exact_zoh_observer_and_immutable_closure(scanner_assembly) -> None:
    controller = scanner_assembly.controllers["scanner_orbit_controller"]
    observer = controller.observer
    assert observer is not None
    bundle = compile_resolved_controller(scanner_assembly, controller.id)
    closure = bundle.manifest["closure"]
    assert closure["assembly_sha256"] == scanner_assembly.assembly_sha256
    assert closure["pmdl_sha256"] == scanner_assembly.system.pmdl_sha256
    assert closure["controller_link_digest"] == controller.controller_link_digest
    assert closure["observer_digest"] == observer.digest
    derivation = closure["observer_derivation"]
    assert derivation["discretization"] == "exact_zero_order_hold_matrix_exponential"
    assert derivation["pmdl_dynamics_classification"] == "nonlinear"
    assert derivation["approximation"] is True
    assert derivation["discrete_transition_spectral_radius"] <= 1.0 + 1e-10
    assert bundle.manifest["observability"][0]["observable"] is True
    assert "OBS_TRANSITION" in bundle.files["scanner_orbit_controller.c"]
    assert "obs_transition" in bundle.files["scanner_orbit_controller.v"]
    assert "input wire signed [47:0] implicit_" not in bundle.files["scanner_orbit_controller.v"]

    digest = observer.digest
    with pytest.raises(TypeError):
        observer.derivation["open_gate_admission"]["assembly_status"] = "complete"  # type: ignore[index]
    assert observer.digest == digest
    with pytest.raises(TypeError):
        bundle.closure["observer_derivation"]["approximation"] = False  # type: ignore[index]


def test_bare_implicit_spec_and_forged_ir_are_rejected(scanner_assembly) -> None:
    controller = scanner_assembly.controllers["scanner_orbit_controller"]
    with pytest.raises(ControlCompilerError, match="ResolvedAssembly"):
        compile_control(controller.spec)

    standalone = parse_control(compiler_document())
    with pytest.raises(ControlCompilerError, match="factory-issued"):
        ControlIR(
            standalone,
            target_identifier(standalone.id),
            HASH,
            _controller_expressions(standalone),
            (),
        )

    controller = scanner_assembly.controllers["scanner_orbit_controller"]
    assert controller.observer is not None
    injected_observer = replace(
        controller.observer,
        A=np.array(controller.observer.A, copy=True) + np.eye(len(controller.observer.A)),
    )
    injected_controller = replace(controller, observer=injected_observer)
    controllers = type(scanner_assembly.controllers)(
        {
            **dict(scanner_assembly.controllers),
            controller.id: injected_controller,
        }
    )
    injected_assembly = replace(scanner_assembly, controllers=controllers)
    with pytest.raises(ControlCompilerError, match="canonical plant-derived observer"):
        compile_resolved_controller(injected_assembly, controller.id)

    issued_ir = ControlIR.from_resolved(scanner_assembly, controller.id)
    forged_transition = replace(
        controller.observer,
        transition=np.array(controller.observer.transition, copy=True)
        + 1e-3 * np.eye(len(controller.observer.transition)),
    )
    with pytest.raises(ControlCompilerError, match="factory-issued"):
        replace(issued_ir, observer=forged_transition)

    changed_document = controller.spec.to_dict()
    changed_document["parameters"][0]["default"] = (  # type: ignore[index]
        float(changed_document["parameters"][0]["default"]) + 0.01  # type: ignore[index]
    )
    changed_spec = parse_control(changed_document)
    from contraption.control.observer import derive_affine_observer

    changed_observer = derive_affine_observer(
        scanner_assembly.system,
        changed_spec,
        explicit_bindings=controller.explicit_input_bindings,
        implicit_bindings=controller.implicit_input_bindings,
        output_bindings=controller.plant_output_bindings,
        assembly_sha256=scanner_assembly.assembly_sha256,
        pmdl_sha256=scanner_assembly.system.pmdl_sha256,
        controller_link_digest=controller.controller_link_digest,
        dynamics_completeness=scanner_assembly.dynamics_completeness.to_dict(),
    )
    changed_controller = replace(
        controller, spec=changed_spec, observer=changed_observer
    )
    changed_assembly = replace(
        scanner_assembly,
        controllers=type(scanner_assembly.controllers)(
            {
                **dict(scanner_assembly.controllers),
                controller.id: changed_controller,
            }
        ),
    )
    with pytest.raises(ControlCompilerError, match="canonical program sha256"):
        compile_resolved_controller(changed_assembly, controller.id)


def test_c99_matches_python_for_one_second_and_holds_state_on_fault(
    scanner_assembly, tmp_path: Path
) -> None:
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("GCC is unavailable")
    controller = scanner_assembly.controllers["scanner_orbit_controller"]
    observer = controller.observer
    assert observer is not None
    runtime = ControlRuntime(
        controller.spec, observer=observer, emit_observability_warnings=False
    )
    records: list[tuple[dict[str, object], object]] = []
    for tick in range(100):
        values: dict[str, object] = {
            item.name: item.default for item in controller.spec.explicit_inputs
        }
        values["armed"] = tick < 80
        values["emergency_stop"] = tick == 50
        values["left_wheel_rate"] = 1.5
        values["right_wheel_rate"] = 1.7
        frame = runtime.step(values)
        records.append((values, frame))
    assert records[-1][1].time == pytest.approx(1.0)
    assert np.isfinite(records[-1][1].implicit_inputs["forward_speed"].variance)

    bundle = compile_resolved_controller(
        scanner_assembly, controller.id, targets=("c99",)
    )
    bundle.write(tmp_path)
    input_rows: list[str] = []
    expected_rows: list[str] = []
    for values, frame in records:
        input_rows.append(
            "{" + ",".join(
                "true" if values[item.name] is True else
                "false" if values[item.name] is False else
                format(float(values[item.name]), ".17g")
                for item in controller.spec.explicit_inputs
            ) + "}"
        )
        expected_rows.append(
            "{" + ",".join(
                [
                    *(
                        ("true" if bool(np.asarray(frame.outputs[item.name])) else "false")
                        if item.dtype == "bool"
                        else format(float(np.asarray(frame.outputs[item.name])), ".17g")
                        for item in controller.spec.outputs
                    ),
                    format(float(np.asarray(frame.implicit_inputs["forward_speed"].mean)), ".17g"),
                    format(float(np.asarray(frame.implicit_inputs["forward_speed"].variance)), ".17g"),
                    str(("idle", "scanning", "emergency").index(frame.next_mode)),
                ]
            ) + "}"
        )
    harness = tmp_path / "harness.c"
    harness.write_text(
        f"""
#include "scanner_orbit_controller.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

typedef struct expected {{ double left, right, lift, tilt; bool record_video; double mean, variance; unsigned mode; }} expected;
static const scanner_orbit_controller_explicit_inputs INPUTS[100] = {{
{','.join(input_rows)}
}};
static const expected EXPECTED[100] = {{
{','.join(expected_rows)}
}};
static int close_value(double left, double right) {{
    return fabs(left - right) <= 1e-8 * fmax(1.0, fmax(fabs(left), fabs(right)));
}}
static int output_matches_state(const scanner_orbit_controller_outputs *output, const scanner_orbit_controller_state *state) {{
    return close_value(output->left_voltage, state->output_left_voltage) &&
           close_value(output->right_voltage, state->output_right_voltage) &&
           close_value(output->lift_target, state->output_lift_target) &&
           close_value(output->tilt_target, state->output_tilt_target) &&
           output->record_video == state->output_record_video;
}}
int main(void) {{
    scanner_orbit_controller_state state, snapshot;
    scanner_orbit_controller_outputs output;
    scanner_orbit_controller_init(&state);
    for (int i = 0; i < 100; ++i) {{
        int status = scanner_orbit_controller_step(&state, &INPUTS[i], &output);
        if (status != SCANNER_ORBIT_CONTROLLER_STATUS_OK) return 10 + i;
        if (!close_value(output.left_voltage, EXPECTED[i].left)) return 120;
        if (!close_value(output.right_voltage, EXPECTED[i].right)) return 121;
        if (!close_value(output.lift_target, EXPECTED[i].lift)) return 122;
        if (!close_value(output.tilt_target, EXPECTED[i].tilt)) return 123;
        if (output.record_video != EXPECTED[i].record_video) return 127;
        if (!close_value(state.implicit_forward_speed_mean, EXPECTED[i].mean)) return 124;
        if (!close_value(state.implicit_forward_speed_variance, EXPECTED[i].variance)) return 125;
        if (state.mode != EXPECTED[i].mode) return 126;
    }}
    snapshot = state;
    scanner_orbit_controller_explicit_inputs invalid = INPUTS[99];
    invalid.target_speed = 999.0;
    output.left_voltage = output.right_voltage = output.lift_target = output.tilt_target = 999.0;
    output.record_video = true;
    if (scanner_orbit_controller_step(&state, &invalid, &output) != SCANNER_ORBIT_CONTROLLER_STATUS_INVALID_INPUT) return 130;
    if (memcmp(&state, &snapshot, sizeof(state)) != 0) return 131;
    if (!output_matches_state(&output, &state)) return 136;
    invalid = INPUTS[99]; invalid.target_speed = NAN;
    output.left_voltage = output.right_voltage = output.lift_target = output.tilt_target = 999.0;
    output.record_video = true;
    if (scanner_orbit_controller_step(&state, &invalid, &output) != SCANNER_ORBIT_CONTROLLER_STATUS_INVALID_INPUT) return 132;
    if (memcmp(&state, &snapshot, sizeof(state)) != 0) return 133;
    if (!output_matches_state(&output, &state)) return 137;
    state.observer_covariance[0] = NAN; snapshot = state;
    output.left_voltage = output.right_voltage = output.lift_target = output.tilt_target = 999.0;
    output.record_video = true;
    if (scanner_orbit_controller_step(&state, &INPUTS[99], &output) != SCANNER_ORBIT_CONTROLLER_STATUS_OBSERVER_FAILURE) return 134;
    if (memcmp(&state, &snapshot, sizeof(state)) != 0) return 135;
    if (!output_matches_state(&output, &state)) return 138;
    return 0;
}}
""".lstrip(),
        encoding="utf-8",
    )
    source = tmp_path / "scanner_orbit_controller.c"
    executable = tmp_path / "scanner_controller"
    result = subprocess.run(
        [
            gcc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            str(source),
            str(harness),
            "-I",
            str(tmp_path),
            "-lm",
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    executed = subprocess.run([str(executable)], check=False)
    assert executed.returncode == 0


def test_verilog_is_vector_observer_with_fault_hold_and_no_unbounded_time_state(
    scanner_assembly,
) -> None:
    source = compile_resolved_controller(
        scanner_assembly, "scanner_orbit_controller", targets=("verilog",)
    ).files["scanner_orbit_controller.v"]
    assert "output reg input_error" in source
    assert "output reg observer_error" in source
    assert "output reg arithmetic_error" in source
    assert "inputs_valid && !observer_failure && !arithmetic_failure" in source
    assert "q_fault" in source
    assert "q_add" in source and "q_sub" in source
    assert "time_q" not in source
    assert "q_sign" not in source
    assert " real " not in source


def test_verilog_executable_trace_matches_quantized_runtime(
    scanner_assembly, tmp_path: Path
) -> None:
    iverilog = shutil.which("iverilog")
    vvp = shutil.which("vvp")
    if iverilog is None or vvp is None:
        if os.environ.get("CI"):
            pytest.fail("Icarus Verilog is required in CI")
        pytest.skip("Icarus Verilog is unavailable")

    controller = scanner_assembly.controllers["scanner_orbit_controller"]
    observer = controller.observer
    assert observer is not None
    fixed = FixedPointFormat()
    runtime = ControlRuntime(
        controller.spec, observer=observer, emit_observability_warnings=False
    )
    defaults = {
        item.name: item.default for item in controller.spec.explicit_inputs
    }
    valid_rows: list[dict[str, object]] = []
    for updates in (
        {},
        {"armed": True},
        {"armed": True, "left_wheel_rate": 1.5, "right_wheel_rate": 1.7},
        {
            "armed": True,
            "emergency_stop": True,
            "left_wheel_rate": 1.5,
            "right_wheel_rate": 1.7,
        },
    ):
        row = dict(defaults)
        row.update(updates)
        valid_rows.append(row)
    invalid_row = dict(valid_rows[-1])
    invalid_row["emergency_stop"] = False
    invalid_row["target_speed"] = 999.0
    final_row = dict(defaults)
    final_row["armed"] = True
    rows = [*valid_rows, invalid_row, final_row]

    state_index = observer.state_names.index("chassis.forward_speed")
    expected: list[tuple[int, ...]] = []
    for index, row in enumerate(rows):
        if index != 4:
            frame = runtime.step(row)
            assert frame.next_mode == runtime.mode
        expected.append(
            (
                *(
                    (1 if bool(runtime.outputs[item.name]) else 0)
                    if item.dtype == "bool"
                    else fixed.quantize(float(runtime.outputs[item.name]))
                    for item in controller.spec.outputs
                ),
                fixed.quantize(float(runtime.implicit_inputs["forward_speed"].mean)),
                fixed.quantize(float(runtime.implicit_inputs["forward_speed"].variance)),
                fixed.quantize(float(np.asarray(runtime.observer_state[state_index]))),
                fixed.quantize(
                    float(
                        np.asarray(
                            runtime.observer_covariance[state_index, state_index]
                        )
                    )
                ),
                ("idle", "scanning", "emergency").index(runtime.mode),
                1 if index == 4 else 0,
                0,
                0,
            )
        )

    bundle = compile_resolved_controller(
        scanner_assembly,
        controller.id,
        targets=("verilog",),
        fixed_point=fixed,
    )
    module_path = tmp_path / "scanner_orbit_controller.v"
    module_path.write_text(
        bundle.files[module_path.name], encoding="utf-8", newline="\n"
    )

    def literal(value: object) -> str:
        if isinstance(value, bool):
            return "1'b1" if value else "1'b0"
        integer = fixed.quantize(float(value))
        return (
            f"-{fixed.total_bits}'sd{abs(integer)}"
            if integer < 0
            else f"{fixed.total_bits}'sd{integer}"
        )

    ports = [".clk(clk)", ".reset_n(reset_n)", ".tick(tick)"]
    ports.extend(
        f".explicit_{item.name}(explicit_{item.name})"
        for item in controller.spec.explicit_inputs
    )
    ports.extend(
        f".output_{item.name}(output_{item.name})"
        for item in controller.spec.outputs
    )
    ports.extend(
        (".input_error(input_error)", ".observer_error(observer_error)", ".arithmetic_error(arithmetic_error)")
    )
    declarations: list[str] = []
    for item in controller.spec.explicit_inputs:
        width = "" if item.dtype == "bool" else f" signed [{fixed.total_bits - 1}:0]"
        declarations.append(f"reg{width} explicit_{item.name};")
    for item in controller.spec.outputs:
        width = "" if item.dtype == "bool" else f" signed [{fixed.total_bits - 1}:0]"
        declarations.append(f"wire{width} output_{item.name};")
    stimulus: list[str] = []
    for index, row in enumerate(rows):
        stimulus.extend(
            [
                "        @(negedge clk);",
                *(f"        explicit_{item.name} = {literal(row[item.name])};" for item in controller.spec.explicit_inputs),
                "        tick = 1'b1;",
                "        @(posedge clk); #1;",
                (
                    f'        $display("TRACE {index} %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d", '
                    "output_left_voltage, output_right_voltage, output_lift_target, output_tilt_target, output_record_video, "
                    "dut.implicit_forward_speed_mean, dut.implicit_forward_speed_variance, "
                    f"dut.observer_state[{state_index}], dut.observer_covariance[{state_index * len(observer.state_names) + state_index}], "
                    "dut.mode, input_error, observer_error, arithmetic_error);"
                ),
                "        @(negedge clk); tick = 1'b0;",
            ]
        )
    testbench = tmp_path / "scanner_tb.v"
    testbench.write_text(
        "\n".join(
            [
                "`timescale 1ns/1ps",
                "module scanner_tb;",
                "reg clk, reset_n, tick;",
                *declarations,
                "wire input_error, observer_error, arithmetic_error;",
                f"scanner_orbit_controller dut ({', '.join(ports)});",
                "always #5 clk = ~clk;",
                "initial begin",
                "        clk = 1'b0; reset_n = 1'b0; tick = 1'b0;",
                *(f"        explicit_{item.name} = {literal(item.default)};" for item in controller.spec.explicit_inputs),
                "        repeat (2) @(posedge clk);",
                "        @(negedge clk); reset_n = 1'b1;",
                *stimulus,
                "        $finish;",
                "end",
                "endmodule",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    executable = tmp_path / "scanner_tb"
    compiled = subprocess.run(
        [iverilog, "-g2005", "-o", str(executable), str(module_path), str(testbench)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(
        [vvp, str(executable)], capture_output=True, text=True, check=False
    )
    assert executed.returncode == 0, executed.stderr
    traces = [line.split() for line in executed.stdout.splitlines() if line.startswith("TRACE ")]
    assert len(traces) == len(expected), executed.stdout
    for index, (tokens, reference) in enumerate(zip(traces, expected, strict=True)):
        assert int(tokens[1]) == index
        actual = tuple(int(value) for value in tokens[2:])
        # Outputs are controller expressions; the coupled observer path has many
        # fixed-point multiply/accumulate roundings.  Both tolerances are explicit
        # in target LSBs and remain far below controller engineering resolution.
        for measured, wanted in zip(actual[:4], reference[:4], strict=True):
            assert abs(measured - wanted) <= 64
        assert actual[4] == reference[4]
        for measured, wanted in zip(actual[5:9], reference[5:9], strict=True):
            assert abs(measured - wanted) <= 16384
        assert actual[9:] == reference[9:]
    # Emergency values bypass slew on the same admitted tick; invalid input holds.
    assert tuple(int(value) for value in traces[3][2:7]) == tuple(expected[3][:5])
    assert tuple(int(value) for value in traces[4][2:11]) == tuple(
        int(value) for value in traces[3][2:11]
    )


def test_smooth_target_specific_math_is_c99_only() -> None:
    document = compiler_document()
    document["derived"][0]["expression"] = "tanh(input.measurement)"  # type: ignore[index]
    spec = parse_control(document)
    assert len(generate_c99(spec)) == 2
    with pytest.raises(ControlCompilerError, match="fixed-point lowering"):
        generate_verilog(spec)


def test_verilog_admission_rejects_intermediate_overflow_and_mode_collision() -> None:
    overflow = compiler_document()
    overflow["derived"][0]["expression"] = "input.measurement ** 8"  # type: ignore[index]
    with pytest.raises(ControlCompilerError, match="cannot be proven inside"):
        generate_verilog(
            parse_control(overflow),
            fixed_point=FixedPointFormat(total_bits=16, fractional_bits=8),
        )

    collision = compiler_document()
    collision["modes"][0]["name"] = "run"  # type: ignore[index]
    collision["modes"][1]["name"] = "RUN"  # type: ignore[index]
    collision["initial_mode"] = "run"
    collision["modes"][0]["transitions"][0]["target"] = "RUN"  # type: ignore[index]
    collision["modes"][1]["transitions"][0]["target"] = "run"  # type: ignore[index]
    with pytest.raises(ControlCompilerError, match="mode names collide"):
        generate_verilog(parse_control(collision))


def test_target_symbol_tables_and_fixed_point_lattice_are_injective_and_safe() -> None:
    collision = compiler_document()
    collision["explicit_inputs"].extend(  # type: ignore[union-attr]
        [
            {"name": "switch", "source": "external"},
            {"name": "control_switch", "source": "external"},
        ]
    )
    with pytest.raises(ControlCompilerError, match="C99 symbol collision"):
        generate_c99(parse_control(collision))

    divisor = compiler_document()
    divisor["explicit_inputs"][0]["bounds"] = [0.001, 1.0]  # type: ignore[index]
    divisor["explicit_inputs"][0]["default"] = 0.001  # type: ignore[index]
    divisor["derived"][0]["expression"] = "1 / input.measurement"  # type: ignore[index]
    with pytest.raises(ControlCompilerError, match="quantizes to zero|exclude zero"):
        generate_verilog(
            parse_control(divisor),
            fixed_point=FixedPointFormat(total_bits=16, fractional_bits=4),
        )

    rail = compiler_document()
    rail["explicit_inputs"][0].update(  # type: ignore[index]
        {"default": 0.0, "bounds": [-1.0, 1.0]}
    )
    rail["outputs"][0].update(  # type: ignore[index]
        {"default": 7.8, "bounds": [0.0, 7.8], "slew_rate": 0.2}
    )
    rail["period_s"] = 1.0
    for mode in rail["modes"]:  # type: ignore[union-attr]
        mode["outputs"]["command"] = "7.8"
    with pytest.raises(
        ControlCompilerError,
        match="slew intermediate|reserved|outside its quantized authored bounds",
    ):
        generate_verilog(
            parse_control(rail),
            fixed_point=FixedPointFormat(total_bits=8, fractional_bits=4),
        )

    directed = compiler_document()
    directed["explicit_inputs"][0].update(  # type: ignore[index]
        {"default": 0.0, "bounds": [-0.1, 0.1]}
    )
    directed["outputs"][0].update(  # type: ignore[index]
        {"default": 0.0, "bounds": [-0.1, 0.1], "slew_rate": None}
    )
    for mode in directed["modes"]:  # type: ignore[union-attr]
        mode["outputs"]["command"] = "input.measurement"
    source = generate_verilog(
        parse_control(directed),
        fixed_point=FixedPointFormat(total_bits=16, fractional_bits=8),
    ).content
    assert "explicit_measurement < -16'sd25" in source
    assert "explicit_measurement > 16'sd25" in source
    assert "q_max(output_command_next, -16'sd25)" in source
    assert "q_min(q_max(output_command_next, -16'sd25), 16'sd25)" in source


def test_scanner_observer_emits_strict_c99(scanner_assembly, tmp_path: Path) -> None:
    header, source = generate_c99(
        ControlIR.from_resolved(scanner_assembly, "scanner_orbit_controller")
    )
    (tmp_path / header.path).write_text(header.content, encoding="utf-8")
    source_path = tmp_path / source.path
    source_path.write_text(source.content, encoding="utf-8")
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("GCC is unavailable")
    result = subprocess.run(
        [gcc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic", "-fsyntax-only", str(source_path), "-I", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_zero_plant_input_observer_emits_strict_c99(
    scanner_assembly, tmp_path: Path
) -> None:
    """Prediction-only plant input is legal and emits no unsigned ``< 0`` loops."""

    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("GCC is unavailable")
    ir = ControlIR.from_resolved(scanner_assembly, "scanner_orbit_controller")
    observer = ir.observer
    assert observer is not None
    nx = len(observer.state_names)
    ny = len(observer.measurement_names)
    nz = len(observer.latent_names)
    zero_input_observer = replace(
        observer,
        input_names=(),
        plant_input_names=(),
        B=np.empty((nx, 0)),
        D=np.empty((ny, 0)),
        M=np.empty((nz, 0)),
        discrete_input=np.empty((nx, 0)),
    )
    assert ir.closure is not None
    closure = dict(ir.closure)
    closure["observer_digest"] = zero_input_observer.digest
    zero_input_ir = _issue_ir(
        ControlIR,
        spec=ir.spec,
        identifier=ir.identifier,
        source_digest=ir.source_digest,
        expressions=ir.expressions,
        observability=zero_input_observer.observability,
        observer=zero_input_observer,
        closure=closure,
    )
    header, source = generate_c99(zero_input_ir)
    (tmp_path / header.path).write_text(header.content, encoding="utf-8")
    source_path = tmp_path / source.path
    source_path.write_text(source.content, encoding="utf-8")
    assert "< SCANNER_ORBIT_CONTROLLER_OBSERVER_NU" not in source.content
    result = subprocess.run(
        [
            gcc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fsyntax-only",
            str(source_path),
            "-I",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fixed_point_constants_and_noise_modes_are_range_checked(scanner_assembly) -> None:
    fixed = FixedPointFormat(total_bits=8, fractional_bits=4)
    assert fixed.quantize(1.5) == 24
    with pytest.raises(ControlCompilerError, match="does not fit"):
        fixed.quantize(100.0)
    with pytest.raises(ControlCompilerError, match="covariance|quantizes to zero|does not fit"):
        compile_resolved_controller(
            scanner_assembly,
            "scanner_orbit_controller",
            targets=("verilog",),
            fixed_point=fixed,
        )
