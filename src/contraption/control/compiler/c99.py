"""Complete, allocation-free C99 controller generator."""

from __future__ import annotations

import math
import json

import numpy as np

from ...physics.dsl import Binary, Call, Comparison, Conditional, Expression, Literal, Symbol, Unary, parse_expression
from ..specs import ControlSpec
from .ir import (
    ControlCompilerError,
    ControlIR,
    GeneratedArtifact,
    admit_target_symbol_table,
    as_ir,
    plain_compiler_data,
    target_identifier,
)


def _admit_c_symbols(ir: ControlIR) -> None:
    spec = ir.spec
    public_inputs = tuple(
        (f"explicit input {item.name}", target_identifier(item.name))
        for item in spec.explicit_inputs
    )
    public_outputs = tuple(
        (f"output {item.name}", target_identifier(item.name))
        for item in spec.outputs
    )
    state_fields: list[tuple[str, str]] = [
        ("internal mode", "mode"),
        ("internal time", "time"),
        ("internal time_in_mode", "time_in_mode"),
    ]
    state_fields.extend(
        (f"register {item.name}", f"register_{item.name}")
        for item in spec.registers
    )
    state_fields.extend(
        (f"output {item.name}", f"output_{item.name}")
        for item in spec.outputs
    )
    for item in spec.implicit_inputs:
        state_fields.extend(
            (
                (f"implicit input {item.name} mean", f"implicit_{item.name}_mean"),
                (
                    f"implicit input {item.name} variance",
                    f"implicit_{item.name}_variance",
                ),
            )
        )
    mode_symbols = tuple(
        (f"mode {item.name}", f"MODE_{item.name.upper()}") for item in spec.modes
    )
    admit_target_symbol_table(
        "C99",
        {
            "explicit-input fields": public_inputs,
            "output fields": public_outputs,
            "controller state fields": tuple(state_fields),
            "mode constants": mode_symbols,
        },
    )


def _number(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ControlCompilerError("C constants must be finite")
    result = format(float(value), ".17g")
    return result if any(character in result for character in ".eE") else result + ".0"


class _CExpression:
    def __init__(self, ir: ControlIR) -> None:
        self.ir = ir
        self.spec = ir.spec
        self.parameters = {item.name: item for item in self.spec.parameters}

    def symbol(self, name: str) -> str:
        if name == "pi":
            return "CTRL_PI"
        if name == "e":
            return "CTRL_E"
        if name in {"time", "time_in_mode"}:
            return f"state->{name}"
        if name == "dt":
            return "dt"
        parts = name.split(".")
        if len(parts) == 2 and parts[0] == "input":
            return f"explicit_input->{target_identifier(parts[1])}"
        if len(parts) == 2 and parts[0] == "output":
            return f"state->output_{parts[1]}"
        if len(parts) == 2 and parts[0] == "register":
            return f"state->register_{parts[1]}"
        if len(parts) == 2 and parts[0] == "parameter":
            try:
                value = self.parameters[parts[1]].default
            except KeyError as exc:
                raise ControlCompilerError(f"unknown parameter symbol {name!r}") from exc
            return "true" if value is True else "false" if value is False else _number(value)
        if len(parts) == 2 and parts[0] == "derived":
            return f"derived_{parts[1]}"
        if len(parts) == 3 and parts[0] == "implicit":
            if parts[2] == "std":
                return f"sqrt(state->implicit_{parts[1]}_variance)"
            if parts[2] in {"mean", "variance"}:
                return f"state->implicit_{parts[1]}_{parts[2]}"
        raise ControlCompilerError(f"cannot lower control symbol {name!r} to C99")

    def render(self, source: str | Expression) -> str:
        node = parse_expression(source) if isinstance(source, str) else source
        if isinstance(node, Literal):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            return _number(node.value)
        if isinstance(node, Symbol):
            return self.symbol(node.name)
        if isinstance(node, Unary):
            operator = "!" if node.operator == "not" else node.operator
            return f"({operator}{self.render(node.operand)})"
        if isinstance(node, Binary):
            operator = {"and": "&&", "or": "||"}.get(node.operator, node.operator)
            if operator == "**":
                return f"pow({self.render(node.left)}, {self.render(node.right)})"
            return f"({self.render(node.left)} {operator} {self.render(node.right)})"
        if isinstance(node, Comparison):
            return f"({self.render(node.left)} {node.operator} {self.render(node.right)})"
        if isinstance(node, Conditional):
            return (
                f"({self.render(node.condition)} ? {self.render(node.when_true)} : "
                f"{self.render(node.when_false)})"
            )
        if isinstance(node, Call):
            args = [self.render(item) for item in node.arguments]
            if node.function == "der":
                raise ControlCompilerError("der() is not legal in controller code")
            unary = {
                "abs": "fabs",
                "sqrt": "sqrt",
                "sin": "sin",
                "cos": "cos",
                "tan": "tan",
                "tanh": "tanh",
                "asin": "asin",
                "acos": "acos",
                "atan": "atan",
                "exp": "exp",
                "log": "log",
                "log10": "log10",
            }
            if node.function in unary:
                return f"{unary[node.function]}({', '.join(args)})"
            if node.function == "atan2":
                return f"atan2({args[0]}, {args[1]})"
            if node.function == "min":
                return f"fmin({args[0]}, {args[1]})"
            if node.function == "max":
                return f"fmax({args[0]}, {args[1]})"
            if node.function == "clip":
                return f"ctrl_clip({args[0]}, {args[1]}, {args[2]})"
            if node.function == "sign":
                return f"ctrl_sign({args[0]})"
            if node.function == "where":
                return f"({args[0]} ? {args[1]} : {args[2]})"
            if node.function == "smooth_abs":
                epsilon = args[1] if len(args) == 2 else "1e-12"
                return f"sqrt(({args[0]}) * ({args[0]}) + ({epsilon}) * ({epsilon}))"
            raise ControlCompilerError(
                f"function {node.function!r} has no C99 controller lowering"
            )
        raise ControlCompilerError(f"cannot lower {type(node).__name__} to C99")


def _ctype(dtype: str) -> str:
    return "bool" if dtype == "bool" else "double"


def _bounded(expression: str, lower: float | None, upper: float | None) -> str:
    result = expression
    if lower is not None:
        result = f"fmax({result}, {_number(lower)})"
    if upper is not None:
        result = f"fmin({result}, {_number(upper)})"
    return result


def _closure_comment(ir: ControlIR) -> list[str]:
    if ir.closure is None:
        return []
    payload = json.dumps(
        plain_compiler_data(ir.closure),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).replace("*/", "* /")
    return [f"/* Resolved closure: {payload} */"]


def _values(value: np.ndarray) -> str:
    return ", ".join(_number(float(item)) for item in value.reshape(-1))


def _observer_source(ir: ControlIR) -> list[str]:
    observer = ir.observer
    if observer is None:
        return []
    name = ir.identifier
    upper = name.upper()
    c_values = _values(observer.C) if observer.C.size else "0.0"
    d_values = _values(observer.D) if observer.D.size else "0.0"
    discrete_input_values = (
        _values(observer.discrete_input)
        if observer.discrete_input.size
        else "0.0"
    )
    latent_input_values = _values(observer.M) if observer.M.size else "0.0"
    measurement_bias_values = (
        _values(observer.measurement_bias)
        if observer.measurement_bias.size
        else "0.0"
    )
    measurement_variance_values = (
        _values(observer.measurement_variance)
        if observer.measurement_variance.size
        else "0.0"
    )
    c_size = max(1, observer.C.size)
    d_size = max(1, observer.D.size)
    measurement_size = max(1, len(observer.measurement_names))
    state_input_lines = (
        [
            f"        for (j = 0; j < {upper}_OBSERVER_NU; ++j)",
            f"            x[i] += OBS_DISCRETE_INPUT[i * {upper}_OBSERVER_NU_STORAGE + j] * input[j];",
        ]
        if observer.input_names
        else []
    )
    measurement_input_lines = (
        [
            f"        for (i = 0; i < {upper}_OBSERVER_NU; ++i)",
            f"            predicted += OBS_D[measurement_index * {upper}_OBSERVER_NU_STORAGE + i] * input[i];",
        ]
        if observer.input_names
        else []
    )
    lines = [
        f"static const double OBS_TRANSITION[{upper}_OBSERVER_NX * {upper}_OBSERVER_NX] = {{ {_values(observer.transition)} }};",
        *(
            [f"static const double OBS_DISCRETE_INPUT[{upper}_OBSERVER_NX * {upper}_OBSERVER_NU_STORAGE] = {{ {discrete_input_values} }};"]
            if observer.input_names
            else []
        ),
        f"static const double OBS_DISCRETE_BIAS[{upper}_OBSERVER_NX] = {{ {_values(observer.discrete_bias)} }};",
        f"static const double OBS_C[{c_size}] = {{ {c_values} }};",
        *(
            [f"static const double OBS_D[{d_size}] = {{ {d_values} }};"]
            if observer.input_names
            else []
        ),
        f"static const double OBS_MEASUREMENT_BIAS[{measurement_size}] = {{ {measurement_bias_values} }};",
        f"static const double OBS_L[{upper}_OBSERVER_NZ * {upper}_OBSERVER_NX] = {{ {_values(observer.L)} }};",
        *(
            [f"static const double OBS_M[{upper}_OBSERVER_NZ * {upper}_OBSERVER_NU_STORAGE] = {{ {latent_input_values} }};"]
            if observer.input_names
            else []
        ),
        f"static const double OBS_LATENT_BIAS[{upper}_OBSERVER_NZ] = {{ {_values(observer.latent_bias)} }};",
        f"static const double OBS_DISCRETE_Q[{upper}_OBSERVER_NX * {upper}_OBSERVER_NX] = {{ {_values(observer.discrete_process_covariance)} }};",
        f"static const double OBS_R[{measurement_size}] = {{ {measurement_variance_values} }};",
        "",
        f"static int {name}_observer_step({name}_state *state, const {name}_explicit_inputs *explicit_input) {{",
        *(
            [f"    double input[{upper}_OBSERVER_NU_STORAGE];"]
            if observer.input_names
            else []
        ),
        f"    double measurement[{measurement_size}];",
        f"    double x[{upper}_OBSERVER_NX];",
        f"    double covariance[{upper}_OBSERVER_NX * {upper}_OBSERVER_NX];",
        f"    double residual[{upper}_OBSERVER_NX * {upper}_OBSERVER_NX];",
        f"    double temporary[{upper}_OBSERVER_NX * {upper}_OBSERVER_NX];",
        f"    double next_covariance[{upper}_OBSERVER_NX * {upper}_OBSERVER_NX];",
        f"    double gain[{upper}_OBSERVER_NX];",
        "    size_t i, j, k, l;",
        "    int measurement_index;",
    ]
    if not observer.measurement_names:
        lines.append("    (void)explicit_input;")
    for index, input_name in enumerate(observer.input_names):
        lines.append(f"    input[{index}] = state->output_{input_name};")
    for index, measurement_name in enumerate(observer.measurement_names):
        lines.append(
            f"    measurement[{index}] = explicit_input->{target_identifier(measurement_name)};"
        )
    lines.extend(
        [
            f"    for (i = 0; i < {upper}_OBSERVER_NX; ++i) {{",
            "        x[i] = OBS_DISCRETE_BIAS[i];",
            f"        for (j = 0; j < {upper}_OBSERVER_NX; ++j)",
            f"            x[i] += OBS_TRANSITION[i * {upper}_OBSERVER_NX + j] * state->observer_state[j];",
            *state_input_lines,
            "    }",
            f"    for (i = 0; i < {upper}_OBSERVER_NX; ++i) {{",
            f"        for (j = 0; j < {upper}_OBSERVER_NX; ++j) {{",
            f"            covariance[i * {upper}_OBSERVER_NX + j] = OBS_DISCRETE_Q[i * {upper}_OBSERVER_NX + j];",
            f"            for (k = 0; k < {upper}_OBSERVER_NX; ++k)",
            f"                for (l = 0; l < {upper}_OBSERVER_NX; ++l)",
            f"                    covariance[i * {upper}_OBSERVER_NX + j] += OBS_TRANSITION[i * {upper}_OBSERVER_NX + k] * state->observer_covariance[k * {upper}_OBSERVER_NX + l] * OBS_TRANSITION[j * {upper}_OBSERVER_NX + l];",
            "        }",
            "    }",
            f"    for (measurement_index = 0; measurement_index < {upper}_OBSERVER_NY; ++measurement_index) {{",
            "        double predicted = OBS_MEASUREMENT_BIAS[measurement_index];",
            "        double innovation_variance = OBS_R[measurement_index];",
            f"        for (i = 0; i < {upper}_OBSERVER_NX; ++i)",
            f"            predicted += OBS_C[measurement_index * {upper}_OBSERVER_NX + i] * x[i];",
            *measurement_input_lines,
            "        const double innovation = measurement[measurement_index] - predicted;",
            f"        for (i = 0; i < {upper}_OBSERVER_NX; ++i)",
            f"            for (j = 0; j < {upper}_OBSERVER_NX; ++j)",
            f"                innovation_variance += OBS_C[measurement_index * {upper}_OBSERVER_NX + i] * covariance[i * {upper}_OBSERVER_NX + j] * OBS_C[measurement_index * {upper}_OBSERVER_NX + j];",
            "        if (!isfinite(innovation_variance) || innovation_variance <= 0.0) return 0;",
            f"        for (i = 0; i < {upper}_OBSERVER_NX; ++i) {{",
            "            gain[i] = 0.0;",
            f"            for (j = 0; j < {upper}_OBSERVER_NX; ++j)",
            f"                gain[i] += covariance[i * {upper}_OBSERVER_NX + j] * OBS_C[measurement_index * {upper}_OBSERVER_NX + j];",
            "            gain[i] /= innovation_variance;",
            "            x[i] += gain[i] * innovation;",
            "        }",
            f"        for (i = 0; i < {upper}_OBSERVER_NX; ++i)",
            f"            for (j = 0; j < {upper}_OBSERVER_NX; ++j)",
            f"                residual[i * {upper}_OBSERVER_NX + j] = (i == j ? 1.0 : 0.0) - gain[i] * OBS_C[measurement_index * {upper}_OBSERVER_NX + j];",
            f"        for (i = 0; i < {upper}_OBSERVER_NX; ++i) {{",
            f"            for (j = 0; j < {upper}_OBSERVER_NX; ++j) {{",
            f"                temporary[i * {upper}_OBSERVER_NX + j] = 0.0;",
            f"                for (k = 0; k < {upper}_OBSERVER_NX; ++k)",
            f"                    temporary[i * {upper}_OBSERVER_NX + j] += residual[i * {upper}_OBSERVER_NX + k] * covariance[k * {upper}_OBSERVER_NX + j];",
            "            }",
            "        }",
            f"        for (i = 0; i < {upper}_OBSERVER_NX; ++i) {{",
            f"            for (j = 0; j < {upper}_OBSERVER_NX; ++j) {{",
            "                next_covariance[i * " + upper + "_OBSERVER_NX + j] = gain[i] * OBS_R[measurement_index] * gain[j];",
            f"                for (k = 0; k < {upper}_OBSERVER_NX; ++k)",
            f"                    next_covariance[i * {upper}_OBSERVER_NX + j] += temporary[i * {upper}_OBSERVER_NX + k] * residual[j * {upper}_OBSERVER_NX + k];",
            "            }",
            "        }",
            f"        for (i = 0; i < {upper}_OBSERVER_NX * {upper}_OBSERVER_NX; ++i)",
            "            covariance[i] = next_covariance[i];",
            "    }",
            f"    for (i = 0; i < {upper}_OBSERVER_NX; ++i) {{",
            "        if (!isfinite(x[i])) return 0;",
            "        state->observer_state[i] = x[i];",
            "    }",
            f"    for (i = 0; i < {upper}_OBSERVER_NX * {upper}_OBSERVER_NX; ++i) {{",
            "        if (!isfinite(covariance[i])) return 0;",
            "        state->observer_covariance[i] = covariance[i];",
            "    }",
        ]
    )
    for latent_index, latent_name in enumerate(observer.latent_names):
        bounds_index = observer.latent_names.index(latent_name)
        mean_expression = f"OBS_LATENT_BIAS[{latent_index}]"
        latent_input_lines = (
            [
                f"    for (i = 0; i < {upper}_OBSERVER_NU; ++i)",
                f"        state->implicit_{latent_name}_mean += OBS_M[{latent_index} * {upper}_OBSERVER_NU_STORAGE + i] * input[i];",
            ]
            if observer.input_names
            else []
        )
        lines.extend(
            [
                f"    state->implicit_{latent_name}_mean = {mean_expression};",
                f"    state->implicit_{latent_name}_variance = 0.0;",
                f"    for (i = 0; i < {upper}_OBSERVER_NX; ++i) {{",
                f"        state->implicit_{latent_name}_mean += OBS_L[{latent_index} * {upper}_OBSERVER_NX + i] * state->observer_state[i];",
                f"        for (j = 0; j < {upper}_OBSERVER_NX; ++j)",
                f"            state->implicit_{latent_name}_variance += OBS_L[{latent_index} * {upper}_OBSERVER_NX + i] * state->observer_covariance[i * {upper}_OBSERVER_NX + j] * OBS_L[{latent_index} * {upper}_OBSERVER_NX + j];",
                "    }",
                *latent_input_lines,
                f"    if (!isfinite(state->implicit_{latent_name}_mean) || !isfinite(state->implicit_{latent_name}_variance)) return 0;",
                f"    state->implicit_{latent_name}_mean = {_bounded(f'state->implicit_{latent_name}_mean', observer.latent_lower_bounds[bounds_index], observer.latent_upper_bounds[bounds_index])};",
                f"    state->implicit_{latent_name}_variance = fmax(state->implicit_{latent_name}_variance, 0.0);",
            ]
        )
    lines.extend(["    return 1;", "}", ""])
    return lines


def _header(ir: ControlIR) -> str:
    spec, name = ir.spec, ir.identifier
    guard = f"CONTRAPTION_{name.upper()}_H"
    lines = [
        "/* Generated by contraption.control.compiler; do not edit. */",
        f"/* Source: {ir.source_digest} */",
        *_closure_comment(ir),
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdbool.h>",
        "#include <stdint.h>",
        "",
        f"#define {name.upper()}_MODE_COUNT {len(spec.modes)}",
        f"#define {name.upper()}_STATUS_OK 0",
        f"#define {name.upper()}_STATUS_NULL_POINTER 1",
        f"#define {name.upper()}_STATUS_INVALID_INPUT 2",
        f"#define {name.upper()}_STATUS_OBSERVER_FAILURE 3",
        f"#define {name.upper()}_STATUS_ARITHMETIC_FAILURE 4",
        "",
        f"typedef struct {name}_explicit_inputs {{",
    ]
    lines.extend(
        f"    {_ctype(item.dtype)} {target_identifier(item.name)};"
        for item in spec.explicit_inputs
    )
    if not spec.explicit_inputs:
        lines.append("    uint8_t _unused;")
    lines.extend(
        [f"}} {name}_explicit_inputs;", "", f"typedef struct {name}_outputs {{"]
    )
    lines.extend(
        f"    {_ctype(item.dtype)} {target_identifier(item.name)};"
        for item in spec.outputs
    )
    lines.extend(
        [
            f"}} {name}_outputs;",
            "",
            f"typedef struct {name}_state {{",
            "    uint32_t mode;",
            "    double time;",
            "    double time_in_mode;",
        ]
    )
    lines.extend(
        f"    {_ctype(item.dtype)} register_{item.name};" for item in spec.registers
    )
    lines.extend(
        f"    {_ctype(item.dtype)} output_{item.name};" for item in spec.outputs
    )
    for item in spec.implicit_inputs:
        lines.extend(
            [
                f"    double implicit_{item.name}_mean;",
                f"    double implicit_{item.name}_variance;",
            ]
        )
    if ir.observer is not None:
        upper = name.upper()
        lines.extend(
            [
                f"    double observer_state[{upper}_OBSERVER_NX];",
                f"    double observer_covariance[{upper}_OBSERVER_NX * {upper}_OBSERVER_NX];",
            ]
        )
    lines.extend(
        [
            f"}} {name}_state;",
            "",
            f"void {name}_init({name}_state *state);",
            f"int {name}_step({name}_state *state, const {name}_explicit_inputs *explicit_input, {name}_outputs *output);",
            "",
            f"#endif /* {guard} */",
        ]
    )
    if ir.observer is not None:
        insert_at = lines.index("")
        upper = name.upper()
        lines[insert_at:insert_at] = [
            f"#define {upper}_OBSERVER_NX {len(ir.observer.state_names)}",
            f"#define {upper}_OBSERVER_NU {len(ir.observer.input_names)}",
            f"#define {upper}_OBSERVER_NU_STORAGE {max(1, len(ir.observer.input_names))}",
            f"#define {upper}_OBSERVER_NY {len(ir.observer.measurement_names)}",
            f"#define {upper}_OBSERVER_NZ {len(ir.observer.latent_names)}",
        ]
    return "\n".join(lines) + "\n"


def _source(ir: ControlIR) -> str:
    spec, name = ir.spec, ir.identifier
    render = _CExpression(ir).render
    mode_index = {mode.name: index for index, mode in enumerate(spec.modes)}
    lines = [
        "/* Generated complete synchronous controller (C99). */",
        f"/* Source: {ir.source_digest} */",
        *_closure_comment(ir),
        f'#include "{name}.h"',
        "#include <math.h>",
        "#include <stddef.h>",
        "",
        "#define CTRL_PI 3.14159265358979323846264338327950288",
        "#define CTRL_E  2.71828182845904523536028747135266250",
        "",
        "static inline double ctrl_clip(double value, double lower, double upper) {",
        "    return fmin(fmax(value, lower), upper);",
        "}",
        "",
        "static inline double ctrl_sign(double value) {",
        "    return (value > 0.0) - (value < 0.0);",
        "}",
        "",
        f"void {name}_init({name}_state *state) {{",
        "    if (state == NULL) return;",
        f"    state->mode = UINT32_C({mode_index[spec.initial_mode]});",
        "    state->time = 0.0;",
        "    state->time_in_mode = 0.0;",
    ]
    for item in spec.registers:
        value = "true" if item.initial is True else "false" if item.initial is False else _number(item.initial)
        lines.append(f"    state->register_{item.name} = {value};")
    for item in spec.outputs:
        value = "true" if item.default is True else "false" if item.default is False else _number(item.default)
        lines.append(f"    state->output_{item.name} = {value};")
    for item in spec.implicit_inputs:
        assert ir.observer is not None
        index = ir.observer.latent_names.index(item.name)
        output_defaults = {output.name: float(output.default) for output in spec.outputs}
        initial_input = np.asarray(
            [output_defaults[name] for name in ir.observer.input_names], dtype=float
        )
        initial_mean = (
            ir.observer.L[index] @ ir.observer.initial_state
            + ir.observer.M[index] @ initial_input
            + ir.observer.latent_bias[index]
        )
        initial_variance = (
            ir.observer.L[index]
            @ ir.observer.initial_covariance
            @ ir.observer.L[index]
        )
        initial_mean = max(
            initial_mean,
            -math.inf if item.bounds.lower is None else item.bounds.lower,
        )
        initial_mean = min(
            initial_mean,
            math.inf if item.bounds.upper is None else item.bounds.upper,
        )
        lines.append(f"    state->implicit_{item.name}_mean = {_number(initial_mean)};")
        lines.append(f"    state->implicit_{item.name}_variance = {_number(initial_variance)};")
    if ir.observer is not None:
        for index, value in enumerate(ir.observer.initial_state):
            lines.append(f"    state->observer_state[{index}] = {_number(value)};")
        for index, value in enumerate(ir.observer.initial_covariance.reshape(-1)):
            lines.append(f"    state->observer_covariance[{index}] = {_number(value)};")
    lines.extend(
        [
            "}",
            "",
            f"static void {name}_write_outputs(const {name}_state *state, {name}_outputs *output) {{",
            "    if (state == NULL || output == NULL) return;",
        ]
    )
    for item in spec.outputs:
        lines.append(
            f"    output->{target_identifier(item.name)} = state->output_{item.name};"
        )
    lines.extend(
        [
            "}",
            "",
            f"int {name}_step({name}_state *state, const {name}_explicit_inputs *explicit_input, {name}_outputs *output) {{",
            f"    const double dt = {_number(spec.period_s)};",
            "    if (state == NULL || explicit_input == NULL || output == NULL) {",
            f"        {name}_write_outputs(state, output);",
            f"        return {name.upper()}_STATUS_NULL_POINTER;",
            "    }",
            f"    const {name}_state original_state = *state;",
            "",
        ]
    )
    for item in spec.explicit_inputs:
        if item.dtype == "real":
            field = f"explicit_input->{target_identifier(item.name)}"
            condition = f"!isfinite({field})"
            if item.bounds.lower is not None:
                condition += f" || {field} < {_number(item.bounds.lower)}"
            if item.bounds.upper is not None:
                condition += f" || {field} > {_number(item.bounds.upper)}"
            lines.extend(
                [
                    f"    if ({condition}) {{",
                    f"        {name}_write_outputs(state, output);",
                    f"        return {name.upper()}_STATUS_INVALID_INPUT;",
                    "    }",
                ]
            )
    if ir.observer is not None:
        lines.extend(
            [
                f"    if (!{name}_observer_step(state, explicit_input)) {{",
                "        *state = original_state;",
                f"        {name}_write_outputs(state, output);",
                f"        return {name.upper()}_STATUS_OBSERVER_FAILURE;",
                "    }",
            ]
        )
        lines.append("")

    for item in spec.derived:
        lines.append(
            f"    const {_ctype(item.dtype)} derived_{item.name} = {render(item.expression)};"
        )
        lines.append(f"    (void)derived_{item.name};")
        if item.dtype == "real":
            lines.extend(
                [
                    f"    if (!isfinite(derived_{item.name})) {{",
                    "        *state = original_state;",
                    f"        {name}_write_outputs(state, output);",
                    f"        return {name.upper()}_STATUS_ARITHMETIC_FAILURE;",
                    "    }",
                ]
            )
    emergency = "false" if spec.emergency_when is None else render(spec.emergency_when)
    lines.extend(
        [
            f"    const bool emergency = {emergency};",
            "    (void)emergency;",
            "    const uint32_t active_mode = state->mode;",
            "    uint32_t next_mode = active_mode;",
        ]
    )
    for item in spec.outputs:
        lines.append(f"    {_ctype(item.dtype)} next_output_{item.name} = state->output_{item.name};")
    for item in spec.registers:
        lines.append(f"    {_ctype(item.dtype)} next_register_{item.name} = state->register_{item.name};")
    lines.extend(["", "    switch (active_mode) {"])
    for mode in spec.modes:
        lines.append(f"    case UINT32_C({mode_index[mode.name]}): /* {mode.name} */")
        for output_name, expression in mode.outputs.items():
            lines.append(f"        next_output_{output_name} = {render(expression)};")
        for register_name, expression in mode.updates.items():
            lines.append(f"        next_register_{register_name} = {render(expression)};")
        transitions = sorted(mode.transitions, key=lambda item: item.priority, reverse=True)
        for index, transition in enumerate(transitions):
            keyword = "if" if index == 0 else "else if"
            lines.append(
                f"        {keyword} ({render(transition.guard)}) next_mode = UINT32_C({mode_index[transition.target]});"
            )
        lines.extend(["        break;", ""])
    lines.extend(["    default:", f"        next_mode = UINT32_C({mode_index[spec.initial_mode]});", "        break;", "    }"])

    for item in spec.registers:
        if item.dtype == "real":
            lines.extend(
                [
                    f"    if (!isfinite(next_register_{item.name})) {{",
                    "        *state = original_state;",
                    f"        {name}_write_outputs(state, output);",
                    f"        return {name.upper()}_STATUS_ARITHMETIC_FAILURE;",
                    "    }",
                ]
            )

    for item in spec.outputs:
        if item.emergency_value is not None:
            value = "true" if item.emergency_value is True else "false" if item.emergency_value is False else _number(item.emergency_value)
            lines.append(f"    if (emergency) next_output_{item.name} = {value};")
        if item.dtype == "real":
            lines.append(
                f"    next_output_{item.name} = {_bounded(f'next_output_{item.name}', item.bounds.lower, item.bounds.upper)};"
            )
            if item.slew_rate is not None:
                delta = _number(item.slew_rate * spec.period_s)
                lines.extend(
                    [
                        f"    if (!emergency && !isfinite(state->output_{item.name})) {{",
                        "        *state = original_state;",
                        f"        {name}_write_outputs(state, output);",
                        f"        return {name.upper()}_STATUS_ARITHMETIC_FAILURE;",
                        "    }",
                        f"    if (!emergency) next_output_{item.name} = ctrl_clip(next_output_{item.name}, state->output_{item.name} - {delta}, state->output_{item.name} + {delta});",
                    ]
                )
    for item in spec.registers:
        if item.dtype == "real":
            lines.append(
                f"    next_register_{item.name} = {_bounded(f'next_register_{item.name}', item.bounds.lower, item.bounds.upper)};"
            )
    for item in spec.outputs:
        if item.dtype == "real":
            lines.extend(
                [
                    f"    if (!isfinite(next_output_{item.name})) {{",
                    "        *state = original_state;",
                    f"        {name}_write_outputs(state, output);",
                    f"        return {name.upper()}_STATUS_ARITHMETIC_FAILURE;",
                    "    }",
                ]
            )
    for item in spec.registers:
        if item.dtype == "real":
            lines.extend(
                [
                    f"    if (!isfinite(next_register_{item.name})) {{",
                    "        *state = original_state;",
                    f"        {name}_write_outputs(state, output);",
                    f"        return {name.upper()}_STATUS_ARITHMETIC_FAILURE;",
                    "    }",
                ]
            )
    lines.extend(
        [
            "    if (!isfinite(state->time + dt) || !isfinite(state->time_in_mode + dt)) {",
            "        *state = original_state;",
            f"        {name}_write_outputs(state, output);",
            f"        return {name.upper()}_STATUS_ARITHMETIC_FAILURE;",
            "    }",
        ]
    )
    for item in spec.outputs:
        lines.extend(
            [
                f"    state->output_{item.name} = next_output_{item.name};",
                f"    output->{target_identifier(item.name)} = next_output_{item.name};",
            ]
        )
    for item in spec.registers:
        lines.append(f"    state->register_{item.name} = next_register_{item.name};")
    lines.extend(
        [
            "    state->mode = next_mode;",
            "    state->time += dt;",
            "    state->time_in_mode = (next_mode == active_mode) ? state->time_in_mode + dt : 0.0;",
            f"    return {name.upper()}_STATUS_OK;",
            "}",
        ]
    )
    observer_lines = _observer_source(ir)
    if observer_lines:
        include_end = lines.index("#define CTRL_PI 3.14159265358979323846264338327950288")
        lines[include_end:include_end] = observer_lines
    return "\n".join(lines) + "\n"


def generate_c99(
    value: ControlSpec | ControlIR, *, identifier: str | None = None
) -> tuple[GeneratedArtifact, GeneratedArtifact]:
    """Generate a complete C99 header/source pair for one controller."""

    ir = as_ir(value, identifier=identifier)
    _admit_c_symbols(ir)
    return (
        GeneratedArtifact(f"{ir.identifier}.h", "text/x-c", _header(ir)),
        GeneratedArtifact(f"{ir.identifier}.c", "text/x-c", _source(ir)),
    )


__all__ = ["generate_c99"]
