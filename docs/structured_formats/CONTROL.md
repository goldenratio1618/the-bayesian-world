# Control DSL (`control-1`)

A `.control` file is a strict, data-only, periodic finite-state controller. It
uses PMDL's safe scalar expression parser, dimensional type system, and
backend-neutral graph. It is never evaluated as Python. The authoritative
schema and validation are in `contraption.control.specs`.

## Top-level fields

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `format` | string | yes | exactly `control-1` |
| `id` | identifier | yes | stable controller id |
| `name` | non-empty string | yes | display name |
| `version` | non-empty string | yes | controller version |
| `period_s` | positive finite number | yes | update period in seconds |
| `explicit_inputs` | array | yes | external or physically wired scalar inputs |
| `outputs` | array | yes | scalar hardware/telemetry outputs |
| `modes` | array | yes | finite-state output/update laws |
| `initial_mode` | symbol | yes | name of a declared mode |
| `parameters` | array | no | immutable tunable constants |
| `registers` | array | no | discrete controller memory |
| `implicit_inputs` | array | no | latent scalar plant quantities |
| `observer` | object or null | conditional | required exactly when implicit inputs exist |
| `derived` | array | no | ordered typed intermediate expressions |
| `emergency_when` | Boolean expression or null | no | dominant emergency override |
| `metadata` | object | no | inert JSON |

Ids match `^[A-Za-z][A-Za-z0-9_.-]{0,127}$`. Symbols match
`^[A-Za-z][A-Za-z0-9_]{0,63}$`.

## Scalar declarations

All numeric records use finite numbers and parseable units. `dtype` is `real`
(default) or `bool`. Boolean values require unit `1` and may not have numeric
bounds.

### Explicit input

| Field | Required | Semantics |
|---|---:|---|
| `name`, `source` | yes | `source` is `external` or `sensor` |
| `dtype`, `unit` | no | default `real` and `1` |
| `default` | no | typed startup value |
| `bounds` | no | `{"lower": number-or-null, "upper": number-or-null}` |
| `measurement_variance` | conditional | positive finite variance for real sensor input; forbidden on external or Boolean input |
| `description` | no | explanatory text |

Every real sensor input needs `measurement_variance` when an observer is present.

### Output

Fields are `name`, optional `dtype`, `unit`, `default`, `bounds`,
`slew_rate`, `emergency_value`, and `description`. A real `slew_rate` is
positive and expressed as output units per second. Boolean outputs forbid it.
Defaults and emergency values must lie within bounds. If `emergency_when` is
declared, every output requires `emergency_value`.

### Parameter and register

A parameter has `name`, optional `dtype`, `unit`, `default`, and `bounds`. A
register has the same shape except `initial` replaces `default`. Registers are
updated simultaneously at a controller tick; expressions read the pre-update
`register.<name>` values.

### Derived value

A derived record has required `name` and `expression` plus optional `dtype` and
`unit`. Derived records are ordered: an expression may use only symbols already
declared, including earlier `derived.<name>` values.

## Implicit inputs and observer

An implicit input is a scalar plant quantity inferred by the resolved
plant-derived observer:

| Field | Required | Rule |
|---|---:|---|
| `name`, `unit` | yes | scalar real state |
| `initial_variance` | no | finite and nonnegative; default 1 |
| `process_variance_per_s` | no | finite and nonnegative; default 0 |
| `bounds` | no | admissible mean interval |

Expressions receive `implicit.<name>.mean`, `implicit.<name>.variance`, and
`implicit.<name>.std`.

The only admitted observer currently has:

| Field | Required/default | Rule |
|---|---|---|
| `kind` | required | exactly `local_affine` |
| `nonlinear_approximation` | required | exactly `approved` |
| `acknowledged_open_gates` | required | unique contraption completeness-gate ids |
| `sample_radius_relative` | required | positive finite |
| `maximum_sampled_remainder` | required | positive finite |
| `relative_step` | `1e-6` | positive finite |
| `newton_tolerance` | `1e-10` | positive finite |
| `newton_max_iterations` | `20` | positive integer |
| `maximum_condition_number` | `1e12` | finite and greater than 1 |

This observer declaration is an explicit approximation admission, not evidence
that every acknowledged physical gate is closed.

## Modes and transitions

A mode has:

- required `name`;
- required `outputs` mapping every output name to one typed expression;
- optional `updates` mapping register names to typed expressions;
- optional `transitions`.

A transition has `target`, Boolean `guard`, and optional integer `priority`
(default 0). Priorities must be unique within a mode. The target must exist.
Higher-priority transition selection is implemented deterministically by the
runtime. Every mode must cover every output exactly; missing or unknown outputs
are invalid.

## Expression environment

Available names are:

- `time`, `time_in_mode`, and `dt` in seconds;
- `input.<name>`, `output.<name>`, `parameter.<name>`, `register.<name>`;
- `implicit.<name>.mean|variance|std`;
- `derived.<name>`.

Allowed arithmetic is `+ - * / **`, comparisons, `and`, `or`, `not`, and a
conditional expression. Numeric calls with common backend lowering are `abs`,
`sqrt`, trigonometric/inverse-trigonometric functions, `tanh`, `exp`, `log`,
`log10`, `min`, `max`, `clip`, `where`, and `smooth_abs`. `der` is PMDL-only;
`sign` is rejected from a numeric controller path. Exponents must be integer
literals from 0 through 8. Units and Boolean/real types are checked statically.
Domain checks reject expressions that can divide by zero or violate functions'
valid domains over declared bounds.

## Compact example

~~~json
{
  "format": "control-1",
  "id": "camera_gate",
  "name": "Camera exposure gate",
  "version": "1.0.0",
  "period_s": 0.01,
  "explicit_inputs": [
    {"name": "armed", "source": "external", "dtype": "bool", "unit": "1", "default": false},
    {"name": "coverage", "source": "sensor", "unit": "1", "default": 0.0,
     "bounds": {"lower": 0.0, "upper": 1.0}}
  ],
  "outputs": [
    {"name": "expose", "dtype": "bool", "unit": "1", "default": false,
     "emergency_value": false}
  ],
  "modes": [
    {
      "name": "idle",
      "outputs": {"expose": "False"},
      "transitions": [{"target": "scan", "guard": "input.armed", "priority": 10}]
    },
    {
      "name": "scan",
      "outputs": {"expose": "input.coverage < 0.95"},
      "transitions": [{"target": "idle", "guard": "not input.armed", "priority": 10}]
    }
  ],
  "initial_mode": "idle",
  "emergency_when": "not input.armed",
  "metadata": {}
}
~~~

## Optical and controller/FPGA boundary

`control-1` is deliberately scalar and bounded. A camera image or
`contraption/optical-observation@1` artifact is not a legal scalar control input.
An optical service may consume artifact streams and expose bounded, unit-typed
features such as confidence, centroid, range, coverage, or a next-view target
through ordinary signal ports. This keeps the controller graph compilable to
C99 and Verilog without making high-bandwidth host/GPU processing part of the
portable controller semantics.
