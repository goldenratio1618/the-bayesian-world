# Physical Model DSL (`pmdl-1`)

PMDL is Contraption's strict, acausal, data-only physical-model language. A
concrete model expresses the descriptor residual system
`F(t, z, zdot, theta, u) = 0`. Assembly assigns computational causality; a
component never encodes an execution order or calls host code. The authoritative
records, parser, dimensional validator, and assembly are in
`contraption.physics.specs`, `contraption.physics.dsl`, and
`contraption.physics.validation`.

## Top-level model

The required fields are `format`, `id`, `name`, `version`, `domains`, and
`implements`. All others are optional with empty/default records. Unknown
fields and duplicate JSON keys are invalid.

| Field | Type | Meaning |
|---|---|---|
| `format` | string | exactly `pmdl-1` |
| `id` | identifier | globally unique concrete-model identity |
| `name` | nonempty string | display name |
| `version` | nonempty string | model version |
| `domains` | nonempty identifier array | every physical domain implemented by the component |
| `implements` | identifier | category/device `pmdl-interface-1` contract |
| `description` | string | scope and physical interpretation |
| `power_ports` | array | acausal effort/flow port declarations |
| `signal_ports` | array | directed scalar/tensor signal declarations |
| `artifact_ports` | array | typed, directed non-scalar artifact streams; see below |
| `states` | array | differential state variables |
| `algebraics` | array | algebraic unknowns |
| `parameters` | array | bounded uncertain constants |
| `relations` | array | residual equations interpreted as equal to zero |
| `stored_energy` | array | declared energy functions |
| `dissipation` | array | declared dissipated-power functions |
| `sources` | array | declared injected-power/source functions |
| `process_noise` | object | accepted-step stochastic increments |
| `modes` | array | explicit discrete relation sets and transitions |
| `initialization` | object | consistent-initialization constraints |
| `validity` | object | operating envelope and numerical step limit |
| `fidelity_levels` | array | named relation/parameter approximations |
| `properties` | array | sampled machine-checkable Boolean properties |
| `trust` | object | separate structural/physical/numerical/empirical status |
| `metadata` | object | inert JSON only |

PMDL ids use `^[A-Za-z][A-Za-z0-9_.-]*$`. Mathematical symbols use
`^[A-Za-z][A-Za-z0-9_]*$`. A model file lives at the category or device layer
whose interface it implements. A model's canonical JSON, id, version, and exact
SHA-256 are bound by every `vN.model` that uses it.

## Variables

### State

| Field | Required/default | Semantics |
|---|---|---|
| `name` | required | scalar differential-state symbol |
| `unit` | `1` | physical unit |
| `initial` | `0` | finite nominal initial value |
| `derivative` | absent | explicit derivative symbol; otherwise `<name>_dot` |
| `description` | empty | physical meaning and reference convention |

Every state derivative must occur in an active residual. `der(state_name)` is
equivalent to the declared/default derivative symbol.

Generated part READMEs assign short, deterministic display symbols by declaration
role (for example, `x_1` for a state and `theta_1` for a parameter) and include
a symbol key that expands each alias back to its exact PMDL identifier. Display
aliases are unique within one PMDL rendering and are presentation only: they are
not PMDL fields, do not change model hashes, and never replace the exact DSL
source shown with each equation. Authored identifiers containing underscores
remain valid and are escaped as single symbols when rendered directly. Greek
keywords such as `alpha` and `gamma_i` render as the corresponding Greek
symbol and subscript.

### Algebraic

An algebraic has `name` plus optional `unit` (`1`), finite `initial` (`0`), and
`description`. It is an unknown solved as part of the descriptor system, not a
procedurally computed temporary.

### Parameter

| Field | Required/default | Rule |
|---|---|---|
| `name` | required | scalar symbol |
| `unit` | `1` | physical unit |
| `default` | `0` | finite and inside bounds |
| `bounds` | unbounded | object with only `lower` and `upper`, each number or null |
| `uncertainty` | fixed | uncertainty record below |
| `learnable` | true | may be inferred/calibrated |
| `description` | empty | physical interpretation |

Uncertainty has `distribution` (`fixed`, `normal`, `lognormal`, `uniform`,
`triangular`, or `empirical`), `parameters`, and optional
`correlation_group`. Normal/lognormal validation requires positive `std` (or
`sigma`); uniform requires numeric `lower < upper`. Distribution parameters do
not override nominal bounds.

## Ports

### Power port

A power port declares:

| Field | Required/default | Meaning |
|---|---|---|
| `name` | required | connector-facing port identity |
| `domain` | required | e.g. `electrical` or `mechanical` |
| `effort`, `flow` | required | scalar symbols used in residuals |
| `effort_unit`, `flow_unit` | required | dimensions whose product is watts |
| `orientation` | `into_component` | `into_component`, `out_of_component`, or `bidirectional` |
| `frame` | `body` | reference frame label |
| `reference` | `declared` | potential/velocity/reference convention |
| `description` | empty | explanatory text |

Power connections enforce common effort and signed flow balance using orientation.
Typical pairs are V/A and N?m/rad/s. Authors must state reference and frame; a
port name alone does not define sign.

### Scalar/tensor signal port

A signal port has required `name` and `direction` (`input` or `output`), optional
`unit` (`1`), `dtype` (`float32`, `float64`, `int32`, or `bool`), positive
integer `shape` dimensions (default scalar), and `description`. Signal ports are
directed and carry no power.

### Artifact stream port

`artifact_ports` declares typed, high-bandwidth or structured values that must
not be forced through scalar PMDL equations.

| Field | Required/default | Validation and meaning |
|---|---|---|
| `name` | required | unique PMDL symbol |
| `direction` | required | `input` or `output` |
| `artifact_type` | required | `namespace/name@major`; lowercase components and positive major version |
| `timing` | `event` | `event` or `sampled` |
| `transport` | `content_addressed` | `in_process`, `content_addressed`, `shared_memory`, `network`, or `controller_stream` |
| `sample_period_s` | absent | positive seconds; required when timing is `sampled` |
| `max_payload_bytes` | absent | positive integer; required for `controller_stream` |
| `description` | empty | stream semantics, producer/consumer, latency, and capability notes |

For example, a content-addressed event stream from an offline sensor is:

~~~json
{
  "name": "frame",
  "direction": "output",
  "artifact_type": "contraption/optical-observation@1",
  "timing": "event",
  "transport": "content_addressed",
  "description": "Hash-bound frame manifest and sidecars"
}
~~~

A bounded future controller link may instead use `timing: sampled`,
`transport: controller_stream`, a positive `sample_period_s`, and
`max_payload_bytes`. These fields specify the contract only; they do not
implement the skipped FPGA path or authorize unbounded data in `control-1`.

Use artifact ports for images, depth maps, point clouds, shape/reconstruction
states, and observation manifests. Use scalar signal ports for exposure
commands and bounded derived features such as range, confidence, centroid,
coverage, or next-view coordinates. Artifact ports are endpoint declarations,
not equation symbols: they cannot occur in PMDL residuals, energy expressions,
or the current scalar controller compiler. Unknown fields are invalid; do not
hide an artifact URI in `metadata` or encode a frame as a signal float array.

## Relations and safe expressions

A relation has required `name` and `expression` plus optional `description`.
The expression is a residual interpreted as `expression = 0`; do not append
`== 0`. All relations are simultaneous.

Available symbols are `t`, `pi`, `e`, states, derivative symbols, algebraics,
parameters, power effort/flow symbols, and scalar signal-port names. Allowed
syntax is:

- finite numeric and Boolean literals;
- dotted safe symbols where a consuming DSL declares them;
- unary `+`, `-`, and `not`;
- `+ - * / **`, `and`, `or`;
- `< <= > >= == !=`;
- conditional `true_value if condition else false_value`;
- `abs`, `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
  `exp`, `log`, `log10`, `tanh`, `min`, `max`, `clip`, `sign`, `where`,
  `smooth_abs`, and `der`.

Expressions are limited to 16,384 characters, 256 syntax nodes, and depth 32.
Calls are direct, positional, and allow-listed. No assignment, indexing,
comprehension, loop, import, object creation, reflection, file/network access,
plugin, callback, eval, or executable host expression is allowed.

Dimensional rules are enforced: addition/comparison branches must be compatible;
multiplication/division combine dimensions; exponents are literal and
dimensionless; transcendental inputs are dimensionless; `atan2` operands match;
`sqrt` halves dimensions; `where` branches match. Relations must be real.
Nonsmooth calls and conditionals in residuals are rejected by model validation;
use `smooth_abs` or explicit discrete modes.

## Energy, dissipation, and sources

`stored_energy`, `dissipation`, and `sources` use the same record:

| Field | Required/default | Meaning |
|---|---|---|
| `name` | required | unique symbol |
| `expression` | required | real expression |
| `unit` | required | declared expression unit |
| `nonnegative` | false | nonnegativity assertion |
| `description` | empty | interpretation |

Stored energy normally uses J; dissipation and source power normally use W.
Declared units must match inferred dimensions. Every dissipation entry must set
`nonnegative: true` and the model must contain a
`kind: nonnegative_dissipation` property. Stored energy nonnegativity should
likewise be backed by a property.

Power ports, stored energy, dissipation, and sources form the explicit energy
accounting boundary. Optical rendering appearance is not an energy source:
illumination and emission must be declared in the optical scene/material
contract, while electrical draw and dynamic scalar behavior remain in PMDL.

## Process noise

Omit `process_noise` (or use an empty object) for no process noise. A nonempty
record requires every field:

~~~json
{
  "seed_policy": "simulation_seed",
  "reproducibility": "same_backend_device",
  "application": "accepted_step_increment",
  "channels": [
    {"name": "roughness", "distribution": "standard_normal"}
  ],
  "increments": [
    {
      "target": "position",
      "expression": "diffusion * sqrt(dt) * roughness"
    }
  ]
}
~~~

Each channel has `name`, exactly `distribution: standard_normal`, and optional
`description`. Each increment has state `target`, `expression`, and optional
`description`. Channels are dimensionless, unique, noncolliding stochastic
symbols. `dt` is the actual accepted time step in seconds. An increment must:

- target one differential state, at most once;
- explicitly use `dt`;
- use at least one declared channel, with every channel used;
- have the same dimension as the target state.

The simulator derives one stream from `simulate(seed=...)`. Replay is guaranteed
only for the same exact model closure, time grid, sample count, numerical
backend, device, and dtype. Increments are applied after accepted deterministic
steps and outputs are reconciled afterward.

## Discrete modes

A mode has `name`, `active_relations`, `transitions`, and `initial`. When any
modes exist, exactly one is initial. Every active relation name must exist.

A transition has required target mode `target` and Boolean `guard`, plus
`resets` mapping state names to real expressions. Reset targets must be states.
Put discontinuous switching in guards/resets rather than in residual
conditionals. Mode equations still need a balanced descriptor system.

## Initialization, validity, and fidelity

`initialization` has:

- `strategy`, default `consistent`;
- `constraints`, relation-shaped residuals;
- `required`, names drawn from states, algebraics, and parameters.

Initial values are guesses until consistency has been solved; they are not a
license to begin with violated algebraic constraints.

`validity` has `ranges` mapping known scalar symbols to bounds,
`assumptions`, and optional positive `max_timestep`. The envelope states where
the equations, fitted parameters, and numerical integration are supported.
Unknown range symbols are invalid. Empty validity is warned.

A fidelity record has `name`, `description`, `active_relations`,
`parameter_overrides`, `approximation_error` (default `unspecified`), and
positive `relative_cost` (default 1). Relations and parameters must exist.
A fidelity is an explicit approximation, not a way to silently remove difficult
physics.

## Properties and trust

A property has `name`, descriptive `kind`, Boolean `expression`, `expected`
(default true), positive integer `samples` (default 32), nonnegative `tolerance`
(default `1e-9`), and `description`. Property expressions use model symbols and
are type-checked. Common kinds include `parameter_bounds`,
`nonnegative_energy`, and `nonnegative_dissipation`; `kind` is a validation/test
classification, not executable dispatch authored by the model.

`trust` separately records `structural`, `physical`, `numerical`, and
`empirical` levels. Each is `unverified`, `reviewed`, `tested`, `validated`, or
`certified`. Evidence entries have required `kind` and `reference`, optional
`summary` and `date`. Do not raise a trust level solely because a simulation ran.

## Acausal electrical example

~~~json
{
  "format": "pmdl-1",
  "id": "electrical.resistor.ideal",
  "name": "Ideal resistor",
  "version": "1.0.0",
  "domains": ["electrical"],
  "implements": "resistor",
  "power_ports": [
    {
      "name": "p", "domain": "electrical", "effort": "v_p", "flow": "i_p",
      "effort_unit": "V", "flow_unit": "A", "orientation": "into_component",
      "frame": "electrical", "reference": "v_p relative to circuit reference"
    },
    {
      "name": "n", "domain": "electrical", "effort": "v_n", "flow": "i_n",
      "effort_unit": "V", "flow_unit": "A", "orientation": "into_component",
      "frame": "electrical", "reference": "v_n relative to circuit reference"
    }
  ],
  "parameters": [
    {
      "name": "resistance", "unit": "ohm", "default": 100.0,
      "bounds": {"lower": 1e-9, "upper": 1e12},
      "uncertainty": {"distribution": "lognormal", "parameters": {"std": 0.1}},
      "learnable": true
    }
  ],
  "relations": [
    {"name": "ohms_law", "expression": "v_p - v_n - resistance * i_p"},
    {"name": "current_conservation", "expression": "i_p + i_n"}
  ],
  "dissipation": [
    {
      "name": "joule_heating", "expression": "resistance * i_p ** 2",
      "unit": "W", "nonnegative": true
    }
  ],
  "properties": [
    {
      "name": "passive", "kind": "nonnegative_dissipation",
      "expression": "resistance * i_p ** 2 >= 0 * resistance * i_p ** 2"
    }
  ],
  "validity": {"ranges": {}, "assumptions": ["Lumped, temperature-invariant element"]},
  "trust": {"structural": "tested", "physical": "reviewed",
            "numerical": "tested", "empirical": "unverified", "evidence": []}
}
~~~

## Optical authoring guidance

PMDL owns scalar coupled behavior: electrical supply draw, exposure/focus state,
thermal drift, shutter timing, calibration parameters, sensor noise parameters,
and bounded command/measurement signals. The optical schemas own rays, spectral
materials, camera geometry, render settings, observations, and reconstruction
artifacts. A typical camera model therefore combines:

- an electrical power port and explicit power/current relation;
- scalar exposure, gain, focus, trigger, status, or timestamp signals as needed;
- learnable bounded calibration/noise parameters with uncertainty and validity;
- an output artifact port typed `contraption/optical-observation@1`;
- modes for discrete shutter/readout behavior if the scalar dynamics require it.

Differentiable inverse solving requires learnable parameters, smooth PMDL
relations within modes, differentiable optical materials/sensors, exact
observation provenance, and an optimizer-facing residual/likelihood outside
the PMDL scalar equation set. Never encode a renderer or neural network as a
PMDL expression or metadata value.

Luna may author these PMDL abstractions. It may not ingest or generate the
geometry, material, calibration, scene, observation, or reconstruction payloads
the artifact ports carry; deterministic host code creates and hash-binds them.
