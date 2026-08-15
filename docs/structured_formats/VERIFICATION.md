# Verification DSL (`verification-1`)

A `.verify` file defines posterior trajectory metrics and probabilistic
acceptance criteria. It is strict JSON and contains only safe scalar
expressions. The authoritative parser is `contraption.verification.specs` and
runtime evaluation is in `contraption.verification.runtime`.

## Top-level fields

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `format` | string | yes | exactly `verification-1` |
| `id` | identifier | yes | `^[A-Za-z][A-Za-z0-9_.-]{0,127}$` |
| `name` | non-empty string | yes | display name |
| `version` | non-empty string | yes | program version |
| `description` | string | no | explanatory text |
| `inputs` | array | yes, nonempty | scalar real trajectories |
| `parameters` | array | no | exact finite constants |
| `metrics` | array | yes, nonempty | per-sample trajectory reductions |
| `criteria` | array | yes, nonempty | posterior acceptance decisions |

Unknown fields, duplicate JSON keys, NaN, and infinity are invalid.

## Inputs and parameters

An input has required `name` and `unit` plus optional `description`. It denotes
one scalar real trajectory supplied by the resolved contraption binding.

A parameter has required `name`, `unit`, and finite numeric `value` plus optional
`description`. Parameters are hash-bound constants of the verification program.

Input, parameter, and metric symbols follow
`^[A-Za-z][A-Za-z0-9_]{0,63}$`, may not be `pi` or `e`, are unique within each
group, and may not collide across those three groups.

## Metrics

A metric has exactly:

| Field | Type | Meaning |
|---|---|---|
| `name` | symbol | criterion-visible metric identity |
| `expression` | real expression | evaluated elementwise over input trajectories |
| `reducer` | enum | `initial`, `final`, `mean`, `min`, `max`, or `rmse` |
| `unit` | string | required output unit/dimension |
| `description` | string, optional | explanatory text |

The expression may use verification inputs and parameters, but not another
metric. Its inferred dimension must equal `unit`. Each posterior simulation
sample produces one scalar metric after reduction. `rmse` computes the root
mean square of the expression itself, so author an error expression such as
`radius - target_radius`.

Real metric expressions are classified as
`smooth_on_valid_domain` or `piecewise_smooth_on_valid_domain`. Comparisons and
other discrete operations are forbidden in a numeric metric path.

## Criteria

A criterion has:

| Field | Type | Rule |
|---|---|---|
| `name` | symbol | unique |
| `expression` | Boolean expression | may use parameters and reduced metrics |
| `minimum_probability` | number | closed interval [0, 1] |
| `confidence_level` | number | open interval (0.5, 1) |
| `description` | string, optional | explanatory text |

The runtime computes passes across posterior samples and compares
`minimum_probability` with a conservative one-sided finite-sample lower
confidence bound, not with the raw pass fraction. Criteria and final admission
are intentionally discrete even when upstream metrics are differentiable.

## Expression language

Verification reuses PMDL arithmetic, comparisons, Boolean logic, conditional
expressions, `pi`/`e`, and backend-native numeric functions:

`abs`, `sqrt`, `sin`, `cos`, `tan`, `tanh`, `asin`, `acos`, `atan`, `atan2`,
`exp`, `log`, `log10`, `min`, `max`, `clip`, `sign`, `where`, and
`smooth_abs`.

Units are checked statically. `der()` is forbidden; expose a derivative as an
explicit input trajectory. No file access, arbitrary attribute access, calls
outside the allow-list, loops, imports, or host code are possible.

## Example

~~~json
{
  "format": "verification-1",
  "id": "scanner.optical_coverage",
  "name": "Scanner optical coverage acceptance",
  "version": "1.0.0",
  "description": "Posterior geometric coverage and reprojection acceptance.",
  "inputs": [
    {"name": "surface_coverage", "unit": "1"},
    {"name": "reprojection_error", "unit": "m"}
  ],
  "parameters": [
    {"name": "minimum_coverage", "unit": "1", "value": 0.95},
    {"name": "maximum_rmse", "unit": "m", "value": 0.002}
  ],
  "metrics": [
    {
      "name": "final_coverage",
      "expression": "surface_coverage",
      "reducer": "final",
      "unit": "1"
    },
    {
      "name": "reprojection_rmse",
      "expression": "reprojection_error",
      "reducer": "rmse",
      "unit": "m"
    }
  ],
  "criteria": [
    {
      "name": "coverage",
      "expression": "final_coverage >= minimum_coverage",
      "minimum_probability": 0.95,
      "confidence_level": 0.95
    },
    {
      "name": "fit",
      "expression": "reprojection_rmse <= maximum_rmse",
      "minimum_probability": 0.95,
      "confidence_level": 0.95
    }
  ]
}
~~~

## Artifact boundary

The language accepts scalar trajectories only. Optical images, depth buffers,
rays, and reconstruction states remain hash-bound artifacts. The optical or
reconstruction runtime must derive explicit scalar diagnostics before binding
them as verification inputs. This keeps statistical admission auditable and
prevents an opaque neural or rendering operation from being hidden in a
criterion expression.
