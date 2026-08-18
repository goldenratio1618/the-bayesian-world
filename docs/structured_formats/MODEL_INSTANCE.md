# Model instances (`model-instance-1`)

A `vN.model` file binds one static catalog part to one exact concrete PMDL
implementation and supplies every parameter value. The authoritative parser and
catalog validation live in `contraption.catalog.instantiations`.

## Placement and identity

The file must be immediately below:

`model_catalog/<domain>/<category>[/<device>]/instantiations/<part-id>/vN.model`

The filename must match `^v[1-9][0-9]*\.model$`. Its `variant` equals the filename
stem, and `id` must equal `<part>.<variant>`. The adjacent `static.part` must have
the same `id` as `part`.

## Top-level fields

All fields except `metadata` are required. Unknown fields are invalid.

| Field | Type | Meaning |
|---|---|---|
| `format` | string | exactly `model-instance-1` |
| `id` | identifier | exactly `<part>.<variant>` |
| `variant` | identifier | exactly the `vN` filename stem |
| `part` | identifier | adjacent `static.part.id` |
| `version` | non-empty string | version of this instantiated hypothesis |
| `model` | model reference | exact PMDL identity/version/canonical hash |
| `parameters` | object | every and only PMDL parameter, initialized |
| `parameter_uncertainty` | object | optional uncertainty records keyed by PMDL parameter; the field itself is required |
| `condition` | enum | `unverified`, `inspected`, `calibrated`, `degraded`, or `retired` |
| `compute` | compute-cost object | model evaluation cost declaration |
| `metadata` | object, optional | inert JSON only |

A model reference has exactly:

| Field | Type | Rule |
|---|---|---|
| `id` | identifier | concrete `pmdl-1.id` |
| `version` | string | exact `pmdl-1.version` |
| `sha256` | string | `sha256:` plus 64 lowercase hex digits over canonical PMDL JSON |

The loader recomputes the canonical PMDL digest and rejects a mismatch. A
semantic version match does not substitute for a byte/canonical-hash match.

`compute` has required positive finite `relative_cost` and optional non-empty
`notes`. Cost is relative scheduling/fidelity information, not a time unit.

## Parameters and uncertainty

`parameters` must initialize every PMDL parameter exactly once. Missing,
unknown, nonnumeric, nonfinite, or out-of-bounds values are rejected.

A scalar parameter is normally a JSON number. The registry also accepts a
mapping with a numeric `value` when a richer initialized parameter record is
needed. `parameter_uncertainty` may name only declared PMDL parameters. Its
values use the PMDL uncertainty object:

| Field | Type | Values |
|---|---|---|
| `distribution` | string | `fixed`, `normal`, `lognormal`, `uniform`, `triangular`, or `empirical` |
| `parameters` | object | distribution parameters such as `std`; interpretation belongs to the UQ layer |
| `correlation_group` | identifier or null | shared-dependence group |

The scalar parameter value remains authoritative for nominal simulation.
Uncertainty does not relax PMDL bounds or validity ranges.

## Example

~~~json
{
  "format": "model-instance-1",
  "id": "C1210C476K8RAC.v1",
  "variant": "v1",
  "part": "C1210C476K8RAC",
  "version": "1.0.0",
  "model": {
    "id": "electrical.capacitor.ideal",
    "version": "1.0.0",
    "sha256": "sha256:611af2c6e3cffc7e2cf7468aae1ab22b59922035fbfbf13e4515681ea6ef8279"
  },
  "parameters": {
    "capacitance": 0.000047
  },
  "parameter_uncertainty": {
    "capacitance": {
      "distribution": "normal",
      "parameters": {"std": 0.0000047}
    }
  },
  "condition": "unverified",
  "compute": {
    "relative_cost": 1.0,
    "notes": "Ideal lumped-capacitance hypothesis."
  },
  "metadata": {
    "dielectric": "X7R"
  }
}
~~~

## Interface and catalog rules

The referenced PMDL must implement the category/device contract owning the
instantiation directory (a device instantiation may use that device or its
parent category implementation). Each instantiation directory contains exactly
one `static.part`, at least one `vN.model`, an optional host-generated
`README.md`, and no unrelated files. The README is a derived human-readable view,
not an authoritative DSL record. It must be regenerated from a fully validated
catalog rather than authored by Luna or used as validation input. It indexes and
names every `vN.model`, so multiple hypotheses remain explicit to a reader who
has no external project documentation.

Different `vN.model` files represent distinct model hypotheses or fidelities for
the same invariant physical part. Do not duplicate static geometry, connectors,
purchasing, deterministic shape references, or provenance inside a model
instance. Do not use variants to bypass a PMDL hash change; update the exact
reference and requalify the closure.
