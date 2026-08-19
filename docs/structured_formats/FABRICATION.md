# Fabrication records

Fabrication records make construction-relevant connector facts typed and
reviewable. The same strict `ConnectorFabricationSpec` is used in two places:

- `static-part-2` `connectors[].fabrication` describes the invariant capability
  or requirement of one part endpoint;
- `contraption-4` `connections[].implementation` describes the hardware and
  process selected for that particular assembly or net.

Part records do not choose assembly-specific fastener length, cable route, or
multi-endpoint topology. Connection implementations do not override incompatible
endpoint constraints. The authoritative implementation is
`contraption.fabrication`.

## Common record

Every non-null record requires:

| Field | Meaning |
|---|---|
| `kind` | `fixed_mount`, `rotary_support`, `electrical_termination`, `optical_alignment`, or `other` |
| `status` | `missing`, `partial`, or `specified` |
| `missing` | unique explicit field paths that are still unknown |

Optional typed payload fields are `standards`, `retention`, `bearing`,
`conductor`, `termination`, `protection`, `route`, `travel`,
`alignment_tolerance_m`, `alignment_tolerance_rad`, and `evidence`. Unknown
fields are invalid.

Status is a claim about completeness:

- `missing` requires a nonempty exhaustive `missing` list and forbids claimed
  fabrication values. It does not cause a default fastener, bearing, wire, or
  tolerance to be selected.
- `partial` requires at least one known value, source evidence, and an exhaustive
  nonempty `missing` list.
- `specified` requires source evidence, an empty `missing` list, and every field
  required for that kind and context.

`connector.fabrication` is optional and may be omitted or null. That means no
fabrication record exists; it is not permission to infer one. Deterministic
imports should preserve the more useful explicit form when the source lacks
details, for example:

~~~json
{
  "kind": "rotary_support",
  "status": "missing",
  "missing": ["retention", "bearing", "travel"]
}
~~~

## Deterministic component-input pathway

The part importer does not treat the modeling agent as an authority for
construction facts. An input may supply optional, strictly typed facts through:

~~~json
{
  "connector_fabrication": [
    {
      "part": "example.motor",
      "connector": "shaft",
      "fabrication": {
        "kind": "rotary_support",
        "status": "partial",
        "missing": ["retention", "bearing", "travel"]
      }
    }
  ]
}
~~~

Each entry must contain exactly `part`, `connector`, and `fabrication`.
Selectors must be unique and must match a connector in the generated part.
The source fragment may not author its own `evidence`: the host validates the
typed record, binds the exact component-input path, JSON locator, and SHA-256,
and only then materializes it. Agent-authored fabrication values without a
matching deterministic entry are replaced with the appropriate typed
`status: missing` record. Omitting `connector_fabrication` is valid and creates
no construction claim.

## Kind and context requirements

| Kind | Endpoint capability is complete when | Assembly implementation additionally requires |
|---|---|---|
| `fixed_mount` | `retention` is complete | nothing beyond the selected retention |
| `rotary_support` | `retention`, `bearing`, and `travel` are complete | nothing additional |
| `electrical_termination` | `conductor` and `termination` are complete | `protection` and `route`; a multi-endpoint net cannot use `point_to_point` |
| `optical_alignment` | at least one standard plus linear and angular alignment tolerances | the same complete alignment specification |
| `other` | at least one standard | the same complete standard identity |

Connector domain/interface determines the allowed kind:

- mechanical and rigid-mechanical connectors use `fixed_mount`, except
  `rotational-shaft`, which uses `rotary_support`;
- electrical and signal connectors use `electrical_termination`;
- optical connectors use `optical_alignment`;
- other domains use `other`.

For a connection implementation, `power` and `signal` require
`electrical_termination`; a fixed attachment requires `fixed_mount`; a revolute
attachment requires `rotary_support`; and a constraint requires `other`.

## Standards and evidence

A standard reference requires `family`, `authority`, `document`, and
`designation`. Optional fields are `revision`, `role` (`internal`, `external`,
`plug`, `receptacle`, or `neutral`), authoritative HTTP(S) `uri`,
`nominal_diameter_m`, `pitch_m`, and `gauge_awg`.

Common families are:

`iso_metric_thread`, `unified_thread`, `iso_bearing`, `abma_bearing`, `awg`,
`iec_conductor`, `ipc_land_pattern`, `manufacturer_connector`,
`manufacturer_package`, `manufacturer_fastener`, and `manufacturer_cable`.

`iso_metric_thread` and `unified_thread` require canonical SI diameter and
pitch. An ISO metric designation such as `M3x0.5` is checked against those
values. `awg` requires an integer gauge from 0 through 40. A rarer standard is
not squeezed into a misleading family: use `family: extension` with its real
authority, document, designation, and mandatory authoritative `uri`. Adding a
new common family is a deliberate schema/code change with validation, tests,
and this guide updated.

Evidence has required `kind` and `source`; optional `locator`, prefixed
`sha256`, and positive `page` pinpoint the claim. Allowed kinds are
`component_input`, `datasheet`, `vendor_page`, `catalog`, `drawing`,
`measurement`, and `manual`. Evidence supports a stated fact; it never fills a
missing one.

## Typed payload summary

| Record | Core content |
|---|---|
| `retention` | method; optional standard hardware, quantity, torque, locking method, and installation process; completeness depends on method |
| `bearing` | method plus standard/designation, bore/outer/width dimensions, radial clearance, axial retention, and lubrication; flexures need a standard or designation |
| `conductor` | conductor and insulation standards, count, material, cross-section, voltage rating, and temperature rating |
| `termination` | method and installation process; method-specific hardware/contact/housing/pin or PCB-land pitch/pad dimensions |
| `protection` | none, fuse, breaker, limiter, fusible link, or other; non-none protection needs standard, part number, current, and voltage ratings |
| `route` | topology, routed length, bend radius, service loop, strain relief, and optional unique waypoints |
| `travel` | `bounded` with ordered limits, or `continuous`; unit is `rad` or `m` |

## Complete fixed-mount example

~~~json
{
  "kind": "fixed_mount",
  "status": "specified",
  "missing": [],
  "retention": {
    "method": "threaded_fastener",
    "hardware": {
      "family": "iso_metric_thread",
      "authority": "ISO",
      "document": "ISO 261",
      "designation": "M3x0.5",
      "role": "external",
      "nominal_diameter_m": 0.003,
      "pitch_m": 0.0005
    },
    "quantity": 2,
    "torque_n_m": 0.5,
    "locking_method": "threadlocker"
  },
  "evidence": [
    {
      "kind": "drawing",
      "source": "Vendor installation drawing",
      "locator": "Detail B",
      "sha256": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ]
}
~~~

Endpoint records with overlapping standard families are checked for dimensional
and mating-role conflicts. Missing records and `status: missing` are not treated
as compatible facts; they simply leave construction gates open. A build is
construction-ready only when each required connection implementation is
`specified` and compatible with all endpoints.
