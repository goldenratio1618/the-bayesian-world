# Physical-model constraints

Read this file completely before proposing a model.

Models are data-only programs in the restricted `.pmdl` language. They use the
universal descriptor residual form `F(t, z, zdot, theta, u) = 0`; computational
causality is assigned by assembly and simulation tooling, not by the component.
Every expression must remain differentiable inside each explicitly declared
discrete mode. Discontinuous mode transitions require guards and reset maps.

Only approved arithmetic, comparison, smooth elementary functions, typed
vectors, and declared symbols are permitted. The complete function allow-list
is `abs`, `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `exp`,
`log`, `log10`, `tanh`, `min`, `max`, `clip`, `sign`, `where`, `smooth_abs`, and
`der`. Function arguments must satisfy the DSL's unit rules; transcendental
functions such as `tanh` require dimensionless inputs. No imports, file access, network
access, dynamic dispatch, reflection, loops, recursion, user-defined functions,
or embedded host-language code are allowed. Units, frames, orientations,
reference conventions, bounds, initialization requirements, validity ranges,
fidelity/approximation metadata, evidence, and property tests are mandatory.

Each power port is bidirectional and pairs effort with flow. Directed command
and measurement ports are separate and cannot carry power. Stored energy must
be nonnegative within the validity envelope; dissipation must be nonnegative;
explicit sources must be declared. Models must be equation-balanced when
assembled, conserve power under the declared sign convention, expose analytic
or autodifferentiable Jacobians, and identify potentially singular limits.

Generated output is staged and will be rejected on any parse, dimensional,
balance, conservation, bounds, initialization, composition, property-test, or
path-safety failure. It cannot modify the simulator or any host-language code.

The only top-level model keys are `format`, `id`, `name`, `version`, `domains`,
`implements`, `description`, `power_ports`, `signal_ports`, `states`, `algebraics`,
`parameters`, `relations`, `stored_energy`, `dissipation`, `sources`, `modes`,
`initialization`, `validity`, `fidelity_levels`, `properties`, `trust`, and
`metadata`. Never invent keys such as `jacobians` or `geometry`; express extra
descriptive detail under `metadata` only when it is data rather than executable
behavior.

Every concrete PMDL must live at its category or device layer in
`model_catalog/<physical-domain>/<category>[/<device>]/` and its `implements`
value must name that directory's abstract `interface.pmdl` contract. Imported
parts live below an adjacent `instantiations/<part-id>/` directory. Each such
directory contains exactly one `static.part` for model-invariant physical data
and one or more `vN.model` files selecting an exact PMDL id/version/hash and
initializing every parameter. A complete import is validated as one bundle;
contraptions reference the model-instance id and never repeat its parameters.

## Exact nested record names

Use these field names exactly; unknown keys are rejected rather than ignored:

- state: `name`, `unit`, `initial`, `derivative`, `description`
- algebraic: `name`, `unit`, `initial`, `description`
- parameter: `name`, `unit`, `default`, `bounds`, `uncertainty`, `learnable`,
  `description`; bounds use only `lower` and `upper`; uncertainty uses only
  `distribution`, `parameters`, and `correlation_group`
- power port: `name`, `domain`, `effort`, `flow`, `effort_unit`, `flow_unit`,
  `orientation`, `frame`, `reference`, `description`
- signal port: `name`, `direction`, `unit`, `dtype`, `shape`, `description`
- relation: `name`, `expression`, `description`
- stored-energy, dissipation, or source entry: `name`, `expression`, `unit`,
  `nonnegative`, `description`
- mode: `name`, `active_relations`, `transitions`, `initial`; each transition
  uses `target`, singular `guard`, and `resets` (a mapping from state names to
  expressions). Never put `guards` or `resets` directly on a mode.
- initialization: `strategy`, `constraints`, `required`
- validity: `ranges`, `assumptions`, `max_timestep`
- fidelity level: `name`, `description`, `active_relations`,
  `parameter_overrides`, `approximation_error`, `relative_cost`
- property: `name`, `kind`, `expression`, `expected`, `samples`, `tolerance`,
  `description`
- trust: `structural`, `physical`, `numerical`, `empirical`, `evidence`; each
  evidence entry uses `kind`, `reference`, `summary`, and optional `date`

Relations are residual expressions interpreted as equal to zero; do not append
`== 0`. Property-test expressions may be Boolean. Use only symbols declared by
the model or its ports, and use `der(state_name)` or the declared derivative
symbol consistently with the supplied gold examples.
