# Architecture contract

This repository is a runnable engineering reference, not a certification claim.
The scanner is a pre-product laboratory fixture rather than a product taxonomy
or a physical domain.

## Filesystem catalog

`model_catalog/` is the only model and part catalog. PMDL means Physical Model
DSL. Its hierarchy is semantic
and validated:

```text
model_catalog/
  <physical-domain>/
    interface.pmdl
    <category>/
      interface.pmdl
      [general-model.pmdl]
      [<device>/
        interface.pmdl
        [device-model.pmdl]
        instantiations/<part-id>/
          static.part
          v1.model
          [v2.model ...]]
```

Every layer declares abstract interface contracts in `interface.pmdl`. Concrete
PMDL classes live beside the most specific contract they implement. A
`static.part` owns model-independent geometry, connectors, provenance, purchase
facts, and metadata. Each `vN.model` selects an exact PMDL id/version/hash and
initializes its parameters, uncertainty, condition, and relative compute cost.

There is no SQL registry. `load_contraption` dynamically builds the Python
`ModelRegistry`, `ModelInterfaceCatalog`, and `PartInstantiationRegistry` from
the catalog roots named by the contraption manifest. Stale hashes, duplicate
ids, and incomplete part/model closures fail before resolution.

## Canonical contraption closure

The complete source is a `contraption-4` bundle rooted at `contraption.json`.
The manifest contains catalog links, component instantiations, typed topology,
the physical root, open-loop external actuators, plural controllers, plural
verification programs, environment, and metadata. Controller and verification
files are content-addressed and must remain inside their bundle; shared catalog
roots are explicit relative links.

```text
contraption-4 manifest
  + catalog roots
  + control-1 programs
  + verification-1 programs
              |
       load_contraption
              |
       ResolvedAssembly
     /       |        \
 PMDL plant  controllers  verification
 simulation  + inference  over posterior
     \       |        /
       hash-bound results
        /             \
 physical scene     build plan
                        \
                 C99 / Verilog controllers
```

Resolution verifies interface conformance, PMDL and artifact hashes, complete
parameter initialization, physical bindings, signal direction, units, sensor
state indices, and actuator destinations. All downstream systems consume the
same `ResolvedAssembly`; none may add or override the represented device.

## Controller boundary

`control-1` is a strict, data-only DSL. It reuses PMDL's safe expression AST and
dimensional type system and adds synchronous control semantics: modes,
prioritized transitions, registers, derived values, bounds, output slew rates,
and same-tick emergency outputs.

The input distinction is architectural:

- `explicit_inputs` are physical sensor wires or external controller pins.
  `contraption.json` binds every sensor input to an exact namespaced PMDL signal
  and every external input to a runtime name. These are the only public inputs
  to generated C99 or Verilog.
- `implicit_inputs` are latent controller state. The current implementation
  declares only the desired namespaced PMDL quantity, prior uncertainty, and
  process-uncertainty rate. `contraption.json` binds that declaration to an
  exact assembled PMDL variable. The loader derives a coupled local-affine
  observer from the complete descriptor system and its explicit sensor wires;
  controller authors do not supply transition or measurement matrices.
  Expressions can consume each posterior `mean`, `variance`, or `std`.
  Observability is calculated for every requested latent, recorded in the
  closure, and reported as a runtime warning when deficient.

The offline simulator creates an independent controller runtime for each
posterior sample. It extracts only wired explicit signals from the simulated
plant and holds controller outputs between controller ticks. Heterogeneous
periods use an exact integer-stride schedule over a common physics subdivision;
external pins are sampled only on the ticks that consume them. The controller
is never passed the plant-state tensor or a hidden-state callback. Multiple
controllers are supported and duplicate actuator drives are rejected. Any
plant input that can influence an observer state, measurement, or latent
projection must be owned at that controller boundary; irrelevance is proven
structurally rather than from a zero derivative at one operating point.
Command echoes from another controller are not yet lowerable as known
exogenous observer inputs, so coupled multi-controller observers fail closed;
observer-free or structurally independent multiple controllers are supported.

## Verification boundary

`verification-1` is another strict data-only DSL built on the PMDL expression
AST and unit system. It declares trajectory inputs, parameters, per-sample
metrics, physical-time reducers, Boolean criteria, and minimum posterior pass
probabilities with required confidence levels. Mean and RMSE use trapezoidal
weighting on the exact trajectory time grid. Admission uses a conservative
Wilson lower confidence bound, not the raw pass fraction. The simulator
automatically evaluates every hash-bound program and records estimates,
confidence intervals, effective sample counts, and overall acceptance.

Numeric expression paths are classified as smooth or piecewise-smooth and
remain native NumPy/Torch graphs. Guards, mode choices, Boolean criteria,
counts, and confidence decisions are the explicit discrete boundary; the
system does not claim gradients through those decisions.

## Execution backends

The NumPy backend is the portable CPU path. It uses ordinary arrays and does not
provide automatic differentiation. The Torch backend retains tensors on the
selected CPU or CUDA device and preserves autograd through PMDL assembly,
controller sensor extraction and output wiring, controller inference, and
verification. Both backends execute the same typed specifications and report
the same controller observability diagnostics.

## Controller compilers

`contraption.control.compiler` lowers `control-1` to target-neutral IR and then
to complete controller implementations:

- C99 uses fixed-allocation structures and double-precision math.
- Verilog-2001 uses synchronous signed fixed-point arithmetic and exposes only
  explicit input and declared output ports.

Both targets include controller state, the PMDL-derived vector observer,
posterior covariance, transitions, registers, output bounds, slew limiting,
same-tick emergency behavior, input validation, and explicit arithmetic and
observer faults. Compilation of a controller with implicit inputs requires its
resolved assembly closure; generated manifests bind the assembly, PMDL,
controller link, observer derivation, operating point, validity record, and
every source digest. Target admission fails closed. For example, `tanh` is
valid in PMDL, the Python controller runtime, and C99, but the generic
synthesizable Verilog target rejects it until a qualified lookup-table or
algebraic lowering is supplied.

## Source ownership

- `contraption.loading` owns device-independent bundle loading and closure
  resolution.
- `contraption.physics` owns PMDL parsing, validation, assembly, simulation,
  fitting, uncertainty, and physical resolution.
- `contraption.control` owns the control DSL, runtime, target-neutral IR, C99,
  and Verilog generation.
- `contraption.verification` owns the verification DSL and posterior evaluator.
- `contraption.catalog` owns interface discovery and part/model instantiation
  registries.
- `contraption.part_import` owns guarded part classification/modeling, budgets,
  and deterministic candidate validation.
- `contraption.visualization` is display-only; `contraption.live` provides the
  generic loopback simulation service.
- `contraption.manufacturing` owns closure-bound build instructions.

System-specific assets and policies live under `assembled_contraptions/`, never
inside `src/contraption`. The former Python RC/RL/DC-motor/planar-body fixtures
are explicit bundles under `assembled_contraptions/examples/test_systems/`.

## Shared guarantees and limits

- Unknown fields, duplicate identifiers, stale hashes, incompatible units, and
  unresolved wiring are errors.
- PMDL uses universal residual form `F(t, z, zdot, theta, u) = 0` and an
  allow-listed expression tree; generated host code is never executed.
- PMDL process noise declares ordered standard-normal channels and unit-checked
  accepted-step increments on differential states. Random draws are the
  intentional nondifferentiable source; increment scale and parameter paths
  remain backend-native. Descriptor algebraics are reconciled after each
  stochastic increment.
- Every accepted timestep preserves assembled network equations and physical
  boundary conditions within declared tolerances.
- Viewer frames, verification reports, controller artifacts, and build plans
  remain bound to the exact assembly closure.
- Controller inference is a coupled local-affine approximation derived from the
  complete assembled PMDL descriptor system at a hash-bound operating point.
  Nonlinear plant relations require explicit approximation approval, and the
  generated closure records those relations, the PMDL validity region, a
  declared perturbation radius/remainder threshold, and the measured coupled
  local residual remainder. This is not an exact nonlinear observer, an
  exhaustive error bound, or a hidden claim of global validity.
- Model instances remain `unverified` until independent evidence changes their
  declared condition.
