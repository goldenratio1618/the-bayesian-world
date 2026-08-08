# Phase 1 implementation and boundaries

## Relationship to the current project plan

The [consumer engineering plan](04_implementation_plan_v2_consumer.md) is the
high-level design authority. This implementation covers a technical subset of
its typed IR, component-package, simulation, validation, compiler, agent, and
artifact layers. It does not claim to complete the card-workstation pilot,
physical qualification sequence, five-graph evidence library, or microfactory
operating model.

The scanner robot is an intentional pre-product laboratory device. It is an
end-to-end harness fixture and a future means to acquire 3D structure and
physical evidence before the first product-level prototype. A successful
simulation is E0/E1 engineering evidence only and does not qualify a part.

## Canonical representation contract

The source of truth is a strict `contraption-2` assembly plus its referenced
component-package registry and exact-hash PMDL registry. Component instances do
not carry independent models or visualization geometry. A component package
defines its model hash, bodies/solids, connector frames, connector-to-model-port
bindings, geometry-parameter bindings, and provenance.

`resolve_assembly` validates and compiles that closure into:

1. an assembled PMDL descriptor system;
2. a physical attachment/joint graph and resolved body/connector poses; and
3. one assembly SHA-256 shared by every downstream artifact.

Every `contraption-2` source must also carry a strict, hash-bound
`metadata.dynamics_completeness` record. A `complete` record may have no open
gates; an `incomplete` record must identify each known missing interaction as an
open gate. Validation rejects an absent, malformed, or self-contradictory
record, and simulation reports, scanner acceptance, build plans, and compiled
C99 artifacts propagate it instead of implying more fidelity than the assembly
declares.

The separation is a projection boundary, not a second representation:

- NumPy and Torch integrate the namespaced residuals of every component model
  plus equations induced by typed connections.
- Runtime network and connector-coincidence checks reject conservation or
  attachment drift after every accepted timestep.
- The viewer accepts only a `ResolvedAssembly` plus, optionally, an actual
  `SimulationResult`; it derives the static scene and hash-bound resolved pose
  frames internally. It performs no simulation, name-based placement, detached
  scene ingestion, or visual override.
- The build planner accepts only `ResolvedAssembly`, derives placement and
  connector distances from resolved poses, and exposes missing fabrication
  facts as release gates.
- The C99 compiler accepts the full `ResolvedAssembly`, derives its local
  dynamics/estimator system directly from the assembled PMDL DAE, and embeds
  assembly, PMDL, and controller provenance. Caller-authored online matrices
  and bare PMDL-system projections are not canonical compilation inputs.

The old scanner aggregate coverage sidecar and authored online-model JSON are
not part of this architecture.

## What is implemented

- A restricted, typed, acausal PMDL with safe expression trees, dimensional
  analysis, descriptor residuals, energy/dissipation declarations, validity
  envelopes, evidence, and property-test records. The portable math allow-list
  includes common dimensionless functions such as `tanh`.
- Strict component-package schemas for physical bodies, explicit connector
  locations, provenance, model-port mappings, and state/parameter geometry
  bindings.
- Fail-closed assembly of electrical, mechanical, signal, control, and explicit
  kinematic-only connections, with equation counts and structural-rank checks.
- Backward-Euler descriptor integration on NumPy and optional GPU-enabled
  PyTorch, with consistent initialization and runtime network invariants.
- Batched uncertainty propagation, fitting, and approximate Bayesian
  candidate-model comparison.
- A restricted onboard control format and a scanner controller that addresses
  namespaced canonical assembly state/control variables.
- DAE-derived, fixed-allocation C99 dynamics plus an uncertainty-aware
  estimator for constrained onboard execution. Controller state-machine C99
  emission is not implemented; the manifest says so explicitly.
- Budgeted, staged classification/modeling agents whose artifacts are inert
  data until deterministic validation and explicit promotion.
- A dependency-free, display-only browser viewer and a deterministic,
  hash-bound build planner.

## Deliberate Phase 1 limitations

"Rigid body" currently means a planar root chassis with a declared tree of
fixed/revolute attachments and three-dimensional visualization transforms. It
does not include a general 3D contact solver, flexible bodies, fracture, fluid,
thermal, or optical physics.

The scanner's hash-bound dynamics record is deliberately `incomplete`. Its
chassis model uses the published **bare-chassis** mass of 0.160 kg (without
batteries) and derives yaw inertia from that mass and the canonical estimated
box envelope in the chassis component package. Fixed battery, electronics,
motor-case, servo-case, and compute-payload mass/inertia are not folded into a
hidden whole-robot parameter. Moving arm/camera inertia is only a
component-local approximation; downstream-body inertial derivation and servo
case-reaction coupling are not modeled. Caster/floor contact, full-solid
keep-out, physical localization/encoder observation chains, and supply/fault
coupling are also open gates.

Those seven omissions are machine-readable and release-blocking. Scanner
acceptance includes dynamics completeness, so `accepted` cannot become true
while any gate remains open. The current
`root_keepout_violation_probability` checks only the planar root position; it is
not labeled or treated as full-body collision evidence.

The scanner component PMDL files are prototype models, not experimentally
qualified gold models. Estimated geometry and connector locations are explicit
in their package provenance and must be replaced or verified with CAD, catalog,
measurement, or scan evidence before construction. The build planner therefore
correctly reports the current design as not build-ready.

The NumPy path is portable but neither GPU-accelerated nor automatically
differentiable. The Torch path retains tensors on the requested CUDA device and
preserves its autograd graph. An explicit unavailable CUDA request is an error;
the tooling does not claim acceleration after silently selecting a CPU.

Monte Carlo intervals describe propagated model, parameter, and process
uncertainty; they are not automatic frequentist coverage guarantees. Model
evidence uses documented approximations and still needs posterior-predictive
checks and physical data.

Generated floating-point C99 is a portable online reference, not a release-ready
firmware image. Target execution, numerical equivalence, timing, range analysis,
hardware-in-the-loop testing, fail-safe behavior, and independent safety
controls remain release gates. FPGA deployment additionally needs fixed-point
selection, synthesis, and timing closure.

Compilation remains available for an incomplete assembly, but the generated
source, model metadata, and manifest all carry the typed completeness status and
open gates. The artifact therefore cannot truthfully be mistaken for a
physically complete device model.

The generated C99 does not execute the declarative `ControlProgram`. It accepts
the resolved actuator-source vector and requires a separately qualified runtime
for the exact controller id/version/hash recorded in its source and manifest.

## Run the scanner fixture

From the repository root in Linux/WSL:

```bash
source .venv/bin/activate
contraption validate
contraption simulate --backend torch --device cuda --output outputs/scanner_demo
contraption view --trajectory outputs/scanner_demo/trajectory.json \
  --output outputs/scanner_demo/viewer
python -m http.server 8000 --directory outputs/scanner_demo/viewer
```

Open <http://127.0.0.1:8000>. The emitted `physical-scene.json`, exact-sample
trajectory, viewer payload, build plan, and compiler manifest carry the same
assembly hash. Viewer poses are reconstructed from that trajectory through the
resolved assembly; the detached physical-scene artifact is diagnostic output,
not another viewer input representation.

For user-set external controls, run `contraption serve --backend torch
--device cuda`. Its same-origin loopback API derives the UI schema from the
hash-bound declarative controller, validates every POST, reruns the Python
assembly, and returns only a complete canonical physical scene. The JavaScript
viewer never integrates dynamics or composes physical transforms.

Compile and inspect the build release gates with:

```bash
contraption compile --output outputs/scanner_demo/online
contraption build --output outputs/scanner_demo/build
```

## Verification ladder

1. Parse the contraption, package registry, and exact-hash PMDL closure.
2. Verify complete connector/model-port and geometry/state/parameter bindings.
3. Reject incompatible domains, interfaces, units, underconstrained attachment
   trees, stale content hashes, or structurally singular networks.
4. Compute a consistent descriptor initial state and check network invariants.
5. Integrate on NumPy and Torch; verify accepted timesteps preserve network and
   physical attachment boundary conditions.
6. Generate pose frames from the physical resolver and reject missing, extra,
   or stale-hash viewer frames.
7. Derive C99 from the assembled DAE, compile it with a host C99 compiler, and
   retain both closure hashes in the manifest/source.
8. Confirm every canonical component/connection appears in the build record and
   that unknown fabrication facts remain unresolved.
9. Treat all physical instances as `unverified` until independent dimensional,
   electrical, dynamic, and safety measurements are recorded.

## Safety boundary

This is engineering software, not a certified controller. Before powering real
hardware, add a physical power switch, appropriate fuse/current limiting,
software and hardware stops, mechanical travel stops, cable strain relief, and
guards. Commission at low energy with the robot lifted or restrained and keep
it clear of people, pets, stairs, liquids, and fragile objects. Do not infer
safe continuous operation from a simulation or catalog stall rating.
