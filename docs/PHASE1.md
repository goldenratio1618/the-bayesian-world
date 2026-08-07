# Phase 1 implementation and boundaries

## Relationship to the current project plan

The [consumer engineering plan](04_implementation_plan_v2_consumer.md) is the
high-level design authority. This implementation covers a technical subset of
its typed IR, simulation, component-package, validation, agent, compiler, and
artifact-tooling layers. It does not claim to complete the card-workstation
pilot, physical qualification sequence, five-graph evidence library, or
microfactory operating model.

The scanner robot is an intentional pre-product lab device. It provides an
end-to-end harness fixture and a future means to acquire 3D structure and
physical evidence before the first product-level prototype. Its successful
simulation is E0/E1 engineering evidence only; it does not qualify a physical
part or supersede the card-workstation family as the strategic pilot.

The PMDL and simulator APIs remain backend-neutral. This Phase 1 reference uses
NumPy for portability and PyTorch for GPU execution and differentiation. The
plan's NumPyro/JAX execution backend and independent Stan oracle remain later
compiler targets; this repository does not imply that PyTorch replaced that
decision.

## What is implemented

Phase 1 is a modular reference stack for electrical and planar rigid-body
digital twins:

* a restricted, typed, acausal physical-model DSL with safe expression trees,
  dimensional analysis, residual descriptor equations, explicit energy and
  dissipation declarations, and schema-level representations for modes,
  fidelity levels, validity envelopes, evidence, and property tests;
* deterministic contraption, connection, internal/external signal, taxonomy,
  category, subcategory, model, and physical-instantiation specifications;
* batched Monte Carlo uncertainty propagation and confidence summaries on a
  NumPy backend, plus an optional PyTorch backend for CUDA vectorization and
  end-to-end automatic differentiation;
* descriptor-system integration, electrical and differential-drive/planar
  mechanics primitives, experimental parameter fitting, and approximate
  Bayesian candidate-model comparison;
* a restricted state-machine/expression format for onboard control logic;
* a conservative online compiler that admits only validated linear or
  linearizable models and generates fixed-allocation C99 plus an
  uncertainty-aware extended Kalman filter;
* budgeted, staged classification and modeling agents whose artifacts are data,
  never imported code, and require strict validation plus explicit promotion;
* a dependency-free interactive browser viewer and deterministic build
  instruction generator; and
* an end-to-end in-silico apartment scanning robot fixture.

## Deliberate Phase 1 limitations

"Rigid-body" currently means a planar chassis with revolute arm coordinates and
3D visualization transforms. It does not yet include a general-purpose 3D
contact solver, flexible bodies, fracture, fluid, thermal, or optical physics.
The camera is mass and geometry only, exactly as requested.

The scanner mission uses a hand-authored `DifferentialDriveArmModel` aggregate;
it is **not** assembled from the component PMDL files. Before the
contraption-level entry point integrates anything, it requires
`examples/scanner_robot/contraption.json` and
`examples/scanner_robot/simulation_coverage.json`. The coverage contract binds,
with a canonical SHA-256 digest, every component ID and model reference plus
every connection ID, kind, domain, and ordered endpoint list. Each physical
element has an explicit state, dynamics-parameter, controller, sensor,
geometry-only, or excluded mapping. Exclusions require a nonempty limitation.
Missing/extra entries, changed models or endpoints, invalid exclusions, and a
stale hash are fatal errors rather than silently reducing the simulation.

`make_scanner_aggregate_model` remains available for isolated tests of the
reduced equations. It carries no physical-coverage or PMDL-composition claim;
physical scanner runs must use the guarded `simulate_scanner_robot` path.

`contraption validate` is structural, type, unit, taxonomy, and reference
validation; its JSON output labels that scope explicitly. Standalone runtime
admission is stricter. The simulator executes validity ranges but rejects
declared modes, initialization constraints, non-identity fidelity selection,
and unexecuted property tests with `UnsupportedPMDLSemanticsError`. Stateless
algebraic components also require network assembly rather than standalone
time integration. These declarations are preserved in the DSL and are never
silently treated as though their runtime semantics had executed.

The NumPy path is runnable everywhere but is not automatically differentiable
and does not use a GPU. Install the `gpu` extra to select PyTorch; the simulator
then keeps tensors on the selected CUDA device and preserves the autograd graph.
This split avoids pretending that a CPU-only machine has GPU acceleration.

Monte Carlo intervals describe propagated model/parameter/process uncertainty;
they are not automatically frequentist coverage guarantees. Tests check sample
moments and analytic special cases. Empirical calibration still requires data
from the built robot.

Model evidence uses a documented Laplace/BIC approximation, not an exact
closed-form Bayes factor. Results are labeled accordingly and should be checked
with posterior predictive diagnostics.

Generated C is a deterministic simulation/control reference suitable for an
MCU toolchain. FPGA use requires downstream fixed-point selection, timing
closure, and hardware-in-the-loop verification; the compiler emits a manifest
that makes those remaining obligations explicit.

## Quick start

From this directory:

```bash
python -m pip install -e .
python -m pytest
contraption demo --output outputs/scanner_robot
```

For CUDA/autograd support:

```bash
python -m pip install -e ".[gpu]"
contraption doctor
```

For a reproducible GPU-first environment, use
[the Linux installation guide](INSTALLATION.md) rather than relying on the
minimal commands above.

The demo produces a trajectory/UQ JSON file, a self-contained viewer, generated
C99 online runtime, compiler manifest, and Markdown assembly instructions.

## Verification ladder

1. Parse and validate every bundled `.pmdl` model and JSON specification.
2. Compare electrical and mechanical special cases with analytic solutions.
3. Check Monte Carlo moments, deterministic seeding, interval ordering, and
   batch invariance.
4. Fit synthetic observations and confirm parameter recovery.
5. Compare candidate-model evidence on generated data.
6. Verify restricted control programs and their safety clamps.
7. Compile generated C with any available C99 compiler and execute a smoke step.
8. Verify the scanner aggregate coverage contract, then run the mission and
   check orbit, pointing, clearance, and uncertainty acceptance criteria.
9. Load the viewer and inspect 3D/electrical views and controls.
10. Treat all physical instances as `unverified` until dimensional, electrical,
    and dynamic measurements are recorded.

## Safety boundary

This is engineering software, not a certified controller. Before powering real
hardware, add a physical power switch, fuse as appropriate, conservative current
limits, a software and hardware stop, mechanical travel stops, cable strain
relief, guarded moving joints, and a low-speed lift test. Keep the robot on the
floor, clear of people, pets, stairs, liquids, and fragile objects. Do not infer
safe continuous operation from stall ratings.
