# Online C99 compiler contract

## Purpose

The offline simulator solves the full namespaced PMDL descriptor system and can
use workstation/GPU resources. The online compiler derives a smaller local
fixed-allocation C99 estimator/runtime for constrained onboard hardware. It is
an explicitly bounded approximation of the same resolved assembly, not a
second physical or behavioral specification.

The generated C99 is useful for deterministic embedded execution, estimator
integration, toolchain checks, and later hardware-in-the-loop comparison. It is
not used to define the browser visualization or to replace the high-fidelity
offline simulation.

## Only canonical resolved assemblies are admitted

`compile_resolved_assembly` accepts only a fully resolved `ResolvedAssembly`
closure. Here "fully resolved" does not mean that its physical fidelity status
must be `complete`; known omissions may remain, but they must be explicit. The
required inputs include:

- a valid `sha256:...` assembly closure hash;
- a valid PMDL closure hash;
- a valid, mandatory `dynamics_completeness` record;
- a square, structurally full-rank assembled residual system;
- at least one differential state;
- finite parameters, controls, timestep, and operating point; and
- a nonsingular, acceptably conditioned local descriptor solve.

The compiler refuses caller-authored `OnlineModelIR` at this entry point. The
scanner no longer loads an `online_model.json`, reviewed aggregate matrix, or
separate topology-coverage document. Every state, input, residual, parameter,
and topology dependency comes from the same component packages/PMDL closure
used by offline simulation.

Both expected hashes can be supplied at the API/CLI boundary. A mismatch is a
hard error. A bare `AssembledPMDLSystem` is refused because it has discarded the
physical/package/controller closure. Generated headers, sources, and manifests
embed the assembly and PMDL hashes plus the canonical controller id, version,
and content hash (or explicit `null`). They also embed the dynamics-completeness
status and open-gate identifiers; the manifest and model metadata retain the
full typed record.

## DAE-derived local model

Let the assembled residual be:

```text
F(t, x, xdot, a, u, θ) = 0
```

where `x` is the differential state, `a` is the algebraic state, `u` is the
declared control-source vector, and `θ` is the package-resolved parameter
vector. At the requested operating point, the compiler solves for
`q = [xdot, a]`, then differentiates the assembled residual to obtain:

```text
G = ∂F/∂q
H = ∂F/∂x
J = ∂F/∂u
```

The implicit-function theorem gives the local differential dynamics from the
first rows of `-G⁻¹[H J]`. Failure to find a consistent operating point,
singular `G`, excessive condition number, nonfinite derivative, or residual
above tolerance stops compilation with diagnostics. No component or connection
may be silently dropped to make the system compilable.

## Generated dynamics/estimator module

Generated C99 uses fixed-size/preallocated arrays and records:

- differential-state and input ordering;
- the derived local matrices/bias and operating point;
- covariance and clamp assumptions;
- nominal/maximum timesteps;
- DAE residual and condition diagnostics; and
- the canonical assembly and PMDL hashes; and
- the declared dynamics-completeness status and open gates.

It does **not** currently compile the declarative `ControlProgram` state machine
into C. The generated functions accept the resolved control-source vector. A
separately qualified controller runtime must execute the exact controller
recorded in the manifest and supply those values. The manifest carries
`controller_execution.emitted: false`, and every generated source/header says
`Controller execution: NOT EMITTED`; therefore this output is not complete
robot firmware.

The portable target uses floating point and an uncertainty-aware estimator.
`contraption compile` requires a local GCC/Clang-compatible C99 syntax check by
default; `--skip-syntax-check` is an explicit artifact-only escape hatch, not a
claim of target validity.

## What remains outside this compiler

The compiler does not certify numerical equivalence outside the declared local
validity region, worst-case execution time, target ABI/toolchain behavior,
overflow policy, memory integrity, actuator safety, or physical behavior.
Release still requires target execution tests, offline/online trajectory
comparison, timing/range analysis, fault injection, and hardware-in-the-loop
qualification. Compilation is allowed while dynamics gates are open, but it
does not close or waive them.

FPGA deployment additionally requires fixed-point selection, quantization and
overflow analysis, scheduling/pipelining, synthesis, timing closure, and HIL
verification. A generated floating-point C model is not an FPGA bitstream.

## CLI

From the repository root after canonical validation:

```bash
contraption validate
contraption compile --output outputs/scanner_demo/online
```

The command resolves the contraption, component-package registry, PMDL registry,
and canonical data-only controller dependency before deriving C99. Inspect the
manifest hashes and compiler diagnostics before using the output in any target
toolchain.
