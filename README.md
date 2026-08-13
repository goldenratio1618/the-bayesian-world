# The Bayesian World: Phase 1 engineering harness

This repository combines the project's consumer-first strategy research with a
typed, validation-led physical engineering harness. The
[current project overview](docs/README.md) and
[consumer engineering plan](docs/04_implementation_plan_v2_consumer.md) remain
the strategic authority. The scanner robot here is an intentional pre-product
lab device for exercising the harness and eventually acquiring 3D structure;
it is not the first product-level market prototype.

## Catalog and canonical assembly

`model_catalog/` is organized strictly as physical domain, category, and
optional device layers. PMDL means Physical Model DSL. Every layer declares an abstract `interface.pmdl`
contract in place; there is no separate JSON taxonomy. Concrete PMDL classes
live at the category or device layer and declare which contract they implement.

Physical part instantiations live below the relevant layer in
`instantiations/<part-id>/`. `static.part` owns model-invariant geometry,
connectors, provenance, purchasing information, and metadata. Each `vN.model`
selects an exact-hash PMDL class and initializes all parameters, uncertainty,
condition, and relative compute cost. Multiple competing model instances may
describe the same physical part.

A contraption is a `contraption-4` bundle rooted at `contraption.json`. The
manifest names its catalog roots and hash-binds every controller and verification
artifact. Components contain only an id and model-instantiation id; the manifest
owns topology, open-loop actuator wiring, controller pin wiring, the physical
root, environment, and metadata. It cannot override part parameters or geometry.

`load_contraption(path)` verifies the complete filesystem closure and emits one
`ResolvedAssembly` with an assembly SHA-256. The PMDL simulator, controller
runtimes, verification evaluator, physical pose solver, display-only visualizer,
build planner, and controller compilers consume that object. There is no
independent scanner aggregate model, visualization layout, acceptance sidecar,
or authored online matrix file.

Every `contraption-4` assembly must declare a strict, hash-bound
`metadata.dynamics_completeness` status. Known omitted interactions are named as
open gates; validation rejects a missing or contradictory record. Simulation
reports, verification results, build planning, and generated controller
manifests all carry the same record, so a locally useful model cannot silently
present itself as a complete physical device. A controller observer on an
incomplete assembly must explicitly acknowledge the exact current open-gate
set; a stale, missing, or extra acknowledgement fails resolution.

The implementation fails closed. Stale hashes, incompatible connectors,
unknown models/ports/parameters, missing geometry bindings, underconstrained
kinematics, singular PMDL networks, attachment drift, mismatched pose frames,
and unavailable requested CUDA execution produce actionable errors. Missing
fabrication facts remain explicit build release gates; the software does not
invent wire routes, gauges, fasteners, torque, or physical qualification.

## Linux/WSL quick start

Use the full [Linux installation guide](docs/INSTALLATION.md). In Ubuntu WSL,
the intended checkout is:

```bash
cd ~/src_bayesian/the-bayesian-world
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

On WSL, install or update the NVIDIA driver on **Windows**; do not install a
Linux NVIDIA display driver inside WSL because the CUDA driver is mapped in
from Windows. Follow NVIDIA's [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/),
then select the GPU-enabled wheel from the official
[PyTorch Linux selector](https://pytorch.org/get-started/locally/) and install
this project:

```bash
python -m pip install -e ".[gpu,agents,dev]"
contraption doctor
python scripts/verify_acceleration.py --expect cuda
contraption validate --spec assembled_contraptions/scanner/contraption.json
python -m pytest
```

Run and visualize the scanner prototype:

```bash
contraption simulate \
  --spec assembled_contraptions/scanner/contraption.json \
  --backend torch --device cuda \
  --controller-input armed=true \
  --output outputs/scanner_demo

contraption view \
  --spec assembled_contraptions/scanner/contraption.json \
  --trajectory outputs/scanner_demo/trajectory.json \
  --output outputs/scanner_demo/viewer

python -m http.server 8000 --directory outputs/scanner_demo/viewer
```

Open <http://127.0.0.1:8000>. Left-drag pans, right-drag rotates, and the mouse
wheel zooms. The browser does not simulate or infer placement: every displayed
body and connector pose is reconstructed from the exact saved simulation
samples through the same resolved assembly used by the simulator.

For interactive mission controls, let the Python process own simulation:

```bash
contraption serve \
  --spec assembled_contraptions/scanner/contraption.json \
  --backend torch --device cuda \
  --controller-input armed=true \
  --host 127.0.0.1 --port 8000
```

The live page obtains its input schema from the exact-hash data-only controllers.
Changing a control POSTs to the loopback Python server, which reruns the same
`ResolvedAssembly` and returns a fully validated, hash-bound physical scene.
JavaScript only validates and displays that response; it contains no dynamics
or kinematics. The offline viewer remains network-disabled by Content Security
Policy.

Generate the other closure-bound artifacts separately:

```bash
contraption compile --spec assembled_contraptions/scanner/contraption.json --output outputs/scanner_demo/online
contraption build --spec assembled_contraptions/scanner/contraption.json --output outputs/scanner_demo/build
```

`contraption demo --spec assembled_contraptions/scanner/contraption.json
--controller-input armed=true` runs simulation, viewer generation, C99/Verilog
controller generation, and build planning together. External controller pins
may instead be collected in a strict JSON object passed with
`--controller-input-file`. The physics `--dt` may be any common subdivision of
all controller periods; when omitted it is derived from their authored decimal
periods. Simulation is engineering evidence only; it does not
authorize construction, deployment, or promotion of any physical evidence
tier.

## Controller execution and compilation

The offline PMDL simulator solves the full assembled descriptor system and runs
each exact-hash `control-1` program inside that simulation. A controller receives
only its explicit sensor wires and external pins. It never receives the offline
simulator state. Each implicit input is separately bound by `contraption.json`
to an exact namespaced PMDL variable. The loader derives one coupled local
affine observer from the complete assembled PMDL descriptor system at a
hash-bound operating point; authors do not supply estimator matrices or hidden
state values. The runtime and generated targets carry the same vector state, a
numerically qualified discrete transition, Joseph-form covariance update, and
PMDL projections. Controller expressions can
use each posterior `mean`, `variance`, or `std`; locally unobservable latents are
reported in diagnostics and emit a runtime warning.

The explicit/implicit split is also a construction boundary. Explicit sensor
inputs correspond to physical wires or pins. Implicit values have no wire: they
exist only as posterior quantities inferred from the wired measurements and the
qualified plant model. Any PMDL actuator input that can influence an observer's
state or projections must be owned by that controller or explicitly represented
at its boundary; an input is never declared irrelevant merely because its first
derivative happens to be zero at the operating point.

The current resolver does not yet lower another controller's command echo as a
known exogenous observer input. Coupled multi-controller plants therefore fail
closed unless each observer owns every plant input that can reach its state or
projections. Observer-free multiple controllers and structurally independent
observer closures remain supported; weakening the hidden-input check is not an
accepted workaround.

Controllers have authored fixed periods. The offline simulator derives or
validates a common physics subdivision, ticks heterogeneous controllers on
integer strides, and holds each output between its own ticks. External pin
providers are sampled only when their controller ticks. Descriptor algebraics
are made consistent before the first sensor sample and reconciled after the
initial command.

The separate simulator `controls=` argument is strictly open-loop: values may be
constants, time series, or callables of time (and optional backend), but never
receive plant state. All feedback belongs in a resolved controller program.

PMDL process uncertainty is also declarative. A model names standard-normal
channels and unit-checked increments for differential states. On every accepted
integration step the simulator draws one canonical channel vector, evaluates
the increments as backend-native expressions, applies them to differential
state, and reconciles descriptor algebraics before publishing the sample.
Seeded replay is guaranteed for the same backend, device, and dtype; NumPy and
Torch are not required to produce identical random streams. Disabling process
noise consumes no draws.

`contraption compile` translates the same controller state machines executed in
silico into allocation-free C99 and synthesizable fixed-point Verilog. Both
targets include modes, transitions, registers, the PMDL-derived observer,
posterior uncertainty, output bounds, slew limiting, emergency behavior, input
validation, and explicit fault reporting. Only explicit inputs become hardware
ports. A controller with implicit inputs can compile only from its resolved
assembly, binding generated files to the assembly, PMDL, controller-link, and
observer digests. Compilation fails when a target lacks a qualified lowering:
for example, smooth transcendental control expressions are supported by the
Python/Torch runtime and C99, while generic synthesizable Verilog rejects them
until a fixed-point algebraic or lookup-table implementation is supplied.

Nonlinear *plant* PMDL is a separate case. Offline NumPy/Torch simulation
executes allow-listed terms such as `tanh` directly. An online observer may use
an explicitly approved local affine linearization of that nonlinear descriptor
system; its manifest records the nonlinear relations, operating point,
derivation, PMDL validity region, open gates, and approximation status. The
controller also declares a relative perturbation radius and maximum sampled
residual remainder; resolution measures coupled deterministic perturbations
and enforces that local remainder. This is bounded local evidence, not an exact
nonlinear hardware simulation or a global error proof.

The verification DSL evaluates every posterior trajectory on its exact time
grid. Mean and RMSE reducers use trapezoidal physical-time weighting, and a
criterion is admitted from a conservative Wilson lower confidence bound rather
than a raw pass fraction. Numeric PMDL/control/verification paths remain
backend-native and are classified as smooth or piecewise-smooth. Typed Boolean
guards, mode choices, criteria, counts, and confidence decisions are the
intentional discrete boundary; gradients are not claimed through those
decisions.

See [the controller compiler contract](docs/ONLINE_COMPILER.md).

## Guarded component agents

Agent operations share a hard `$100` lifetime ledger under `outputs/` and never
promote their own results:

```bash
contraption budget
contraption agent-canary --kind both --env-file ../.env
contraption agent-run classification-all --job-file assembled_contraptions/scanner/agent_jobs.json --env-file ../.env
contraption agent-run modeling-one --job-file assembled_contraptions/scanner/agent_jobs.json --target romi_drive --env-file ../.env
```

The modeling agent writes complete inert catalog bundles: any required
interfaces, `static.part`, and at least `v1.model`. It reuses an exact existing
PMDL class/hash when its physics fits, and adds a concrete PMDL only when needed. It may
call the dedicated bundle validator repeatedly for path, interface, PMDL,
parameter, physical, and property feedback. Protected inputs are hash-checked
before and after a run. PMDL math uses a portable allow-list; arbitrary Python
imports are not executed. The parent `../.env`, agent receipts, staging data,
and budget ledger are machine-local and gitignored.

## Design records and limits

- [Phase 1 scope and artifact contract](docs/PHASE1.md)
- [Online C99 compiler contract](docs/ONLINE_COMPILER.md)
- [Component-agent safety contract](docs/COMPONENT_AGENTS.md)
- [Linux/GPU installation](docs/INSTALLATION.md)
- [Purchasing and physical safety guidance](docs/PURCHASING.md)

The current mechanics are deliberately bounded: planar chassis motion plus a
declared rigid attachment/joint tree, not general 3D contact, flexible bodies,
fracture, fluids, thermal behavior, or optical physics. Every physical instance
starts `unverified`; measured acceptance and independent safety controls remain
mandatory.

The scanner is explicitly dynamics-incomplete. It models a published 0.160 kg
bare chassis and component-local dynamics without hiding payload mass in a
whole-robot override. Fixed-payload inertia, moving arm/camera inertial
derivation, servo case reactions, caster contact, full-body keep-out,
controller/sensor observation fidelity, and supply/fault coupling remain seven
named open gates. Consequently its overall `accepted` result remains false even
when the currently modeled orbit and pointing checks pass.
