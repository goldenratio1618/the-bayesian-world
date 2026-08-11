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
optional device layers. Every layer declares an abstract `interface.pmdl`
contract in place; there is no separate JSON taxonomy. Concrete PMDL classes
live at the category or device layer and declare which contract they implement.

Physical part instantiations live below the relevant layer in
`instantiations/<part-id>/`. `static.part` owns model-invariant geometry,
connectors, provenance, purchasing information, and metadata. Each `vN.model`
selects an exact-hash PMDL class and initializes all parameters, uncertainty,
condition, and relative compute cost. Multiple competing model instances may
describe the same physical part.

A contraption is a `contraption-3` document whose components contain only an id
and a model-instantiation id. It owns connectivity, controls, the physical root,
environment, and metadata; it cannot override part parameters or geometry.

Resolution verifies that complete closure and emits one assembly SHA-256. The
PMDL simulator, physical pose solver, display-only visualizer, build planner,
and C99 compiler all consume that resolved assembly. There is no independent
scanner aggregate model, visualization layout, simulation-coverage sidecar, or
authored online matrix file.

Every `contraption-3` assembly must declare a strict, hash-bound
`metadata.dynamics_completeness` status. Known omitted interactions are named as
open gates; validation rejects a missing or contradictory record. Simulation
reports, acceptance metrics, build planning, and C99 artifacts all carry the
same record, so a locally useful model cannot silently present itself as a
complete physical device.

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
contraption validate
python -m pytest
```

Run and visualize the scanner prototype:

```bash
contraption simulate \
  --backend torch --device cuda \
  --output outputs/scanner_demo

contraption view \
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
  --backend torch --device cuda \
  --host 127.0.0.1 --port 8000
```

The live page obtains its input schema from the exact-hash data-only controller.
Changing a control POSTs to the loopback Python server, which reruns the same
`ResolvedAssembly` and returns a fully validated, hash-bound physical scene.
JavaScript only validates and displays that response; it contains no dynamics
or kinematics. The offline viewer remains network-disabled by Content Security
Policy.

Generate the other closure-bound artifacts separately:

```bash
contraption compile --output outputs/scanner_demo/online
contraption build --output outputs/scanner_demo/build
```

`contraption demo` runs simulation, viewer generation, C99 derivation, and
build planning together. Simulation is engineering evidence only; it does not
authorize construction, deployment, or promotion of any physical evidence
tier.

## What the C99 is for

The offline PMDL simulator can spend more compute solving the full assembled
descriptor system. `contraption compile` derives a local, fixed-allocation C99
dynamics/estimator module from that same verified DAE closure for constrained
onboard hardware. It accepts only `ResolvedAssembly`, records the assembly,
PMDL, and controller hashes, and refuses caller-authored matrices or partial
assembly projections. The lower-power module is an approximation of the same
representation, not a second physical object specification.

Phase 1 does **not** translate the declarative controller state machine into
C99. The generated module exposes the resolved control-source inputs; a
separately qualified runtime must execute the exact hash-bound `ControlProgram`
and supply those inputs. Source comments and the manifest mark controller
execution as not emitted so this artifact cannot be mistaken for complete
firmware.

## Guarded component agents

Agent operations share a hard `$100` lifetime ledger under `outputs/` and never
promote their own results:

```bash
contraption budget
contraption agent-canary --kind both --env-file ../.env
contraption agent-run classification-all --env-file ../.env
contraption agent-run modeling-one --target romi_drive --env-file ../.env
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
