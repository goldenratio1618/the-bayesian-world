# The Bayesian World: Phase 1 engineering harness

This repository combines the project's consumer-first strategy research with a
modular Phase 1 implementation of its internal engineering harness. The current
code provides probabilistic, differentiable electrical and planar
rigid-mechanical digital twins; the restricted PMDL physical-model language;
strict contraption and controller contracts; uncertainty propagation, fitting,
and approximate model selection; guarded component-ingestion agents; a
fixed-allocation C99/EKF compiler path; an offline viewer; and a deterministic
build planner.

The [current project overview](docs/README.md) and
[consumer engineering plan](docs/04_implementation_plan_v2_consumer.md) are the
strategic design authority. The scanner robot is intentionally a pre-product
lab device and end-to-end acceptance fixture: it exercises motion modeling,
controls, visualization, compilation, and build instructions while supporting
future acquisition of 3D structure and physical evidence. It is not the first
product-level market pilot. The plan's guarded card-workstation family remains
that pilot.

The implementation fails closed. Unknown models, parameters, controls,
components, ports, incompatible domains or units, ambiguous environment files,
unsupported active PMDL semantics, stale compiler coverage, unsafe agent
artifacts, and explicit-but-unavailable CUDA requests produce actionable
errors. Missing fabrication facts such as verified wire gauge or fastener
torque remain visible safety-gate items; they are never invented.

## Install and verify on Linux

Follow [the Linux installation guide](docs/INSTALLATION.md). It covers common
distribution packages, an isolated Python environment, GPU-first PyTorch
installation, CPU fallback, agent dependencies, and verification.

After activation, the main checks are:

```bash
contraption doctor
contraption validate
python -m pytest
contraption demo --backend torch --device cuda --output outputs/scanner_demo
```

The full scanner demo writes a probabilistic trajectory and acceptance report,
an offline 3D/electrical viewer, a reviewed online-model manifest and generated
C99, and a BOM/assembly plan. Its current motion simulator and online IR are
explicit, reviewed aggregate abstractions bound to the complete contraption by
an exact-hash coverage contract; they are not silently presented as automatic
PMDL network assembly.

## Guarded component agents

Agent operations share a hard `$100` lifetime ledger under `outputs/` and never
promote their own results:

```bash
contraption budget
contraption agent-canary --kind classification
contraption agent-run classification-all
contraption agent-run modeling-one --target romi_drive
```

The CLI reads only `OPENAI_API_KEY`. It accepts `--env-file`; otherwise it
discovers exactly one `.env` in the repository or its parent directory and
rejects an ambiguous two-file setup. Generated receipts, staging workspaces,
logs, and the budget ledger are local runtime state and are gitignored.

The modeling harness drafts only in an isolated candidate directory and gives
the model a dedicated validator with deterministic parser, symbol, unit,
balance, and property diagnostics. Protected inputs are hash-checked before and
after the run. Successful files remain staged until an explicit, independently
revalidated promotion. PMDL math is a portable allow-list, including
dimensionless `tanh`; arbitrary Python imports are never executed.

## Design records and limits

- [Phase 1 architecture](ARCHITECTURE.md) explains module and trust boundaries.
- [Phase 1 scope](docs/PHASE1.md) maps this implementation to the wider plan and
  records deliberate limitations.
- [Component agents](docs/COMPONENT_AGENTS.md) documents validation, staging,
  promotion, resume behavior, and budget accounting.
- [Online compiler](docs/ONLINE_COMPILER.md) distinguishes generated C99/EKF
  functionality from future PMDL network assembly and FPGA work.
- [Purchasing guidance](docs/PURCHASING.md) lists apartment-suitable parts,
  tools, safety gates, and alternatives. The software purchases nothing.

Optical physics, general 3D contact, flexible bodies, exact Bayesian evidence,
automatic PMDL network/DAE reduction, regulatory certification, and
hardware-in-the-loop qualification remain outside this Phase 1 implementation.
Every physical instantiation starts `unverified`; simulation does not promote
its evidence tier or authorize a physical release.
