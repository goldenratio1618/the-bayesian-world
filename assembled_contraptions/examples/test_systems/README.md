# Declarative test-system bundles

Each child directory in this folder is a complete, loadable contraption fixture,
not a Python dynamics implementation or a placeholder. The only public entry
point is `<system>/contraption.json`:

```python
from contraption import load_contraption

resolved = load_contraption(path / "contraption.json")
system = resolved.system
```

The manifest uses `contraption-4`. It identifies contained catalog roots with
`catalogs`, selects exact model-instance IDs in `components`, declares assembly
topology in `physical_root` and `connections`, and binds open-loop plant inputs
with `actuators`. A controller belongs in `controllers` only when the fixture is
actually closed-loop; actuator bindings must not be represented as fake
controllers.

Each contained catalog uses the normal catalog layout: domain and category
`interface.pmdl` contracts, concrete `*.pmdl` models, and
`instantiations/<part>/{static.part,vN.model}`. Every model instance initializes
all PMDL parameters and pins the canonical model hash. Verification programs are
`verification-1` artifacts whose canonical hash and output-signal bindings are
declared by the manifest. Verification inputs bind PMDL output signals, so a
model must expose any internal state or applied actuator value it intends to
verify as an explicit diagnostic output.

PMDL stochastic dynamics use a model-level `process_noise` block. Its named
`standard_normal` channels drive unit-checked accepted-step `increments` that
must explicitly reference `dt` and may target differential states only. After
an increment, descriptor algebraics are reconciled before outputs, invariants,
controllers, or verification observe the state. `simulate(seed=...)` provides
the process-noise seed stream; exact replay is promised for an unchanged PMDL
closure, time grid, sample count, backend, device, dtype, and seed. NumPy and
Torch do not promise identical draws to one another. The planar rigid-body
fixture demonstrates `sqrt(dt)` diffusion and a physical reference length for
converting translational roughness to yaw roughness.

Bundles must:

- load through the same device-independent contraption loader as production
  assemblies;
- use only validated, data-only DSL artifacts with explicit units, bounds, and
  content hashes;
- expose controller and verification inputs through declared signal endpoints
  rather than simulator-only state access;
- include verification criteria that can be evaluated from simulation
  results; and
- remain self-contained or declare every external catalog dependency
  explicitly.

Add a system directory only when its complete bundle and physics-based tests
are present and passing.
