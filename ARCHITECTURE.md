# Phase 1 architecture contract

This repository is a runnable engineering reference, not a certification claim.
The consumer engineering plan remains the strategic authority; the scanner is
a pre-product laboratory/3D-evidence fixture rather than the market pilot.

## Canonical closure

The only complete device representation is a `contraption-2` specification
resolved with its component-package, exact-hash PMDL, and data-only controller
registries. Packages own physical bodies, connector frames, model-port and
geometry bindings, and provenance. Resolution rejects stale, missing,
incompatible, underconstrained, or singular dependencies and produces one
assembly closure hash.

Simulation, physical pose generation, visualization, build planning, and C99
derivation consume that `ResolvedAssembly`. They may produce different artifact
formats, but none may add or override the represented object:

```text
contraption-2 + packages + PMDL + controller
                    |
              resolve/validate
                    |
             ResolvedAssembly
          /         |          \
   PMDL simulation  poses      DAE-derived C99
          |          |                |
   runtime checks   viewer          manifest
          \          |                /
             hash-bound artifacts
                    |
                 build plan
```

The viewer is display-only and receives fully resolved body and connector poses.
The build planner accepts only the resolved closure and leaves unknown routing,
ratings, and retention details unresolved. The online compiler derives its
local dynamics/estimator from the assembled descriptor residual; it does not
admit an authored scanner matrix or bare assembled-system projection as an
alternate model. It retains controller provenance but does not yet emit
controller state-machine C99, which remains an explicit external-runtime gate.

## Layers

1. `specs`, `units`, `dsl`, `controls`, `physical`, and `validation` define
   strict inert-data contracts.
2. `resolved` verifies package/model/controller hashes and compiles physical and
   PMDL projections with one identity.
3. `assembly`, `backend`, and `simulator` compose component residuals, integrate
   on NumPy/Torch, and enforce network/attachment invariants.
4. `visualization`, `build`, and `compiler` consume only verified projections
   carrying the canonical closure hash.
5. `agents` may stage inert proposals; deterministic validation and explicit
   promotion are required before they enter a registry.
6. `examples/scanner_robot` is the bounded end-to-end acceptance fixture.

## Shared guarantees

- Unknown fields and duplicate identifiers are errors.
- PMDL has universal residual form `F(t, z, zdot, theta, u) = 0` and an
  allow-listed expression tree; arbitrary generated Python is never executed.
- Connector domain/interface mismatches and incomplete port/binding coverage
  fail before simulation.
- Every accepted timestep preserves assembled network equations and resolved
  physical boundary conditions within declared tolerances.
- Dynamic viewer frames require exact body/connector key coverage and the
  matching assembly hash.
- Generated C99 embeds both the assembly and PMDL hashes and remains subject to
  target/HIL qualification.
- Physical instances remain `unverified` until independent measurement and
  acceptance evidence says otherwise.
