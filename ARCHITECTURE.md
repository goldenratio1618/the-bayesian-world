# Architecture contract

This repository is a runnable engineering reference, not a certification
claim. The scanner is a pre-product laboratory fixture rather than a product
taxonomy or a physical domain.

## Model catalog

`model_catalog/` is the only model and part catalog. Its filesystem hierarchy
is semantic and validated:

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

Every top-level directory declares one or more kinds of physics through an
abstract domain `interface.pmdl`. Categories describe general object classes;
the optional device layer describes a more specific device type. Concrete PMDL
classes declare the category or device contract they implement and must live at
that layer. There is no parallel JSON taxonomy.

`static.part` owns model-invariant geometry, connector frames, provenance,
purchasing information, and metadata. Each `vN.model` selects an exact
id/version/hash PMDL class, initializes all of its parameters, declares
parameter uncertainty and condition, and records relative compute cost. Several
model files may coexist for one physical part so inference can compare their
likelihood and cost.

## Canonical closure

The only complete contraption source is a `contraption-3` document. Components
contain exactly an id and a catalog model-instantiation id. Parameters and
geometry cannot be authored in a contraption. Connections, control bindings,
the physical root, controller reference, environment, and metadata complete
the source.

```text
contraption-3 + model_catalog + controller
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

Resolution verifies every interface, catalog path, PMDL hash, initialized
parameter, physical binding, and controller hash before producing one assembly
closure hash. Simulation, physical poses, visualization, build planning, and
C99 derivation consume that `ResolvedAssembly`; none may add or override the
represented object.

## Source modules

- `contraption.physics` owns PMDL parsing, validation, assembly, simulation,
  fitting, uncertainty, physical resolution, and the electrical/mechanical
  reference systems.
- `contraption.catalog` owns abstract interface discovery and model
  instantiations.
- `contraption.part_import` owns classification, guarded modeling, budgets, and
  deterministic candidate validation.
- `contraption.visualization` owns the display-only viewer, scanner scene
  projection, and loopback live server.
- `contraption.manufacturing` owns closure-bound build instructions.
- `contraption.applications` owns scanner-specific runtime policy.

## Shared guarantees

- Unknown fields, duplicate identifiers, invalid catalog layers, and stale
  hashes are errors.
- PMDL has universal residual form `F(t, z, zdot, theta, u) = 0` and an
  allow-listed expression tree; generated host code is never executed.
- Connector and interface mismatches, incomplete parameter initialization, and
  geometry/parameter disagreement fail before simulation.
- Every accepted timestep preserves assembled network equations and physical
  boundary conditions within declared tolerances.
- Viewer frames and generated C99 remain bound to the exact assembly closure.
- Model instances remain `unverified` until independent evidence changes their
  declared condition.
