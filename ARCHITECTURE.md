# Phase 1 architecture contract

This repository is a runnable reference implementation, not a claim of
production certification.  The safety boundary is explicit: DSL and agent
outputs are parsed into data-only intermediate representations and never
executed as Python.

The repository's consumer engineering plan remains the strategic authority.
This stack is an internal harness foundation, and the scanner example is a
pre-product laboratory/3D-evidence fixture rather than the market pilot. Its
simulation results remain unqualified engineering evidence until independent
physical tests establish a named validity envelope.

## Layers

1. `specs`, `units`, `dsl`, `validation`, and `taxonomy` define immutable data
   contracts and the restricted acausal model language.
2. `backend`, `simulator`, and `fitting` assemble descriptor residuals,
   integrate trajectories, propagate distributions, and infer parameters.
   NumPy is the portable backend; PyTorch is selected when installed for GPU
   vectorization and reverse-mode gradients.
3. `controls`, `compiler`, `visualization`, and `build` consume validated IR.
   They never import agent code.
4. `agents` may propose taxonomy and model artifacts into a staging directory.
   Strict validation and an explicit promotion step are required before those
   files can enter a registry.
5. `examples/scanner_robot` is the end-to-end acceptance fixture.

## Shared API expectations

* Specifications serialize as deterministic JSON and reject unknown keys.
* A physical model has the universal residual form
  `F(t, z, zdot, theta, u) = 0`; the DSL expression tree is allow-listed.
* Simulation results expose time, named samples, mean, covariance, quantiles,
  and confidence intervals.
* Online compilation accepts only models proven linear or linearizable over a
  declared validity envelope and emits a fixed-allocation C99 runtime plus an
  extended Kalman filter description.
* Control programs use a separate restricted expression/state-machine format;
  arbitrary Python callbacks are not accepted.
* Tests use `unittest`-compatible assertions so the bundled runtime can execute
  them even when `pytest` is unavailable.
