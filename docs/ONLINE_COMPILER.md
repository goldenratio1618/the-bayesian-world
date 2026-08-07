# Online compiler contract

The Phase 1 online compiler emits a fixed-allocation C99 estimator from an
already assembled affine or reviewed linearized model. It does **not** assemble
component PMDL residuals, reduce a DAE, prove a residual Jacobian nonsingular,
or compile the declarative `ControlProgram`. Those remain explicit upstream or
future compiler stages.

## Required assembly coverage

Every assembled IR used with `compile_contraption` must contain
`metadata.assembly_coverage` with schema
`contraption.online-assembly-coverage/v1`. The object lists the exact
`component_ids` and `connection_ids` represented, maps every component instance
to its exact model reference in `component_models`, and carries a
`topology_sha256`. The digest binds component/model pairs and every connection
identifier, kind, declared domain, and endpoint. Missing entries, extra entries,
model substitutions, endpoint changes, malformed endpoints, unknown component
references, duplicate identifiers, and stale topology digests are hard errors.

An `admitted_models` string list, a component-local admission flag, or an
unstructured `review_status` string is not an admission contract.

## Two explicit admission paths

1. **Validated model registry.** Supply a complete mapping from every referenced
   model ID to a full PMDL `ModelSpec`. The compiler validates each PMDL model,
   requires its affirmative structured `online_admission`, and runs contraption
   compatibility validation against that registry. Unknown power/signal ports,
   unit/domain mismatches, unknown parameter overrides, and invalid signal
   direction fail compilation. Only this path reports that all referenced models
   were registry-validated.

2. **Reviewed abstraction boundary.** When full component PMDL contracts are not
   available, `assembly_coverage.review` must identify the review, reviewer, and
   evidence basis; affirm review of component contracts, ports/connections, and
   IR coverage; and list non-empty limitations. This is an explicit trust
   boundary, not equivalent to PMDL validation. The generated manifest labels it
   `reviewed_abstraction` and states that referenced PMDL models were not
   registry-validated.

The scanner fixture uses the second path. Its matrices are a hand-derived,
reviewed aggregate `DifferentialDriveArmModel` linearization tied to the vendor
manifest and exact scanner topology. It is not a network assembled from the
bundled PMDL library, and its generated C99 does not include the scanner
`ControlProgram`; both limitations are recorded in the fixture and output
manifest.

## Numeric subset and generated runtime

The accepted numeric IR has fixed state, input, and measurement dimensions;
finite affine matrices and biases; positive-semidefinite process covariance;
positive-definite measurement covariance; bounded positive timesteps; and, for
a `linearized` IR, a declared operating point. Flexible/non-rigid mechanics are
rejected when declared by a validated model admission.

Generated C99 uses preallocated arrays, explicit dimensions, forward-Euler
prediction, a fixed-matrix Kalman measurement update, Joseph-form covariance
updates, innovation gating, and state clamps. The manifest records state/input
ordering, matrices, numeric assumptions, validity bounds, the topology digest,
and the admission level. Optional host compiler checking verifies C99 syntax;
target execution, timing, numerical equivalence, and hardware-in-the-loop tests
remain release gates.

Floating-point C is the Phase 1 portable target. A future FPGA flow must add
range analysis, fixed-point quantization, overflow policy, pipeline scheduling,
synthesis, timing closure, and hardware-in-the-loop verification.
