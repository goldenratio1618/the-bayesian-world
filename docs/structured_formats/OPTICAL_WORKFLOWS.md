# Offline optical capture and reconstruction workflows

This guide is the operational contract for the two optical CLI workflows:

- `contraption simulate --optical-capture` renders one exact resolved assembly
  frame and writes `optical-observation-1` artifacts;
- `contraption optical-reconstruct` verifies and fuses one or more depth
  observations into a `reconstruction-state-1` sparse Bayesian map.

The authoritative implementations are `contraption.cli`,
`contraption.optics.assembly`, and `contraption.optics.workflow`. The
[optical observation](./OPTICAL_OBSERVATION.md) and
[reconstruction state](./RECONSTRUCTION_STATE.md) guides define the emitted
artifact fields. Use a fresh output directory for every evidence-bearing run.

## End-to-end CPU scanner workflow

From the repository root, set `WORLD_SCENE` to an existing strict
`optical-scene-1` whose hash-bound target is in the selected camera frustum.
The scene and every referenced shape must pass their own content verification.
This deterministic run needs no GPU:

~~~bash
WORLD_SCENE=path/to/world.optical-scene.json
test -f "$WORLD_SCENE"

contraption validate \
  --spec assembled_contraptions/scanner/contraption.json

contraption simulate \
  --spec assembled_contraptions/scanner/contraption.json \
  --duration 0.02 \
  --dt 0.01 \
  --samples 1 \
  --seed 71 \
  --backend numpy \
  --controller-input armed=true \
  --output outputs/scanner_optical_capture \
  --optical-capture \
  --optical-scene "$WORLD_SCENE" \
  --optical-sensor camera:scanner.camera.optical \
  --optical-width 32 \
  --optical-height 18 \
  --optical-backend numpy \
  --optical-sample-index 0 \
  --optical-time-index -1 \
  --optical-seed 7
~~~

The final trajectory index is 2 for this duration/step, so the emitted names
are deterministic. Fuse that target-bearing frame with:

~~~bash
contraption optical-reconstruct \
  --sensor outputs/scanner_optical_capture/optical/sensors/camera-scanner.camera.optical.optical.json \
  --observation outputs/scanner_optical_capture/optical/observations/camera-scanner.camera.optical-frame-00000002.optical-observation.json \
  --output outputs/scanner_optical_reconstruction \
  --id scanner-map \
  --voxel-size 0.01 \
  --block-size 8 \
  --origin 0 0 0 \
  --truncation-distance 0.04 \
  --pixel-stride 1
~~~

Use fresh output directories instead of overwriting prior evidence. The
reconstruction output may be absent or an existing empty directory; a nonempty
directory fails closed.

Omitting `--optical-scene` still permits an assembly-only capture, but that
camera view may have no finite target hit. Reconstruction now requires at least
one eligible occupancy/TSDF cell because every successful run emits a canonical
surface; a target-free observation therefore fails instead of producing a
volume-only artifact. The external scene supplies world geometry and is not a
field of `contraption-4`. The scanner's `environment.object_bounding_cube`
remains a mission/verification value and is explicitly not interpreted as
optical geometry.

## Capture option semantics

`--optical-capture` opts the simulation into one optical capture after the
trajectory is complete. Physics and optics have independent backend options.

| Option | Default | Exact behavior |
|---|---|---|
| `--optical-scene PATH` | absent | load and verify one external `optical-scene-1` plus every referenced shape, then merge it with assembly geometry |
| `--optical-sensor ID` | all bound sensors | select a static-part sensor binding by local id or `component:id`; repeat to select several |
| `--optical-width N` | 160 | positive derived sensor width |
| `--optical-height N` | 90 | positive derived sensor height |
| `--optical-backend` | `numpy` | `numpy`, `torch`, or `auto` |
| `--optical-device DEVICE` | backend choice | NumPy permits only CPU; Torch accepts its supported device such as `cpu` or `cuda` |
| `--optical-sample-index N` | 0 | stochastic simulation sample whose poses are used |
| `--optical-time-index N` | -1 | trajectory frame; negative indices count from the end |
| `--optical-seed N` | simulation `--seed` | optical noise seed; sensor number `i` uses `N+i` |

The CLI always applies its width/height pair. It independently scales source
`fx`/`fy` and `cx`/`cy` into a derived descriptor, retaining the source
descriptor hash and recording the acquisition override in metadata. Invalid
sample/time indices, unknown requested sensors, or an assembly with no selected
optical sensor fail.

For explicit CUDA execution, install the GPU extra and replace the backend
arguments with:

~~~text
--backend torch --device cuda --optical-backend torch --optical-device cuda
~~~

The first pair controls PMDL simulation and the second controls optical
rendering. Explicit CUDA fails when unavailable. `auto` may fall back to NumPy
only when a requested device does not require otherwise; use explicit
`torch/cuda` when hardware placement is part of the claim.

## Capture outputs

For `--output RUN` the simulation writes its normal trajectory, physical scene,
and `RUN/report.json` plus:

~~~text
RUN/optical/report.json
RUN/optical/sensors/<binding>.optical.json
RUN/optical/observations/<capture>.optical-observation.json
RUN/optical/observations/<capture>.<product>.npy
~~~

`RUN/optical/report.json` has format `assembly-optical-capture-1` and exactly
indexes:

- assembly id and prefixed assembly SHA-256;
- sample index, resolved nonnegative time index, and time in seconds;
- backend, device, runtime-scene digest, and nullable external-scene digest;
- each binding id, sensor id, configured sensor digest, source sensor digest,
  qualified mount connector, pose digest, and descriptor path;
- each observation id, canonical observation digest, and manifest path.

The top-level simulation report points to this report and repeats the assembly,
backend/device, scene digests, and sensor/observation counts in its metrics.

The capture report is a run receipt with absolute paths, not a relocatable
content-addressed bundle. Moving a run makes those receipt paths stale. Each
observation manifest, however, uses relative content references to its Numpy
sidecars and can be moved together with those sidecars. Retain the complete
assembly/catalog closure, optional external scene and its shape closure,
configured sensor descriptor, observation manifests, and all referenced
sidecars for an evidence or replay claim.

## Capture hash closure and admission

Capture fails closed in this order:

1. `contraption-4`, its catalog, PMDL, controller, verification, and physical
   closure resolve to one assembly SHA-256.
2. The chosen trajectory sample/time reconstructs every body and connector
   pose from that same assembly.
3. Every rendered assembly solid must be `kind: shape`; its static-part
   manifest-file digest must still match, its `shape-artifact-1` content must
   verify, and its selected CTMESH surface must declare `ray_trace`.
4. Each sensor file must still match the static-part
   `descriptor_sha256` and its mount connector must have a resolved pose.
5. An external scene, when supplied, verifies its own digest and every
   referenced shape before being merged. The merged runtime scene receives a
   separate canonical digest.
6. The observation binds the configured sensor digest, runtime-scene digest,
   exact pose, timing, seed, assembly id/digest, qualified mount, pose digest,
   and every output byte count/digest.

There is no box fallback, guessed material, detached camera pose, or unverified
external scene.

## Reconstruction option semantics

| Option | Default | Exact behavior |
|---|---|---|
| `--sensor PATH` | required | load one strict `optical-sensor-1`; repeat for heterogeneous cameras |
| `--observation PATH` | required | load and content-verify one `optical-observation-1`; repeat for multi-view fusion |
| `--output PATH` | required | fresh reconstruction directory |
| `--id ID` | `optical-reconstruction` | nonempty reconstruction identity |
| `--voxel-size M` | 0.01 | positive finite cubic voxel edge in metres |
| `--block-size N` | 8 | integer in [2,64] voxels per block axis |
| `--origin X Y Z` | 0 0 0 | world-space location of voxel index [0,0,0], in metres |
| `--truncation-distance M` | four voxels | finite distance at least one voxel |
| `--pixel-stride N` | 1 | positive image-axis subsampling stride |
| `--surface-occupancy-threshold P` | 0.55 | eligible voxel occupancy probability; finite and in (0,1) |
| `--surface-maximum-abs-tsdf X` | 0.5 | maximum absolute normalized TSDF support; finite and in [0,1] |
| `--surface-maximum-occupied-voxels N` | 250000 | positive fail-closed extraction bound |
| `--surface-maximum-triangles N` | 2000000 | positive fail-closed extraction bound |

Sensors are indexed by their canonical descriptor SHA-256, never by filename
or display id. Every observation must find an exact sensor digest match and
must contain `depth_m`. The loader verifies every observation sidecar before an
update. Optional uncertainty controls depth precision; otherwise sensor depth
noise or half a voxel is used. Optional linear RGB updates surface color.
Invalid, clipped, or nonpositive-uncertainty depth samples are skipped.

All assembly-bound observations in one run must have the same assembly id,
assembly digest, and assembly frame. Assembly-bound and unbound observations
cannot be mixed. Different sensors and mounts are allowed when those closure
values agree. Every observation must have `assembly_frame: world`; other frame
names are rejected because sparse fusion operates in the canonical world frame.

The same workflows are available to trusted host code through
`build_assembly_optical_frame`, `capture_assembly`, `capture_result`,
`reconstruct_capture`, and `reconstruct_observations` from
`contraption.optics`. Their returned `AssemblyOpticalFrame`,
`AssemblyOpticalCapture`, and `ReconstructionArtifact` records preserve the
same hashes and validation boundaries as the CLI. They are host APIs, not DSL
callbacks or Luna tools.

## Reconstruction outputs and closure

The command writes:

| Path | Meaning |
|---|---|
| `reconstruction.state.json` | `reconstruction-state-1` manifest and continuation parameters |
| `block_<bx>_<by>_<bz>.svox` | only allocated, independently hash-bound sparse posterior blocks |
| `canonical.surface.ctmesh` | deterministic canonical CTMESH extracted in the world frame |
| `surface.position-standard-deviation.npy` | little-endian float32 per-vertex position standard deviation in metres |
| `shape.artifact.json` | independently verified unified `shape-artifact-1` containing the surface, sparse volume, uncertainty, material, and local evidence |
| `shape-volume.json` | convenience copy of the artifact's `sparse_tsdf` volume; not the unified model |
| `evidence/sensors/*` | exact sensor descriptor source closure |
| `evidence/observation-*/*` | exact observation manifests and their original NPY sidecars |
| `report.json` | `optical-reconstruction-run-1` receipt indexing every canonical output |

The state records the ordered canonical digest of every fused observation. Its
canonical `artifact_sha256` in the report hashes normalized state JSON; each
block has an independent byte hash in that state. The shape-volume content
reference hashes the written `reconstruction.state.json` file bytes.

The unified shape has id `<reconstruction-id>.shape`, one
`reconstruction-surface` CTMESH with purposes `render`, `ray_trace`, and
`analysis`, and one `reconstruction` sparse-TSDF volume with purposes
`reconstruction` and `ray_trace`. Its `posterior-color` Lambertian material
leaves CTMESH vertex RGBA as the authoritative Gaussian posterior color mean.
The `surface-position-standard-deviation` per-vertex field combines TSDF
posterior variance with voxel quantization variance `voxel_size_m^2 / 12`;
the surface's empirical uncertainty points to that field.

Extraction method `occupancy-tsdf-voxel-boundary-v1` selects updated,
positive-precision cells meeting both surface thresholds. Every exposed voxel
face becomes an outward-wound two-triangle quad with shared grid corners and no
sub-voxel interpolation. The surface declares exact vertex/triangle counts,
world-frame metric bounds, and truthfully computed watertight/manifold flags;
a sparse single-view result need not be watertight or manifold. No eligible
cells or either explicit resource bound being exceeded fails the command.

The shape sources use formats `optical_sensor` and `optical_observation`.
Their provenance preserves canonical sensor, observation, scene, assembly, and
mount hashes without retaining original absolute paths. Loading the shape with
content verification transitively verifies copied observation NPY sidecars and
the reconstruction state plus every SVOX block. Thus scanned and imported
models share the same canonical artifact contract and standard CTMESH consumer
path; no scan-specific render adapter is used.

`report.json` has format `optical-reconstruction-run-1`. Its
`reconstruction_state` object has exactly `path`, `artifact_sha256`,
`format`, `update_count`, `block_count`, and ordered
`observation_sha256`. The `shape_volume`, `shape_surface`, and
`surface_uncertainty` objects each have `path` and full `value`. The
`shape_artifact` object has `path`, `artifact_sha256`, `format`,
`source_count`, `surface_ids`, and `volume_ids`.

The report paths are absolute run receipts. CLI stdout adds one absolute
`report` path to an otherwise identical object. The shape artifact's content
references are relative and relocatable with its complete directory; keep that
directory intact.

## Current limitations

- `optical-reconstruct` is Bayesian occupancy/TSDF/color fusion. It does not
  invoke the differentiable multi-view inverse solver described in
  [Optical observations](./OPTICAL_OBSERVATION.md); that full inverse API is
  currently a host Python workflow.
- Each CLI call creates a new map. The engine can load and continue a state,
  but the CLI has no resume option.
- `pixel_stride` affects exact replay but is not stored in the current state or
  run receipt. Preserve the command/launch receipt when making a reproducibility
  claim.
- The derived CTMESH is a voxel-boundary posterior projection, not exact CAD or
  physical metrology. Preserve the mutable sparse posterior and uncertainty.
- One simulation invocation captures one sample/time frame. Multi-view CLI
  fusion requires observations from multiple fresh captures or a physical
  adapter.
- These workflows run offline on the host. PMDL `controller_stream` reserves a
  future transport contract, but no optical controller/FPGA bridge is present.
- A successful render or reconstruction is model evidence, not physical
  calibration, collision safety, object identification, or device acceptance.

## Luna boundary

This guide is bundled so Luna understands the runtime capabilities and can
model compatible PMDL artifact ports and scalar behavior. It does not authorize
Luna to invoke either workflow, inspect observation content, author optical
evidence or reconstruction artifacts, or replace deterministic host assets.
