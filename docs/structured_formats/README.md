# Structured formats and DSLs

This directory is the human- and Luna-readable contract for Contraption's
authoritative authoring formats and the standardized shape/optical
interoperability records that humans or Luna must understand. Runtime parsers
and validators remain normative; these guides explain the same contracts,
units, semantics, physics boundaries, examples, and validation rules.

Internal derived operational records such as trajectories, physical/live
scenes, build outputs, and general simulation reports remain normative in code.
They are not Luna authoring surfaces and are outside this format index.

For exact assembly-capture and reconstruction commands, see
[Offline optical workflows](./OPTICAL_WORKFLOWS.md).

## Format index

| Guide | Format or payload | Primary owner |
|---|---|---|
| [PMDL](./PMDL.md) | `pmdl-1` | human or Luna model author |
| [PMDL interfaces](./PMDL_INTERFACES.md) | `pmdl-interface-1` | catalog taxonomy/contract author |
| [Static part](./STATIC_PART.md) | `static-part-2` | catalog author plus deterministic shape importer |
| [Fabrication](./FABRICATION.md) | connector fabrication constraints and connection implementations | catalog/assembly author plus deterministic importer |
| [Model instance](./MODEL_INSTANCE.md) | `model-instance-1` | catalog author |
| [Procurement](./PROCUREMENT.md) | `procurement-record-1` | deterministic identity/procurement importer or evidence-backed catalog author |
| [Contraption](./CONTRAPTION.md) | `contraption-4` | assembly author |
| [Control](./CONTROL.md) | `control-1` | controller author |
| [Verification](./VERIFICATION.md) | `verification-1` | acceptance-test author |
| [Deterministic ingestion](./DETERMINISTIC_INGESTION.md) | `deterministic-part-ingestion-1` and staged v1 | host part-import harness |
| [Triangle mesh](./TRIANGLE_MESH.md) | CTMESH and inline triangle mesh v1 | deterministic shape pipeline |
| [Shape artifact](./SHAPE_ARTIFACT.md) | `shape-artifact-1` | deterministic shape pipeline |
| [Optical material](./OPTICAL_MATERIAL.md) | `optical-material-1` | deterministic optical-property pipeline |
| [Optical sensor](./OPTICAL_SENSOR.md) | `optical-sensor-1` | deterministic sensor ingestion/calibration |
| [Optical scene](./OPTICAL_SCENE.md) | `optical-scene-1` | optical runtime/scene compiler |
| [Optical observation](./OPTICAL_OBSERVATION.md) | `optical-observation-1` | optical runtime or physical sensor adapter |
| [Reconstruction state](./RECONSTRUCTION_STATE.md) | `reconstruction-state-1` | Bayesian reconstruction engine |
| [Optical CLI workflows](./OPTICAL_WORKFLOWS.md) | assembly capture and reconstruction run reports | host operator |
| [Offline render bundle](./RENDER_BUNDLE.md) | `contraption.render-bundle/v1` | deterministic viewer materializer |

## Common rules

Unless a guide explicitly says otherwise:

- Documents are UTF-8 JSON objects. Duplicate keys, unknown fields, non-finite
  numbers, executable hooks, and implicit unit or axis conventions are invalid.
- Identifiers are machine identities, not display labels. Preserve the exact
  identifier grammar stated by the relevant guide.
- Scalar physical values use declared units. Geometry and poses use SI metres,
  kilograms, seconds, radians, and a right-handed coordinate system.
- Quaternions are ordered WXYZ, normalized, and sign-canonicalized.
- Hash syntax is format-specific: PMDL/artifact links use `sha256:` followed by
  64 lowercase hexadecimal digits, while shape and optical content references
  use the 64 hexadecimal digits without a prefix. A reference binds exact bytes,
  not merely a filename or semantic version.
- `metadata` is inert JSON. It may record information but cannot introduce
  physics, ports, validation exceptions, executable behavior, or hidden defaults.
  Fabrication semantics belong in typed connector/connection fields; purchasing,
  supplier, lifecycle, and product identity belong in `.procurement` records.
- Provenance and uncertainty are capabilities, not decorations. A model may use
  a representation only for operations its quality/capability declaration supports.
- Absence is explicit. A renderer, solver, or importer must not replace an
  unavailable exact representation with an undocumented box, guessed material,
  zero uncertainty, or default calibration.

## Authored and deterministic boundaries

PMDL, interfaces, static parts, model instances, contraptions, controls, and
verification programs are authored declarative records. Luna may propose only
the catalog formats that the modeling harness permits, subject to deterministic
validation. Procurement records are host-owned evidence projections rather than
dynamic facts embedded in physical parts.

Source geometry and optical evidence never pass through Luna. The protected
host declaration and staging contract is defined in
[Deterministic ingestion](./DETERMINISTIC_INGESTION.md). Supported meshes,
textures, images, point clouds, scan frames, material tables, and camera
calibration are ingested by explicit deterministic host code. The current
built-in optical-property extractor parses OBJ/MTL plus strict optical
sidecars; other CAD and GLB/glTF geometry or material properties require an
explicit deterministic tessellator/property importer. The host owns
byte hashing, coordinate/unit normalization, topology diagnostics, derived
surfaces and volumes, optical property extraction, and schema validation. It
then bundles those validated artifacts with Luna's independently validated
catalog proposal.

Luna may:

- model optical power, scalar signals, state, noise, timing, validity, and
  artifact-stream ports in PMDL;
- use documented optical abstractions when selecting or creating a physical model;
- preserve an opaque, host-supplied exact identifier or hash reference where a
  catalog schema explicitly permits it.

Luna must not:

- open or interpret a CAD, mesh, image, texture, point-cloud, or scan payload;
- create or alter `shape-artifact-1`, optical material/sensor/scene/observation,
  reconstruction, CTMESH, GLB, acceleration, or derived mass-property payloads;
- infer an optical constant, calibration, geometry, scale, axis transform, or
  content hash;
- fabricate a reference to a deterministic artifact.

## Validation order

A complete closure is validated in this order:

1. Parse each record strictly and reject duplicate/unknown fields.
2. Validate identifiers, finite numbers, units, enum values, shapes, and hashes.
3. Resolve catalog/interface ancestry and exact PMDL identity.
4. Validate ports, expressions, dimensions, initialization, and equation balance.
5. Resolve physical bodies, connectors, shape references, and assemblies.
6. Validate fabrication constraints and selected connection implementations.
7. Validate procurement provisions against exact static-part versions/hashes.
8. Resolve optical material, sensor, scene, and observation capabilities.
9. Validate control and verification bindings against the resolved closure.
10. Admit a derived artifact only when every exact dependency and capability is present.

Use `contraption validate` for the complete project closure and the dedicated
part-import validator for a staged catalog bundle. Passing syntax alone never
claims physical or empirical qualification.
