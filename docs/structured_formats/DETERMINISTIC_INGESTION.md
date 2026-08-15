# Deterministic part ingestion (`deterministic-part-ingestion-1`)

This host-owned contract declares source geometry and calibrated optical-sensor
descriptors that must be bundled with a Luna-authored catalog proposal without
exposing source bytes to Luna. It has two related forms:

- `deterministic-part-ingestion-1` is authored inside the component-information
  JSON consumed by the host;
- `deterministic-part-ingestion-staged-1` is an internal, protected snapshot
  produced before Luna dispatch.

The authoritative implementation is
`contraption.part_import.deterministic_assets`. This format is orchestration
data, not PMDL and not a shape representation.

## Component-information plan

The optional component-information field `deterministic_ingestion` has exactly:

| Field | Required | Meaning |
|---|---:|---|
| `format` | yes | exactly `deterministic-part-ingestion-1` |
| `shapes` | no | shape-import array; default empty |
| `optical_sensors` | no | optical-sensor binding array; default empty |

Each shape item allows only:

| Field | Required/default | Semantics |
|---|---|---|
| `source` | required | safe POSIX path relative to the component-information directory |
| `catalog_directory` | required | safe catalog-relative instantiation directory below the candidate root |
| `body` | required | target body identifier in Luna's final `static.part` |
| `solid` | required | target solid identifier in that body |
| `artifact_id` | required | output `shape-artifact-1` id |
| `version` | `1.0.0` | nonempty artifact version |
| `metres_per_source_unit` | required | positive finite unit conversion |
| `density_kg_m3` | absent | optional positive finite density for mass properties |
| `surface_uncertainty` | absent | optional strict `ShapeUncertainty` for the canonical surface |
| `provenance` | `{}` | inert JSON merged into deterministic import provenance |

Each optical-sensor item allows exactly these required fields:

| Field | Semantics |
|---|---|
| `source` | safe POSIX path, relative to component information, containing one strict `optical-sensor-1` descriptor |
| `catalog_directory` | safe catalog-relative target instantiation directory |
| `body` | body identifier for the emitted static-part sensor binding |
| `pose_connector` | optical connector identifier; must exactly equal the descriptor's `mount_connector` |
| `artifact_port` | PMDL output artifact-port identifier for the observation stream |

Identifiers match `^[A-Za-z][A-Za-z0-9_.-]*$`. Paths must be nonempty POSIX
relative paths with no backslash, NUL, absolute root, `.`, or `..` component.
Sources must be existing regular non-symlink files. At least one of `shapes` or
`optical_sensors` must be nonempty. Shape target triples
`(catalog_directory, body, solid)` are unique case-insensitively. At most one
deterministic optical-sensor descriptor may target an instantiation
`catalog_directory`, also compared case-insensitively.

When present, `surface_uncertainty` allows exactly `distribution`,
`parameters`, and optional `correlation_group`, with the values defined in
[Shape artifact](./SHAPE_ARTIFACT.md). It is strict-parsed before staging,
preserved in the protected plan, and passed unchanged to the canonical surface
and any density-derived mass properties. When absent, `import_shape` emits a conservative uniform position interval
of +/- `max(1e-7 m, 0.005 * maximum_model_extent_m)`; the interval and its
basis are recorded in the artifact.

The host parses each sensor descriptor before dispatch, derives its `sensor_id`,
and rejects an invalid schema or a mount mismatch. Unknown fields, non-list
arrays, non-finite values, an empty overall plan, and other format versions fail.

## Source closure

The protected input closure always includes the primary source. For OBJ, every
space-separated `mtllib` target is read from the same directory; traversal,
missing files, and linked directories are rejected. The first existing optical
sidecar among `<source-filename>.optical.json` and
`<source-stem>.optical.json` is also included. Duplicate paths are removed
without changing order.

Each optical-sensor item contributes exactly its descriptor file to the closure.

This closure is determined by host code. Luna neither sees nor interprets these
source files. It receives the textual component-information record and the
structured-format guides so it can create matching body/solid, connector, and
PMDL artifact-port targets.

## Example

~~~json
{
  "manufacturer": "fixture",
  "deterministic_ingestion": {
    "format": "deterministic-part-ingestion-1",
    "shapes": [
      {
        "source": "assets/optical_cube.obj",
        "catalog_directory": "optical/cameras/powered_rotational_cameras/instantiations/example_camera",
        "body": "camera",
        "solid": "module",
        "artifact_id": "example.camera-module",
        "version": "1.0.0",
        "metres_per_source_unit": 0.001,
        "density_kg_m3": 2500.0,
        "surface_uncertainty": {
          "distribution": "normal",
          "parameters": {"standard_deviation_m": 0.0002},
          "correlation_group": "fixture-metrology"
        },
        "provenance": {
          "kind": "vendor",
          "reference": "vendor CAD archive"
        }
      }
    ],
    "optical_sensors": [
      {
        "source": "assets/example-camera.optical.json",
        "catalog_directory": "optical/cameras/powered_rotational_cameras/instantiations/example_camera",
        "body": "camera",
        "pose_connector": "optical_axis",
        "artifact_port": "optical_observation"
      }
    ]
  }
}
~~~

The proposed `static.part` must contain body `camera` and solid `module`.
Their initial geometry may follow the modeling contract, but the host replaces
that solid's geometry with the imported exact shape binding before generic
validation.

The same part must expose connector `optical_axis` on body `camera` and PMDL
output artifact port `optical_observation`. The source descriptor must declare
`mount_connector: optical_axis`; its id is host-derived, not repeated in the plan.

## Staged plan

Before Luna dispatch, `stage_plan` creates a new protected directory, one
`shape-NNN/` subdirectory per shape and one `optical-sensor-NNN/` subdirectory
per sensor, copies the corresponding exact source closures, and writes
`plan.json`:

~~~json
{
  "format": "deterministic-part-ingestion-staged-1",
  "shapes": [
    {
      "source": "shape-000/optical_cube.obj",
      "catalog_directory": "optical/cameras/powered_rotational_cameras/instantiations/example_camera",
      "body": "camera",
      "solid": "module",
      "artifact_id": "example.camera-module",
      "version": "1.0.0",
      "metres_per_source_unit": 0.001,
      "density_kg_m3": 2500.0,
      "surface_uncertainty": {
        "distribution": "normal",
        "parameters": {"standard_deviation_m": 0.0002},
        "correlation_group": "fixture-metrology"
      },
      "provenance": {"kind": "vendor"},
      "input_sha256": {
        "optical_cube.obj": "64-lowercase-hex-digest",
        "optical_cube.mtl": "64-lowercase-hex-digest"
      }
    }
  ],
  "optical_sensors": [
    {
      "source": "optical-sensor-000/example-camera.optical.json",
      "catalog_directory": "optical/cameras/powered_rotational_cameras/instantiations/example_camera",
      "body": "camera",
      "pose_connector": "optical_axis",
      "artifact_port": "optical_observation",
      "sensor_id": "example.camera.optical",
      "input_sha256": "64-lowercase-hex-digest"
    }
  ]
}
~~~

The staged format has top-level `format`, `shapes`, and `optical_sensors`;
`stage_plan` emits both arrays even when one is empty. Each normalized shape
retains the plan fields, uses a staged relative `source`, and adds an
`input_sha256` object mapping every closure basename to a bare digest. Its
`surface_uncertainty` is the normalized strict record when declared and
`null` when absent.
Each sensor retains its plan fields, uses a staged relative `source`, and adds
the host-derived `sensor_id` plus one bare-string `input_sha256` for the
descriptor bytes.

The staged record is internal data emitted only by `stage_plan`, not a
separately accepted user or Luna authoring surface. The plan and copied closure
join the workspace's protected integrity set; the host checks them
before/after dispatch and on failure paths.

## External CAD/GLB tessellator contract

Built-in OBJ/STL/CTMESH readers return canonical triangles directly. STEP,
BREP, FreeCAD, PLY, glTF, and GLB require an explicit deterministic backend.
That backend must return one `TessellatedShape`: its canonical `TriangleMesh`,
every extracted `OpticalMaterial`, and any linked source files needed to prove
the import. A bare mesh is rejected; this prevents an external geometry backend
from silently discarding embedded optical properties. A strict
`*.optical.json` library or explicit host material records may override the
backend table, while the original source closure remains immutable evidence.
Texture/UV or vendor properties a backend cannot represent must fail closed or
be normalized into a strict optical sidecar; they are never guessed.

## Post-Luna bundling

After the structured response passes its Luna-only ownership gate, the host:

1. materializes Luna's text files into a private candidate directory;
2. reopens the protected staged plan and verifies each closure file digest;
3. resolves `catalog_directory` below that candidate root;
4. requires Luna's target `static.part` and exact body/solid ids;
5. calls deterministic `import_shape` into
   `<catalog_directory>/shape/<solid>/`;
6. imports source/sidecar optical properties, canonical CTMESH, a disposable
   runtime GLB cache, uncertainty, and density-derived mass properties when
   available;
7. selects the imported analysis surface and replaces the target solid geometry
   with `kind: shape`, exact bounds-derived `dimensions_m`, relative
   `shape_uri`, prefixed manifest-file `shape_sha256`, and `surface_id`;
8. verifies each staged sensor digest, strict descriptor schema, derived id,
   and declared mount;
9. refuses a Luna-supplied `sensor.optical.json`, then copies the exact protected
   descriptor bytes to that fixed target name;
10. replaces `static.part.optical_sensors` with the host-owned singleton
    binding; and
11. writes `static.part` as strict finite JSON and runs the ordinary complete
    artifact/catalog validator over the combined bundle.

The host-owned binding has exactly:

~~~json
{
  "id": "<descriptor.id>",
  "body": "<plan.body>",
  "pose_connector": "<plan.pose_connector>",
  "artifact_port": "<plan.artifact_port>",
  "descriptor_uri": "sensor.optical.json",
  "descriptor_sha256": "sha256:<digest-of-copied-bytes>"
}
~~~

The complete catalog validator then requires the bound body and optical
connector to exist, and the PMDL artifact port to be an output of exact type
`contraption/optical-observation@1`. The static-part loader rechecks the copied
descriptor bytes, id, mount, binding, and prefixed file hash.

Deterministic provenance begins with
`{"kind":"deterministic-luna-bundle", ...}`; explicit plan provenance is merged
after it and therefore must remain auditable. Any source hash mismatch, escaped
path, missing static part/body/solid/connector/artifact port, changed sensor
identity or mount, Luna-supplied sensor destination, or downstream
shape/catalog validation error rejects the entire proposal. There is no
fallback box and no legacy mesh adapter.

## Luna exclusion

Luna must understand this contract only to create correctly named textual
catalog targets. It must not open the source closure, stage or hash inputs, call
`import_shape`, compute dimensions/transforms/materials/mass properties, author
either ingestion format, emit deterministic shape/optical artifacts or sensor
calibration, supply `sensor.optical.json`, or modify host-provided ingestion
references. The structured-response and recovery gates reject agent-authored
shape/optical formats before the trusted host bundler appends them. Generic
validation then admits only the verified combined output.
