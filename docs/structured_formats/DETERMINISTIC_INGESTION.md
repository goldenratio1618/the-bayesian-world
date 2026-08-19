# Deterministic part ingestion (`deterministic-part-ingestion-1`)

This host-owned contract declares source geometry, calibrated optical-sensor
descriptors, and deterministic source documents used with a Luna-authored
catalog proposal without exposing raw document, archive, mesh, or CAD bytes to
Luna. It has two
related forms:

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
| `documents` | no | host-parsed PDF, DXF, or structured-ECAD source array; default empty |

Each shape item allows only:

| Field | Required/default | Semantics |
|---|---|---|
| `source` | required | safe POSIX path relative to the component-information directory |
| `archive_member` | required for `.zip` only | safe POSIX path of the selected geometry inside a bounded ZIP; forbidden for non-ZIP sources |
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

Each document item allows only:

| Field | Required/default | Semantics |
|---|---:|---|
| `source` | required | safe POSIX path relative to the component-information directory |
| `source_format` | `auto` | one of `auto`, `pdf`, `dxf`, `kicad`, or `librepcb` |

`auto` recognizes PDF by its `%PDF-` magic, ASCII DXF by its `.dxf` suffix, and
otherwise requires a supported KiCad or LibrePCB root S-expression. A `.pdf`
extension with wrong magic, binary DXF, an unknown binary or text format, and a
declared/detected format mismatch fail closed. No generic "send the bytes to
Luna" fallback exists.

Identifiers match `^[A-Za-z][A-Za-z0-9_.-]*$`. Paths must be nonempty POSIX
relative paths with no backslash, NUL, absolute root, `.`, or `..` component.
Sources must be existing regular non-symlink files. At least one of `shapes`,
`optical_sensors`, or `documents` must be nonempty. Shape target triples
`(catalog_directory, body, solid)` are unique case-insensitively. At most one
deterministic optical-sensor descriptor may target an instantiation
`catalog_directory`, also compared case-insensitively.
Document sources must be unique case-insensitively.

## Deterministic document parsers

PDF support is provided by `contraption.part_import.document_ingestion` and the
optional `documents` dependency (`pypdf>=6,<7`). Before any extracted text can
become modeling context, the host enforces a strict PDF magic header, configured
byte/page/text/object/depth limits, a readable catalog and page tree, and rejects
encryption, JavaScript, launch/submit/import/external-target/rendition actions,
embedded files, file attachments, XFA, and multimedia objects. The parsed PDF
object graph is inspected, so incidental marker-like bytes inside compressed
image or content streams do not produce false positives. Each page is normalized
to NFKC plain text with layout controls removed and receives its own SHA-256;
the joined text and original PDF bytes are also hashed. Parser name/version and
all configured evidence are recorded in `deterministic-pdf-extraction-1` JSON.

`contraption.part_import.dxf_ingestion` strict-parses bounded ASCII DXF as
code/value group pairs. It requires complete pairs, well-formed unique sections,
an `ENTITIES` section, a final `0/EOF`, bounded value lengths, and already-NFKC
values without control characters. Every pair is preserved exactly in
`deterministic-dxf-extraction-1`; `$INSUNITS` and entity counts are indexed
without replacing the drawing with a bounding box or inventing extrusion
thickness. Binary DXF is deliberately unsupported.

`contraption.part_import.ecad_ingestion` uses a bounded strict S-expression
parser for `kicad_symbol_lib` and `librepcb_*` records. It extracts only explicit
format-defined facts:

- KiCad symbol ids, absolute HTTP(S) `Datasheet` properties, and explicit
  `Manufacturer`/`MFR` and `MPN`/`Manufacturer Part Number` properties;
- LibrePCB device-level HTTP(S) resources and each `(part "<MPN>"
  (manufacturer "<name>"))` record.

It does not promote display values, descriptions, filenames, URL substrings, or
package names to manufacturer part numbers. Missing properties stay absent,
conflicting explicit properties are reported as omissions, and non-HTTP(S)
document references are not converted into purchase links. Every extracted
identity fact retains the exact source SHA-256, S-expression locator, line,
column, and value SHA-256 in `deterministic-ecad-extraction-1` JSON.

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

For glTF, every non-data buffer and image URI is percent-decoded, constrained
below the source directory, bounded by file-count and total-byte limits, and
included in the closure. GLB chunk structure and embedded JSON are validated.
A ZIP shape source is accepted only with an explicit `archive_member`. The host
rejects traversal, backslashes, duplicate case-insensitive names, links, special
files, encryption, unsupported compression, excessive file counts/sizes/ratios,
and CRC/length failures; it then extracts the entire bounded linked-resource
closure into the protected staging tree. A failed extraction removes only its
new staging destination.

Each optical-sensor item contributes exactly its descriptor file to the closure.
Each document item contributes its raw source to the input-hash closure, but the
raw source is not copied into the staged document directory. Only the canonical
extraction JSON is staged and eligible for inert modeling context. Shape input
trees and their precomputed canonical artifacts remain protected siblings; no
shape/archive/CAD path is returned by `modeling_context_paths`.

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
    ],
    "documents": [
      {
        "source": "assets/example-camera-datasheet.pdf",
        "source_format": "pdf"
      },
      {
        "source": "assets/example-camera.kicad_sym",
        "source_format": "kicad"
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
per sensor. It copies and hashes the exact input closure, converts every shape
to a complete validated `shape-artifact-1`, hashes that prepared tree, and only
then writes `plan.json`. A missing optional backend, malformed source, or
unsupported representation therefore fails before any Luna request or budget
reservation:

The primary source is read into that host-private tree exactly once before its
active-content or link declarations are parsed. OBJ/glTF links are discovered
from the private primary snapshot, each linked file is then snapshotted once,
and preflight, FreeCAD/trimesh conversion, evidence copying, and hashing use
only those private paths. ZIP validation and extraction likewise consume the
same in-memory byte snapshot that is hashed; neither path is reopened from the
mutable downloaded catalog during conversion.

~~~json
{
  "format": "deterministic-part-ingestion-staged-1",
  "shapes": [
    {
      "source": "shape-000/input/optical_cube.obj",
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
      "input_root": "shape-000/input",
      "input_sha256": {
        "optical_cube.obj": "64-lowercase-hex-digest",
        "optical_cube.mtl": "64-lowercase-hex-digest"
      },
      "prepared_root": "shape-000/prepared",
      "prepared_sha256": {
        "canonical.ctmesh": "64-lowercase-hex-digest",
        "runtime.glb": "64-lowercase-hex-digest",
        "shape.artifact.json": "64-lowercase-hex-digest",
        "source/content-addressed-optical_cube.obj": "64-lowercase-hex-digest",
        "source/content-addressed-optical_cube.mtl": "64-lowercase-hex-digest"
      },
      "backend": {
        "id": "contraption-native-obj",
        "version": "1"
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
  ],
  "documents": [
    {
      "source_name": "example-camera-datasheet.pdf",
      "source_format": "pdf",
      "input_bytes": 12345,
      "input_sha256": "64-lowercase-hex-digest",
      "extraction": "document-000/extraction.json",
      "extraction_format": "deterministic-pdf-extraction-1",
      "extraction_sha256": "64-lowercase-hex-digest"
    }
  ]
}
~~~

The staged format has top-level `format`, `shapes`, `optical_sensors`, and
`documents`; `stage_plan` emits all three arrays even when one is empty. Each
normalized shape retains the plan fields and adds a staged relative `source`,
`input_root`, exact relative-path `input_sha256` closure, `prepared_root`,
exact `prepared_sha256` closure, and versioned `backend` identity. FreeCAD
records also bind the converter executable SHA-256. ZIP inputs additionally
record the raw archive name/size/hash and selected member. The protected
canonical artifact contains the same backend/archive evidence in provenance.
`surface_uncertainty` is the normalized strict record when declared and
`null` when absent.
Each sensor retains its plan fields, uses a staged relative `source`, and adds
the host-derived `sensor_id` plus one bare-string `input_sha256` for the
descriptor bytes.
Each document entry records the original basename, declared format, byte length
and hash plus a safe relative canonical extraction path, extraction format and
hash. `modeling_context_paths` revalidates those fields, strict-parses the JSON,
and checks its embedded source length/hash before returning any context path.
It never returns a raw PDF/DXF/ECAD path, any raw shape/CAD/archive path, or a
prepared CTMESH path. Before returning the document contexts it revalidates the
complete shape input and prepared trees, archive-member binding, backend
identity, artifact identity, manifest, and every content hash. Any changed
extraction or inconsistent source evidence rejects the run.

The staged record is internal data emitted only by `stage_plan`, not a
separately accepted user or Luna authoring surface. The plan and copied closure
join the workspace's protected integrity set; the host checks them
before/after dispatch and on failure paths.

## Geometry support, dependencies, and corpus audit

The 2026-08-01 `D:\WorldModelCatalogs` extension inventory and the resulting
host behavior are:

| Source | Files | Pre-Luna pathway | Dependency / fail-closed boundary |
|---|---:|---|---|
| STL | 2,551 | built-in strict ASCII/binary triangle reader | no optional dependency |
| OBJ | 2 | built-in geometry plus bounded MTL reader | `mtllib` remains below the source directory; UV/texture maps fail |
| CTMESH | 0 source files | built-in canonical decoder | canonical packing/topology/hash checks |
| PLY | 0 | built-in strict ASCII or endian-aware binary PLY 1.0 | exactly vertex then triangle-face elements; integer list topology; only XYZ, complete normals, and RGB[A] vertex fields are accepted; UV, unknown properties, and ambiguous polygon triangulation fail |
| STEP/STP | 10,890 | automatic FreeCAD/OpenCascade worker | requires `FreeCADCmd` or `freecadcmd`; current project host has neither |
| IGES/IGS | 3 | same FreeCAD/OpenCascade worker | same system dependency |
| BREP | 33 | same FreeCAD/OpenCascade worker | same system dependency |
| FCStd | 3,202 | strict ZIP/document preflight, then isolated FreeCAD worker in safe mode | rejects macros/Python proxies, unsafe members, compression bombs, CRC failures; requires FreeCAD command line |
| glTF/GLB | 0 | bounded glTF/GLB closure parser plus automatic trimesh scene adapter | install project extra `[geometry]` (`trimesh>=4.8,<5`); current project environment lacks it |
| WRL/VRML | 3 | bounded VRML preflight plus the same trimesh adapter | scripts, external nodes and image/movie textures fail; line-only files fail because no triangle surface |
| DXF | 12 | lossless bounded ASCII group-pair document extraction | binary DXF and interpreting a 2-D outline as a 3-D solid fail; no thickness is invented |
| ZIP | 1,046 total; 1,030 Google Scanned Objects | bounded extraction only when a shape declares exact `archive_member` | generic software/data ZIPs are not shape inputs; unsafe/bomb archives fail |

The corpus also contains 102 `.fcstd1` recovery backups, four `.sldprt`, one
`.sldasm`, one `.ipt`, one `.skp`, one `.3ds`, and one `.dwg`. These
have no audited deterministic converter and are rejected before Luna. Renaming a
backup or routing a proprietary format through an arbitrary desktop converter
is not accepted. A future adapter must satisfy the contract below and identify
its exact parser/kernel version.

The optional FreeCAD adapter launches a checked-in worker without a shell, in a
fresh temporary home and safe mode, with source/output/vertex/triangle/time and
process-output limits. It tessellates actual CAD topology at a fixed metric chord
deflection; it never substitutes extents or a bounding box. Its evidence records
the successful FreeCAD version string and SHA-256 of the resolved executable.
Because FreeCAD is not installed on the current host, STEP/STP/IGES/IGS/BREP/
FCStd imports presently stop during `stage_plan` with exact install guidance.

The optional scene adapter is version-pinned by the `geometry` extra. It
validates every linked URI before the optional loader is imported or called,
applies every scene-node transform, preserves vertex colors or bounded source
materials, includes linked glTF resources, and rejects UV coordinates, textures,
or mixed visual coverage that CTMESH cannot represent. Because trimesh is not installed in the
current project environment, glTF/GLB/WRL imports presently stop during
`stage_plan`.

An explicit deterministic backend remains possible through the typed
`Tessellator(Path, metres_per_source_unit) -> TessellatedShape` contract. It
must provide a stable non-generic `backend_id` and exact `backend_version`,
return one canonical `TriangleMesh`, every representable
`OpticalMaterial`, and every linked source needed for evidence. A bare mesh,
unversioned backend, missing source, NaN/infinity, unbounded output, material
index mismatch, unsupported texture, or invalid topology fails before Luna.
A strict `*.optical.json` library may supply measured material records, but it
does not turn a spatial UV texture into a uniform color.

Google Scanned Objects ZIPs are therefore structurally supported and safely
extractable, but the current objects use `map_Kd` UV textures. Their exact
geometry is not promoted until the canonical shape schema can preserve that
spatial texture (or an authoritative compatible optical representation is
provided). The importer fails closed at MTL parsing; it does not silently strip
the texture.

## Post-Luna bundling

After the structured response passes its Luna-only ownership gate, the host:

1. materializes Luna's text files into a private candidate directory;
2. reopens the protected staged plan and revalidates every exact input and
   prepared-tree digest, archive-member binding, backend identity, artifact
   identity, manifest, and content hash;
3. resolves `catalog_directory` below that candidate root;
4. requires Luna's target `static.part` and exact body/solid ids;
5. refuses a Luna-supplied shape destination and byte-copies the already
   prepared, complete `shape-artifact-1` tree into
   `<catalog_directory>/shape/<solid>/`;
6. reloads that copied artifact and rechecks its identity, manifest, backend,
   source evidence, canonical CTMESH, uncertainty, and density-derived mass
   properties when available;
7. selects the verified analysis surface and replaces the target solid geometry
   with `kind: shape`, exact bounds-derived `dimensions_m`, relative
   `shape_uri`, prefixed manifest-file `shape_sha256`, and `surface_id`;
8. verifies each staged sensor digest, strict descriptor schema, derived id,
   and declared mount;
9. verifies every deterministic document extraction and its embedded source
   evidence without copying the original document into the candidate;
10. refuses a Luna-supplied `sensor.optical.json`, then copies the exact protected
   descriptor bytes to that fixed target name;
11. replaces `static.part.optical_sensors` with the host-owned singleton
    binding; and
12. writes `static.part` as strict finite JSON and runs the ordinary complete
    artifact/catalog validator over the combined bundle.

Before the candidate is admitted, the host writes a shape receipt beside the
modeling run and outside the candidate-writable tree. Its strict schema binds
the protected staged-plan path/size/hash, every nonoverlapping shape root, and
the exact sorted candidate-relative file path/size/hash set (manifest,
canonical/runtime geometry, primary sources, and linked resources). Promotion
grants unknown-extension trust only to paths verified by this receipt. It
revalidates the receipt before source admission, maps only those verified paths
into the atomic snapshot, and re-hashes the snapshot against the same external
receipt before copying. A candidate-authored manifest cannot expand that set;
missing, extra, changed, linked, or escaping files reject promotion.

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

Deterministic provenance begins with the host-owned
`{"kind":"deterministic-preflight", "backend": ..., ...}` record written while
the protected artifact is prepared before dispatch; declared provenance remains
nested and auditable rather than being merged by Luna. Any source hash mismatch, escaped
path, missing static part/body/solid/connector/artifact port, changed sensor
identity or mount, Luna-supplied sensor destination, or downstream
shape/catalog validation error rejects the entire proposal. There is no
fallback box and no legacy mesh adapter.

## Luna exclusion

Luna must understand this contract only to create correctly named textual
catalog targets. It must not open the source closure (including raw PDF or ECAD
bytes, DXF, CAD, mesh, or archive members), stage or hash inputs, run a geometry
backend or call `import_shape`, compute dimensions/transforms/materials/mass properties, author
either ingestion format, emit deterministic shape/optical artifacts or sensor
calibration, supply `sensor.optical.json`, or modify host-provided ingestion
references. The structured-response and recovery gates reject agent-authored
shape/optical formats before the trusted host bundler appends them. Generic
validation then admits only the verified combined output.
