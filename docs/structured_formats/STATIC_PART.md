# Static parts (`static-part-1`)

`static.part` stores the physical facts that do not change when a different PMDL
hypothesis or fidelity is selected for the same part: bodies, geometry/shape
references, connectors, geometric parameter bindings, hash-bound optical sensor
bindings, provenance, and purchasing
identity. Dynamic equations and initialized parameter values belong in PMDL and
`vN.model`. The authoritative parser is
`contraption.catalog.instantiations.StaticPartSpec` and physical records are in
`contraption.physics.physical`.

## Placement and top-level fields

A file lives at:

`model_catalog/<domain>/<category>[/<device>]/instantiations/<part-id>/static.part`

Required fields are everything through `provenance` below. Unknown fields and
duplicate JSON keys are invalid.

| Field | Type | Meaning |
|---|---|---|
| `format` | string | exactly `static-part-1` |
| `id` | identifier | stable physical part id; directory/model-instance identity |
| `name` | nonempty string | display name |
| `version` | nonempty string | static physical record version |
| `physical_role` | enum | `part`, `boundary`, or `software` |
| `bodies` | body array | explicit material body geometry |
| `connectors` | connector array | spatial interfaces and PMDL-port bindings |
| `parameter_bindings` | array | typed geometric measurements bound to PMDL parameters |
| `optical_sensors` | array, optional | host-owned descriptor/pose/artifact-port bindings; default empty |
| `provenance` | provenance | part-level source |
| `purchasing` | object, optional | manufacturer/order/sourcing data |
| `metadata` | object, optional | inert JSON |

Identifiers match `^[A-Za-z][A-Za-z0-9_.-]*$`. PMDL symbols match
`^[A-Za-z][A-Za-z0-9_]*$`.

A `part` requires at least one body and every connector is spatial. A
`boundary` or `software` requires `bodies: []`, nonspatial connectors, no
physical parameter bindings, and matching `boundary` or `software` provenance.
The validator never creates placeholder geometry.

## Transforms and coordinate convention

Every body, solid, and spatial connector pose has exactly:

~~~json
{
  "translation_m": [0.0, 0.0, 0.0],
  "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
}
~~~

Translation contains three finite metres. The quaternion is finite, normalized
within `1e-9`, WXYZ ordered, and sign-canonicalized so the first nonzero
component is positive. Transforms use parent-from-child composition in a
right-handed frame. The part itself does not redefine the shape artifact's
canonical +Z-up/+X-forward convention; any source-to-canonical conversion is
recorded by deterministic shape ingestion.

## Bodies and solids

A body has exactly required `id`, `local_pose`, and nonempty unique `solids`.
Its pose locates the material body in the part frame. Each solid has exactly
`id`, `geometry`, `local_pose`, and `provenance`.

`geometry` always requires `kind` and three positive local XYZ extents in
`dimensions_m`. Allowed kinds are `box`, `cylinder`, `sphere`, and `shape`.
Cylinder +Z is its axis and its X/Y diameters must match exactly within
`1e-12`. Sphere diameters must all match.

A detailed standardized solid uses:

~~~json
{
  "kind": "shape",
  "dimensions_m": [0.025, 0.024, 0.0124],
  "shape_uri": "shape/shape.json",
  "shape_sha256": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "surface_id": "mechanical-and-optical"
}
~~~

For `kind: shape` all three additional fields are required:

| Field | Rule |
|---|---|
| `shape_uri` | safe POSIX path relative to the `static.part` directory; no absolute path, backslash, `.`, or `..` component |
| `shape_sha256` | `sha256:` plus the exact manifest-file byte digest |
| `surface_id` | identifier naming one canonical CTMESH surface in the manifest |

They are forbidden on primitive kinds. Raw CAD and ad-hoc `mesh`/`mesh_uri`
geometry are not accepted; there is no legacy adapter. On catalog load, the
host requires a regular in-tree manifest, verifies its exact file-byte hash,
loads and verifies the complete `shape-artifact-1` content closure, requires the
selected surface to have `kind: ctmesh`, and compares its bounds-derived XYZ
extent with `dimensions_m` (relative tolerance `1e-6`, absolute `1e-10` m).

Note the two intentional hashes: static `shape_sha256` has a `sha256:` prefix
and binds the exact pretty/compact manifest file bytes, while a shape artifact's
own `artifact_sha256` is a bare digest of canonical compact JSON. Neither can be
substituted for the other.

Primitive solids remain intentional exact geometry for genuinely ideal/simple
parts. They must not be used as an undeclared fallback when detailed geometry
exists or is required. Detailed surfaces, optical materials, physical fields,
volumes, uncertainty, and derived mass properties live in the referenced
[shape artifact](./SHAPE_ARTIFACT.md), not duplicated in `static.part`.

## Provenance

Every part, connector, and solid uses:

| Field | Required | Values |
|---|---:|---|
| `kind` | yes | `estimated`, `catalog`, `vendor`, `cad`, `scan`, `measured`, `manual`, `derived`, `boundary`, or `software` |
| `source` | yes | nonempty human/audit description |
| `reference` | no | source identifier, document, URL, or measurement reference |

Provenance does not itself make a representation accurate. Detailed shape
payloads also have exact source/content hashes and conversion settings.

## Connectors

A connector allows exactly:

| Field | Required | Semantics |
|---|---:|---|
| `id` | yes | part-local connector id |
| `model_port` | yes, nullable | PMDL power/signal/artifact port name, or null for a physical-only frame |
| `body` | yes, nullable | body id |
| `domain` | yes | physical/semantic domain |
| `interface` | yes | compatibility identity such as `rotational-shaft` |
| `local_pose` | yes, nullable | connector frame in body coordinates |
| `provenance` | yes | connector-frame evidence |
| `kinematics` | no | special frame kinematics |
| `joint_coordinate_state` | conditional | PMDL state symbol for `rotational-shaft` |

`body` and `local_pose` are either both present or both null. A physical
`part` requires both; boundaries/software require both null. Connector ids,
bound model ports, and body references are validated and unique. One model port
may be bound only once.

A `rotational-shaft` requires `joint_coordinate_state`. It identifies the local
PMDL angular state associated with the shaft coordinate.

The only special connector kinematics is:

~~~json
{"kind": "counter_rotation", "state": "angle"}
~~~

It keeps a non-material connector frame fixed by undoing local joint rotation.
It is permitted only on a spatial connector.

A camera's optical device frame is an ordinary spatial connector with
`domain: optical` and a declared interface such as `camera-view-axis`. Its local
+Z axis should follow the optical sensor/scene convention. Viewpoint switching
and sensor rendering use this exact frame; an approximate face-center guess
must retain estimated/derived provenance.

## Optical sensor bindings

`optical_sensors` is optional and defaults to an empty array. Each unique-id
binding has exactly:

| Field | Required semantics |
|---|---|
| `id` | sensor identity; must equal the `optical-sensor-1` descriptor's `id` |
| `body` | body carrying the sensor |
| `pose_connector` | optical-domain connector on that same body |
| `artifact_port` | PMDL artifact output that emits observations |
| `descriptor_uri` | safe POSIX relative path; admitted files are exactly the regular instantiation-local `sensor.optical.json` |
| `descriptor_sha256` | `sha256:` plus exact descriptor-file byte digest |

Catalog validation rejects duplicate sensor ids, unknown bodies, a connector on
another body or outside the optical domain, path traversal, symlinks, missing or
renamed descriptors, descriptor-byte hash mismatch, and a sensor whose
`mount_connector` differs from `pose_connector`. When resolved with its model,
`artifact_port` must name an output with exactly
`artifact_type: contraption/optical-observation@1`.

A validated scanner binding is structurally:

~~~json
{
  "optical_sensors": [
    {
      "id": "scanner.camera.optical",
      "body": "camera",
      "pose_connector": "optical_axis",
      "artifact_port": "optical_observation",
      "descriptor_uri": "sensor.optical.json",
      "descriptor_sha256": "sha256:e28fb792f34fd7c90eb674ed9cbac72bd3f8c15df756d55576ef623f9a711114"
    }
  ]
}
~~~

The digest above is tied to the checked-in descriptor and must be recomputed
after any deterministic descriptor change. Intrinsic/timing/noise fields are
defined in [Optical sensor](./OPTICAL_SENSOR.md); they are not duplicated here.

## Physical parameter bindings

Bindings prevent a PMDL geometric parameter from silently disagreeing with the
part record. A binding has:

| Field | Rule |
|---|---|
| `model_parameter` | PMDL parameter symbol |
| `unit` | parseable length unit compatible with metres |
| `absolute_tolerance` | finite and nonnegative in that unit |
| `measure` | one supported typed measure |

Supported measures are:

~~~json
{
  "kind": "solid_radius",
  "body": "wheel",
  "solid": "tire",
  "axis": "x"
}
~~~

and:

~~~json
{
  "kind": "connector_distance",
  "first_connector": "left",
  "second_connector": "right"
}
~~~

`solid_radius` requires a cylinder or sphere. Axis is `x`, `y`, or `z`; a
cylinder radius may not use its axial Z extent. `connector_distance` requires
two distinct spatial connectors and measures positive Euclidean distance in the
part frame. Binding names are unique and must name model parameters when the
part/model instance is resolved.

Detailed shape-derived parameters such as mass, center of mass, and inertia come
from `shape-artifact-1` physical fields and derived mass-property records rather
than pretending they are primitive-radius measurements.

## Example with an optical frame

~~~json
{
  "format": "static-part-1",
  "id": "scanner.camera",
  "name": "Scanner camera payload",
  "version": "1.0.0",
  "physical_role": "part",
  "bodies": [
    {
      "id": "camera",
      "local_pose": {
        "translation_m": [0, 0, 0],
        "rotation_quaternion_wxyz": [1, 0, 0, 0]
      },
      "solids": [
        {
          "id": "module",
          "geometry": {
            "kind": "shape",
            "dimensions_m": [0.025, 0.024, 0.0124],
            "shape_uri": "shape/shape.json",
            "shape_sha256": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "surface_id": "mechanical-and-optical"
          },
          "local_pose": {
            "translation_m": [0.0125, 0, 0],
            "rotation_quaternion_wxyz": [1, 0, 0, 0]
          },
          "provenance": {
            "kind": "cad",
            "source": "Deterministically imported vendor CAD"
          }
        }
      ]
    }
  ],
  "connectors": [
    {
      "id": "optical_axis",
      "model_port": null,
      "body": "camera",
      "domain": "optical",
      "interface": "camera-view-axis",
      "local_pose": {
        "translation_m": [0.025, 0, 0],
        "rotation_quaternion_wxyz": [0.7071067811865476, 0, 0.7071067811865475, 0]
      },
      "provenance": {
        "kind": "derived",
        "source": "Calibrated optical center and +Z viewing axis"
      }
    }
  ],
  "parameter_bindings": [],
  "optical_sensors": [
    {
      "id": "scanner.camera.optical",
      "body": "camera",
      "pose_connector": "optical_axis",
      "artifact_port": "optical_observation",
      "descriptor_uri": "sensor.optical.json",
      "descriptor_sha256": "sha256:e28fb792f34fd7c90eb674ed9cbac72bd3f8c15df756d55576ef623f9a711114"
    }
  ],
  "provenance": {
    "kind": "catalog",
    "source": "Vendor mechanical drawing"
  },
  "purchasing": {},
  "metadata": {}
}
~~~

This example binds detailed geometry, an optical connector frame, and the exact
host-owned `optical-sensor-1` descriptor/PMDL artifact port. Neither record
hides the other inside `metadata`.

## Luna/deterministic bundling boundary

Luna may propose bodies, connector abstractions, PMDL bindings, provenance
claims supported by its textual inputs, and ordinary catalog metadata. It must
not parse a source geometry/texture/calibration file, compute transforms,
generate canonical surfaces, infer optical properties, or author a shape,
material, sensor, scene, observation, or reconstruction payload.

The deterministic importer produces those payloads, verifies exact hashes and
units, and bundles host-owned static-part references with Luna's proposal.
Materialization rejects any Luna top-level deterministic format even if the
generic host admission pipeline supports that format.
