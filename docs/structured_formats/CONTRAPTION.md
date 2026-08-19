# Contraption assembly (`contraption-4`)

A `contraption.json` declares a hash-closed assembly of catalog model instances,
physical connections, optional external actuation, deployable controllers,
verification programs, and environment data. It does not repeat part parameters
or geometry. The authoritative records are in `contraption.physics.specs`;
catalog/hash resolution is in `contraption.loading` and
`contraption.physics.resolved`.

## Top-level fields

All listed fields through `components` and `metadata` are required. Optional
arrays/objects default empty. Unknown fields and duplicate JSON keys are invalid.

| Field | Type | Meaning |
|---|---|---|
| `format` | string | exactly `contraption-4` |
| `id` | identifier | assembly identity |
| `name` | nonempty string | display name |
| `version` | nonempty string | assembly version |
| `catalogs` | nonempty catalog-link array | catalog roots in the closure |
| `physical_root` | object | world pose and optional state binding of one root component |
| `components` | nonempty component array | local ids bound to catalog model instances |
| `connections` | array | power, signal, attachment, or constraint networks |
| `actuators` | array | explicit external output bindings |
| `controllers` | array | exact control artifacts and wiring |
| `verifications` | array | exact verification artifacts and observable bindings |
| `environment` | object | typed consumer-specific environment/scenario data |
| `metadata` | object | condition, completeness gates, and inert audit data |

Ids use `^[A-Za-z][A-Za-z0-9_.-]*$` unless a nested field is explicitly a PMDL
symbol. Catalog, controller, and verification files are resolved relative to
the contraption document. Resolution rejects containment escapes and stale exact
hashes.

## Catalogs and components

A catalog link is either a relative path string or:

~~~json
{"path": "../../model_catalog"}
~~~

Every linked root must contain a complete interface/model/instantiation catalog.
A component has exactly:

~~~json
{"id": "camera", "instantiation": "scanner.camera.v1"}
~~~

`id` is local to this contraption. `instantiation` resolves one exact
`model-instance-1`, which in turn binds the static part and exact PMDL model,
hash, parameters, uncertainty, and condition. Component ids are unique. Do not
place model ids, parameters, or geometry in a component record.

## Physical root

The root has exactly `component`, `pose`, and nullable `state_binding`:

~~~json
{
  "component": "chassis",
  "pose": {
    "translation_m": [0.85, 0.0, 0.0],
    "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
  },
  "state_binding": {
    "kind": "planar",
    "x": "chassis.position_x",
    "y": "chassis.position_y",
    "yaw": "chassis.yaw"
  }
}
~~~

The root component must exist. Pose uses the static-part transform convention.
A planar binding has exactly `kind: planar` and three distinct namespaced PMDL
states on the root component. It replaces world X, Y, and yaw during dynamic
visualization/physical configuration while preserving root Z, roll, and pitch.
Use null when the assembly root is fixed.

## Port references

Most endpoints use the compact string `component.port`. The parser also accepts
`{"component": "...", "port": "..."}` where that record is explicitly used.
The component id may contain dots; parsing splits at the final dot. Port names
are PMDL symbols.

Physical attachments use the same endpoint strings but resolve static-part
connector ids, which must also agree with any PMDL-port binding and physical
domain/interface.

## Connections

A connection allows exactly `id`, `kind`, `endpoints`, optional `domain`,
conditional `joint`, optional typed `implementation`, and optional `metadata`.
Physical resolution requires `metadata` to be absent or empty; active connection
semantics belong in typed fields.

| `kind` | Semantics |
|---|---|
| `power` | acausal network with common effort, signed flow balance, compatible domain/units/orientation |
| `signal` | one directed output to compatible input(s), matching unit/dtype/shape |
| `attachment` | exactly two physical connectors plus a typed fixed/revolute joint |
| `constraint` | typed constraint endpoints consumed by an admitting engine |

Every connection needs at least two unique endpoints. Endpoint components and
ports/connectors must exist. Connection ids are unique. `domain`, when supplied,
must match resolved port domains.

### Construction implementation

`implementation` is optional/nullable and uses the shared
[fabrication record](./FABRICATION.md). Static connector fabrication records
describe endpoint capabilities or requirements; this connection record selects
assembly-specific hardware, bearing arrangement, wire termination/protection,
route, or alignment details.

Its required `kind` follows connection semantics:

| Connection | Required fabrication kind |
|---|---|
| `power` or `signal` | `electrical_termination` |
| fixed `attachment` | `fixed_mount` |
| revolute `attachment` | `rotary_support` |
| `constraint` | `other` |

A construction-ready implementation has `status: specified`, `missing: []`,
complete context-specific fields, and evidence. `missing` and `partial` records
are accepted only when every absent construction field is named explicitly;
they remain build-release gates. Electrical connection implementations require
both endpoint termination facts and selected `protection` and `route`. A route
for more than two endpoints cannot use `point_to_point` topology.

Resolution checks known endpoint records against one another and against the
selected implementation. Conflicting standard dimensions or non-complementary
mating roles fail. Missing facts are not invented and do not count as
compatibility evidence.

### Fixed joint

~~~json
{
  "kind": "fixed",
  "behavior_binding": "kinematic_only",
  "coordinate_bindings": []
}
~~~

A fixed joint forbids `coordinate`, nonempty bindings, and nonzero
`zero_angle_rad`. `behavior_binding` is `kinematic_only` or `pmdl`; use
`kinematic_only` when the attachment contributes pose constraints only.

### Revolute joint

~~~json
{
  "kind": "revolute",
  "behavior_binding": "pmdl",
  "coordinate": "wheel.angle",
  "zero_angle_rad": 0.0,
  "coordinate_bindings": [
    {
      "state": "wheel.angle",
      "joint_angle_at_state_zero_rad": 0.0
    }
  ]
}
~~~

A revolute joint requires `coordinate` and nonempty unique
`coordinate_bindings`. The first binding names `coordinate` and its zero angle
equals `zero_angle_rad` within `1e-12`. Each binding references a PMDL state on
one of the two endpoint components. Coordinates are radians. Connector frames
must be coincident within the assembly resolver's translation/angular tolerance,
and the declared shaft axes determine rotation.

## External actuators

An actuator has required `id`, string `source`, target port reference `target`,
optional `settings`, and optional `external` (default false). In a top-level
`contraption-4` actuator, `external` must be true. Controller-owned outputs
belong in `controllers[].outputs` rather than duplicated as actuators.

External actuators are intentionally explicit unmodeled boundary inputs. Their
settings are inert until interpreted by a qualified runtime.

## Hash-bound controller links

A controller has exactly:

| Field | Meaning |
|---|---|
| `id` | unique link id; normally matches the control program identity |
| `program` | contained `path` plus `sha256:` and 64 lowercase hex digits |
| `explicit_inputs` | mapping from control input names to exactly one `signal` or `external` binding |
| `implicit_inputs` | mapping from control latent names to namespaced PMDL state/algebraic targets |
| `outputs` | mapping from control output names to exactly one `signal` or `external` binding |

All names and types must match the loaded `control-1` artifact. Signal bindings
are `component.port` strings. External bindings are stable machine symbols. The
resolved controller closure includes the exact PMDL plant and open physical
completeness gates acknowledged by its observer.

Artifact-stream image/shape traffic does not flow through `control-1` scalar
bindings. An optical service consumes artifact ports and exposes bounded scalar
features through ordinary component signals, preserving future FPGA/controller
compilation.

## Hash-bound verification links

A verification has exactly:

| Field | Meaning |
|---|---|
| `id` | unique verification link id |
| `program` | contained relative path and exact `sha256:...` |
| `inputs` | mapping from verification input symbols to namespaced scalar component trajectories |

Bindings must match the `verification-1` program. Images, depth maps, and
reconstruction artifacts must first yield explicit scalar diagnostic
trajectories such as coverage, reprojection error, or uncertainty.

## Environment, fixed world objects, optical scene, and metadata

`environment` and `metadata` are canonical inert JSON at the base parser layer;
a simulator feature that depends on them must apply its own strict typed
projection. Existing scanner environment records gravity, floor/contact
assumptions, object bounds, and mission settings.

The physical/optical runtime admits a strict `environment.world_objects`
projection for catalog geometry that remains fixed in the world while the
contraption moves. Each entry has exactly `id`, `instantiation`, and `pose`:

~~~json
{
  "world_objects": [
    {
      "id": "scan-target",
      "instantiation": "scanner.icosahedron.v1",
      "pose": {
        "translation_m": [0.0, 0.0, 0.35],
        "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
      }
    }
  ]
}
~~~

The id must not collide with an assembled component. The instantiation must be
a verified spatial catalog part with canonical shape artifacts. A world object
adds fixed body/connector poses to every trajectory frame and participates in
the physical assembly hash, viewer render bundle, optical ray-tracing scene,
build BOM, and placement plan. Its PMDL model is catalog identity only: it does
not add plant states, ports, or attachment constraints. Unknown fields,
unresolved instantiations, invalid poses, or duplicate ids fail closed.

The contraption schema intentionally has no `optical_scene` field. The optical
engine constructs assembly objects and sensor poses through the runtime bridge,
using resolved component/world-object shape artifacts and connector frames. Optional CLI
`--optical-scene` supplies a separate, exact `optical-scene-1` closure for
additional external evidence geometry; each external object hash-binds its own
shape artifact and is merged with the derived assembly scene. Offline CPU/GPU
backends consume the resulting runtime scene.

`metadata` is the correct place for explicit qualification state and open-gate
descriptions, but not for physics. An optical Boolean flag cannot replace an
optical scene, sensor, material, observation, or validated capability.

## Closure and fail-closed validation

Complete loading verifies:

1. contained catalog/artifact paths and exact hashes;
2. interface/model/model-instance/static-part identity;
3. complete initialized parameters and uncertainty;
4. component, port, connector, joint, controller, and verification references;
5. unit/domain/direction/shape compatibility;
6. PMDL equation balance and active semantics;
7. physical assembly connectivity, root reachability, connector compatibility,
   and pose coincidence;
8. shape/optical content hashes and capabilities when those links are present.
9. fixed world-object identities, canonical poses, and collision-free ids.

Unsupported active semantics, stale hashes, disconnected components, unknown
ports, missing detailed surfaces, or absent optical capabilities are errors.
No part or surface is silently dropped and no missing mesh becomes a box.

The physical assembly hash covers canonical contraption assembly data plus all
used resolved component and world-object parts. Runtime joint-coordinate values
are configuration and do not change assembly identity.

## Compact example

~~~json
{
  "format": "contraption-4",
  "id": "camera-rig",
  "name": "Camera and target fixture",
  "version": "1.0.0",
  "catalogs": [{"path": "../../model_catalog"}],
  "physical_root": {
    "component": "base",
    "pose": {
      "translation_m": [0, 0, 0],
      "rotation_quaternion_wxyz": [1, 0, 0, 0]
    },
    "state_binding": null
  },
  "components": [
    {"id": "base", "instantiation": "fixture.base.v1"},
    {"id": "camera", "instantiation": "scanner.camera.v1"}
  ],
  "connections": [
    {
      "id": "camera_mount",
      "kind": "attachment",
      "domain": "mechanical",
      "endpoints": ["base.camera_mount", "camera.mount"],
      "joint": {
        "kind": "fixed",
        "behavior_binding": "kinematic_only",
        "coordinate_bindings": []
      },
      "implementation": {
        "kind": "fixed_mount",
        "status": "missing",
        "missing": ["retention"]
      }
    }
  ],
  "actuators": [],
  "controllers": [],
  "verifications": [],
  "environment": {"gravity_m_s2": [0, 0, -9.80665]},
  "metadata": {"condition": "unverified"}
}
~~~
