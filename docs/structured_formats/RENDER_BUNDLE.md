# Offline render bundles (`contraption.render-bundle/v1`)

The offline viewer consumes a materialized, self-contained projection of the
canonical physical assembly and shape artifacts. A render bundle is disposable
viewer data, not an authored model and not a new source of physical truth. The
authoritative contracts are `shape-artifact-1`, CTMESH, the resolved physical
scene, and the optical sensor/observation formats.

By default, `generate_viewer(assembly)` uses the exact verified
`assembly.instantiations` registry retained in the resolved closure, resolves the authored
`shape_uri` below each owning `PartInstantiation.directory` and builds this
projection deterministically. Callers without catalog context may pass
`shape_artifacts` explicitly. `part_instantiations=registry` remains available
for deliberately constructed assemblies. Every `shape_artifacts` key is
`component/body/solid` and points to a loaded `ShapeArtifact` or manifest path.
The key set must exactly equal every solid in the physical scene.

## Shape materialization

For each solid the materializer:

1. loads the strict `ShapeArtifact` manifest;
2. selects the exact `surface_id` bound by the solid, requires CTMESH and the
   `render` purpose, and verifies the solid's `shape_sha256`;
3. resolves its content only below the manifest directory;
4. verifies exact byte length and SHA-256;
5. decodes and validates CTMESH topology, counts, and bounds;
6. checks its XYZ extent against the resolved solid's declared metric extent;
7. embeds vertices, triangles, normals, vertex colors, ordered material
   assignments, and uncertainty in the browser payload.

Source representations are never opened or interpreted by the renderer. A
GLB-only artifact fails closed: CTMESH is the canonical surface from which the
self-contained viewer's exact triangle payload is derived; GLB is a disposable
runtime cache. The viewer never substitutes a box for missing mesh geometry.

Every embedded surface retains:

- the exact raw manifest-file `shape_manifest_sha256` bound by `static.part`;
- the canonical `shape_artifact_sha256`;
- `shape_id` and selected `surface_id`;
- the exact CTMESH `source_surface_sha256`;
- its own hash over the materialized browser projection.

The bundle itself is bound to the exact `assembly_sha256` and has a canonical
content hash. The Python validator and browser validator both require complete
solid bindings and reject unknown fields or unreferenced surfaces.

## Optical viewpoints

An optical sensor view contains:

| Field | Meaning |
|---|---|
| `id` | unique viewer-local sensor id |
| `display_name` | human-readable selector label |
| `connector` | exact `component.connector` pose key; it must be a spatial connector in domain `optical` |
| `projection.kind` | currently exactly `pinhole` |
| `projection.resolution_px` | positive `[width, height]` |
| `projection.focal_length_px` | positive `[fx, fy]` |
| `projection.principal_point_px` | `[cx, cy]` inside the sensor |
| `projection.clipping_m` | positive increasing `[near, far]` |
| `descriptor_sha256` | exact optical-sensor descriptor hash |

Automatic catalog discovery walks only parsed `static.part.optical_sensors`
bindings. It ignores an ad hoc `sensor.optical.json` that has no binding and
revalidates the bound body, optical connector, descriptor path/bytes/hash,
descriptor id/mount, exact PMDL identity/hash, and named output artifact port
of type `contraption/optical-observation@1`. Thus the viewer cannot gain a
camera merely because a plausibly named file appears beside an instantiation.

The connector's local +Z is the camera forward direction; local +X is image
right and local +Y is image down (the canonical CV/pixel convention). Its hash-bound connector pose is taken from the
selected physical pose frame. The viewer never estimates camera extrinsics or
field of view from component geometry.

World-orbit mode shows calibrated sensor frustums. Selecting a sensor projects
canonical triangles with the declared intrinsics from that exact connector
pose. Orbit drag and zoom are disabled in a sensor POV so user navigation
cannot masquerade as sensor output.

## Observation layers

An observation is bound to one exact artifact hash, sensor descriptor hash,
optical-scene hash, physical assembly id/hash, assembly-qualified mount
connector, and the exact world-from-sensor matrix plus its binary float64 hash.
That matrix must equal the physical connector pose at its `frame_index`. Its `layers` may
contain `rgb`, `depth`, `segmentation`, `uncertainty`, or `reconstruction`.
The rotation block must be orthonormal and proper right-handed with determinant
+1; reflection matrices fail before pose comparison.

Raster layers retain the original `.npy` output hash, media type, dtype, and
shape, plus deterministic derived PNG bytes, canonical base64, exact dimensions,
derived-byte SHA-256, and an explicit display transform/range. The Python
validator verifies the original observation content before conversion and checks
the PNG header/dimensions and payload digest before generation. The browser shows
those exact embedded bytes; it does not recolor or fabricate a missing mode.

A reconstruction layer references one embedded canonical surface plus an
explicit world pose and observation hash. Surface layers are permitted only for
`reconstruction`. Selecting an unavailable layer displays a clear unavailable
state instead of falling back to the ordinary scene.

## API example

~~~python
artifact = generate_viewer(
    assembly,
    optical_sensors=[sensor_view],
    optical_observations=[frame_observation],
)
artifact.write("outputs/scanner/viewer")
~~~

The actual mapping must cover all physical solids, not only those shown above.
Callers may instead provide a previously validated `render_bundle`, but may not
provide both forms.

## Controller boundary

Render bundles and image payloads are host/offline artifacts. They do not enter
the scalar controller compiler. Future onboard/FPGA paths should consume typed,
bounded optical summaries or explicitly designed pixel streams while retaining
the same sensor/observation hashes. This preserves one optical evidence model
without forcing large browser or image arrays through the current controller
IR.
