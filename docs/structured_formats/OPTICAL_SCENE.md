# Optical scenes (`optical-scene-1`)

`optical-scene-1` binds canonical shape artifacts, instance poses, segmentation
identities, lights, and environment radiance for optical simulation. It contains
no source CAD or duplicated triangle arrays. The authoritative parser and
content verifier are `contraption.optics.schemas.OpticalScene`.

## Top-level fields

| Field | Required/default | Meaning |
|---|---|---|
| `format` | required | exactly `optical-scene-1` |
| `id` | required | nonempty trimmed scene identity |
| `canonical_frame` | required | exactly `right_handed_z_up_x_forward_metres` |
| `objects` | required | nonempty array with unique object and segmentation ids |
| `lights` | `[]` | lights with unique ids |
| `environment_linear_rgb` | `[0.02,0.02,0.02]` | three nonnegative linear environment-radiance values |
| `metadata` | `{}` | inert finite JSON |

Unknown keys, non-finite numbers, path traversal, and unsupported frame/version
values are rejected.

## Scene objects

| Field | Required/default | Semantics |
|---|---|---|
| `id` | required | unique nonempty object id |
| `shape_artifact_uri` | required | safe POSIX path relative to the scene manifest |
| `shape_artifact_sha256` | required | shape manifest canonical digest: 64 lowercase hex digits, no prefix |
| `segmentation_id` | required | positive integer unique in the scene |
| `transform_world_from_object_row_major` | identity | finite, rigid, right-handed local/object-to-world homogeneous 4x4 transform |
| `surface_id` | absent | exact surface id within the shape artifact; otherwise select its `ray_trace` surface |
| `surface_uncertainty_m` | 0 | nonnegative surface standard-deviation contribution in metres |

`transform_world_from_object_row_major` contains 16 finite row-major numbers
and defaults to identity when absent; serialization always emits it. Its final
row must be `[0,0,0,1]`, translations are metres, and the 3x3 rotation block
must have orthonormal rows and determinant `+1`. Scale, shear, and reflection
are invalid. Unit/axis normalization of the source asset must be recorded in
the shape artifact rather than hidden here.

Loading with verification resolves every shape below the scene directory,
validates all content hashes in the `shape-artifact-1` manifest, compares the
manifest's canonical digest with `shape_artifact_sha256`, and verifies
`surface_id` when present.

## Lights

Every light has required `id`, `kind`, `color_linear_rgb`, and `intensity`.
`kind` is `point` or `directional`. Color contains three nonnegative linear
values and intensity is finite and nonnegative.

- A point light requires `position_m: [x,y,z]`. The current backend applies
  inverse-square attenuation as `intensity/(4*pi*r^2)`.
- A directional light requires a nonzero `direction_world`. The parser
  normalizes the vector and the renderer treats it as the direction traveled by
  incoming light.
- The numerical intensity scale is the renderer's declared linear radiometric
  scale. Do not label it as calibrated SI power without calibration metadata
  and a backend that implements that interpretation.

## Example

~~~json
{
  "format": "optical-scene-1",
  "id": "scanner-bench",
  "canonical_frame": "right_handed_z_up_x_forward_metres",
  "objects": [
    {
      "id": "target",
      "shape_artifact_uri": "shapes/target/shape.json",
      "shape_artifact_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "segmentation_id": 1,
      "transform_world_from_object_row_major": [
        1, 0, 0, 0.50,
        0, 1, 0, 0.00,
        0, 0, 1, 0.02,
        0, 0, 0, 1
      ],
      "surface_id": "ray-surface",
      "surface_uncertainty_m": 0.0005
    }
  ],
  "lights": [
    {
      "id": "inspection-light",
      "kind": "point",
      "color_linear_rgb": [1.0, 0.96, 0.90],
      "intensity": 12.0,
      "position_m": [0.2, -0.2, 0.5]
    }
  ],
  "environment_linear_rgb": [0.01, 0.01, 0.01],
  "metadata": {"purpose": "scanner calibration"}
}
~~~

## Runtime and inverse physics

`RuntimeScene.from_manifest` converts the selected canonical CTMESH surfaces and
embedded optical materials into backend-neutral instances. The NumPy renderer
provides an exact CPU baseline; the Torch renderer performs vectorized
CPU/CUDA primary-ray rendering. Both produce only products requested by the
sensor. Visibility selection is discontinuous at occlusion boundaries; the
Torch path is differentiable through the currently selected intersections,
shading, intrinsics, poses, materials, and lights.

The inverse solver exposes these scene targets:

- `geometry.vertices`;
- `materials.base_color`, `roughness`, `metallic`, `transmission`,
  `refractive_index`, and `emission`;
- `lights.position`, `direction`, `color`, and `intensity`;
- `environment.radiance`.

Per-view camera and sensor-intrinsic targets are documented with observations.
Use priors, parameter transforms, multiple views, and held-out residual checks.
A successful numerical fit does not overwrite a source shape: it produces new
posterior/reconstruction evidence with provenance and uncertainty.

## Ownership boundary

The scene compiler is deterministic host code. Luna may describe optical
behavioral abstractions and typed ports, but it must not choose source geometry
bytes, compute object transforms from assets, calculate hashes, or author this
format.
