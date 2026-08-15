# Shape artifact (shape-artifact-1)

`shape-artifact-1` is the source-independent manifest shared by mechanics,
optics, reconstruction, and visualization. Immutable original files remain
evidence under `sources`. Runtime systems consume canonical `surfaces`,
`volumes`, `optical_materials`, and `physical_fields` instead of interpreting
STEP, FCStd, OBJ, scans, or other source formats. The authoritative records are
in `contraption.shape.artifacts`.

## Top-level manifest

Required keys are `format`, `id`, `version`, `canonical_frame`, `sources`, and
`surfaces`. Unknown keys are invalid.

| Field | Type | Rule |
|---|---|---|
| `format` | string | exactly `shape-artifact-1` |
| `id` | trimmed nonempty string | stable shape identity |
| `version` | trimmed nonempty string | manifest version |
| `canonical_frame` | string | exactly `right_handed_z_up_x_forward_metres` |
| `sources` | source array | nonempty immutable input evidence |
| `surfaces` | surface array | canonical CTMESH surfaces; may be empty only when a volume exists |
| `volumes` | volume array | optional canonical volumetric forms |
| `optical_materials` | embedded optical-material array | optional surface material definitions |
| `physical_fields` | field array | density, elasticity, or other spatial quantities |
| `derived_mass_properties` | object or null | reproducible result linked to a surface and density field |
| `caches` | content-reference array | disposable backend-specific data |
| `provenance` | object | manifest-level evidence and derivation record |
| `metadata` | object | inert JSON |

Source, surface, volume, optical-material, and physical-field ids are unique
within their respective arrays. Every `surface.material_ids` entry must resolve
to an embedded optical material. At least one canonical surface or volume is
required.

## Content reference

Every external payload is referenced with exactly:

| Field | Type | Rule |
|---|---|---|
| `uri` | string | contained POSIX path relative to the manifest; no absolute path, backslash, NUL, `.`, or `..` segment |
| `media_type` | trimmed string | MIME/media type |
| `sha256` | string | exactly 64 lowercase hex characters, without a `sha256:` prefix |
| `byte_length` | integer | nonnegative exact byte count |

`ShapeArtifact.verify()` resolves each content reference below the artifact
directory, reads exact bytes, and verifies length and SHA-256. It verifies
sources, surfaces, volumes, array-backed physical fields, and caches. Every
surface is decoded as strict CTMESH; vertex/triangle counts, bounds,
watertight/manifold diagnostics, and face-material indices are checked against
the manifest. Multiple materials require an explicit per-face index map.

For sources with format `optical_sensor` or `optical_observation`, verification
also parses the nested strict manifest; observation verification includes every
relative output sidecar. Every `sparse_tsdf` volume must use the canonical
reconstruction-state media type; its state and every referenced SVOX block are
parsed and verified. Derived mass properties are recomputed from the named
closed, oriented CTMESH and positive constant `mass_density` field. These
transitive checks make an artifact invalid when geometry metadata, nested
evidence, or posterior bytes change.

## Uncertainty

`ShapeUncertainty` is used on canonical representations, materials, fields, and
derived mass properties:

| Field | Type | Values/default |
|---|---|---|
| `distribution` | string | `fixed` (default), `normal`, `lognormal`, `uniform`, `triangular`, or `empirical` |
| `parameters` | object | finite JSON distribution parameters; default empty |
| `correlation_group` | trimmed string or absent | shared-dependence group |

An uncertainty record states uncertainty in that representation or value. It
does not grant a capability such as watertightness or mechanical validity.
An empirical surface uncertainty must name `parameters.field_id`. That field
must be a `per_vertex` `surface_position_standard_deviation` in metres backed
by a canonical little-endian float32 NPY vector with exactly one finite,
nonnegative value per CTMESH vertex.

The deterministic part importer strict-parses an explicitly supplied
`surface_uncertainty`, preserves it on the canonical CTMESH surface, and
copies the same record onto any density-derived mass properties. If the source
plan omits it, the importer records a conservative uniform positional interval
with lower/upper bounds
`+/- max(1e-7 m, 0.005 * maximum_model_extent_m)` and a basis explaining that
the source declared no metrology uncertainty. That default is intentionally not
`fixed`. OBJ/MTL material values are likewise nominal rather than measured and
receive non-fixed uniform uncertainty unless an explicit strict optical library
supplies a better record.

## Source representation

| Field | Required | Meaning |
|---|---:|---|
| `id` | yes | source identity |
| `format` | yes | one supported source kind |
| `content` | yes | exact original payload reference |
| `metres_per_source_unit` | yes | positive conversion to metres |
| `transform_to_canonical_row_major` | no | 16-number 4x4 row-major transform; identity default |
| `provenance` | no | inert evidence/derivation object |
| `license` | no | trimmed license expression/text |

Supported `format` values are `step`, `brep`, `fcstd`, `cadquery`,
`openscad`, `stl`, `obj`, `ply`, `gltf`, `glb`, `point_cloud`,
`depth_frames`, `scan_frames`, `optical_sensor`,
`optical_observation`, `ctmesh`, `procedural`, `material_library`, and
`texture`.

A source is never silently canonical. Its explicit scale and transform record
the normalization operation. Sources remain unchanged and may be regenerated
through a newer deterministic importer.

## Surface representation

| Field | Required | Rule |
|---|---:|---|
| `id` | yes | unique surface identity |
| `kind` | yes | exactly `ctmesh` |
| `content` | yes | exact canonical payload |
| `purposes` | yes | nonempty unique list drawn from `analysis`, `ray_trace`, `render`, `collision` |
| `vertex_count` | yes | positive integer |
| `triangle_count` | yes | positive integer |
| `bounds_m` | yes | six finite values [min-x, min-y, min-z, max-x, max-y, max-z] |
| `watertight` | yes | explicit Boolean diagnostic |
| `manifold` | yes | explicit Boolean diagnostic |
| `material_ids` | no | unique references to embedded materials |
| `uncertainty` | no | representation uncertainty |

Bounds minima may not exceed maxima. A purpose is a validated intended use,
not a cosmetic label. `surface_for(purpose)` fails when no canonical CTMESH
declares that purpose. GLB may be preserved as original source evidence or
stored under `caches` as a disposable runtime projection derived from CTMESH; it
is never a canonical `surfaces` entry.

## Volume representation

| Field | Required | Rule |
|---|---:|---|
| `id` | yes | unique volume identity |
| `kind` | yes | `tetrahedral_mesh`, `sparse_tsdf`, `sparse_sdf`, `sparse_occupancy`, or `nanovdb` |
| `content` | yes | exact canonical payload |
| `purposes` | yes | nonempty list using `mechanics`, `ray_trace`, `reconstruction`, or `collision` |
| `voxel_size_m` | no | positive finite resolution |
| `dimensions` | no | three positive integer extents |
| `mutable_topology` | no | Boolean, default false |
| `uncertainty` | no | volume uncertainty |

The mutable Bayesian reconstruction representation normally uses sparse blocks
with `mutable_topology: true`. NanoVDB is normally a compiled immutable snapshot
or cache, not the authoritative mutable posterior.

## Embedded optical material

The manifest embeds ordered `OpticalMaterial` records so CTMESH
`face_material` indices and `surface.material_ids` have a source-independent
meaning. Each record has:

- required trimmed `id`;
- `model`: `lambertian`, `principled` (default), `dielectric`, `conductor`,
  `emissive`, or `measured`;
- linear `base_color_linear_rgba` in [0,1], default [0.5,0.5,0.5,1];
- scalar `roughness`, `metallic`, and `transmission` in [0,1];
- `refractive_index` at least 1 and nonnegative `extinction_coefficient`;
- nonnegative three-channel `absorption_per_m` and `scattering_per_m`;
- `phase_anisotropy` in [-1,1];
- nonnegative `emission_linear_rgb`;
- Boolean `double_sided`;
- increasing, unique `spectrum` samples;
- `uncertainty` and inert `provenance`.

A spectral sample requires `wavelength_nm` in [100, 1,000,000]. Optional
`reflectance` and `transmittance` are in [0,1]. Optional `refractive_index`,
`extinction_coefficient`, and `emission_w_sr_m2_nm` are nonnegative.

The standalone `optical-material-1` format adds an artifact identity/envelope
around the same optical abstraction; see [Optical materials](./OPTICAL_MATERIAL.md).

## Physical fields

A field declares `id`, physical `quantity`, parseable/documented `unit`, and one
`representation`:

| Representation | Required data |
|---|---|
| `constant` | `constant_value` |
| `per_material` | nonempty `material_values` mapping |
| `per_vertex` | `content` |
| `per_cell` | `content` |
| `voxel_grid` | `content` |

Optional `uncertainty` applies in all cases. Numeric values are finite. A
density field must use quantity `mass_density` and unit `kg/m^3`; mechanics must
not infer density from optical appearance.

## Derived mass properties

The record requires:

- `source_surface`: exact surface id used for integration;
- `density_field`: exact physical-field id;
- positive `mass_kg` and `volume_m3`;
- three-value `center_of_mass_m`;
- nine-value row-major `inertia_kg_m2_row_major` about the center of mass;
- optional `uncertainty`.

The current deterministic mass integrator requires a watertight, consistently
oriented CTMESH and a positive constant density. It integrates signed
tetrahedra, rejects nonpositive signed volume/mass, symmetrizes inertia, and
records the result as derived data. Partial scan surfaces do not support this
derivation until an uncertainty-aware closed volume is qualified.

## Caches and authority

Caches are only content references. Examples include OptiX/Vulkan acceleration
data, collision BVHs, and NanoVDB snapshots. They are replaceable, tied to exact
canonical inputs, and never authoritative evidence.

The manifest's `artifact_sha256` is SHA-256 of compact, key-sorted canonical
manifest JSON and is returned without a `sha256:` prefix. Whitespace changes do
not change that digest; semantic field changes do.

## Minimal example

~~~json
{
  "format": "shape-artifact-1",
  "id": "example.bracket",
  "version": "1.0.0",
  "canonical_frame": "right_handed_z_up_x_forward_metres",
  "sources": [
    {
      "id": "vendor_step",
      "format": "step",
      "content": {
        "uri": "source/bracket.step",
        "media_type": "model/step",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "byte_length": 0
      },
      "metres_per_source_unit": 0.001,
      "provenance": {"kind": "vendor"}
    }
  ],
  "surfaces": [
    {
      "id": "analysis",
      "kind": "ctmesh",
      "content": {
        "uri": "canonical/analysis.ctmesh",
        "media_type": "application/vnd.contraption.ctmesh",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "byte_length": 0
      },
      "purposes": ["analysis", "ray_trace", "collision"],
      "vertex_count": 8,
      "triangle_count": 12,
      "bounds_m": [-0.01, -0.01, -0.005, 0.01, 0.01, 0.005],
      "watertight": true,
      "manifold": true,
      "material_ids": ["paint"]
    }
  ],
  "volumes": [],
  "optical_materials": [
    {"id": "paint", "model": "principled", "roughness": 0.7}
  ],
  "physical_fields": [
    {
      "id": "density",
      "quantity": "mass_density",
      "unit": "kg/m^3",
      "representation": "constant",
      "constant_value": 1200.0
    }
  ],
  "derived_mass_properties": null,
  "caches": [],
  "provenance": {},
  "metadata": {}
}
~~~

The zero hashes/lengths above are structural placeholders only; a real manifest
must reference existing exact bytes and pass `verify()`.

## Luna boundary

Luna receives this guide so it understands shape capabilities and can author
compatible PMDL behavior. It never emits or edits this manifest or any payload.
Deterministic ingestion creates it and bundles its validated references with
the separately validated Luna catalog proposal.
