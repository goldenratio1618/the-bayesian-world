# Reconstruction states (`reconstruction-state-1`)

`reconstruction-state-1` is the immutable, hash-bound snapshot of the sparse
Bayesian occupancy/TSDF map produced while a scanner explores a scene. The
manifest stays small; independently addressable compressed voxel blocks permit
incremental updates and content reuse. Authoritative implementations are
`contraption.optics.schemas.ReconstructionState` and
`contraption.optics.reconstruction.SparseBayesianReconstruction`.
For exact capture/fusion commands and run-report closure, see
[Offline optical workflows](./OPTICAL_WORKFLOWS.md).

## Manifest fields

| Field | Required/default | Meaning |
|---|---|---|
| `format` | required | exactly `reconstruction-state-1` |
| `id` | required | nonempty trimmed reconstruction identity |
| `representation` | required | exactly `sparse_bayesian_tsdf_occupancy` |
| `canonical_frame` | required | exactly `right_handed_z_up_x_forward_metres` |
| `voxel_size_m` | required | positive finite cubic voxel edge |
| `block_size` | required | integer in [2,64]; voxels per block axis |
| `origin_world_m` | required | finite world-space location of voxel index [0,0,0] |
| `truncation_distance_m` | required | at least one voxel; converts metric signed distance to normalized TSDF |
| `occupancy_prior_probability` | required | probability strictly between 0 and 1 |
| `occupied_probability` | required | inverse-sensor probability with `prior < occupied < 1` |
| `free_probability` | required | inverse-sensor probability with `0 < free < prior` |
| `min_log_odds` | required | finite lower occupancy-evidence clamp |
| `max_log_odds` | required | finite upper clamp greater than minimum |
| `update_count` | required | nonnegative number of fused frame updates |
| `blocks` | required | unique signed block-index references; may be empty |
| `observation_sha256` | `[]` | ordered exact observation-manifest digests |
| `metadata` | `{}` | inert finite JSON describing posterior/update provenance |

A block reference contains `index: [bx,by,bz]` with three signed integers and a
content reference. Content references use a safe relative POSIX URI, media type,
bare 64-lowercase-hex `sha256`, and nonnegative `byte_length`. Loading with
verification rejects traversal and checks every referenced payload byte count
and digest. The canonical media type is
`application/vnd.contraption.sparse-voxel-block`.

## Example

~~~json
{
  "format": "reconstruction-state-1",
  "id": "scanner-map",
  "representation": "sparse_bayesian_tsdf_occupancy",
  "canonical_frame": "right_handed_z_up_x_forward_metres",
  "voxel_size_m": 0.005,
  "block_size": 8,
  "origin_world_m": [-0.5, -0.5, 0.0],
  "truncation_distance_m": 0.02,
  "occupancy_prior_probability": 0.5,
  "occupied_probability": 0.7,
  "free_probability": 0.35,
  "min_log_odds": -8.0,
  "max_log_odds": 8.0,
  "update_count": 42,
  "blocks": [
    {
      "index": [12, -2, 4],
      "content": {
        "uri": "block_12_-2_4.svox",
        "media_type": "application/vnd.contraption.sparse-voxel-block",
        "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "byte_length": 8192
      }
    }
  ],
  "observation_sha256": [
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  ],
  "metadata": {
    "posterior": "independent Bernoulli occupancy and Gaussian TSDF/color"
  }
}
~~~

The byte length is illustrative; exact compression depends on posterior values.

## Sparse voxel block v2

A block payload is:

1. ASCII magic `SVOXBLK2\n`;
2. little-endian unsigned 32-bit JSON-header length;
3. compact UTF-8 header;
4. zlib level-6 compressed array body.

The header has exactly `schema`
(`contraption.sparse-voxel-block/v2`), `index`, `block_size`, `arrays`,
`uncompressed_byte_length`, and `uncompressed_crc32`. Each array descriptor has
exactly `name`, NumPy `dtype`, `shape`, byte `offset`, and `byte_length`.
The embedded three-integer `index` must exactly match the manifest binding.
Arrays occur in the table order below and occupy one contiguous body with no
gaps or trailing bytes. The reader checks exact header fields, semantic array
layout, CRC, size, dtype, shape, completeness, and block size, and accepts one
well-framed zlib stream with no trailing compressed data.

Every block contains:

| Name | Type/shape | Posterior meaning |
|---|---|---|
| `occupancy_log_odds` | little-endian f32 `[B,B,B]` | bounded Bernoulli log odds |
| `tsdf_mean` | little-endian f32 `[B,B,B]` | normalized signed-distance mean |
| `tsdf_precision` | little-endian f32 `[B,B,B]` | nonnegative inverse variance |
| `color_mean` | little-endian f32 `[B,B,B,3]` | linear RGB posterior mean |
| `color_precision` | little-endian f32 `[B,B,B]` | shared nonnegative RGB precision |
| `update_count` | little-endian u32 `[B,B,B]` | voxel update count |

All floating arrays must be finite. Empty voxels begin at prior occupancy,
TSDF mean 1, zero TSDF/color precision, zero color, and zero update count.

## Bayesian update physics

`update_observation` first verifies the sensor id/hash and all observation
sidecars and requires `depth_m`. Optional `uncertainty` supplies per-pixel depth
standard deviation; otherwise the update uses the larger of declared sensor
depth noise and half a voxel. Optional linear RGB updates the surface voxel.

For each valid depth ray:

- voxels before the measured surface receive a free-space log-odds increment;
- the surface neighborhood receives an occupied log-odds increment;
- log odds are clipped to configured bounds;
- normalized truncated signed distance is fused by Gaussian
  mean/precision addition;
- surface color is fused by independent Gaussian mean/precision addition;
- duplicate consecutive voxels along one ray are removed deterministically.

Only observed blocks are allocated. The default operational settings are
voxel 0.005 m, block size 8, truncation four voxels, prior 0.5, occupied
probability 0.7, free probability 0.35, and log-odds bounds [-8,8]. The manifest
persists every one of these continuation parameters, so loading resumes with
the same inverse-sensor model and clamping. Acquisition choices such as
`pixel_stride` still belong in provenance when exact replay matters.

`pixel_stride` can trade spatial sampling for speed without changing the schema.
Fusing independent Gaussian measurements is order-independent up to
floating-point roundoff; clamped occupancy log odds and finite precision still
make backend/dtype/order part of a strict replay claim.

## Posterior queries and active scanning

`voxel_posterior` returns `occupancy_probability`, `tsdf_mean`,
`tsdf_standard_deviation_m`, `color_mean`, and `update_count`.
`surface_points` extracts likely near-zero-TSDF
voxel centers with color and metric standard deviation.
`expected_information_gain` sums Bernoulli occupancy entropy along candidate
camera rays, and `rank_candidate_views` orders poses for next-best-view
selection. These are planning scores, not proof that a view is collision-free
or controller-safe.

`as_shape_volume` exposes the written manifest as a
`shape-artifact-1` `VolumeRepresentation` of kind `sparse_tsdf`, with purposes
`reconstruction` and `ray_trace`, mutable topology, voxel dimensions inferred
from allocated blocks, and one-voxel normal uncertainty. A meshing/extraction
step must create and hash a new canonical surface; consumers must not relabel
the sparse posterior as an exact CAD surface.

## Relationship to differentiable inverse solving

Sparse fusion provides a scalable online Bayesian map. The separate
differentiable inverse solver performs joint multi-view MAP optimization of
continuous geometry, material, light, camera, and focal-length parameters and
can return a local Laplace covariance. Either result may seed the other, but
they remain different posterior approximations with explicit provenance. Never
discard observation hashes or collapse their uncertainty into an unqualified
mesh.

## Ownership boundary

Only the reconstruction engine writes manifests/blocks. Luna may declare typed
reconstruction artifact ports and scalar exploration policies, but it may not
read observations, update voxel arrays, invent posterior values, or author this
format.
