# Canonical triangle meshes (CTMESH v1)

CTMESH is Contraption's deterministic, source-independent triangle-surface
payload. The in-memory and serialization implementation is
`contraption.shape.mesh.TriangleMesh`. Source CAD/OBJ/STL/glTF files are evidence;
mechanics, optics, reconstruction, and visualization consume canonical surfaces.

## Coordinate and topology contract

- `vertices_m` is an `N x 3` finite float array in metres in a right-handed local
  frame. At least three vertices are required.
- `triangles` is an `M x 3` integer array of zero-based vertex indices. At least
  one face is required; every index is in range and a face may not repeat an index.
- Counter-clockwise winding defines the outward front face.
- `vertex_normals`, when present, is `N x 3`, finite, nonzero, and normalized by
  the constructor. If absent, deterministic area-weighted vertex normals are
  computed; zero-area faces or undefined vertex normals are errors.
- `vertex_rgba_linear`, when present, is `N x 4` with values in [0, 1] in linear
  color space.
- `face_material`, when present, is a length-`M` array of nonnegative integer
  indices into the shape artifact's ordered optical-material list.
- `dimensions_m` is the axis-aligned local bounding-box extent computed from
  vertices; it is not an independently authored scale.
- `watertight` is true only when each undirected mesh edge occurs exactly twice.
  It does not by itself prove outward orientation, absence of self-intersection,
  manifold vertices, or suitability for mass integration.

## Inline JSON form

The inline form is intended for small fixtures and tests:

~~~json
{
  "schema": "contraption.triangle-mesh/v1",
  "vertices_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
  "triangles": [[0, 1, 2]],
  "vertex_normals": [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
  "vertex_rgba_linear": [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
  "face_material": [0]
}
~~~

Allowed keys are exactly `schema`, `vertices_m`, `triangles`,
`vertex_normals`, optional `vertex_rgba_linear`, and optional `face_material`.
`schema` is exactly `contraption.triangle-mesh/v1`.

## Binary CTMESH1 form

The byte layout is:

1. eight-byte magic `CTMESH1\n`;
2. little-endian unsigned 32-bit JSON-header length;
3. compact UTF-8 JSON header;
4. contiguous binary array payloads.

The header has exactly:

~~~json
{
  "schema": "contraption.ctmesh/v1",
  "arrays": [
    {
      "name": "vertices_m",
      "dtype": "<f4",
      "shape": [8, 3],
      "offset": 0,
      "bytes": 96
    }
  ]
}
~~~

Every array descriptor has exactly `name`, NumPy `dtype` string, `shape`,
byte `offset` relative to the payload body, and byte count. Required arrays are
`vertices_m` (`<f4`), `triangles` (`<u4`), and `vertex_normals` (`<f4`).
Optional arrays are `vertex_rgba_linear` (`<f4`) and `face_material` (`<u4`).
The reader validates array bounds, reshape length, allowed names, and the
in-memory topology contract.

Canonical analysis retains float64 vertices in memory. Serialization deliberately
uses little-endian float32 positions/normals and uint32 topology for efficient
CPU/GPU transfer. Therefore derived mechanical properties must record the exact
CTMESH content hash and numerical method; they must not imply exact BREP
integration.

## GLB source or runtime cache

`TriangleMesh.to_glb_bytes()` can emit deterministic glTF 2.0 binary with
float32 POSITION/NORMAL attributes, uint32 triangle indices, computed metric
bounds, one default opaque PBR material, and no source scale/axis transform.

GLB is never a canonical surface. An original GLB remains under
`SourceRepresentation` evidence; a GLB emitted from canonical CTMESH belongs in
the artifact's disposable `caches`. Render/ray backends may consume that cache
only when its dependency on the exact CTMESH is retained, and must be able to
regenerate it. Backend-specific BVHs and acceleration structures follow the
same cache rule. Mechanics, optical truth, and source-independent hashes bind
the one CTMESH surface contract.

## Mechanical and optical use

Mass-property integration requires a closed, consistently oriented, qualified
surface plus a density/material field. A partial or non-watertight optical scan
does not acquire mass by being triangulated.

Ray traversal may use any qualified surface regardless of watertightness, but
optical behavior comes from explicit `optical-material-1` assignments, not
vertex color alone. Vertex color is display/radiometric evidence and never
silently becomes density, opacity, roughness, or refractive index.

## Deterministic ownership

Luna may understand and reference the CTMESH abstraction but never emits,
repairs, simplifies, or hashes CTMESH or derived GLB payloads. The deterministic
ingestion
pipeline owns conversion settings, topology diagnostics, normals, material
indices, LOD generation, hashes, and source provenance.
