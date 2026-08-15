# Optical materials (`optical-material-1`)

An `optical-material-1` JSON document is a standalone, deterministic library of
surface and participating-medium properties imported beside a source shape.
The same `OpticalMaterial` record may be embedded in `shape-artifact-1`. The
authoritative parser is `contraption.shape.artifacts.OpticalMaterialLibrary`.
Unknown fields, duplicate JSON keys, non-finite numbers, and unsupported model
names are invalid.

This record describes optical response. Density, elasticity, thermal response,
and mass properties belong in shape physical fields or PMDL.

## Library fields

| Field | Required/default | Meaning |
|---|---|---|
| `format` | required | exactly `optical-material-1` |
| `id` | required | nonempty trimmed library identity |
| `version` | required | nonempty trimmed version |
| `materials` | required | nonempty array of materials with unique `id` values |
| `provenance` | `{}` | inert JSON describing exact sources and extraction |

## Material fields

Only `id` is syntactically required; omitted physical values take the explicit
defaults below.

| Field | Default | Units and semantics |
|---|---:|---|
| `id` | required | unique nonempty material identity |
| `model` | `principled` | `lambertian`, `principled`, `dielectric`, `conductor`, `emissive`, or `measured` |
| `base_color_linear_rgba` | `[0.5,0.5,0.5,1]` | four linear channels, each in [0,1]; never sRGB-encoded |
| `roughness` | 0.5 | microsurface roughness in [0,1] |
| `metallic` | 0 | conductor blend in [0,1] |
| `transmission` | 0 | transmitted energy fraction in [0,1] |
| `refractive_index` | 1.5 | real refractive index, at least 1 |
| `extinction_coefficient` | 0 | nonnegative imaginary-index coefficient |
| `absorption_per_m` | `[0,0,0]` | nonnegative RGB absorption coefficients in m^-1 |
| `scattering_per_m` | `[0,0,0]` | nonnegative RGB scattering coefficients in m^-1 |
| `phase_anisotropy` | 0 | scattering asymmetry in [-1,1] |
| `emission_linear_rgb` | `[0,0,0]` | nonnegative linear emitted-radiance coefficients |
| `double_sided` | false | Boolean two-sided surface response |
| `spectrum` | `[]` | ordered measured spectral samples |
| `uncertainty` | fixed | `ShapeUncertainty` record |
| `provenance` | `{}` | inert source/extraction JSON |

`spectrum` wavelengths must be strictly increasing and unique. Each sample
requires `wavelength_nm` in [100, 1,000,000] nm and may contain
`reflectance`/`transmittance` in [0,1], or nonnegative `refractive_index`,
`extinction_coefficient`, and `emission_w_sr_m2_nm`. A consumer must document
its interpolation and out-of-band policy; it must not silently reinterpret
spectral samples as sRGB.

Uncertainty has only `distribution`, `parameters`, and optional
`correlation_group`. Distribution is `fixed`, `normal`, `lognormal`,
`uniform`, `triangular`, or `empirical`. Parameters are finite JSON values
whose meaning must be stated by the producing importer; matching correlation
groups identify values that should not be sampled independently.

## Example

~~~json
{
  "format": "optical-material-1",
  "id": "camera-window-materials",
  "version": "1.0.0",
  "materials": [
    {
      "id": "borosilicate",
      "model": "dielectric",
      "base_color_linear_rgba": [0.96, 0.98, 1.0, 1.0],
      "roughness": 0.02,
      "metallic": 0.0,
      "transmission": 0.93,
      "refractive_index": 1.47,
      "extinction_coefficient": 0.0,
      "absorption_per_m": [0.03, 0.02, 0.01],
      "scattering_per_m": [0.0, 0.0, 0.0],
      "phase_anisotropy": 0.0,
      "emission_linear_rgb": [0.0, 0.0, 0.0],
      "double_sided": false,
      "spectrum": [
        {"wavelength_nm": 460, "transmittance": 0.91, "refractive_index": 1.48},
        {"wavelength_nm": 620, "transmittance": 0.94, "refractive_index": 1.47}
      ],
      "uncertainty": {
        "distribution": "normal",
        "parameters": {"roughness_standard_deviation": 0.005},
        "correlation_group": "window-coating-batch"
      },
      "provenance": {
        "kind": "vendor",
        "source": "vendor spectral table",
        "conversion": "deterministic linear-value import"
      }
    }
  ],
  "provenance": {"source_sha256": "64-lowercase-hex-digits"}
}
~~~

## Physics and backend behavior

The exact forward model is a backend capability, not implied by a field name.
The current primary-ray NumPy and Torch backends use linear base color,
roughness, metallic, transmission, refractive index, and emission. They use a
Lambertian-plus-specular direct-light approximation and a one-surface
transmission split; they do not claim full volumetric transport, wavelength
dispersion, multiple refraction, or polarization. Those richer fields remain
standardized evidence for a capable path/spectral backend.

The differentiable backend exposes base color, roughness, metallic,
transmission, refractive index, and emission as inverse-problem targets. A
solver should carry the declared uncertainty into priors rather than treating
an imported nominal value as exact.

## Deterministic ownership

The built-in deterministic shape importer currently extracts optical properties
from OBJ/MTL and from a strict adjacent `*.optical.json` sidecar. Its MTL
projection records linearized diffuse color, opacity/transmission, roughness
(or a shininess-derived value), metallic response, refractive index, emission,
and illumination-derived material classification with explicit provenance.
Because MTL contains nominal values but no metrology uncertainty, imported MTL
materials receive a non-fixed `uniform` uncertainty record stating that
limitation. A sidecar supplies the full strict optical-material library.
Explicit host materials take precedence over a sidecar, which takes precedence
over MTL-discovered materials.

Other CAD, GLB/glTF, vendor-table, or measured spectral inputs require an
explicit deterministic tessellator/property importer before they enter this
contract. They are not automatically parsed by the current built-in workflow,
and their properties must not be guessed from filenames or visual appearance.
