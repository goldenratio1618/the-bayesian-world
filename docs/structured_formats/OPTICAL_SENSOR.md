# Optical sensors (`optical-sensor-1`)

`optical-sensor-1` is the strict, backend-neutral intrinsic, timing, noise, and
bounded-transport contract for a pinhole optical sensor. It is consumed by the
offline NumPy/Torch renderers and can describe a physical sensor adapter without
changing observation semantics. The authoritative parser is
`contraption.optics.schemas.OpticalSensor`.

## Coordinate and projection convention

Sensor coordinates are +X image-right, +Y image-down, and +Z forward. Pixels use
`u = fx * X/Z + cx` and `v = fy * Y/Z + cy`. Resolution is always
`[width, height]`; focal length is `[fx, fy]`; principal point is `[cx, cy]`.
The world frame is right-handed, +Z up, +X forward, in metres. A runtime
`Pose` supplies the 4x4 row-major transform from sensor to world coordinates;
it is not stored in this intrinsic descriptor.

## Top-level fields

| Field | Required/default | Meaning |
|---|---|---|
| `format` | required | exactly `optical-sensor-1` |
| `id` | required | nonempty trimmed sensor identity |
| `projection` | required | exactly `pinhole` |
| `resolution_px` | required | positive integer `[width,height]`; at most 16,777,216 pixels |
| `focal_length_px` | required | positive finite `[fx,fy]` in pixels |
| `principal_point_px` | required | finite `[cx,cy]` in pixels |
| `near_clip_m` | 0.01 | positive near distance in metres |
| `far_clip_m` | 100 | far distance in metres and greater than near |
| `exposure_duration_s` | 0.01 | nonnegative integration duration |
| `readout_duration_s` | 0 | nonnegative frame-readout duration |
| `processing_latency_s` | 0 | nonnegative post-readout latency |
| `outputs` | all four | nonempty unique subset of supported products |
| `spectral_channels` | RGB defaults | nonempty channels with unique ids |
| `noise` | default record | sensor noise contract |
| `wire` | default record | bounded future transport contract |
| `display_name` | absent | optional nonempty human label |
| `mount_connector` | absent | optional nonempty physical connector identity |
| `metadata` | `{}` | inert finite JSON |

Supported output names are exactly:

- `rgb_linear`: nonnegative linear RGB radiance after exposure;
- `depth_m`: optical ray distance in metres;
- `segmentation`: integer scene-object label;
- `uncertainty`: predicted depth standard deviation in metres.

## Spectral channels

Each channel requires `id`, `center_wavelength_nm`, and `bandwidth_nm`.
Center wavelength is in [100,3000] nm; bandwidth is positive; optional
`relative_response` is nonnegative and defaults to 1. The default channels are
red 620/100 nm, green 540/100 nm, and blue 460/100 nm. Channel declarations are
calibration evidence; a backend that implements only RGB response must not
claim spectral rendering.

## Noise record

| Field | Default | Validation/semantics |
|---|---:|---|
| `model` | `gaussian_poisson` | `none` or `gaussian_poisson` |
| `seed` | 0 | nonnegative integer mixed with frame/run seed |
| `read_noise_std_linear` | 0 | nonnegative linear-RGB Gaussian standard deviation |
| `shot_noise_scale` | 0 | nonnegative variance scale proportional to signal |
| `depth_noise_std_m` | 0 | nonnegative Gaussian depth standard deviation, metres |
| `depth_quantization_m` | 0 | nonnegative depth quantum, metres; zero disables |
| `dropout_probability` | 0 | probability in [0,1] |

The exact stochastic replay boundary includes backend, device/dtype, scene,
sensor, frame index, and seed. `model: none` disables stochastic application.

## Wire payload record

`wire` reserves a bounded contract for offboard/onboard streaming; it does not
implement an FPGA or controller bridge.

| Field | Default | Allowed values |
|---|---:|---|
| `schema` | `contraption.optical-frame/v1` | exactly that version |
| `encoding` | `cbor-arrays` | `cbor-arrays` or `raw-chunks` |
| `max_payload_bytes` | 8,388,608 | integer at least 256 |
| `max_frame_rate_hz` | 30 | positive finite rate |

A PMDL artifact port intended for controller transport must independently set
`transport: controller_stream` and a positive `max_payload_bytes`. The current
scalar control DSL does not carry image arrays.

### OPFR v1 binary envelope

`contraption.optics.encode_wire_frame` and `decode_wire_frame` implement a
bounded framing envelope for one product payload. The fixed 32-byte header uses
the exact Python `struct` layout `<4sBBHQQII`: little-endian, standard sizes,
and no alignment padding.

| Offset | Width | Header value |
|---:|---:|---|
| 0 | 4 | ASCII magic `OPFR` |
| 4 | 1 | version, exactly unsigned 1 |
| 5 | 1 | unsigned payload-kind id |
| 6 | 2 | unsigned application `flags`, currently opaque |
| 8 | 8 | unsigned `frame_index` |
| 16 | 8 | unsigned `timestamp_ns` |
| 24 | 4 | unsigned payload byte length |
| 28 | 4 | unsigned CRC-32 of payload bytes |

Kind ids are stable:

| Id | `WireFrame.kind` | Payload contract |
|---:|---|---|
| 1 | `rgb_linear` | encoded product bytes selected by `wire.encoding` |
| 2 | `depth_m` | encoded depth product bytes |
| 3 | `segmentation` | encoded segmentation product bytes |
| 4 | `uncertainty` | encoded uncertainty product bytes |
| 5 | `observation_manifest` | encoded optical-observation manifest bytes |

`flags` is any integer in [0,65535]; no flag bits currently have standardized
semantics. `frame_index` and `timestamp_ns` are integers in [0, 2^64-1].
The timestamp's unit is nanoseconds, but the envelope does not define its clock
epoch; a transport integration must define and preserve that clock-domain
contract.

The payload length is in [0, `wire.max_payload_bytes`]. The descriptor requires
that bound to be at least 256; the default is 8,388,608. The encoder computes
`zlib.crc32(payload) & 0xffffffff`, packs the header, and appends the exact
payload bytes. The CRC covers only the payload, not the header. One envelope is
exactly `32 + payload_length` bytes.

Decoding rejects a frame shorter than 32 bytes, wrong magic/version, kind ids
outside 1 through 5, a declared length above the descriptor bound, trailing or
missing bytes, and a payload CRC mismatch. Total input length must equal the
header plus its declared payload.

The OPFR envelope does not serialize arrays by itself. `wire.encoding` states
whether the body contract is `cbor-arrays` or `raw-chunks`; the caller must
encode/decode and validate that product body separately. The envelope also does
not enforce `max_frame_rate_hz`, order frames, retransmit, authenticate, encrypt,
or provide flow control.

This is serialization and validation only. It implements no MIPI CSI-2,
Bluetooth, network, DMA, shared-memory, generated-controller, or FPGA
transport. A future adapter may carry OPFR bytes only after separately
satisfying PMDL artifact-port bounds, clocking, buffering, integrity, safety,
and hardware-interface requirements.

## Example

~~~json
{
  "format": "optical-sensor-1",
  "id": "scanner-camera",
  "projection": "pinhole",
  "resolution_px": [640, 480],
  "focal_length_px": [615.0, 614.5],
  "principal_point_px": [319.5, 239.5],
  "near_clip_m": 0.03,
  "far_clip_m": 4.0,
  "exposure_duration_s": 0.008,
  "readout_duration_s": 0.004,
  "processing_latency_s": 0.006,
  "outputs": ["rgb_linear", "depth_m", "segmentation", "uncertainty"],
  "spectral_channels": [
    {"id": "red", "center_wavelength_nm": 620, "bandwidth_nm": 100, "relative_response": 1},
    {"id": "green", "center_wavelength_nm": 540, "bandwidth_nm": 100, "relative_response": 1},
    {"id": "blue", "center_wavelength_nm": 460, "bandwidth_nm": 100, "relative_response": 1}
  ],
  "noise": {
    "model": "gaussian_poisson",
    "seed": 17,
    "read_noise_std_linear": 0.001,
    "shot_noise_scale": 0.002,
    "depth_noise_std_m": 0.0015,
    "depth_quantization_m": 0.0005,
    "dropout_probability": 0.002
  },
  "wire": {
    "schema": "contraption.optical-frame/v1",
    "encoding": "cbor-arrays",
    "max_payload_bytes": 8388608,
    "max_frame_rate_hz": 30
  },
  "display_name": "Scanner calibrated camera",
  "mount_connector": "camera.optical_axis",
  "metadata": {"calibration_temperature_c": 22.0}
}
~~~

## Hashing, validation, and inverse use

`artifact_sha256` is SHA-256 of canonical compact JSON (sorted keys, no NaN),
represented as 64 lowercase hexadecimal digits without a prefix. Intrinsic
arrays and timing are finite; unknown fields fail.

The differentiable renderer can optimize `sensor.focal_length` as `[fx,fy]`,
`sensor.principal_point` as `[cx,cy]`, and per-view camera
translation/rotation deltas. Distortion, rolling-shutter geometry, and spectral
response are not currently inverse targets. A calibration workflow must select
only exposed targets, use multi-view evidence, declare priors/likelihoods, and
retain the original sensor
hash in each observation.

## Ownership boundary

Calibration ingestion is deterministic host work. Luna may model exposure
commands, trigger timing, bandwidth limits, optical power, scalar derived
features, and typed observation ports, but it must not infer calibration from
images or author an `optical-sensor-1` file.
