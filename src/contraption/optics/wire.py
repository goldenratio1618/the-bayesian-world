"""Bounded optical frame wire envelope reserved for future controllers.

This module intentionally implements only serialization and validation. It
does not claim a MIPI, FPGA, Bluetooth, or other hardware transport path.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib

from .schemas import WirePayloadSpec


_MAGIC = b"OPFR"
_VERSION = 1
_HEADER = struct.Struct("<4sBBHQQII")
_KINDS = {"rgb_linear": 1, "depth_m": 2, "segmentation": 3, "uncertainty": 4, "observation_manifest": 5}
_KINDS_BY_ID = {value: key for key, value in _KINDS.items()}


class WirePayloadError(ValueError):
    """Raised when a wire payload is unsupported, corrupt, or out of bounds."""


@dataclass(frozen=True, slots=True)
class WireFrame:
    kind: str
    frame_index: int
    timestamp_ns: int
    payload: bytes
    flags: int = 0


def encode_wire_frame(frame: WireFrame, spec: WirePayloadSpec = WirePayloadSpec()) -> bytes:
    if frame.kind not in _KINDS:
        raise WirePayloadError(f"unsupported optical wire payload kind {frame.kind!r}")
    if isinstance(frame.frame_index, bool) or frame.frame_index < 0 or frame.frame_index >= 2**64:
        raise WirePayloadError("wire frame index is outside uint64")
    if isinstance(frame.timestamp_ns, bool) or frame.timestamp_ns < 0 or frame.timestamp_ns >= 2**64:
        raise WirePayloadError("wire timestamp is outside uint64")
    if isinstance(frame.flags, bool) or not 0 <= frame.flags < 2**16:
        raise WirePayloadError("wire flags are outside uint16")
    payload = bytes(frame.payload)
    if len(payload) > spec.max_payload_bytes:
        raise WirePayloadError(f"wire payload has {len(payload)} bytes; limit is {spec.max_payload_bytes}")
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    return _HEADER.pack(_MAGIC, _VERSION, _KINDS[frame.kind], frame.flags, frame.frame_index, frame.timestamp_ns, len(payload), checksum) + payload


def decode_wire_frame(payload: bytes, spec: WirePayloadSpec = WirePayloadSpec()) -> WireFrame:
    raw = bytes(payload)
    if len(raw) < _HEADER.size:
        raise WirePayloadError("optical wire frame is truncated")
    magic, version, kind_id, flags, frame_index, timestamp_ns, length, checksum = _HEADER.unpack(raw[: _HEADER.size])
    if magic != _MAGIC or version != _VERSION:
        raise WirePayloadError("unsupported optical wire magic or version")
    if kind_id not in _KINDS_BY_ID:
        raise WirePayloadError("unknown optical wire payload type")
    if length > spec.max_payload_bytes or len(raw) != _HEADER.size + length:
        raise WirePayloadError("optical wire payload length is invalid or exceeds its bound")
    body = raw[_HEADER.size :]
    if zlib.crc32(body) & 0xFFFFFFFF != checksum:
        raise WirePayloadError("optical wire payload CRC mismatch")
    return WireFrame(_KINDS_BY_ID[kind_id], frame_index, timestamp_ns, body, flags)


__all__ = ["WireFrame", "WirePayloadError", "decode_wire_frame", "encode_wire_frame"]
