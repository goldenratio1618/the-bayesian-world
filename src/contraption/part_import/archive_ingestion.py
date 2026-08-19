"""Bounded, fail-closed extraction of declared shape members from ZIP archives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


class DeterministicArchiveError(ValueError):
    """Raised when an archive cannot be treated as deterministic shape evidence."""


@dataclass(frozen=True, slots=True)
class ZipLimits:
    max_archive_bytes: int = 64 * 1024 * 1024
    max_entries: int = 2_048
    max_member_bytes: int = 256 * 1024 * 1024
    max_total_uncompressed_bytes: int = 512 * 1024 * 1024
    max_compression_ratio: int = 200

    def __post_init__(self) -> None:
        for name in (
            "max_archive_bytes",
            "max_entries",
            "max_member_bytes",
            "max_total_uncompressed_bytes",
            "max_compression_ratio",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DeterministicArchiveError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ExtractedShapeArchive:
    archive_sha256: str
    archive_bytes: int
    selected_member: str
    selected_path: Path
    extracted_files: tuple[Path, ...]


def _member_path(name: str, *, directory: bool) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise DeterministicArchiveError("ZIP members must use nonempty POSIX paths")
    raw = name[:-1] if directory and name.endswith("/") else name
    pieces = raw.split("/")
    if not raw or any(piece in {"", ".", ".."} for piece in pieces):
        raise DeterministicArchiveError(f"unsafe ZIP member path {name!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or (path.parts and ":" in path.parts[0]):
        raise DeterministicArchiveError(f"unsafe ZIP member path {name!r}")
    return path


def _validated_infos(
    archive: zipfile.ZipFile,
    *,
    limits: ZipLimits,
) -> tuple[zipfile.ZipInfo, ...]:
    infos = tuple(archive.infolist())
    if not infos or len(infos) > limits.max_entries:
        raise DeterministicArchiveError(
            f"ZIP entry count must be between 1 and {limits.max_entries}"
        )
    names: set[str] = set()
    total = 0
    for info in infos:
        path = _member_path(info.filename, directory=info.is_dir())
        folded = path.as_posix().casefold()
        if folded in names:
            raise DeterministicArchiveError(
                f"ZIP contains duplicate case-insensitive member {info.filename!r}"
            )
        names.add(folded)
        if info.flag_bits & 0x1:
            raise DeterministicArchiveError("encrypted ZIP members are not accepted")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise DeterministicArchiveError(
                f"unsupported ZIP compression method for {info.filename!r}"
            )
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if kind == stat.S_IFLNK:
            raise DeterministicArchiveError("ZIP symbolic links are not accepted")
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise DeterministicArchiveError("ZIP special-file entries are not accepted")
        if info.is_dir():
            if info.file_size != 0:
                raise DeterministicArchiveError("ZIP directory entries must be empty")
            continue
        if info.file_size <= 0 or info.file_size > limits.max_member_bytes:
            raise DeterministicArchiveError(
                f"ZIP member {info.filename!r} exceeds the member size limit"
            )
        total += info.file_size
        if total > limits.max_total_uncompressed_bytes:
            raise DeterministicArchiveError("ZIP exceeds the total extraction size limit")
        if info.compress_size <= 0:
            raise DeterministicArchiveError("nonempty ZIP member has no compressed bytes")
        if info.file_size > info.compress_size * limits.max_compression_ratio:
            raise DeterministicArchiveError(
                f"ZIP member {info.filename!r} exceeds the compression-ratio limit"
            )
    return infos


def extract_shape_archive(
    source: str | Path,
    destination: str | Path,
    *,
    member: str,
    limits: ZipLimits = ZipLimits(),
) -> ExtractedShapeArchive:
    """Extract one validated ZIP into a new host-owned directory.

    Every member is checked and extracted because linked OBJ/MTL/glTF resources
    may be needed to prove the selected shape. No archive path is ever placed
    in a Luna workspace.
    """

    archive_path = Path(source).resolve()
    if not archive_path.is_file() or archive_path.is_symlink():
        raise DeterministicArchiveError(
            f"archive source is missing or not a regular file: {archive_path}"
        )
    archive_size = archive_path.stat().st_size
    if archive_size <= 0 or archive_size > limits.max_archive_bytes:
        raise DeterministicArchiveError(
            f"archive size must be between 1 and {limits.max_archive_bytes} bytes"
        )
    payload = archive_path.read_bytes()
    if len(payload) != archive_size:
        raise DeterministicArchiveError("archive changed while being read")
    if not payload.startswith(b"PK\x03\x04"):
        raise DeterministicArchiveError("shape archive does not have ZIP file magic")
    selected = _member_path(member, directory=False).as_posix()
    output = Path(destination).resolve()
    if output.exists():
        raise DeterministicArchiveError(f"archive destination already exists: {output}")
    output.mkdir(parents=True)
    extracted: list[Path] = []
    try:
        # Parse and extract the same immutable byte snapshot that is hashed.
        # Reopening archive_path here would permit a path-swap between evidence
        # capture and validation/extraction.
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = _validated_infos(archive, limits=limits)
            by_name = {
                _member_path(info.filename, directory=info.is_dir()).as_posix(): info
                for info in infos
            }
            selected_info = by_name.get(selected)
            if selected_info is None or selected_info.is_dir():
                raise DeterministicArchiveError(
                    f"declared archive member {selected!r} is missing or is a directory"
                )
            for relative, info in by_name.items():
                target = (output / Path(*PurePosixPath(relative).parts)).resolve()
                if target != output and output not in target.parents:
                    raise DeterministicArchiveError("ZIP extraction path escaped destination")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with archive.open(info, "r") as source_stream, target.open("xb") as target_stream:
                    while True:
                        chunk = source_stream.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > info.file_size or written > limits.max_member_bytes:
                            raise DeterministicArchiveError(
                                f"ZIP member {info.filename!r} exceeded its declared size"
                            )
                        target_stream.write(chunk)
                if written != info.file_size:
                    raise DeterministicArchiveError(
                        f"ZIP member {info.filename!r} length changed during extraction"
                    )
                extracted.append(target)
        selected_path = (output / Path(*PurePosixPath(selected).parts)).resolve()
        if not selected_path.is_file() or selected_path.is_symlink():
            raise DeterministicArchiveError(
                "selected ZIP shape member is not a regular file"
            )
        return ExtractedShapeArchive(
            archive_sha256=hashlib.sha256(payload).hexdigest(),
            archive_bytes=len(payload),
            selected_member=selected,
            selected_path=selected_path,
            extracted_files=tuple(sorted(extracted)),
        )
    except DeterministicArchiveError:
        shutil.rmtree(output, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(output, ignore_errors=True)
        raise DeterministicArchiveError(f"ZIP extraction failed: {exc}") from exc


__all__ = [
    "DeterministicArchiveError",
    "ExtractedShapeArchive",
    "ZipLimits",
    "extract_shape_archive",
]
