"""Isolated FreeCADCmd worker for the deterministic host adapter.

This module is executed by FreeCAD's Python runtime, not imported by the
contraption process. It emits only an exact triangle record; the host performs
all schema, topology, hash, and resource validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys


def _shape_records(source: Path):
    import FreeCAD  # type: ignore
    import Part  # type: ignore

    suffix = source.suffix.casefold()
    document = None
    if suffix == ".fcstd":
        document = FreeCAD.openDocument(str(source), hidden=True)
        candidates = [
            item
            for item in document.Objects
            if hasattr(item, "Shape")
            and not item.Shape.isNull()
            and bool(getattr(getattr(item, "ViewObject", None), "Visibility", True))
        ]
        candidate_ids = {id(item) for item in candidates}
        # Keep displayed/top-level results and avoid re-tessellating every
        # intermediate feature inside a PartDesign body or compound.
        records = [
            item.Shape
            for item in candidates
            if not any(id(parent) in candidate_ids for parent in getattr(item, "InList", ()))
        ]
        if not records:
            records = [item.Shape for item in candidates]
        return records, document
    if suffix == ".brep":
        shape = Part.Shape()
        if not shape.read(str(source)):
            raise ValueError("OpenCascade rejected BREP source")
        return [shape], None
    if suffix in {".step", ".stp", ".iges", ".igs"}:
        shape = Part.read(str(source))
        if shape is None or shape.isNull():
            raise ValueError("OpenCascade rejected STEP/IGES source")
        return [shape], None
    raise ValueError(f"unsupported FreeCAD source extension {suffix!r}")


def main() -> int:
    if len(sys.argv) < 7:
        raise ValueError("FreeCAD worker arguments are missing")
    source = Path(sys.argv[-6]).resolve()
    output = Path(sys.argv[-5]).resolve()
    scale = float(sys.argv[-4])
    deflection = float(sys.argv[-3])
    max_vertices = int(sys.argv[-2])
    max_triangles = int(sys.argv[-1])
    if (
        not source.is_file()
        or not math.isfinite(scale)
        or scale <= 0
        or not math.isfinite(deflection)
        or deflection <= 0
    ):
        raise ValueError("invalid FreeCAD worker source or numeric arguments")
    shapes, document = _shape_records(source)
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    try:
        for shape in shapes:
            points, facets = shape.tessellate(deflection)
            if not points or not facets:
                continue
            offset = len(vertices)
            for point in points:
                coordinate = [float(point.x) * scale, float(point.y) * scale, float(point.z) * scale]
                if any(not math.isfinite(value) for value in coordinate):
                    raise ValueError("FreeCAD emitted NaN or infinity")
                vertices.append(coordinate)
                if len(vertices) > max_vertices:
                    raise ValueError("FreeCAD output exceeds the vertex safety limit")
            for facet in facets:
                if len(facet) != 3:
                    raise ValueError("FreeCAD emitted a non-triangle facet")
                triangles.append([offset + int(value) for value in facet])
                if len(triangles) > max_triangles:
                    raise ValueError("FreeCAD output exceeds the triangle safety limit")
    finally:
        if document is not None:
            import FreeCAD  # type: ignore

            FreeCAD.closeDocument(document.Name)
    if len(vertices) < 3 or not triangles:
        raise ValueError("FreeCAD source contains no tessellated surface")
    output.write_text(
        json.dumps(
            {
                "format": "contraption-freecad-mesh-1",
                "vertices_m": vertices,
                "triangles": triangles,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"contraption FreeCAD worker: {exc}", file=sys.stderr)
        raise SystemExit(2)
