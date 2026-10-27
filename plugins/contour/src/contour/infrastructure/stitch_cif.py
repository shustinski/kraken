"""Strict Contour CIF exchange profile for physically registered stitching.

Reuse Contour primitive conversion and authored keyhole decoding. Unknown CIF
commands, scaling, calls, multiple layers and malformed polygons fail closed;
the interactive reader's permissive skipping is inappropriate for stitching.
"""
import hashlib
import re
from collections import Counter


class CifStitchError(ValueError):
    def __init__(self, code, detail, *, polygon_id=None, command_index=None):
        self.code, self.polygon_id, self.command_index = code, polygon_id, command_index
        location = (f" polygon {polygon_id}" if polygon_id is not None else "") + (f" command {command_index}" if command_index is not None else "")
        super().__init__(f"CIF [{code}]{location}: {detail}")


def load_stitch_cif(path, *, expected_sha256, image_size, max_bytes=16*1024**2, max_vertices=1000000):
    from kraken_core.safe_files import open_regular_read
    from shapely import Polygon, is_valid_reason
    from shapely.errors import ShapelyError

    from contour.infrastructure.cif_primitives import CifBox, CifComment, CifPolygon
    from contour.serializers_parts.cif_box_primitive_polygon import _cif_primitives_to_polygons
    from contour.serializers_parts.split_linked_polygon_rings import _split_linked_polygon_rings

    with open_regular_read(path) as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("CIF exceeds its input budget or its pinned SHA-256 differs")
    text = data.decode("utf-8-sig")
    primitives, layers, sizes = [], [], []
    position = vertices = 0
    definition = finished = ended = False
    body_commands = []
    for command_index, match in enumerate(re.finditer(r"\([^()]*\)\s*;?|[^();]+;|\bE\s*$", text), 1):
        if text[position:match.start()].strip():
            raise ValueError("Unsupported or malformed CIF syntax")
        position = match.end()
        command = match.group().strip().rstrip(";").strip()
        if ended:
            raise ValueError("CIF contains commands after E")
        if command.startswith("("):
            content = command[1:command.rfind(")")].strip()
            tokens = content.split()
            if tokens and tokens[0] == "S":
                if len(tokens) != 3:
                    raise ValueError("Malformed CIF image dimensions")
                sizes.append(tuple(map(int, tokens[1:])))
            primitives.append(CifComment("( " + content + " )"))
            continue
        tokens = command.split()
        if tokens == ["L"]:
            raise CifStitchError("empty_layer", "CIF layer name is empty", command_index=command_index)
        if tokens == ["DS", "1", "1", "1"] and not definition and not finished:
            definition = True
        elif tokens == ["DF"] and definition and not finished:
            finished = True
        elif tokens == ["E"] and finished:
            ended = True
        elif tokens and tokens[0] == "L" and len(tokens) == 2 and definition and not finished:
            layers.append(tokens[1])
        elif tokens and tokens[0] in {"P", "B"} and definition and not finished and layers:
            if tokens[0] == "P" and vertices + (len(tokens)-1)//2 > max_vertices:
                raise ValueError("CIF exceeds its vertex budget")
            values = list(map(int, tokens[1:]))
            if tokens[0] == "P":
                if len(values) < 6 or len(values) % 2:
                    raise CifStitchError("malformed_polygon", "Expected at least three coordinate pairs", command_index=command_index)
                vertices += len(values)//2
                points = tuple(zip(values[::2], values[1::2], strict=True))
                if len(set(points)) < 3:
                    raise CifStitchError("degenerate_polygon", "At least three distinct vertices are required",
                        polygon_id=len(body_commands)+1, command_index=command_index)
                primitives.append(CifPolygon(points))
            elif len(values) in {4, 6} and values[0] > 0 and values[1] > 0 and (len(values) == 4 or values[4] or values[5]):
                primitives.append(CifBox(*values, *(() if len(values) == 6 else (1, 0))))
                vertices += 4
            else:
                raise ValueError("Malformed CIF box")
            if vertices > max_vertices:
                raise ValueError("CIF exceeds its vertex budget")
            body_commands.append(command_index)
        else:
            raise ValueError("Unsupported CIF exchange command; no partial vector result was produced")
    if text[position:].strip() or not ended or len(layers) != 1 or sizes != [tuple(image_size)]:
        raise ValueError("CIF requires one layer, exact calibrated dimensions and a complete definition")
    _, decoded_size, polygons = _cif_primitives_to_polygons(path, primitives)
    if len(polygons) != len(body_commands):
        raise CifStitchError("incomplete_decode", "A CIF primitive was omitted by the decoder")
    if decoded_size != tuple(image_size):
        raise ValueError("Decoded CIF dimensions differ from the image calibration")
    result = []
    for polygon in polygons:
        try:
            geometry = Polygon(polygon.points)
        except (ValueError, ShapelyError) as exc:
            raise CifStitchError("degenerate_polygon", str(exc), polygon_id=polygon.id,
                command_index=body_commands[polygon.id-1]) from exc
        if not geometry.is_valid:
            # Authored doubled bridge edges encode holes in this dialect.
            # Decode the explicit rings; never make_valid arbitrary crossings.
            points = polygon.points[:-1] if polygon.points[0] == polygon.points[-1] else polygon.points
            counts = Counter(points)
            start = next((i for i, point in enumerate(points) if counts[point] == 1), 0)
            # A slit crossing the serialized ring's closing edge is the same
            # authored ring after rotation; keep the bridge inside the sequence.
            outer, holes = _split_linked_polygon_rings(points[start:] + points[:start])
            try:
                geometry = Polygon(outer, holes)
            except (ValueError, ShapelyError) as exc:
                raise CifStitchError("degenerate_polygon", str(exc), polygon_id=polygon.id,
                    command_index=body_commands[polygon.id-1]) from exc
        if not geometry.is_valid or geometry.is_empty or geometry.area <= 0:
            raise CifStitchError("ambiguous_topology", "CIF polygon topology is invalid or ambiguous: " + is_valid_reason(geometry),
                polygon_id=polygon.id, command_index=body_commands[polygon.id-1])
        result.append((polygon.category, geometry))
    return result
