"""Shared PNG integrity and allocation limits for local and server analysis maps."""
import hashlib
from io import BytesIO


def validate_map_bytes(data, sha256, size_bytes):
    from PIL import Image

    if not 0 < size_bytes <= 256 * 1024 * 1024:
        raise ValueError("Rendered map exceeds its size limit")
    if len(data) != size_bytes or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Stored rendered map failed checksum validation")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("Rendered map must be a PNG image")
            if image.width * image.height > 64 * 1024 * 1024:
                raise ValueError("Rendered map exceeds the decoded pixel limit")
            image.verify()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("Rendered map is not a valid bounded PNG image") from exc


def read_verified_map(blobs, reference):
    if not 0 < reference.size_bytes <= 256 * 1024 * 1024:
        raise ValueError("Rendered map exceeds its size limit")
    with blobs.open(reference) as stream:
        data = stream.read(reference.size_bytes + 1)
    validate_map_bytes(data, reference.sha256, reference.size_bytes)
    return data
