"""Media types for the engineering formats supported by Kraken."""
import mimetypes
from pathlib import PurePath


def artifact_media_type(filename: str) -> str:
    # Python/system registries assign chemical/x-cif to this extension. In
    # Kraken's vector workflow CIF means Caltech Intermediate Form instead.
    if PurePath(filename).suffix.casefold() == ".cif":
        return "application/x-cif"
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"
