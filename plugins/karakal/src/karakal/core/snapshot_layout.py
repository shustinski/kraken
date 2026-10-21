"""What a snapshot folder holds: the layers to check, found from its subfolders. Qt-free.

A snapshot is laid out in folders:

- ``jpg``: the source photos of the frames;
- ``binary_cif``: the network's masks of the frames;
- ``confidence``: the network's confidence maps;
- ``<NAME>_inv``: the second network of a ones/zeros pair (``binary_cif_inv``, ``confidence_inv``).

A mask folder and its ``_inv`` twin are a pair: the plain one paints ones, the ``_inv`` one zeros.
A mask folder without a confidence folder of its own (``confidence`` / ``confidence_inv``) takes
the plain confidence folder, where the files of both networks may lie side by side. A snapshot
whose images lie in the folder itself is one layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
INV_SUFFIX = "_inv"
SOURCE_NAMES = ("jpg", "jpeg", "src", "source", "sources", "original", "originals", "исходники")
CONFIDENCE_MARKERS = ("confidence", "conf")


@dataclass(frozen=True, slots=True)
class SnapshotLayer:
    label: str
    mask_folder: Path
    confidence_folder: Path | None = None
    source_folder: Path | None = None
    cell_class: str = ""  # "ones" / "zeros" for a pair, empty for a single network
    mask_count: int = 0
    confidence_count: int = 0


@dataclass(frozen=True, slots=True)
class SnapshotLayout:
    root: Path
    layers: tuple[SnapshotLayer, ...] = ()
    source_folder: Path | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_pair(self) -> bool:
        return {layer.cell_class for layer in self.layers} >= {"ones", "zeros"}


def count_images(folder: Path | None) -> int:
    if folder is None or not folder.is_dir():
        return 0
    return sum(1 for item in folder.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def _base(name: str) -> str:
    lowered = name.lower()
    return name[: -len(INV_SUFFIX)] if lowered.endswith(INV_SUFFIX) else name


def _is_inv(name: str) -> bool:
    return name.lower().endswith(INV_SUFFIX)


def _is_confidence(name: str) -> bool:
    base = _base(name).lower()
    return any(marker in base for marker in CONFIDENCE_MARKERS)


def _is_source(name: str) -> bool:
    return _base(name).lower() in SOURCE_NAMES


def discover_snapshot(root: Path | str) -> SnapshotLayout:
    """Layers of a snapshot folder: masks, their confidence maps and the source photos."""

    root = Path(root)
    if not root.is_dir():
        return SnapshotLayout(root=root, notes=("missing",))
    folders = sorted((item for item in root.iterdir() if item.is_dir() and count_images(item)), key=lambda p: p.name.lower())
    sources = [folder for folder in folders if _is_source(folder.name)]
    confidences = [folder for folder in folders if _is_confidence(folder.name)]
    masks = [folder for folder in folders if folder not in sources and folder not in confidences]
    source = next((folder for folder in sources if not _is_inv(folder.name)), sources[0] if sources else None)
    plain_confidence = next((folder for folder in confidences if not _is_inv(folder.name)), None)
    inv_confidence = next((folder for folder in confidences if _is_inv(folder.name)), None)
    inv_sources = {_base(folder.name).lower(): folder for folder in sources if _is_inv(folder.name)}
    if not masks and count_images(root):
        # Images in the snapshot folder itself: one network.
        return SnapshotLayout(
            root=root,
            layers=(SnapshotLayer(root.name, root, plain_confidence, source, "", count_images(root), count_images(plain_confidence)),),
            source_folder=source,
        )
    by_base: dict[str, list[Path]] = {}
    for folder in masks:
        by_base.setdefault(_base(folder.name).lower(), []).append(folder)
    layers: list[SnapshotLayer] = []
    for _key, group in sorted(by_base.items()):
        plain = next((folder for folder in group if not _is_inv(folder.name)), None)
        inv = next((folder for folder in group if _is_inv(folder.name)), None)
        paired = plain is not None and inv is not None
        for folder, inverted in ((plain, False), (inv, True)):
            if folder is None:
                continue
            confidence = (inv_confidence or plain_confidence) if inverted else plain_confidence
            layer_source = inv_sources.get(_base(source.name).lower(), source) if (inverted and source is not None) else source
            layers.append(
                SnapshotLayer(
                    label=folder.name,
                    mask_folder=folder,
                    confidence_folder=confidence,
                    source_folder=layer_source,
                    cell_class=("zeros" if inverted else "ones") if paired else "",
                    mask_count=count_images(folder),
                    confidence_count=count_images(confidence),
                )
            )
    notes = []
    if not layers:
        notes.append("no_masks")
    if layers and plain_confidence is None and inv_confidence is None:
        notes.append("no_confidence")
    return SnapshotLayout(root=root, layers=tuple(layers), source_folder=source, notes=tuple(notes))
