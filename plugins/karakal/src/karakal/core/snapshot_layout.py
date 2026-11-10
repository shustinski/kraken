"""What a snapshot folder holds: the layers to check, found from its subfolders. Qt-free.

Every subfolder with images gets a setting: what it is (masks, confidence maps, source photos, not
used) and which layer it belongs to; mask folders also tell their cell class (ones, zeros or none).
The settings come from ``karakal.json`` in the snapshot folder when it is there (written when the
user sets a folder by hand on the home page); a subfolder it does not list is guessed by its name:

- ``jpg`` (``src``, ``sources``…): the source photos of the frames;
- ``binary_cif`` (``binary…``, ``mask…``, ``маски…``, ``ones``, ``zeros``…): the network's masks;
- ``confidence`` (``conf…``): the network's confidence maps;
- ``<NAME>_inv``: the second network of a ones/zeros pair (``binary_cif_inv``, ``confidence_inv``).

A mask folder and its ``_inv`` twin are a pair: the plain one paints ones, the ``_inv`` one zeros.
A mask folder without a confidence folder of its own takes the one that serves all layers
(the plain ``confidence``, where the files of both networks may lie side by side). A snapshot
whose images lie in the folder itself is one layer. Any other name is not recognised: the snapshot
cannot be run until the user tells what it is (or that it is not used).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
INV_SUFFIX = "_inv"
SOURCE_NAMES = ("jpg", "jpeg", "src", "source", "sources", "original", "originals", "исходники")
CONFIDENCE_MARKERS = ("confidence", "conf")
MASK_PREFIXES = ("binary", "bin_", "mask", "маск")
ONES_NAMES = ("ones", "единицы", "единичные")
ZEROS_NAMES = ("zeros", "нули", "нулевые")

MANIFEST_FILE = "karakal.json"
MANIFEST_SCHEMA = "karakal.snapshot.v1"
ROOT_FOLDER = "."

KIND_MASKS = "masks"
KIND_CONFIDENCE = "confidence"
KIND_SOURCES = "sources"
KIND_SKIP = "skip"
KIND_UNKNOWN = ""
KINDS = (KIND_MASKS, KIND_CONFIDENCE, KIND_SOURCES, KIND_SKIP)
CLASSES = ("", "ones", "zeros")
# Layer of a confidence or source folder that serves every layer without one of its own.
ALL_LAYERS = ""


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
class SnapshotFolder:
    """One subfolder of the snapshot and what it is."""

    name: str  # the subfolder's name; ROOT_FOLDER for images in the snapshot folder itself
    count: int
    kind: str = KIND_UNKNOWN
    # Masks: the layer's name. Confidence and sources: the layer they serve (ALL_LAYERS: every
    # layer without one of its own).
    layer: str = ""
    cell_class: str = ""  # masks only
    # False: told by the user (karakal.json), not guessed from the name.
    guessed: bool = True


@dataclass(frozen=True, slots=True)
class SnapshotLayout:
    root: Path
    layers: tuple[SnapshotLayer, ...] = ()
    source_folder: Path | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    folders: tuple[SnapshotFolder, ...] = ()
    # The settings came (at least partly) from karakal.json.
    from_manifest: bool = False

    @property
    def is_pair(self) -> bool:
        return {layer.cell_class for layer in self.layers} >= {"ones", "zeros"}

    @property
    def unrecognized(self) -> tuple[str, ...]:
        return tuple(folder.name for folder in self.folders if folder.kind == KIND_UNKNOWN)

    @property
    def ready(self) -> bool:
        """Every folder told and at least one layer: the snapshot can be run."""

        return bool(self.layers) and not self.unrecognized

    def folder_path(self, name: str) -> Path:
        return self.root if name == ROOT_FOLDER else self.root / name


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


def _mask_class_by_name(name: str) -> str | None:
    """"ones"/"zeros" for folders named after the class, "" for other mask names, None: not masks."""

    base = _base(name).lower()
    if base in ONES_NAMES:
        return "zeros" if _is_inv(name) else "ones"
    if base in ZEROS_NAMES:
        return "ones" if _is_inv(name) else "zeros"
    if base.startswith(MASK_PREFIXES):
        return ""
    return None


# Guessing from the names --------------------------------------------------------------------------


def _guess(names: list[str], counts: dict[str, int]) -> list[SnapshotFolder]:
    masks = [name for name in names if not _is_source(name) and not _is_confidence(name) and _mask_class_by_name(name) is not None]
    by_base: dict[str, list[str]] = {}
    for name in masks:
        by_base.setdefault(_base(name).lower(), []).append(name)
    inv_masks = [name for name in masks if _is_inv(name)]
    lone_inv = inv_masks[0] if len(inv_masks) == 1 else ALL_LAYERS
    folders: list[SnapshotFolder] = []
    for name in names:
        count = counts[name]
        if _is_source(name):
            folders.append(SnapshotFolder(name, count, KIND_SOURCES, lone_inv if _is_inv(name) else ALL_LAYERS))
        elif _is_confidence(name):
            folders.append(SnapshotFolder(name, count, KIND_CONFIDENCE, lone_inv if _is_inv(name) else ALL_LAYERS))
        elif name in masks:
            by_name = _mask_class_by_name(name) or ""
            group = by_base[_base(name).lower()]
            paired = any(not _is_inv(item) for item in group) and any(_is_inv(item) for item in group)
            cell_class = by_name or (("zeros" if _is_inv(name) else "ones") if paired else "")
            folders.append(SnapshotFolder(name, count, KIND_MASKS, name, cell_class))
        else:
            folders.append(SnapshotFolder(name, count, KIND_UNKNOWN))
    return folders


# karakal.json -----------------------------------------------------------------------------------


def read_manifest(root: Path | str) -> dict[str, SnapshotFolder]:
    """Folder settings written into the snapshot folder; empty when there are none (or unreadable)."""

    path = Path(root) / MANIFEST_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("folders"), list):
        return {}
    told: dict[str, SnapshotFolder] = {}
    for entry in data["folders"]:
        if not isinstance(entry, dict) or not str(entry.get("name") or ""):
            continue
        kind = str(entry.get("kind") or "")
        cell_class = str(entry.get("class") or "")
        told[str(entry["name"])] = SnapshotFolder(
            name=str(entry["name"]),
            count=0,
            kind=kind if kind in KINDS else KIND_UNKNOWN,
            layer=str(entry.get("layer") or ""),
            cell_class=cell_class if cell_class in CLASSES else "",
            guessed=False,
        )
    return told


def write_manifest(layout: SnapshotLayout) -> Path | None:
    """The folder settings of ``layout`` into its snapshot folder; None when it cannot be written
    (a read-only share): the settings then hold only until the snapshot is chosen again."""

    path = layout.root / MANIFEST_FILE
    payload = {
        "schema": MANIFEST_SCHEMA,
        "folders": [
            {"name": folder.name, "kind": folder.kind, "layer": folder.layer, "class": folder.cell_class}
            for folder in layout.folders
        ],
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return None
    return path


# Layers from the folder settings ----------------------------------------------------------------


def _serving(folders: list[SnapshotFolder], kind: str, layer: str, *, prefer_inv: bool) -> SnapshotFolder | None:
    own = [folder for folder in folders if folder.kind == kind and folder.layer == layer]
    if own:
        return own[0]
    shared = [folder for folder in folders if folder.kind == kind and folder.layer == ALL_LAYERS]
    if not shared:
        return None
    plain = [folder for folder in shared if not _is_inv(folder.name)]
    inv = [folder for folder in shared if _is_inv(folder.name)]
    if prefer_inv and inv:
        return inv[0]
    return (plain or inv)[0]


def build_layout(root: Path, folders: list[SnapshotFolder] | tuple[SnapshotFolder, ...], *, from_manifest: bool = False) -> SnapshotLayout:
    root = Path(root)
    folders = list(folders)
    masks = sorted(
        (folder for folder in folders if folder.kind == KIND_MASKS),
        key=lambda folder: (_base(folder.name).lower(), _is_inv(folder.name), folder.name.lower()),
    )
    layers: list[SnapshotLayer] = []
    path_of = {folder.name: (root if folder.name == ROOT_FOLDER else root / folder.name) for folder in folders}
    for folder in masks:
        label = folder.layer or (root.name if folder.name == ROOT_FOLDER else folder.name)
        inverted = folder.cell_class == "zeros" or _is_inv(folder.name)
        confidence = _serving(folders, KIND_CONFIDENCE, folder.layer or folder.name, prefer_inv=inverted)
        source = _serving(folders, KIND_SOURCES, folder.layer or folder.name, prefer_inv=inverted)
        confidence_path = path_of[confidence.name] if confidence is not None else None
        layers.append(
            SnapshotLayer(
                label=label,
                mask_folder=path_of[folder.name],
                confidence_folder=confidence_path,
                source_folder=path_of[source.name] if source is not None else None,
                cell_class=folder.cell_class,
                mask_count=folder.count,
                confidence_count=count_images(confidence_path),
            )
        )
    sources = [folder for folder in folders if folder.kind == KIND_SOURCES]
    shared_source = next((folder for folder in sources if not _is_inv(folder.name)), sources[0] if sources else None)
    notes = []
    if not layers:
        notes.append("no_masks")
    if layers and not any(folder.kind == KIND_CONFIDENCE for folder in folders):
        notes.append("no_confidence")
    return SnapshotLayout(
        root=root,
        layers=tuple(layers),
        source_folder=path_of[shared_source.name] if shared_source is not None else None,
        notes=tuple(notes),
        folders=tuple(folders),
        from_manifest=from_manifest,
    )


def discover_snapshot(root: Path | str) -> SnapshotLayout:
    """Layers of a snapshot folder: masks, their confidence maps and the source photos."""

    root = Path(root)
    if not root.is_dir():
        return SnapshotLayout(root=root, notes=("missing",))
    subfolders = sorted((item for item in root.iterdir() if item.is_dir() and count_images(item)), key=lambda p: p.name.lower())
    names = [folder.name for folder in subfolders]
    counts = {folder.name: count_images(folder) for folder in subfolders}
    told = read_manifest(root)
    guessed = {folder.name: folder for folder in _guess(names, counts)}
    folders = [
        replace(told[name], count=counts[name]) if name in told else guessed[name]
        for name in names
    ]
    root_count = count_images(root)
    if root_count:
        if ROOT_FOLDER in told:
            folders.insert(0, replace(told[ROOT_FOLDER], count=root_count))
        elif not any(folder.kind == KIND_MASKS for folder in folders):
            # Images in the snapshot folder itself and no mask folder: one network.
            folders.insert(0, SnapshotFolder(ROOT_FOLDER, root_count, KIND_MASKS, root.name))
    return build_layout(root, folders, from_manifest=bool(told))


def with_folder(layout: SnapshotLayout, name: str, *, kind: str, layer: str = "", cell_class: str = "") -> SnapshotLayout:
    """``layout`` with one folder told by the user; the layers are made again."""

    folders = []
    for folder in layout.folders:
        if folder.name == name:
            folder = replace(
                folder,
                kind=kind if kind in KINDS else KIND_UNKNOWN,
                layer=str(layer or "") if kind in (KIND_MASKS, KIND_CONFIDENCE, KIND_SOURCES) else "",
                cell_class=cell_class if kind == KIND_MASKS and cell_class in CLASSES else "",
                guessed=False,
            )
        folders.append(folder)
    return build_layout(layout.root, folders, from_manifest=True)
