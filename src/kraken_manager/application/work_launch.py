"""Which installed plugin opens a piece of issued work.

The identifiers are the same ones the desktop already passes to the plugin
launcher (Contour, NeuralImage, Karakal). This module does not start processes.
"""

from __future__ import annotations

from kraken_manager.domain.work_orders import WorkKind

_CAPABILITY_PLUGINS = {
    "frames.vectorize.v1": "contour",
    "frames.dataset.prepare.v1": "contour",
    "frames.binary-segment.v1": "neuralimage",
    "frames.binary-segment.v2": "neuralimage",
    "frames.confidence.v1": "neuralimage",
    "dataset.model.train.v1": "neuralimage",
    "layer.confidence.analyze.v1": "karakal",
}

_KIND_PLUGINS = {
    WorkKind.VECTOR_REVIEW: "contour",
    WorkKind.VECTORIZE: "contour",
    WorkKind.TRAIN: "neuralimage",
}


def plugin_id_for_capability(capability: str) -> str | None:
    return _CAPABILITY_PLUGINS.get(str(capability or ""))


def plugin_id_for_work_kind(kind: WorkKind | str) -> str | None:
    try:
        resolved = kind if isinstance(kind, WorkKind) else WorkKind(str(kind))
    except ValueError:
        return None
    return _KIND_PLUGINS.get(resolved)
