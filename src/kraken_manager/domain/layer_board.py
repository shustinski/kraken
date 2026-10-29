"""Pure rules for the layer board: names, selection actions, CIF review order, legend."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

IMAGE_KINDS = frozenset({"source", "binary"})
VECTOR_KINDS = frozenset({"vector"})
CIF_KINDS = frozenset({"cif"})
ASSIGNMENT_ACTION = "assign_cif_review_layers"
DATASET_PREPARE_CAPABILITY = "frames.dataset.prepare.v1"

_PARAMETER_HIDDEN = frozenset(
    {
        "source_representation_id",
        "plugin_job_id",
        "format",
        "version",
        "input_media_types",
        "execution_host",
        "input_transport",
        "model_version",
    }
)

_STATUS_LABELS = {
    "empty": "Пусто",
    "image_ready": "Изображение",
    "processing": "Обработка",
    "vectorized": "Векторизовано",
    "in_review": "На проверке",
    "returned_unchanged": "Возврат без изменений",
    "returned_changed": "Возврат с изменениями",
    "approved": "Принято",
    "changes_requested": "Нужны исправления",
    "conflict": "Конфликт",
    "error": "Ошибка",
    "not_checked": "Не проверено",
    "checked": "Проверено",
}

_LEGEND_EMPTY = "Нет данных"


@dataclass(frozen=True, slots=True)
class BoardNode:
    node_id: str
    kind: str
    title: str = ""
    capability: str = ""
    parameters: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ContextAction:
    label: str
    code: str


@dataclass(frozen=True, slots=True)
class LegendEntry:
    label: str
    color: str = ""


@dataclass(frozen=True, slots=True)
class LegendSection:
    title: str
    entries: tuple[LegendEntry, ...] = ()
    empty_text: str = _LEGEND_EMPTY

    @property
    def empty(self) -> bool:
        return not self.entries


def collapsed_block_title(nodes: Sequence[BoardNode]) -> str:
    """Name a collapsed pipeline block after the step the user created."""

    preferred = ("job", "operation", "dataset", "model", "binary", "stitching", "karakal", "confidence")
    titles: list[str] = []
    for kind in preferred:
        for node in nodes:
            title = str(node.title or "").strip()
            if node.kind == kind and title and title not in titles:
                titles.append(title)
    if not titles:
        for node in nodes:
            title = str(node.title or "").strip()
            if title and title not in titles:
                titles.append(title)
    if not titles:
        return "Обработка слоя"
    if len(titles) == 1:
        return titles[0]
    return " · ".join(titles)


def parameter_lines(parameters: Mapping[str, object] | None, *, capability: str = "") -> tuple[str, ...]:
    """Human lines for a parameterized block. Hidden identifiers are omitted."""

    visible: list[str] = []
    if isinstance(parameters, Mapping):
        for key, value in parameters.items():
            name = str(key)
            if name in _PARAMETER_HIDDEN or "sha256" in name.casefold() or name.endswith("_id"):
                continue
            if isinstance(value, (Mapping, list, tuple, set)) or value in ("", None):
                continue
            visible.append(f"{name}: {value}")
    if visible:
        return tuple(visible[:4])
    if capability == DATASET_PREPARE_CAPABILITY:
        return ("Параметры: не заданы",)
    return ()


def dataset_prepare_is_parameter_block(node: BoardNode) -> bool:
    return node.capability == DATASET_PREPARE_CAPABILITY or node.kind == "dataset"


def selection_kind(kind: str, subtitle: str = "") -> str:
    """CIF variants are the vector blocks labeled CIF. A vector block stays a vector."""

    if kind == "vector" and str(subtitle).strip().casefold() == "cif":
        return "cif"
    return kind


def multi_selection_actions(nodes: Sequence[BoardNode]) -> tuple[ContextAction, ...] | None:
    """Actions for a multi-selection.

    Several images and exactly one vector open Contour, issue work, and export.
    Several CIF variants, with any number of images, only export.
    ``None`` means the caller should keep the single-node menu.
    An empty tuple means this combination has no extra actions.
    """

    if len(nodes) <= 1:
        return None
    images = sum(1 for node in nodes if node.kind in IMAGE_KINDS)
    vectors = sum(1 for node in nodes if node.kind in VECTOR_KINDS)
    cifs = sum(1 for node in nodes if node.kind in CIF_KINDS)
    recognized = IMAGE_KINDS | VECTOR_KINDS | CIF_KINDS
    if any(node.kind not in recognized for node in nodes):
        return ()
    if images >= 2 and vectors == 1 and cifs == 0:
        return (
            ContextAction("Открыть в Contour", "open_in_contour"),
            ContextAction("Выдать в работу", "issue_work"),
            ContextAction("Выгрузить", "export_selection"),
        )
    if cifs >= 2 and vectors == 0:
        return (ContextAction("Выгрузить", "export_selection"),)
    return ()


def order_review_images(order: Sequence[str], representation_id: str, index: int) -> tuple[str, ...]:
    """Insert or move one image representation, preserving the rest of the order."""

    identifier = str(representation_id).strip()
    if not identifier:
        raise ValueError("Изображение для проверки не указано")
    remaining = tuple(item for item in order if str(item) != identifier and str(item).strip())
    bounded = max(0, min(int(index), len(remaining)))
    return remaining[:bounded] + (identifier,) + remaining[bounded:]


def review_image_ids(events: Iterable[object], vector_ids: Sequence[str]) -> tuple[str, ...]:
    """Latest saved order that covers any of the selected CIF representations."""

    wanted = {str(value) for value in vector_ids if str(value).strip()}
    if not wanted:
        return ()
    found: tuple[str, ...] = ()
    for event in events:
        payload = _payload(event)
        if str(payload.get("action", "")) != ASSIGNMENT_ACTION:
            continue
        parameters = payload.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        recorded = {
            str(value)
            for value in parameters.get("vector_representation_ids", ())
            if str(value).strip()
        }
        recorded.add(str(payload.get("node_id", "")))
        recorded.discard("")
        if not recorded & wanted:
            continue
        found = tuple(
            dict.fromkeys(
                str(value)
                for value in parameters.get("image_representation_ids", ())
                if str(value).strip()
            )
        )
    return found


def companion_export_plan(
    ordered_image_ids: Sequence[str],
    *,
    primary_image_id: str,
    names_by_id: Mapping[str, tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Folders to export with a CIF layer, in the saved right-column order.

    The primary image is already part of the work package, so it is not repeated.
    """

    primary = str(primary_image_id or "")
    plan: list[tuple[str, str]] = []
    used: set[str] = set()
    for raw in ordered_image_ids:
        image_id = str(raw).strip()
        if not image_id or image_id == primary or any(image_id == item[1] for item in plan):
            continue
        layer_name, image_name = names_by_id.get(image_id, ("слой", image_id[:8]))
        folder = _unique_folder(f"{layer_name} — {image_name}", used)
        plan.append((folder, image_id))
    return tuple(plan)


def next_image_representation_name(existing: Iterable[str]) -> str:
    base = "Исходные изображения"
    names = {str(item).strip() for item in existing}
    if base not in names:
        return base
    index = 2
    while f"{base} ({index})" in names:
        index += 1
    return f"{base} ({index})"


def matrix_legend(cells: Sequence[Mapping[str, object]]) -> tuple[LegendSection, ...]:
    """Legend for the main-window thumbnails. Missing fields stay empty."""

    return (
        LegendSection("Исполнитель", _performer_entries(cells)),
        LegendSection("Время", _time_entries(cells)),
        LegendSection("Качество", _quality_entries(cells)),
        LegendSection("Статус", _status_entries(cells)),
    )


def workflow_view_signature(
    *,
    project_revision: object,
    representation_ids: Sequence[str],
    history_count: int,
    pipeline_revision: object,
    pipeline_status: object,
    node_statuses: Sequence[tuple[str, str]],
) -> tuple[object, ...]:
    return (
        project_revision,
        tuple(representation_ids),
        int(history_count),
        pipeline_revision,
        str(pipeline_status or ""),
        tuple(node_statuses),
    )


def _performer_entries(cells: Sequence[Mapping[str, object]]) -> tuple[LegendEntry, ...]:
    entries: list[LegendEntry] = []
    seen: set[tuple[str, str]] = set()
    for cell in cells:
        label = str(cell.get("performer_name") or cell.get("performer_initials") or "").strip()
        if not label:
            continue
        color = str(cell.get("performer_color") or "").strip()
        key = (label, color)
        if key in seen:
            continue
        seen.add(key)
        entries.append(LegendEntry(label, color))
    return tuple(entries)


def _time_entries(cells: Sequence[Mapping[str, object]]) -> tuple[LegendEntry, ...]:
    stamps = [stamp for cell in cells if (stamp := _timestamp(cell.get("modified_at"))) is not None]
    if not stamps:
        return ()
    oldest, newest = min(stamps), max(stamps)
    return (
        LegendEntry("Раньше", _scale_color(0.0)),
        LegendEntry("Позже", _scale_color(1.0 if newest > oldest else 0.0)),
    )


def _quality_entries(cells: Sequence[Mapping[str, object]]) -> tuple[LegendEntry, ...]:
    values: list[float] = []
    for cell in cells:
        if "quality" not in cell or cell.get("quality") is None:
            continue
        try:
            values.append(float(cell["quality"]))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    if not values:
        return ()
    return (LegendEntry("Ниже", _quality_color(0.0)), LegendEntry("Выше", _quality_color(1.0)))


def _status_entries(cells: Sequence[Mapping[str, object]]) -> tuple[LegendEntry, ...]:
    entries: list[LegendEntry] = []
    seen: set[str] = set()
    for cell in cells:
        if "review_status" in cell and str(cell.get("review_status") or "").strip():
            code = str(cell.get("review_status"))
        elif "status" in cell and str(cell.get("status") or "").strip():
            code = str(cell.get("status"))
        else:
            continue
        if code in seen:
            continue
        seen.add(code)
        entries.append(LegendEntry(_STATUS_LABELS.get(code, code), _STATUS_COLORS.get(code, "")))
    return tuple(entries)


_STATUS_COLORS = {
    "empty": "#1f2937",
    "image_ready": "#2563eb",
    "processing": "#7c3aed",
    "vectorized": "#0891b2",
    "in_review": "#d97706",
    "returned_unchanged": "#65a30d",
    "returned_changed": "#ca8a04",
    "approved": "#15803d",
    "changes_requested": "#ea580c",
    "conflict": "#dc2626",
    "error": "#b91c1c",
    "not_checked": "#64748b",
    "checked": "#22c55e",
}


def _quality_color(value: float) -> str:
    bounded = max(0.0, min(1.0, value))
    return _rgb(1.0 - bounded, bounded, 0.10)


def _scale_color(ratio: float) -> str:
    bounded = max(0.0, min(1.0, ratio))
    return _rgb(1.0 - bounded, bounded, 0.08)


def _rgb(red: float, green: float, blue: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(red * 255),
        int(green * 255),
        int(blue * 255),
    )


def _timestamp(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _payload(event: object) -> Mapping[str, object]:
    if isinstance(event, Mapping):
        payload = event.get("payload", event)
        return payload if isinstance(payload, Mapping) else {}
    payload = getattr(event, "payload", None)
    return payload if isinstance(payload, Mapping) else {}


def _unique_folder(folder: str, used: set[str]) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_ —" else "_" for character in folder)
    cleaned = " ".join(cleaned.split()).strip(" _") or "слой"
    candidate = cleaned
    index = 2
    while candidate.casefold() in used:
        candidate = f"{cleaned} ({index})"
        index += 1
    used.add(candidate.casefold())
    return candidate
