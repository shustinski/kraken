"""Request journal for Kraken Server.

Levels accumulate: medium includes low, high includes every request.

Lines describe what happened in words and name the project, layer, job and
file by their names instead of their identifiers, so an operator can follow
the server without decoding UUIDs.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Callable, Mapping

LOG_LEVELS = ("none", "low", "medium", "high")
_RANK = {name: index for index, name in enumerate(LOG_LEVELS)}
LOGGER = logging.getLogger("kraken_server.access")
_HANDLER_NAME = "kraken-server-operation"

_API_PREFIX = "/api/v1"
_PROJECT_CREATE = "/api/v1/projects"
_ACCOUNT_CREATE = frozenset({"/api/v1/auth/accounts"})
_LAYER_CREATE = re.compile(r"/api/v1/projects/[^/]+/layers\Z")
_PLACEHOLDER = re.compile(r"\{([^}:]+)(?::[^}]*)?\}")


@dataclass
class _Threshold:
    rank: int = 0


_threshold = _Threshold()


def parse_log_level(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _RANK:
        allowed = ", ".join(LOG_LEVELS)
        raise ValueError(f"server.log_level must be one of: {allowed}")
    return normalized


def configure_operation_log(level_name: str) -> str:
    """Apply the journal level. ``none`` records nothing."""
    level = parse_log_level(level_name)
    _threshold.rank = _RANK[level]
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.disabled = level == "none"
    if not any(getattr(handler, "name", "") == _HANDLER_NAME for handler in LOGGER.handlers):
        handler = logging.StreamHandler()
        handler.name = _HANDLER_NAME
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        LOGGER.addHandler(handler)
    return level


def request_rank(method: str, path: str) -> int:
    """Minimum level rank that records this call. High-only calls rank as 3."""
    normalized = path.split("?", 1)[0].rstrip("/") or "/"
    verb = method.upper()
    if verb == "POST" and (normalized == _PROJECT_CREATE or normalized in _ACCOUNT_CREATE):
        return _RANK["low"]
    if verb == "POST" and _LAYER_CREATE.fullmatch(normalized):
        return _RANK["medium"]
    if verb in {"POST", "PUT"} and _is_data_upload(normalized):
        return _RANK["medium"]
    return _RANK["high"]


def should_record(method: str, path: str, status_code: int) -> bool:
    required = request_rank(method, path)
    if _threshold.rank < required:
        return False
    return not (_threshold.rank < _RANK["high"] and status_code >= 400)


@dataclass
class Requester:
    """Who made the request: a signed-in user (by principal id) or an agent (by its name)."""

    principal_id: str | None = None
    agent_name: str | None = None


_requester: ContextVar[Requester | None] = ContextVar("kraken_journal_requester", default=None)


def begin_request() -> tuple[Requester, Token]:
    """Open a requester slot for this request. Auth dependencies fill it in."""
    holder = Requester()
    return holder, _requester.set(holder)


def end_request(token: Token) -> None:
    _requester.reset(token)


def note_user(principal_id: str) -> None:
    holder = _requester.get()
    if holder is not None and holder.agent_name is None:
        holder.principal_id = principal_id


def note_agent(name: str) -> None:
    holder = _requester.get()
    if holder is not None:
        holder.agent_name = name
        holder.principal_id = None


def record_http_request(
    method: str,
    path: str,
    status_code: int,
    *,
    route: str | None = None,
    params: Mapping[str, Any] | None = None,
    names: "NameBook | None" = None,
    requester: Requester | None = None,
) -> None:
    """Journal one request. ``route`` is the matched path template, ``params`` its values."""
    if not should_record(method, path, status_code):
        return
    LOGGER.info(
        "%s",
        describe_request(
            method, path, status_code, route=route, params=params, names=names, requester=requester
        ),
    )


def describe_request(
    method: str,
    path: str,
    status_code: int,
    *,
    route: str | None = None,
    params: Mapping[str, Any] | None = None,
    names: "NameBook | None" = None,
    requester: Requester | None = None,
) -> str:
    verb = method.upper()
    clean_path = path.split("?", 1)[0]
    template = _short(route) if route else None
    values = {key: str(value) for key, value in (params or {}).items()}
    action = _ACTIONS.get((verb, template)) if template else None
    if action is None:
        action = _fallback_action(verb, clean_path, template)
    who = _who(requester, names)
    if requester is not None and requester.agent_name and action.startswith("агент "):
        action = action.removeprefix("агент ")
    context = _context(values, names)
    text = action if not context else f"{action}: {', '.join(context)}"
    text = text if who is None else f"{who} — {text}"
    return f"{text} → {_status(status_code)}"


def _who(requester: Requester | None, names: "NameBook | None") -> str | None:
    if requester is None:
        return None
    if requester.agent_name:
        return f"агент «{requester.agent_name}»"
    if requester.principal_id:
        name = names.user(requester.principal_id) if names else None
        return _name_or_short(name, requester.principal_id)
    return None


def _status(code: int) -> str:
    return str(code) if code < 400 else f"ошибка {code}"


def _short(route: str) -> str:
    template = route[len(_API_PREFIX):] if route.startswith(_API_PREFIX) else route
    return _PLACEHOLDER.sub(lambda match: "{" + match.group(1) + "}", template.rstrip("/") or "/")


def _fallback_action(verb: str, path: str, template: str | None) -> str:
    rank = request_rank(verb, path)
    normalized = path.rstrip("/") or "/"
    if rank == _RANK["low"]:
        return "создание проекта" if normalized == _PROJECT_CREATE else "создание учётной записи"
    if rank == _RANK["medium"]:
        return "создание слоя" if _LAYER_CREATE.fullmatch(normalized) else "загрузка данных"
    shown = template if template is not None else _hide_identifiers(path[len(_API_PREFIX):] if path.startswith(_API_PREFIX) else path)
    shown = _PLACEHOLDER.sub("…", shown)
    return f"{_VERBS.get(verb, verb)} {shown}"


_UUIDISH = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[0-9a-fA-F]{32,}")


def _hide_identifiers(path: str) -> str:
    return _UUIDISH.sub("…", path)


_VERBS = {"GET": "запрос", "POST": "действие", "PUT": "запись", "PATCH": "изменение", "DELETE": "удаление"}


def _is_data_upload(path: str) -> bool:
    if "/uploads" in path or "/outputs/" in path:
        return True
    return "/artifacts/" in path and path.endswith("/versions")


# ---------------------------------------------------------------- names

_SHORT_NOUNS = {
    "run_id": "запуск",
    "other_run_id": "второй запуск",
    "frame_id": "кадр",
    "series_id": "серия",
    "representation_id": "представление",
    "issue_id": "замечание",
    "note_id": "заметка",
    "batch_id": "пакет",
    "plan_id": "план",
    "template_id": "шаблон",
    "order_id": "задание",
    "output_id": "результат",
    "principal_id": "пользователь",
    "role": "роль",
    "column": "столбец",
    "row": "строка",
    "product": "продукт",
    "channel": "канал",
    "filename": "файл",
}
_SKIPPED = frozenset({"digest"})


def _context(values: Mapping[str, str], names: "NameBook | None") -> list[str]:
    parts: list[str] = []
    project_id = values.get("project_id")
    job_label = None
    job_id = values.get("job_id")
    if job_id:
        summary = names.job(job_id) if names else None
        if summary is not None:
            project_id = project_id or summary[0]
            job_label = summary[1]
    if project_id:
        parts.append(f"проект «{_name_or_short(names.project(project_id) if names else None, project_id)}»")
    layer_id = values.get("layer_id")
    if layer_id:
        layer = names.layer(project_id, layer_id) if names and project_id else None
        parts.append(f"слой «{_name_or_short(layer, layer_id)}»")
    if job_id:
        parts.append(f"задача «{job_label or _shorten(job_id)}»")
    version_id = values.get("version_id")
    if version_id:
        filename = names.artifact(project_id, version_id) if names and project_id else None
        parts.append(f"файл «{_name_or_short(filename, version_id)}»")
    for key, value in values.items():
        if key in {"project_id", "layer_id", "job_id", "version_id"} or key in _SKIPPED:
            continue
        noun = _SHORT_NOUNS.get(key, key)
        parts.append(f"{noun} {_shorten(value)}")
    return parts


def _name_or_short(name: str | None, identifier: str) -> str:
    return _shorten(name) if name else _shorten(identifier)


def _shorten(value: str) -> str:
    return value[:8] if _UUIDISH.fullmatch(value) else value


class NameBook:
    """Cached identifier → name lookups. Any lookup failure falls back to a short id."""

    def __init__(
        self,
        *,
        project: Callable[[str], str | None] | None = None,
        layer: Callable[[str, str], str | None] | None = None,
        job: Callable[[str], tuple[str, str] | None] | None = None,
        artifact: Callable[[str, str], str | None] | None = None,
        user: Callable[[str], str | None] | None = None,
        ttl_seconds: float = 300.0,
        max_entries: int = 4096,
    ) -> None:
        self._lookups = {"project": project, "layer": layer, "job": job, "artifact": artifact, "user": user}
        self._ttl = ttl_seconds
        self._max = max_entries
        self._cache: OrderedDict[tuple, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def project(self, project_id: str) -> str | None:
        return self._get("project", project_id)

    def layer(self, project_id: str, layer_id: str) -> str | None:
        return self._get("layer", project_id, layer_id)

    def job(self, job_id: str) -> tuple[str, str] | None:
        return self._get("job", job_id)

    def artifact(self, project_id: str, version_id: str) -> str | None:
        return self._get("artifact", project_id, version_id)

    def user(self, principal_id: str) -> str | None:
        return self._get("user", principal_id)

    def _get(self, kind: str, *key: str) -> Any:
        lookup = self._lookups.get(kind)
        if lookup is None:
            return None
        cache_key = (kind, *key)
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(cache_key)
            if hit is not None and hit[0] > now:
                self._cache.move_to_end(cache_key)
                return hit[1]
        try:
            value = lookup(*key)
        except Exception:  # noqa: BLE001 - the journal must never break a request
            value = None
        with self._lock:
            self._cache[cache_key] = (now + self._ttl, value)
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return value


def backend_name_book(backend: Any, agent_gateway: Any = None) -> NameBook:
    """Name lookups backed by the server services and the agent queue."""

    def project(project_id: str) -> str | None:
        return str(backend.get_project(project_id).get("name") or "") or None

    def layer(project_id: str, layer_id: str) -> str | None:
        for item in backend.list_layers(project_id, include_archived=True):
            if str(item.get("layer_id")) == layer_id:
                return str(item.get("name") or "") or None
        return None

    def artifact(project_id: str, version_id: str) -> str | None:
        metadata = backend.get_artifact_version(project_id, version_id)
        return str(metadata.get("filename") or "") or None

    def job(job_id: str) -> tuple[str, str] | None:
        if agent_gateway is None or not hasattr(agent_gateway, "job_summary"):
            return None
        summary = agent_gateway.job_summary(job_id)
        if summary is None:
            return None
        return summary["project_id"], job_label(summary)

    def user(principal_id: str) -> str | None:
        for item in backend.list_principals(include_inactive=True):
            if str(item.get("principal_id")) == principal_id:
                return str(item.get("display_name") or item.get("email") or "") or None
        return None

    return NameBook(project=project, layer=layer, job=job, artifact=artifact, user=user)


_JOB_KINDS = {"analysis": "анализ", "stitch": "сшивка"}


def job_label(summary: Mapping[str, Any]) -> str:
    label = _JOB_KINDS.get(str(summary.get("kind") or "")) or str(summary.get("capability") or "задача")
    index, count = summary.get("partition_index"), summary.get("partition_count")
    if isinstance(index, int) and isinstance(count, int) and count > 1:
        label = f"{label}, часть {index + 1} из {count}"
    return label


# ---------------------------------------------------------------- actions

_P = "/projects/{project_id}"
_J = "/agent/jobs/{job_id}"
_A = _P + "/analysis-runs"
_ACTIONS: dict[tuple[str, str], str] = {
    ("GET", "/health"): "проверка доступности сервера",
    ("GET", "/session"): "проверка сеанса",
    ("PATCH", "/principals/me/profile"): "изменение профиля",
    ("GET", "/principals"): "список пользователей",
    ("GET", "/performers"): "список исполнителей",
    ("POST", "/auth/accounts"): "создание учётной записи",
    ("POST", "/auth/sessions"): "вход в систему",
    ("GET", "/updates/{product}/{channel}/{filename}"): "загрузка обновления",
    # agent queue
    ("POST", "/agent/lease"): "агент запрашивает задачу",
    ("POST", _J + "/lease"): "агент берёт задачу",
    ("POST", _J + "/heartbeat"): "агент сообщает, что задача выполняется",
    ("POST", _J + "/execution-state"): "агент сохраняет состояние задачи",
    ("GET", _J + "/inputs/{version_id}"): "агент скачивает входной файл",
    ("GET", _J + "/inputs/{version_id}/download-ticket"): "агент запрашивает ссылку на входной файл",
    ("GET", _J + "/checkpoint-files/{digest}"): "агент скачивает контрольную точку",
    ("POST", _J + "/publications"): "агент публикует результат",
    ("POST", _J + "/outputs/{output_id}"): "агент выгружает результат",
    ("POST", _J + "/outputs/{output_id}/upload-ticket"): "агент запрашивает ссылку для выгрузки результата",
    ("POST", _J + "/outputs/{output_id}/upload-complete"): "агент завершил выгрузку результата",
    ("POST", _J + "/complete"): "агент завершил задачу",
    ("POST", _J + "/fail"): "агент сообщает об ошибке задачи",
    # projects
    ("GET", "/projects"): "список проектов",
    ("POST", "/projects"): "создание проекта",
    ("GET", _P): "открытие проекта",
    ("PATCH", _P): "изменение проекта",
    ("POST", _P + "/archive"): "архивирование проекта",
    ("POST", _P + "/restore"): "восстановление проекта",
    ("POST", _P + "/commands"): "команда проекта",
    ("GET", _P + "/history"): "история проекта",
    ("GET", _P + "/stream-revision"): "ревизия потока",
    ("GET", _P + "/events/{event_id}"): "событие проекта",
    ("GET", _P + "/statistics"): "статистика проекта",
    ("GET", _P + "/workspace"): "рабочая папка проекта",
    ("GET", _P + "/frames"): "список кадров",
    ("GET", _P + "/viewport"): "просмотр матрицы кадров",
    ("GET", _P + "/viewport-mosaic"): "мозаика матрицы кадров",
    ("POST", _P + "/import-plan"): "план импорта",
    ("GET", _P + "/deletion-requests"): "запросы на удаление",
    ("POST", _P + "/deletion-requests"): "запрос на удаление",
    ("GET", _P + "/acl/{principal_id}"): "права пользователя в проекте",
    ("PUT", _P + "/acl/{principal_id}/{role}"): "выдача роли",
    ("DELETE", _P + "/acl/{principal_id}/{role}"): "снятие роли",
    # layers
    ("GET", _P + "/layers"): "список слоёв",
    ("POST", _P + "/layers"): "создание слоя",
    ("POST", _P + "/layers/external"): "подключение внешнего слоя",
    ("POST", _P + "/layers/import"): "импорт слоя",
    ("POST", _P + "/layers/scan"): "поиск слоёв в папке",
    ("POST", _P + "/layers/reorder"): "изменение порядка слоёв",
    ("POST", _P + "/layers/{layer_id}/archive"): "архивирование слоя",
    ("POST", _P + "/layers/{layer_id}/delete"): "удаление слоя",
    ("POST", _P + "/layers/{layer_id}/rename"): "переименование слоя",
    ("POST", _P + "/layers/{layer_id}/reorder"): "перемещение слоя",
    ("GET", _P + "/layers/{layer_id}/files"): "файлы слоя",
    ("GET", _P + "/layers/{layer_id}/representations"): "представления слоя",
    ("PATCH", _P + "/layers/{layer_id}/representations/{representation_id}"): "изменение представления слоя",
    ("POST", _P + "/layers/{layer_id}/representations/{representation_id}/import"): "импорт в представление слоя",
    ("GET", _P + "/layers/{layer_id}/pipeline/workflow"): "конвейер слоя",
    ("PUT", _P + "/layers/{layer_id}/pipeline/workflow"): "сохранение конвейера слоя",
    ("POST", _P + "/layers/{layer_id}/pipeline/workflow/commands"): "команда конвейера слоя",
    ("POST", _P + "/pipeline-actions"): "действие конвейера",
    # artifacts
    ("GET", _P + "/artifacts"): "список файлов проекта",
    ("POST", _P + "/artifacts"): "создание серии файлов",
    ("PATCH", _P + "/artifacts/{series_id}"): "изменение серии файлов",
    ("POST", _P + "/artifacts/{series_id}/uploads"): "загрузка файла",
    ("GET", _P + "/artifacts/{series_id}/versions"): "версии файла",
    ("GET", _P + "/artifacts/{series_id}/active"): "текущая версия файла",
    ("GET", _P + "/artifacts/{series_id}/revision"): "ревизия файла",
    ("GET", _P + "/artifacts/versions/{version_id}"): "скачивание файла",
    ("GET", _P + "/artifacts/versions/{version_id}/download-ticket"): "ссылка на скачивание файла",
    ("GET", _P + "/artifacts/versions/{version_id}/metadata"): "сведения о файле",
    # jobs
    ("GET", _P + "/jobs"): "список задач",
    ("GET", _P + "/jobs/{job_id}"): "задача",
    ("POST", _P + "/jobs"): "запуск задачи",
    ("POST", _P + "/jobs/{job_id}/cancel"): "отмена задачи",
    ("POST", _P + "/jobs/{job_id}/retry"): "повтор задачи",
    ("POST", _P + "/jobs/{job_id}/confirm-partial"): "принятие частичного результата",
    ("POST", _P + "/jobs/{job_id}/desktop-lease"): "задача берётся на этом компьютере",
    ("GET", _P + "/stitch-renders/{job_id}/vectors"): "векторы сшивки",
    ("GET", _P + "/stitch-renders/{job_id}/blocks/{column}/{row}"): "блок сшивки",
    ("GET", _P + "/stitch-renders/{job_id}/vector-blocks/{column}/{row}"): "векторный блок сшивки",
    # analysis
    ("GET", _A): "список запусков анализа",
    ("POST", _A): "запуск анализа",
    ("GET", _A + "/{run_id}"): "состояние запуска анализа",
    ("GET", _A + "/{run_id}/frames"): "кадры запуска анализа",
    ("POST", _A + "/{run_id}/frames/{frame_id}/map"): "построение карты кадра",
    ("GET", _A + "/{run_id}/frames/{frame_id}/map"): "карта кадра",
    ("GET", _A + "/{run_id}/compare/{other_run_id}"): "сравнение запусков анализа",
    ("POST", _A + "/{run_id}/commands"): "команда запуска анализа",
    ("POST", _P + "/analyses/karakal"): "запуск анализа Karakal",
    ("GET", _P + "/quality-analysis-runs"): "запуски анализа качества",
    ("GET", _P + "/quality-analysis-runs/{run_id}/readiness"): "готовность анализа качества",
    ("GET", _P + "/quality-artifacts"): "файлы анализа качества",
    ("GET", _P + "/quality-issues"): "список замечаний",
    ("POST", _P + "/quality-issues"): "создание замечания",
    ("POST", _P + "/quality-issues/from-analysis"): "замечания по результатам анализа",
    ("GET", _P + "/quality-issues/{issue_id}"): "открытие замечания",
    ("GET", _P + "/quality-issues/{issue_id}/history"): "история замечания",
    ("POST", _P + "/quality-issues/{issue_id}/commands"): "действие с замечанием",
    ("POST", _P + "/quality-issues/{issue_id}/analysis-recheck"): "перепроверка замечания",
    # notes, reviews, work
    ("GET", _P + "/notes"): "список заметок",
    ("POST", _P + "/notes"): "создание заметки",
    ("PATCH", _P + "/notes/{note_id}"): "изменение заметки",
    ("GET", _P + "/reviews"): "список проверок",
    ("POST", _P + "/reviews"): "создание проверки",
    ("PATCH", _P + "/reviews/{batch_id}"): "изменение проверки",
    ("POST", _P + "/reviews/{batch_id}/package"): "упаковка проверки",
    ("POST", _P + "/reviews/{batch_id}/return/preflight"): "предпроверка возврата",
    ("POST", _P + "/reviews/{batch_id}/return/commit"): "возврат проверки",
    ("GET", _P + "/work-orders"): "список заданий исполнителям",
    ("POST", _P + "/work-orders"): "выдача задания исполнителю",
    ("POST", _P + "/work-orders/{order_id}/cancel"): "отмена задания исполнителю",
    ("POST", _P + "/work-orders/{order_id}/complete"): "завершение задания исполнителю",
    ("GET", _P + "/work-orders/legacy-migration"): "план переноса старых заданий",
    ("POST", _P + "/work-orders/legacy-migration"): "перенос старых заданий",
    ("GET", _P + "/work-plans"): "список планов работ",
    ("POST", _P + "/work-plans"): "создание плана работ",
    ("GET", _P + "/work-plans/{plan_id}"): "открытие плана работ",
    ("POST", _P + "/work-plans/{plan_id}/commands"): "действие с планом работ",
    ("GET", _P + "/work-plan-templates"): "шаблоны планов работ",
    ("POST", _P + "/work-plan-templates"): "создание шаблона плана работ",
    ("GET", _P + "/work-plan-templates/{template_id}"): "открытие шаблона плана работ",
    ("PUT", _P + "/work-plan-templates/{template_id}"): "сохранение шаблона плана работ",
    ("POST", _P + "/work-planning/analytics"): "аналитика планирования",
    ("GET", _P + "/multilayer"): "многослойный просмотр",
    ("POST", _P + "/multilayer/commands"): "команда многослойного просмотра",
    ("GET", _P + "/electrical"): "электрическая схема",
    ("GET", _P + "/electrical/export"): "экспорт электрической схемы",
    ("POST", _P + "/electrical/commands"): "команда электрической схемы",
}
