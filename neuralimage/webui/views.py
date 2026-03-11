from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from django.contrib import messages
from django.http import HttpRequest, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from lib.ui_texts import get_ui_section, normalize_ui_language
from lib.version import APP_NAME, APP_VERSION, get_app_title
from .forms import MainWindowForm, SettingsForm, defaults_from_main_state, defaults_from_settings_state
from .services.training_session import get_session_service


_LOG = logging.getLogger(__name__)
_PICKER_LOCK = threading.Lock()


def _resolve_ui_language(request: HttpRequest) -> str:
    requested = (
        request.GET.get('lang')
        or request.POST.get('ui_lang')
        or request.session.get('ui_language')
        or request.headers.get('Accept-Language', '')
    )
    language = normalize_ui_language(requested)
    request.session['ui_language'] = language
    return language


def _webui_texts(language: str) -> dict[str, object]:
    return get_ui_section('webui', language)


def _status_display(status: str, texts: dict[str, object]) -> str:
    status_texts = texts.get('status_values', {})
    if isinstance(status_texts, dict):
        value = status_texts.get(status)
        if isinstance(value, str) and value.strip():
            return value
    return status


def _load_model_choices() -> list[tuple[str, str]]:
    fallback = [('M 720k', 'M 720k')]
    try:
        from model.NeuralNetwork.registrator import get_registered_models

        names = list(get_registered_models().keys())
        if not names:
            return fallback
        return [(name, name) for name in names]
    except Exception:
        _LOG.exception('Failed to load registered model choices for WebUI')
        return fallback


def _build_forms(request: HttpRequest, language: str, texts: dict[str, object]):
    session = get_session_service()
    main_state, settings_state = session.load_initial_states()

    if request.method == 'POST':
        main_form = MainWindowForm(request.POST, prefix='main', language=language, ui_texts=texts)
        settings_form = SettingsForm(request.POST, prefix='settings', language=language, ui_texts=texts)
    else:
        main_form = MainWindowForm(
            initial=defaults_from_main_state(main_state),
            prefix='main',
            language=language,
            ui_texts=texts,
        )
        settings_form = SettingsForm(
            initial=defaults_from_settings_state(settings_state),
            prefix='settings',
            language=language,
            ui_texts=texts,
        )

    settings_form.fields['model'].widget.attrs['list'] = 'model-list'

    return main_form, settings_form


def _is_local_request(request: HttpRequest) -> bool:
    if str(os.getenv('NEURALIMAGE_WEBUI_PICKER_ALLOW_REMOTE', '0')).strip().lower() in {'1', 'true', 'yes', 'on'}:
        return True
    remote_addr = str(request.META.get('REMOTE_ADDR', '')).strip().lower()
    host = str(request.get_host() or '').split(':', 1)[0].strip().lower()
    local_addrs = {'127.0.0.1', '::1', 'localhost'}
    return (remote_addr in local_addrs) and (host in local_addrs)


def _pick_path_via_dialog(kind: str, file_filter: str, texts: dict[str, object]) -> str:
    import tkinter as tk
    from tkinter import filedialog

    dialog_texts = texts.get('dialogs', {})
    if not isinstance(dialog_texts, dict):
        dialog_texts = {}

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        if kind == 'folder':
            selected = filedialog.askdirectory(
                title=str(dialog_texts.get('pick_folder_title', 'Выберите папку'))
            )
        else:
            if file_filter == 'model':
                selected = filedialog.askopenfilename(
                    title=str(dialog_texts.get('pick_model_title', 'Выберите файл модели')),
                    filetypes=[
                        (str(dialog_texts.get('model_files', 'Файлы модели')), '*.pth'),
                        (str(dialog_texts.get('all_files', 'Все файлы')), '*.*'),
                    ],
                )
            else:
                selected = filedialog.askopenfilename(
                    title=str(dialog_texts.get('pick_file_title', 'Выберите файл')),
                    filetypes=[(str(dialog_texts.get('all_files', 'Все файлы')), '*.*')],
                )
    finally:
        root.destroy()
    return str(selected or '')


def _build_dashboard_context(
    main_form: MainWindowForm,
    settings_form: SettingsForm,
    language: str,
    texts: dict[str, object],
    status: str,
) -> dict[str, object]:
    return {
        'main_form': main_form,
        'settings_form': settings_form,
        'model_choices': _load_model_choices(),
        'status': status,
        'status_display': _status_display(status, texts),
        'app_name': APP_NAME,
        'app_version': APP_VERSION,
        'app_title': get_app_title(),
        'texts': texts,
        'js_texts': texts.get('js', {}),
        'ui_language': language,
    }


@require_GET
def dashboard(request: HttpRequest):
    language = _resolve_ui_language(request)
    texts = _webui_texts(language)
    main_form, settings_form = _build_forms(request, language, texts)
    status = get_session_service().snapshot().get('status', 'idle')
    return render(request, 'webui/dashboard.html', _build_dashboard_context(main_form, settings_form, language, texts, status))


@require_POST
def start_processing(request: HttpRequest):
    language = _resolve_ui_language(request)
    texts = _webui_texts(language)
    main_form, settings_form = _build_forms(request, language, texts)

    if not main_form.is_valid() or not settings_form.is_valid():
        return render(
            request,
            'webui/dashboard.html',
            _build_dashboard_context(
                main_form,
                settings_form,
                language,
                texts,
                get_session_service().snapshot().get('status', 'idle'),
            ),
            status=400,
        )

    session = get_session_service()
    ok, error = session.start(main_form.to_state(), settings_form.to_state())
    if not ok:
        messages.error(request, error or str(texts.get('start_error', 'Не удалось запустить обработку.')))
    else:
        messages.success(request, str(texts.get('start_ok', 'Обработка запущена.')))

    return redirect('webui:dashboard')


@require_POST
def stop_processing(request: HttpRequest):
    language = _resolve_ui_language(request)
    texts = _webui_texts(language)
    session = get_session_service()
    ok, error = session.stop()
    if not ok:
        messages.error(request, error or str(texts.get('stop_error', 'Не удалось остановить обработку.')))
    else:
        messages.success(request, str(texts.get('stop_ok', 'Запрос на остановку отправлен.')))
    return redirect('webui:dashboard')


@require_GET
def status_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    texts = _webui_texts(language)
    after = request.GET.get('after', '0')
    try:
        after_id = int(after)
    except ValueError:
        after_id = 0

    session = get_session_service()
    snapshot = session.snapshot(after_event_id=after_id)
    status = str(snapshot.get('status', 'idle'))
    snapshot['status_display'] = _status_display(status, texts)
    return JsonResponse(snapshot)


@require_POST
def pick_path_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    texts = _webui_texts(language)
    if not _is_local_request(request):
        return JsonResponse(
            {'ok': False, 'error': str(texts.get('picker_local_only', 'Выбор пути доступен только с localhost.'))},
            status=403,
        )

    kind = str(request.POST.get('kind', '')).strip().lower()
    file_filter = str(request.POST.get('filter', '')).strip().lower()
    if kind not in {'folder', 'file'}:
        return JsonResponse(
            {'ok': False, 'error': str(texts.get('picker_invalid_kind', 'Некорректный тип выбора пути.'))},
            status=400,
        )

    try:
        with _PICKER_LOCK:
            selected_path = _pick_path_via_dialog(kind=kind, file_filter=file_filter, texts=texts)
    except Exception as error:
        _LOG.exception('Failed to open host file dialog for WebUI path picker')
        error_template = str(texts.get('picker_dialog_failed', 'Не удалось открыть диалог выбора файла: {error}'))
        return JsonResponse({'ok': False, 'error': error_template.format(error=error)}, status=500)

    if not selected_path:
        return JsonResponse({'ok': False, 'cancelled': True, 'path': ''})

    normalized = str(Path(selected_path))
    return JsonResponse({'ok': True, 'path': normalized})
