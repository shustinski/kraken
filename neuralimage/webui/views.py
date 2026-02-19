from __future__ import annotations

import logging
from django.contrib import messages
from django.http import HttpRequest, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from lib.ui_texts import get_ui_section
from lib.version import APP_NAME, APP_VERSION, get_app_title
from .forms import MainWindowForm, SettingsForm, defaults_from_main_state, defaults_from_settings_state
from .services.training_session import get_session_service


_WEBUI_TEXTS = get_ui_section('webui')
_LOG = logging.getLogger(__name__)


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


def _build_forms(request: HttpRequest):
    session = get_session_service()
    main_state, settings_state = session.load_initial_states()

    if request.method == 'POST':
        main_form = MainWindowForm(request.POST, prefix='main')
        settings_form = SettingsForm(request.POST, prefix='settings')
    else:
        main_form = MainWindowForm(initial=defaults_from_main_state(main_state), prefix='main')
        settings_form = SettingsForm(initial=defaults_from_settings_state(settings_state), prefix='settings')

    settings_form.fields['model'].widget.attrs['list'] = 'model-list'
    settings_form.fields['model'].label = str(_WEBUI_TEXTS.get('model_label', 'Модель'))

    return main_form, settings_form


@require_GET
def dashboard(request: HttpRequest):
    main_form, settings_form = _build_forms(request)
    status = get_session_service().snapshot().get('status', 'idle')
    return render(
        request,
        'webui/dashboard.html',
        {
            'main_form': main_form,
            'settings_form': settings_form,
            'model_choices': _load_model_choices(),
            'status': status,
            'app_name': APP_NAME,
            'app_version': APP_VERSION,
            'app_title': get_app_title(),
        },
    )


@require_POST
def start_processing(request: HttpRequest):
    main_form, settings_form = _build_forms(request)
    model_choices = _load_model_choices()

    if not main_form.is_valid() or not settings_form.is_valid():
        return render(
            request,
            'webui/dashboard.html',
            {
                'main_form': main_form,
                'settings_form': settings_form,
                'model_choices': model_choices,
                'status': get_session_service().snapshot().get('status', 'idle'),
                'app_name': APP_NAME,
                'app_version': APP_VERSION,
                'app_title': get_app_title(),
            },
            status=400,
        )

    session = get_session_service()
    ok, error = session.start(main_form.to_state(), settings_form.to_state())
    if not ok:
        messages.error(request, error or str(_WEBUI_TEXTS.get('start_error', 'Не удалось запустить обработку.')))
    else:
        messages.success(request, str(_WEBUI_TEXTS.get('start_ok', 'Обработка запущена.')))

    return redirect('webui:dashboard')


@require_POST
def stop_processing(request: HttpRequest):
    session = get_session_service()
    ok, error = session.stop()
    if not ok:
        messages.error(request, error or str(_WEBUI_TEXTS.get('stop_error', 'Не удалось остановить обработку.')))
    else:
        messages.success(request, str(_WEBUI_TEXTS.get('stop_ok', 'Запрос на остановку отправлен.')))
    return redirect('webui:dashboard')


@require_GET
def status_api(request: HttpRequest):
    after = request.GET.get('after', '0')
    try:
        after_id = int(after)
    except ValueError:
        after_id = 0

    session = get_session_service()
    return JsonResponse(session.snapshot(after_event_id=after_id))

