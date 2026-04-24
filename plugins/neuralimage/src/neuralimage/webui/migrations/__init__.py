from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import tempfile
import threading
import gc
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.db import transaction
from django.http import FileResponse, HttpRequest, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.text import get_valid_filename
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from neuralimage.infrastructure.config.state_store import create_workflow_snapshot_payload, load_workflow_snapshot
from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.lib.data_interfaces import CutSettings, SampleCutMode, build_synthetic_defect_generator_parameters
from neuralimage.lib.images import SampleWorker
from neuralimage.lib.message_bus import MessageBus
from neuralimage.lib.runtime_paths import resolve_resource_path
from neuralimage.lib.ui_texts import get_ui_section, normalize_ui_language
from neuralimage.lib.update_checker import (
    collect_release_history,
    fetch_update_info,
    load_selected_update_channel,
    load_update_client_config,
    normalize_update_channel,
    save_selected_update_channel,
)
from neuralimage.lib.version import APP_NAME, APP_VERSION, get_app_title
from .forms import MainWindowForm, SettingsForm, defaults_from_main_state, defaults_from_settings_state
from .services.broadcast_notifications import get_broadcast_notification_store
from .services.training_session import get_session_service
from neuralimage.application.services.workflow_mapper import build_workflow_parameters
from neuralimage.lib.data_interfaces import WorkMode


_LOG = logging.getLogger(__name__)
_PICKER_LOCK = threading.Lock()
_WEBUI_UI_MODE_SESSION_KEY = 'webui_ui_mode'
_LOCAL_SUPERUSER_USERNAME = 'not_admin'
_LOCAL_SUPERUSER_PASSWORD = 'Aa123456'
_BROADCAST_LIMIT = 20
_UPLOAD_CATEGORY_BY_FIELD = {
    'main-result_folder': 'results',
    'main-sample_folder': 'samples',
    'main-label_folder': 'labels',
    'main-model_path': 'models',
    'settings-validation_image_folder': 'validation_images',
    'settings-validation_label_folder': 'validation_labels',
}


@dataclass(frozen=True)
class LDAPAuthConfig:
    server_uri: str
    bind_dn: str
    bind_password: str
    user_search_base: str
    user_search_filter: str
    user_dn_template: str
    connect_timeout: int


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
    return get_ui_section('neuralimage.webui', language)


def _desktop_texts(language: str) -> dict[str, object]:
    return get_ui_section('main_window', language)


def _task_texts(language: str) -> dict[str, object]:
    return get_ui_section('task_properties_dialog', language)


def _status_display(status: str, texts: dict[str, object]) -> str:
    status_texts = texts.get('status_values', {})
    if isinstance(status_texts, dict):
        value = status_texts.get(status)
        if isinstance(value, str) and value.strip():
            return value
    return status


def _auth_ui_texts(language: str) -> dict[str, str]:
    if language == 'en':
        return {
            'authorize_button': 'Sign in with LDAP',
            'superuser_button': 'Sign in with local account',
            'auth_title': 'Sign in to NeuralImage WebUI',
            'auth_subtitle': 'Use your LDAP account to access the shared processing queue.',
            'auth_not_configured': 'LDAP authentication is not configured on this host.',
            'auth_local_note': 'Local account: login not_admin / password Aa123456.',
            'username_label': 'Login',
            'password_label': 'Password',
            'logout_button': 'Sign out',
            'current_user': 'User',
        }
    return {
        'authorize_button': 'Войти через LDAP',
        'superuser_button': 'Войти по логину и паролю',
        'auth_title': 'Вход в NeuralImage WebUI',
        'auth_subtitle': 'Используйте учетную запись LDAP для доступа к общей очереди обработки.',
        'auth_not_configured': 'LDAP-аутентификация не настроена на этом хосте.',
        'auth_local_note': 'Локальная аварийная учётная запись: логин not_admin / пароль Aa123456.',
        'username_label': 'Логин',
        'password_label': 'Пароль',
        'logout_button': 'Выйти',
        'current_user': 'Пользователь',
    }


def _ldap_auth_config() -> LDAPAuthConfig | None:
    server_uri = str(os.getenv('NEURALIMAGE_LDAP_SERVER_URI', '') or '').strip()
    if not server_uri:
        return None
    try:
        connect_timeout = int(os.getenv('NEURALIMAGE_LDAP_CONNECT_TIMEOUT', '10'))
    except ValueError:
        connect_timeout = 10
    return LDAPAuthConfig(
        server_uri=server_uri,
        bind_dn=str(os.getenv('NEURALIMAGE_LDAP_BIND_DN', '') or '').strip(),
        bind_password=str(os.getenv('NEURALIMAGE_LDAP_BIND_PASSWORD', '') or ''),
        user_search_base=str(os.getenv('NEURALIMAGE_LDAP_USER_SEARCH_BASE', '') or '').strip(),
        user_search_filter=str(os.getenv('NEURALIMAGE_LDAP_USER_SEARCH_FILTER', '(sAMAccountName={username})') or '').strip(),
        user_dn_template=str(os.getenv('NEURALIMAGE_LDAP_USER_DN_TEMPLATE', '') or '').strip(),
        connect_timeout=max(1, connect_timeout),
    )


def _ldap_imports():
    try:
        from ldap3 import ALL, SUBTREE, Connection, Server
        from ldap3.utils.conv import escape_filter_chars
    except ImportError as error:
        raise RuntimeError('ldap3 is not installed. Install the web dependencies before enabling LDAP auth.') from error
    return ALL, SUBTREE, Connection, Server, escape_filter_chars


def _first_ldap_attr(entry, names: tuple[str, ...]) -> str:
    for name in names:
        try:
            value = entry[name].value
        except Exception:
            value = None
        if value:
            return str(value).strip()
    return ''


def _ldap_authenticate(username: str, password: str, config: LDAPAuthConfig) -> dict[str, str]:
    username = str(username or '').strip()
    if not username or not password:
        raise ValueError('LDAP login and password are required.')

    ALL, SUBTREE, Connection, Server, escape_filter_chars = _ldap_imports()
    server = Server(config.server_uri, get_info=ALL, connect_timeout=config.connect_timeout)

    if config.user_dn_template:
        user_dn = config.user_dn_template.format(username=username)
        user_connection = Connection(server, user=user_dn, password=password, auto_bind=True)
        user_connection.unbind()
        return {'username': username, 'display_name': username, 'email': ''}

    if not config.user_search_base:
        raise ValueError('LDAP user search base is required when user DN template is not configured.')
    if not config.bind_dn:
        raise ValueError('LDAP bind DN is required when user DN template is not configured.')

    service_connection = Connection(server, user=config.bind_dn, password=config.bind_password, auto_bind=True)
    try:
        safe_username = escape_filter_chars(username)
        search_filter = config.user_search_filter.format(username=safe_username)
        service_connection.search(
            search_base=config.user_search_base,
            search_filter=search_filter,
            search_scope=SUBTREE,
            attributes=['cn', 'displayName', 'mail', 'userPrincipalName', 'sAMAccountName', 'uid'],
            size_limit=2,
        )
        entries = list(service_connection.entries)
        if len(entries) != 1:
            raise ValueError('LDAP user was not found or search returned multiple users.')
        entry = entries[0]
        user_dn = str(entry.entry_dn)
        display_name = _first_ldap_attr(entry, ('displayName', 'cn')) or username
        email = _first_ldap_attr(entry, ('mail', 'userPrincipalName'))
        resolved_username = _first_ldap_attr(entry, ('sAMAccountName', 'uid', 'userPrincipalName')) or username
    finally:
        service_connection.unbind()

    user_connection = Connection(server, user=user_dn, password=password, auto_bind=True)
    user_connection.unbind()
    return {'username': resolved_username, 'display_name': display_name, 'email': email}


def _current_user_display_name(request: HttpRequest) -> str:
    user = request.user
    if not user.is_authenticated:
        return ''
    full_name = str(getattr(user, 'get_full_name', lambda: '')() or '').strip()
    if full_name:
        return full_name
    first_name = str(getattr(user, 'first_name', '') or '').strip()
    if first_name:
        return first_name
    return str(getattr(user, 'username', '') or '').strip()


@transaction.atomic
def _ensure_local_superuser():
    user_model = get_user_model()
    user, created = user_model.objects.get_or_create(
        username=_LOCAL_SUPERUSER_USERNAME,
        defaults={
            'is_staff': True,
            'is_superuser': True,
            'first_name': 'Local Admin',
        },
    )
    update_fields: list[str] = []
    if not user.is_staff:
        user.is_staff = True
        update_fields.append('is_staff')
    if not user.is_superuser:
        user.is_superuser = True
        update_fields.append('is_superuser')
    if not str(user.first_name or '').strip():
        user.first_name = 'Local Admin'
        update_fields.append('first_name')
    user.set_password(_LOCAL_SUPERUSER_PASSWORD)
    update_fields.append('password')
    if created:
        user.save()
    else:
        user.save(update_fields=update_fields)
    return user


@transaction.atomic
def _resolve_or_create_user_from_ldap(profile: dict[str, str]):
    ldap_username = str(profile.get('username', '') or '').strip()
    if not ldap_username:
        raise ValueError('LDAP profile does not contain username.')

    user_model = get_user_model()
    user, created = user_model.objects.get_or_create(username=ldap_username[:150])
    update_fields: list[str] = []

    if created:
        user.set_unusable_password()
        update_fields.append('password')

    email = str(profile.get('email', '') or '').strip()
    display_name = str(profile.get('display_name', '') or ldap_username).strip()
    if user.email != email:
        user.email = email
        update_fields.append('email')
    if user.first_name != display_name[:150]:
        user.first_name = display_name[:150]
        update_fields.append('first_name')
    if not user.is_active:
        user.is_active = True
        update_fields.append('is_active')

    if created:
        user.save()
    elif update_fields:
        user.save(update_fields=update_fields)
    return user


def _load_model_choices() -> list[tuple[str, str]]:
    fallback = [('M 720k', 'M 720k')]
    try:
        from neuralimage.model.NeuralNetwork.registrator import get_registered_models

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
            selected = filedialog.askdirectory(title=str(dialog_texts.get('pick_folder_title', 'Выберите папку')))
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
    request: HttpRequest,
    main_form: MainWindowForm,
    settings_form: SettingsForm,
    language: str,
    texts: dict[str, object],
    status: str,
) -> dict[str, object]:
    desktop_texts = _desktop_texts(language)
    settings_panel_texts = get_ui_section('settings_panel', language)
    task_texts = _task_texts(language)
    auth_texts = _auth_ui_texts(language)
    js_texts = dict(texts.get('js', {}))
    js_texts.update(
        {
            'work_mode_labels': texts.get('main_form', {}).get('work_modes', {}),
            'queue_status_values': task_texts.get('status_values', {}),
            'queue_empty': 'No tasks in queue.' if language == 'en' else 'Очередь пуста.',
            'preview_unavailable': 'Preview is not available yet.' if language == 'en' else 'Предпросмотр пока недоступен.',
            'memory_label_default': desktop_texts.get('memory_label_default', 'Memory: -'),
            'memory_unit': desktop_texts.get('memory_unit', 'MB'),
            'speed_unit': desktop_texts.get('speed_unit', 'batch/s'),
            'runtime_ram_label': desktop_texts.get('runtime_ram_label', 'RAM'),
            'runtime_vram_label': desktop_texts.get('runtime_vram_label', 'VRAM'),
            'runtime_speed_label': desktop_texts.get('runtime_speed_label', 'Speed'),
            'validation_quality_default': desktop_texts.get('validation_quality_default', 'Validation quality: -'),
            'performance_label_default': desktop_texts.get('performance_label_default', 'Performance: -'),
            'recognition_speed_default': desktop_texts.get('recognition_speed_default', 'Recognition speed: -'),
            'recognition_speed_label': desktop_texts.get('recognition_speed_label', 'Recognition speed'),
            'recognition_speed_unit': desktop_texts.get('recognition_speed_unit', 'img/s'),
            'preview_current_frame': desktop_texts.get('preview_current_frame', 'Frame: {name}'),
            'preview_current_frame_default': desktop_texts.get('preview_current_frame_default', 'Frame: -'),
            'current_username': str(request.user.username),
            'menu_ui_mode_simple': desktop_texts.get('menu_ui_mode_simple', 'Simple'),
            'menu_ui_mode_advanced': desktop_texts.get('menu_ui_mode_advanced', 'Advanced'),
            'simple_workflow_conductors': desktop_texts.get('simple_workflow_conductors', 'Conductor recognition'),
            'simple_workflow_contacts': desktop_texts.get('simple_workflow_contacts', 'Contact recognition'),
            'simple_workflow_memory': desktop_texts.get('simple_workflow_memory', 'Memory recognition'),
            'workflow_import_ok': 'Configuration imported.' if language == 'en' else 'Конфигурация загружена.',
            'workflow_restore_ok': 'Task parameters restored to the form.' if language == 'en' else 'Параметры задачи восстановлены в форму.',
            'update_not_configured': desktop_texts.get('update_check_not_configured', 'The update source is not configured.'),
            'queue_properties_title': task_texts.get('window_title', 'Task Properties'),
            'restore_button': task_texts.get('restore_button', 'Restore'),
            'close_button': 'Close' if language == 'en' else 'Закрыть',
            'samples_count': desktop_texts.get('samples_count', 'Dataset frames: 0'),
            'samples_count_loading': desktop_texts.get('samples_count_loading', 'Calculating...'),
            'samples_count_template': desktop_texts.get('samples_count_template', 'Dataset frames: {count}'),
            'menu_metrics': desktop_texts.get('menu_metrics', 'Metrics panel'),
            'menu_log_panel': desktop_texts.get('menu_log_panel', 'Log panel'),
            'menu_settings_panel': desktop_texts.get('menu_settings_panel', 'Settings panel'),
            'menu_batch_preview': desktop_texts.get('menu_batch_preview', 'Training batch preview'),
            'menu_release_memory': desktop_texts.get('menu_release_memory', 'Release GPU memory'),
            'menu_theme': desktop_texts.get('menu_theme', 'Theme'),
            'theme_dark': desktop_texts.get('theme_dark', 'Dark'),
            'theme_light': desktop_texts.get('theme_light', 'Light'),
            'menu_sample': desktop_texts.get('menu_sample', 'Dataset'),
            'menu_train': desktop_texts.get('menu_train', 'Training'),
            'menu_pred': desktop_texts.get('menu_pred', 'Recognition'),
            'reset_defaults': settings_panel_texts.get('reset_defaults', 'Reset to defaults'),
            'augmentation_preview_button': settings_panel_texts.get('augmentation_preview_button', 'Augmentation preview'),
            'rare_patch_editor': settings_panel_texts.get('rare_patch_group', 'Rare patch oversampling'),
            'menu_open_validation_gradient': desktop_texts.get('menu_open_validation_gradient', 'Open Validation gradient'),
            'release_memory_ok': 'GPU memory release requested.' if language == 'en' else 'Запрошено освобождение памяти GPU.',
            'reset_defaults_ok': 'Default parameters restored.' if language == 'en' else 'Параметры по умолчанию восстановлены.',
            'webui_tool_unavailable_title': 'Tool status' if language == 'en' else 'Статус инструмента',
            'webui_tool_unavailable': (
                'This Qt desktop tool has no direct browser implementation yet. Use the corresponding WebUI controls or run the Qt client on the server console.'
                if language == 'en'
                else 'У этого Qt-инструмента пока нет прямой браузерной реализации. Используйте соответствующие элементы WebUI или запустите Qt-клиент на консоли сервера.'
            ),
        }
    )
    ui_mode = _resolve_ui_mode(request)
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
        'desktop_texts': desktop_texts,
        'settings_panel_texts': settings_panel_texts,
        'auth_texts': auth_texts,
        'js_texts': js_texts,
        'ui_language': language,
        'current_user_display_name': _current_user_display_name(request),
        'ui_mode': ui_mode,
        'workflow_presets': _workflow_presets(language),
        'workflow_import_url': reverse('webui:workflow_import_api'),
        'workflow_preset_url': reverse('webui:workflow_preset_api'),
        'streaming_recognition_url': reverse('webui:streaming_recognition_api'),
        'queue_properties_url': reverse('webui:queue_properties_api'),
        'queue_restore_url': reverse('webui:queue_restore_api'),
        'help_content_url': reverse('webui:help_content_api'),
        'changelog_content_url': reverse('webui:changelog_content_api'),
        'update_info_url': reverse('webui:update_info_api'),
        'ui_mode_url': reverse('webui:ui_mode_api'),
        'sample_count_url': reverse('webui:sample_count_api'),
        'pick_path_url': reverse('webui:pick_path_api'),
        'release_memory_url': reverse('webui:release_memory_api'),
        'reset_defaults_url': reverse('webui:reset_defaults_api'),
        'tool_status_url': reverse('webui:tool_status_api'),
    }


def _resolve_ui_mode(request: HttpRequest) -> str:
    raw = str(request.session.get(_WEBUI_UI_MODE_SESSION_KEY, '')).strip().lower()
    if raw in {'simple', 'advanced'}:
        return raw
    main_state, _settings_state = get_session_service().load_initial_states()
    state_mode = str(getattr(main_state, 'ui_mode', '') or '').strip().lower()
    if state_mode in {'simple', 'advanced'}:
        request.session[_WEBUI_UI_MODE_SESSION_KEY] = state_mode
        return state_mode
    request.session[_WEBUI_UI_MODE_SESSION_KEY] = 'simple'
    return 'simple'


def _workflow_presets(language: str) -> list[dict[str, str]]:
    desktop_texts = _desktop_texts(language)
    return [
        {
            'key': 'conductors',
            'label': str(desktop_texts.get('simple_workflow_conductors', 'Conductor recognition')),
        },
        {
            'key': 'contacts',
            'label': str(desktop_texts.get('simple_workflow_contacts', 'Contact recognition')),
        },
        {
            'key': 'memory',
            'label': str(desktop_texts.get('simple_workflow_memory', 'Memory recognition')),
        },
    ]


def _workflow_preset_path(preset_key: str) -> Path | None:
    mapping = {
        'conductors': resolve_resource_path('conductors_workflow.json'),
        'contacts': resolve_resource_path('contacts_workflow.json'),
        'memory': resolve_resource_path('memory_workflow.json'),
    }
    return mapping.get(str(preset_key or '').strip().lower())


def _states_payload(main_state, settings_state) -> dict[str, object]:
    return {
        'main': defaults_from_main_state(main_state),
        'settings': defaults_from_settings_state(settings_state),
        'ui_mode': str(getattr(main_state, 'ui_mode', 'simple') or 'simple'),
    }


def _load_workflow_states_from_upload(uploaded_file) -> tuple[object, object]:
    suffix = Path(str(getattr(uploaded_file, 'name', '') or 'workflow.json')).suffix or '.json'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        for chunk in uploaded_file.chunks():
            temp_file.write(chunk)
        temp_path = Path(temp_file.name)
    try:
        return load_workflow_snapshot(temp_path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            _LOG.warning('Failed to delete temporary workflow snapshot: %s', temp_path)


def _webui_upload_root() -> Path:
    root = Path(getattr(settings, 'WEBUI_UPLOAD_ROOT', Path(settings.BASE_DIR) / 'webui_uploads'))
    root.mkdir(parents=True, exist_ok=True)
    _cleanup_old_webui_uploads(root)
    return root


def _cleanup_old_webui_uploads(root: Path) -> None:
    try:
        max_age_seconds = int(getattr(settings, 'WEBUI_UPLOAD_MAX_AGE_SECONDS', 7 * 24 * 60 * 60))
    except (TypeError, ValueError):
        max_age_seconds = 7 * 24 * 60 * 60
    if max_age_seconds <= 0 or not root.exists():
        return

    cutoff = time.time() - max_age_seconds
    for child in root.iterdir():
        try:
            if child.stat().st_mtime >= cutoff:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            _LOG.warning('Failed to cleanup stale WebUI upload path: %s', child)


def _safe_path_part(value: str, fallback: str) -> str:
    safe = get_valid_filename(str(value or '').strip()).strip(' .')
    return safe or fallback


def _safe_relative_parts(relative_path: str, fallback_name: str) -> list[str]:
    normalized = str(relative_path or fallback_name or '').replace('\\', '/').strip()
    candidate = PurePosixPath(normalized or fallback_name)
    if candidate.is_absolute():
        raise ValueError('Absolute upload paths are not allowed.')

    parts: list[str] = []
    for index, part in enumerate(candidate.parts):
        if part in {'', '.', '..'}:
            raise ValueError('Unsafe upload path component.')
        parts.append(_safe_path_part(part, fallback=f'part_{index}'))
    if not parts:
        parts.append(_safe_path_part(fallback_name, fallback='uploaded_file'))
    return parts


def _upload_category_for_request(request: HttpRequest) -> str:
    target = str(request.POST.get('target', '') or '').strip()
    category = _UPLOAD_CATEGORY_BY_FIELD.get(target, '')
    if category:
        return category
    kind = str(request.POST.get('kind', '') or '').strip().lower()
    file_filter = str(request.POST.get('filter', '') or '').strip().lower()
    if file_filter == 'model':
        return 'models'
    return 'folders' if kind == 'folder' else 'files'


def _new_upload_destination(request: HttpRequest, category: str) -> Path:
    user_name = _safe_path_part(str(getattr(request.user, 'username', '') or 'user'), fallback='user')[:80]
    destination = _webui_upload_root() / user_name / _safe_path_part(category, fallback='files') / secrets.token_hex(12)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _is_managed_result_target(request: HttpRequest) -> bool:
    target = str(request.POST.get('target', '') or '').strip()
    kind = str(request.POST.get('kind', '') or '').strip().lower()
    return kind == 'folder' and target == 'main-result_folder'


def _save_uploaded_file(uploaded_file, destination_root: Path, relative_path: str) -> Path:
    parts = _safe_relative_parts(relative_path, fallback_name=str(getattr(uploaded_file, 'name', '') or 'uploaded_file'))
    target_path = destination_root.joinpath(*parts)
    resolved_root = destination_root.resolve()
    resolved_target = target_path.resolve()
    if resolved_root != resolved_target and resolved_root not in resolved_target.parents:
        raise ValueError('Upload path escapes destination root.')

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open('wb') as output_file:
        for chunk in uploaded_file.chunks():
            output_file.write(chunk)
    return target_path


def _is_zip_upload(uploaded_file) -> bool:
    name = str(getattr(uploaded_file, 'name', '') or '').strip().lower()
    return name.endswith('.zip')


def _extract_zip_upload(uploaded_file, destination_root: Path) -> int:
    temp_zip = destination_root / f'_upload_{secrets.token_hex(8)}.zip'
    with temp_zip.open('wb') as output_file:
        for chunk in uploaded_file.chunks():
            output_file.write(chunk)

    extracted_count = 0
    try:
        with zipfile.ZipFile(temp_zip) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                parts = _safe_relative_parts(member.filename, fallback_name=Path(member.filename).name or 'file')
                target_path = destination_root.joinpath(*parts)
                resolved_root = destination_root.resolve()
                resolved_target = target_path.resolve()
                if resolved_root != resolved_target and resolved_root not in resolved_target.parents:
                    raise ValueError('Zip member escapes destination root.')
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, 'r') as source_file, target_path.open('wb') as output_file:
                    shutil.copyfileobj(source_file, output_file)
                extracted_count += 1
    finally:
        temp_zip.unlink(missing_ok=True)
    return extracted_count


def _store_uploaded_path_selection(request: HttpRequest, *, kind: str) -> dict[str, object]:
    files = request.FILES.getlist('files')
    category = _upload_category_for_request(request)
    destination = _new_upload_destination(request, category)

    if not files:
        if _is_managed_result_target(request):
            return {'ok': True, 'path': str(destination), 'count': 0}
        raise ValueError('No files were uploaded.')

    if kind == 'folder' and len(files) == 1 and _is_zip_upload(files[0]):
        extracted_count = _extract_zip_upload(files[0], destination)
        return {'ok': True, 'path': str(destination), 'count': extracted_count}

    relative_paths = request.POST.getlist('relative_paths')
    saved_paths: list[Path] = []
    for index, uploaded_file in enumerate(files):
        relative_path = relative_paths[index] if index < len(relative_paths) else str(getattr(uploaded_file, 'name', ''))
        saved_paths.append(_save_uploaded_file(uploaded_file, destination, relative_path))

    if kind == 'file':
        return {'ok': True, 'path': str(saved_paths[0]), 'count': 1}
    return {'ok': True, 'path': str(destination), 'count': len(saved_paths)}


def _save_streamed_source_files(request: HttpRequest, destination_root: Path) -> int:
    files = request.FILES.getlist('source_files')
    if not files:
        raise ValueError('Source files are required for streaming recognition.')
    if len(files) == 1 and _is_zip_upload(files[0]):
        return _extract_zip_upload(files[0], destination_root)

    relative_paths = request.POST.getlist('source_relative_paths')
    saved_count = 0
    for index, uploaded_file in enumerate(files):
        relative_path = relative_paths[index] if index < len(relative_paths) else str(getattr(uploaded_file, 'name', ''))
        _save_uploaded_file(uploaded_file, destination_root, relative_path)
        saved_count += 1
    return saved_count


def _zip_directory(source_dir: Path, destination_zip: Path) -> int:
    archived_count = 0
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination_zip, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_dir.rglob('*'):
            if not path.is_file():
                continue
            archive.write(path, path.relative_to(source_dir).as_posix())
            archived_count += 1
    return archived_count


def _build_streaming_forms(request: HttpRequest, source_dir: Path, result_dir: Path, language: str, texts: dict[str, object]):
    form_data = request.POST.copy()
    form_data['main-source_folder'] = str(source_dir)
    form_data['main-result_folder'] = str(result_dir)
    main_form = MainWindowForm(form_data, prefix='main', language=language, ui_texts=texts)
    settings_form = SettingsForm(form_data, prefix='settings', language=language, ui_texts=texts)
    return main_form, settings_form


def _run_streaming_recognition(main_state: MainWindowState, settings_state: SettingsState) -> None:
    work_mode, _training_parameters, recognition_parameters = build_workflow_parameters(main_state, settings_state)
    if work_mode != WorkMode.recognition_only:
        raise ValueError('Streaming source upload is supported only for recognition-only mode.')

    from neuralimage.model.NeuralNetwork.model_train_and_recognition import NeuralRecognizer

    message_bus = MessageBus()
    recognizer = NeuralRecognizer(recognition_parameters, message_bus)
    recognizer.run(
        multithreading=bool(getattr(recognition_parameters, 'recognition_multiprocessing_enabled', True)),
    )


def _resolve_markdown_content(section_name: str, language: str) -> str:
    texts = get_ui_section(section_name, language)
    content_ref = str(texts.get('content', '')).strip()
    if not content_ref:
        return ''
    if content_ref.lower().endswith('.md'):
        path = Path(content_ref)
        if not path.is_absolute():
            if path.parts and path.parts[0] == 'resources':
                path = resolve_resource_path(*path.parts[1:])
            else:
                path = Path(__file__).resolve().parent.parent / path
        try:
            return path.read_text(encoding='utf-8')
        except OSError as error:
            _LOG.warning('Failed to read markdown content for %s: %s', section_name, error)
            return ''
    return content_ref


def _task_payload_for_properties(task) -> dict[str, object]:
    queue_status = 'running' if getattr(task, 'task_id', None) == getattr(get_session_service()._processing_session.active_task, 'task_id', None) else ('paused' if getattr(task, 'paused', False) else 'queued')
    snapshot_payload = create_workflow_snapshot_payload(task.main_window_state, task.settings_state)
    return {
        'task_id': int(task.task_id),
        'status': queue_status,
        'owner_username': str(getattr(task, 'owner_username', '') or ''),
        'owner_display_name': str(getattr(task, 'owner_display_name', '') or ''),
        'workflow': snapshot_payload,
        'form_state': _states_payload(task.main_window_state, task.settings_state),
    }


def _post_value(mapping, key: str, default: str = '') -> str:
    return str(mapping.get(key, default) or default)


def _post_bool(mapping, key: str, default: bool = False) -> bool:
    value = mapping.get(key)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _post_int(mapping, key: str, default: int) -> int:
    try:
        return int(str(mapping.get(key, default)).strip())
    except (TypeError, ValueError):
        return int(default)


def _post_float(mapping, key: str, default: float) -> float:
    try:
        return float(str(mapping.get(key, default)).strip())
    except (TypeError, ValueError):
        return float(default)


def _sample_count_settings_from_post(post_data) -> tuple[str, CutSettings, object]:
    sample_folder = _post_value(post_data, 'main-sample_folder', '').strip()
    patch_x = max(1, _post_int(post_data, 'settings-sample_x', 256))
    patch_y = max(1, _post_int(post_data, 'settings-sample_y', 256))
    cut_mode = _post_value(post_data, 'settings-sample_cut_mode', SampleCutMode.online.value).strip().lower()
    online_mode = cut_mode == SampleCutMode.online.value
    settings = CutSettings(
        step=max(1, _post_int(post_data, 'settings-step', 100)),
        x_size=patch_x,
        y_size=patch_y,
        vertical_rotation=_post_bool(post_data, 'settings-vertical_rotation', True),
        horizontal_rotation=_post_bool(post_data, 'settings-horizontal_rotation', True),
        flip_x=_post_bool(post_data, 'settings-flip_x', False),
        flip_y=_post_bool(post_data, 'settings-flip_y', False),
        color_mode=_post_value(post_data, 'settings-color_mode', 'RGB'),
        model=_post_value(post_data, 'settings-model', 'M 720k'),
        additional_augmentation=_post_bool(post_data, 'settings-additional_augmentation', False),
        augmentation_gamma_strength=_post_float(post_data, 'settings-augmentation_gamma_strength', 0.15),
        augmentation_blur_probability=_post_float(post_data, 'settings-augmentation_blur_probability', 0.25),
        augmentation_blur_radius=_post_float(post_data, 'settings-augmentation_blur_radius', 1.0),
        random_crop=online_mode and _post_bool(post_data, 'settings-random_crop', False),
        crops_per_image=max(1, _post_int(post_data, 'settings-crops_per_image', 64)),
        scale_augmentation=online_mode and _post_bool(post_data, 'settings-scale_augmentation', False),
        scale_augmentation_strength=_post_float(post_data, 'settings-scale_augmentation_strength', 0.2),
    )
    synthetic_payload = json.loads(_post_value(post_data, 'settings-synthetic_defect_generator_json', '{}') or '{}')
    return sample_folder, settings, synthetic_payload


def _build_login_context(request: HttpRequest, language: str, texts: dict[str, object]) -> dict[str, object]:
    auth_texts = _auth_ui_texts(language)
    return {
        'app_name': APP_NAME,
        'app_version': APP_VERSION,
        'app_title': get_app_title(),
        'texts': texts,
        'auth_texts': auth_texts,
        'ui_language': language,
        'ldap_login_url': reverse('webui:auth_ldap_login'),
        'superuser_login_url': reverse('webui:auth_superuser_login'),
        'ldap_auth_configured': _ldap_auth_config() is not None,
    }


def _require_login_page(request: HttpRequest, *, language: str, texts: dict[str, object]):
    return render(request, 'webui/login.html', _build_login_context(request, language, texts), status=401)


def _require_authenticated_api(request: HttpRequest, *, language: str) -> JsonResponse | None:
    if request.user.is_authenticated:
        return None
    message = 'Authentication required.' if language == 'en' else 'Требуется авторизация.'
    return JsonResponse({'ok': False, 'error': message}, status=401)


def _is_authorized_broadcast_request(request: HttpRequest) -> bool:
    token = str(os.getenv('NEURALIMAGE_WEBUI_ADMIN_TOKEN', '') or '').strip()
    provided = str(request.headers.get('X-NeuralImage-Admin-Token', '') or '').strip()
    if token and secrets.compare_digest(token, provided):
        return True
    return bool(request.user.is_authenticated and request.user.is_superuser)


def _serialize_notifications(after_id: int = 0) -> tuple[list[dict[str, object]], int]:
    notifications, latest_id = get_broadcast_notification_store().after(after_id, limit=_BROADCAST_LIMIT)
    return (
        [
            {
                'id': item.id,
                'message': item.message,
                'created_by': item.created_by,
                'created_at': item.created_at.isoformat(),
            }
            for item in notifications
        ],
        latest_id,
    )


@require_GET
def dashboard(request: HttpRequest):
    language = _resolve_ui_language(request)
    texts = _webui_texts(language)
    if not request.user.is_authenticated:
        return _require_login_page(request, language=language, texts=texts)
    main_form, settings_form = _build_forms(request, language, texts)
    status = get_session_service().snapshot(current_username=str(request.user.username)).get('status', 'idle')
    return render(
        request,
        'webui/dashboard.html',
        _build_dashboard_context(request, main_form, settings_form, language, texts, status),
    )


@require_POST
def auth_ldap_login(request: HttpRequest):
    language = _resolve_ui_language(request)
    config = _ldap_auth_config()
    if config is None:
        messages.error(request, _auth_ui_texts(language)['auth_not_configured'])
        return redirect('webui:dashboard')

    username = str(request.POST.get('username', '') or '').strip()
    password = str(request.POST.get('password', '') or '')
    try:
        profile = _ldap_authenticate(username, password, config)
        user = _resolve_or_create_user_from_ldap(profile)
    except Exception as error:
        _LOG.exception('LDAP authentication failed')
        message = f'LDAP authentication failed: {error}' if language == 'en' else f'Ошибка авторизации LDAP: {error}'
        messages.error(request, message)
        return redirect('webui:dashboard')

    login(request, user, backend='django.contrib.auth.backends.ModelBackend')
    success_message = 'Signed in via LDAP.' if language == 'en' else 'Вход через LDAP выполнен.'
    messages.success(request, success_message)
    return redirect('webui:dashboard')


@require_POST
def auth_superuser_login(request: HttpRequest):
    language = _resolve_ui_language(request)
    _ensure_local_superuser()
    username = str(request.POST.get('username', '')).strip()
    password = str(request.POST.get('password', ''))
    user = authenticate(
        request,
        username=username,
        password=password,
    )
    if user is None:
        messages.error(
            request,
            'Invalid login or password.' if language == 'en' else 'Неверный логин или пароль.',
        )
        return redirect('webui:dashboard')
    login(request, user, backend='django.contrib.auth.backends.ModelBackend')
    messages.success(
        request,
        'Signed in as local superuser.' if language == 'en' else 'Выполнен вход как локальный суперпользователь.',
    )
    return redirect('webui:dashboard')


@require_POST
def auth_logout(request: HttpRequest):
    logout(request)
    language = _resolve_ui_language(request)
    messages.success(request, 'Signed out.' if language == 'en' else 'Вы вышли из системы.')
    return redirect('webui:dashboard')


@require_POST
def start_processing(request: HttpRequest):
    language = _resolve_ui_language(request)
    texts = _webui_texts(language)
    if not request.user.is_authenticated:
        return _require_login_page(request, language=language, texts=texts)

    main_form, settings_form = _build_forms(request, language, texts)
    if not main_form.is_valid() or not settings_form.is_valid():
        return render(
            request,
            'webui/dashboard.html',
            _build_dashboard_context(
                request,
                main_form,
                settings_form,
                language,
                texts,
                get_session_service().snapshot(current_username=str(request.user.username)).get('status', 'idle'),
            ),
            status=400,
        )

    session = get_session_service()
    ok, error, success_message = session.start(
        main_form.to_state(),
        settings_form.to_state(),
        owner_username=str(request.user.username),
        owner_display_name=_current_user_display_name(request),
    )
    if not ok:
        messages.error(request, error or str(texts.get('start_error', 'Не удалось запустить обработку.')))
    else:
        messages.success(request, success_message or str(texts.get('start_ok', 'Обработка запущена.')))
    return redirect('webui:dashboard')


@require_POST
def stop_processing(request: HttpRequest):
    language = _resolve_ui_language(request)
    texts = _webui_texts(language)
    if not request.user.is_authenticated:
        return _require_login_page(request, language=language, texts=texts)

    session = get_session_service()
    ok, error = session.stop(owner_username=str(request.user.username))
    if not ok:
        messages.error(request, error or str(texts.get('stop_error', 'Не удалось остановить обработку.')))
    else:
        messages.success(request, str(texts.get('stop_ok', 'Запрос на остановку отправлен.')))
    return redirect('webui:dashboard')


@require_GET
def status_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    texts = _webui_texts(language)
    after = request.GET.get('after', '0')
    try:
        after_id = int(after)
    except ValueError:
        after_id = 0
    notification_after = request.GET.get('notification_after', '0')
    try:
        notification_after_id = int(notification_after)
    except ValueError:
        notification_after_id = 0

    session = get_session_service()
    snapshot = session.snapshot(after_event_id=after_id, current_username=str(request.user.username))
    status = str(snapshot.get('status', 'idle'))
    snapshot['status_display'] = _status_display(status, texts)
    notifications, last_notification_id = _serialize_notifications(notification_after_id)
    snapshot['notifications'] = notifications
    snapshot['last_notification_id'] = last_notification_id
    return JsonResponse(snapshot)


@csrf_exempt
@require_POST
def broadcast_notification_api(request: HttpRequest):
    if not _is_authorized_broadcast_request(request):
        return JsonResponse({'ok': False, 'error': 'Forbidden.'}, status=403)
    message = str(request.POST.get('message', '') or '').strip()
    if not message:
        return JsonResponse({'ok': False, 'error': 'Message is required.'}, status=400)
    if len(message) > 4000:
        return JsonResponse({'ok': False, 'error': 'Message is too long.'}, status=400)
    created_by = str(request.POST.get('created_by', '') or '').strip()
    if not created_by and request.user.is_authenticated:
        created_by = str(request.user.username or '')
    notification = get_broadcast_notification_store().add(message=message, created_by=created_by[:150])
    return JsonResponse(
        {
            'ok': True,
            'notification': {
                'id': notification.id,
                'message': notification.message,
                'created_by': notification.created_by,
                'created_at': notification.created_at.isoformat(),
            },
        }
    )


@require_POST
def queue_remove_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    task_id_raw = str(request.POST.get('task_id', '')).strip()
    try:
        task_id = int(task_id_raw)
    except ValueError:
        return JsonResponse({'ok': False, 'error': 'Invalid task id.'}, status=400)

    ok, error = get_session_service().remove_task(task_id, owner_username=str(request.user.username))
    if not ok:
        return JsonResponse({'ok': False, 'error': error or 'Failed to remove task.'}, status=400)
    return JsonResponse({'ok': True})


@require_POST
def queue_pause_toggle_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    task_id_raw = str(request.POST.get('task_id', '')).strip()
    try:
        task_id = int(task_id_raw)
    except ValueError:
        return JsonResponse({'ok': False, 'error': 'Invalid task id.'}, status=400)

    ok, error = get_session_service().toggle_pause_task(task_id, owner_username=str(request.user.username))
    if not ok:
        return JsonResponse({'ok': False, 'error': error or 'Failed to change queue state.'}, status=400)
    return JsonResponse({'ok': True})


@require_GET
def queue_properties_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    task_id_raw = str(request.GET.get('task_id', '')).strip()
    try:
        task_id = int(task_id_raw)
    except ValueError:
        return JsonResponse({'ok': False, 'error': 'Invalid task id.'}, status=400)

    service = get_session_service()
    task = service.get_task(task_id)
    if task is None:
        return JsonResponse({'ok': False, 'error': 'Task not found.'}, status=404)
    payload = _task_payload_for_properties(task)
    payload['can_restore'] = bool(str(getattr(task, 'owner_username', '') or '') == str(request.user.username or ''))
    return JsonResponse({'ok': True, 'task': payload})


@require_POST
def queue_restore_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    task_id_raw = str(request.POST.get('task_id', '')).strip()
    try:
        task_id = int(task_id_raw)
    except ValueError:
        return JsonResponse({'ok': False, 'error': 'Invalid task id.'}, status=400)

    service = get_session_service()
    task = service.get_task(task_id)
    if task is None:
        return JsonResponse({'ok': False, 'error': 'Task not found.'}, status=404)
    if str(getattr(task, 'owner_username', '') or '') != str(request.user.username or ''):
        return JsonResponse({'ok': False, 'error': 'You can restore only your own tasks.'}, status=403)
    return JsonResponse({'ok': True, 'state': _states_payload(task.main_window_state, task.settings_state)})


@require_POST
def workflow_import_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    upload = request.FILES.get('workflow_file')
    if upload is None:
        return JsonResponse({'ok': False, 'error': 'Workflow file is required.'}, status=400)
    try:
        main_state, settings_state = _load_workflow_states_from_upload(upload)
    except (OSError, ValueError) as error:
        return JsonResponse({'ok': False, 'error': str(error)}, status=400)
    return JsonResponse({'ok': True, 'state': _states_payload(main_state, settings_state)})


@require_GET
def workflow_preset_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    preset_key = str(request.GET.get('preset', '')).strip().lower()
    preset_path = _workflow_preset_path(preset_key)
    if preset_path is None or not preset_path.is_file():
        return JsonResponse({'ok': False, 'error': 'Unknown workflow preset.'}, status=404)
    try:
        main_state, settings_state = load_workflow_snapshot(preset_path)
    except (OSError, ValueError) as error:
        return JsonResponse({'ok': False, 'error': str(error)}, status=400)
    return JsonResponse({'ok': True, 'state': _states_payload(main_state, settings_state), 'preset': preset_key})


@require_POST
def streaming_recognition_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    texts = _webui_texts(language)
    with tempfile.TemporaryDirectory(prefix='neuralimage_stream_source_') as source_temp, tempfile.TemporaryDirectory(
        prefix='neuralimage_stream_result_'
    ) as result_temp:
        source_dir = Path(source_temp)
        result_dir = Path(result_temp)
        try:
            source_count = _save_streamed_source_files(request, source_dir)
            if source_count <= 0:
                raise ValueError('No source images were provided.')

            main_form, settings_form = _build_streaming_forms(request, source_dir, result_dir, language, texts)
            if not main_form.is_valid() or not settings_form.is_valid():
                return JsonResponse(
                    {
                        'ok': False,
                        'error': 'Invalid recognition settings.',
                        'main_errors': main_form.errors,
                        'settings_errors': settings_form.errors,
                    },
                    status=400,
                )

            _run_streaming_recognition(main_form.to_state(), settings_form.to_state())
            downloads_root = _webui_upload_root() / 'downloads'
            archive_path = downloads_root / f'neuralimage_result_{secrets.token_hex(8)}.zip'
            archived_count = _zip_directory(result_dir, archive_path)
            if archived_count <= 0:
                return JsonResponse({'ok': False, 'error': 'Recognition produced no result files.'}, status=500)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            _LOG.exception('Streaming recognition request failed')
            return JsonResponse({'ok': False, 'error': str(error)}, status=400)
        except Exception as error:
            _LOG.exception('Streaming recognition execution failed')
            return JsonResponse({'ok': False, 'error': str(error)}, status=500)

    return FileResponse(
        archive_path.open('rb'),
        as_attachment=True,
        filename=archive_path.name,
        content_type='application/zip',
    )


@require_POST
def ui_mode_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    ui_mode = str(request.POST.get('ui_mode', '')).strip().lower()
    if ui_mode not in {'simple', 'advanced'}:
        return JsonResponse({'ok': False, 'error': 'Invalid UI mode.'}, status=400)
    request.session[_WEBUI_UI_MODE_SESSION_KEY] = ui_mode
    return JsonResponse({'ok': True, 'ui_mode': ui_mode})


@require_GET
def help_content_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized
    texts = get_ui_section('help_dialog', language)
    return JsonResponse(
        {
            'ok': True,
            'title': str(texts.get('window_title', 'Help')),
            'content': _resolve_markdown_content('help_dialog', language),
        }
    )


@require_GET
def changelog_content_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized
    texts = get_ui_section('changelog_dialog', language)
    return JsonResponse(
        {
            'ok': True,
            'title': str(texts.get('window_title', 'Changelog')),
            'content': _resolve_markdown_content('changelog_dialog', language),
        }
    )


@require_GET
def update_info_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    desktop_texts = _desktop_texts(language)
    config = load_update_client_config()
    selected_channel = normalize_update_channel(
        request.GET.get('channel') or load_selected_update_channel(
            config.default_channel,
            available_channels=config.available_channels,
        )
    )
    if selected_channel not in tuple(normalize_update_channel(item) for item in config.available_channels):
        selected_channel = normalize_update_channel(config.default_channel)
    save_selected_update_channel(selected_channel)
    manifest_url = config.get_manifest_url(selected_channel)
    if not manifest_url:
        return JsonResponse(
            {
                'ok': True,
                'configured': False,
                'title': str(desktop_texts.get('menu_check_updates', 'Check for updates')),
                'message': str(desktop_texts.get('update_check_not_configured', 'The update source is not configured.')),
                'channels': list(config.available_channels),
                'selected_channel': selected_channel,
            }
        )
    update_info = fetch_update_info(manifest_url, expected_channel=selected_channel)
    if update_info is None:
        return JsonResponse(
            {
                'ok': False,
                'error': str(desktop_texts.get('update_check_failed', 'Failed to check for updates.')),
            },
            status=502,
        )
    message_template = str(
        desktop_texts.get(
            'update_manual_text',
            'Installed version: {current_version}\nSelected channel: {channel}\nServer version: {new_version}.',
        )
    )
    return JsonResponse(
        {
            'ok': True,
            'configured': True,
            'title': str(desktop_texts.get('menu_check_updates', 'Check for updates')),
            'message': message_template.format(
                current_version=APP_VERSION,
                channel=selected_channel,
                new_version=update_info.version,
            ),
            'release_history': collect_release_history(update_info),
            'channels': list(config.available_channels),
            'selected_channel': selected_channel,
            'current_version': APP_VERSION,
            'new_version': update_info.version,
            'download_url': update_info.download_url,
        }
    )


@require_POST
def sample_count_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    sample_folder, settings, synthetic_payload = _sample_count_settings_from_post(request.POST)
    sample_path = Path(sample_folder)
    if not sample_folder or not sample_path.is_dir():
        return JsonResponse({'ok': True, 'count': 0})
    try:
        image_paths = SampleWorker.collect_image_paths(sample_path)
        image_sizes = SampleWorker.collect_image_sizes(image_paths)
        total_samples = SampleWorker.calculate_total_samples(image_sizes, settings)
        synthetic_generator = build_synthetic_defect_generator_parameters(synthetic_payload)
        if synthetic_generator.enabled and float(synthetic_generator.epoch_size_factor) > 0.0 and image_sizes:
            synthetic_frame_count = max(1, int(round(len(image_sizes) * float(synthetic_generator.epoch_size_factor))))
            synthetic_size_xy = (
                max(int(settings.x_size), int(synthetic_generator.image_size_xy[0])),
                max(int(settings.y_size), int(synthetic_generator.image_size_xy[1])),
            )
            total_samples += synthetic_frame_count * SampleWorker.calculate_image_parts_for_settings(
                (int(synthetic_size_xy[1]), int(synthetic_size_xy[0])),
                settings,
            )
    except Exception as error:
        _LOG.exception('Failed to calculate sample count for WebUI')
        return JsonResponse({'ok': False, 'error': str(error)}, status=500)
    return JsonResponse({'ok': True, 'count': int(total_samples)})


@require_POST
def release_memory_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    gc.collect()
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()
    except Exception as error:
        _LOG.warning('GPU memory release failed: %s', error)
        return JsonResponse({'ok': False, 'error': str(error)}, status=500)
    return JsonResponse({'ok': True, 'cuda_available': cuda_available})


@require_POST
def reset_defaults_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized
    return JsonResponse({'ok': True, 'state': _states_payload(MainWindowState(), SettingsState())})


@require_GET
def tool_status_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    tool = str(request.GET.get('tool', '')).strip()
    labels = {
        'augmentation_preview': get_ui_section('settings_panel', language).get('augmentation_preview_button', 'Augmentation preview'),
        'rare_patch_editor': get_ui_section('settings_panel', language).get('rare_patch_group', 'Rare patch oversampling'),
        'validation_gradient': _desktop_texts(language).get('menu_open_validation_gradient', 'Open Validation gradient'),
        'developer': _desktop_texts(language).get('menu_developer', 'Developer'),
    }
    title = str(labels.get(tool, 'Tool status'))
    message = (
        'This Qt desktop tool has no direct browser implementation yet. Use the corresponding WebUI controls or run the Qt client on the server console.'
        if language == 'en'
        else 'У этого Qt-инструмента пока нет прямой браузерной реализации. Используйте соответствующие элементы WebUI или запустите Qt-клиент на консоли сервера.'
    )
    return JsonResponse({'ok': True, 'title': title, 'message': message, 'tool': tool})


@require_POST
def pick_path_api(request: HttpRequest):
    language = _resolve_ui_language(request)
    unauthorized = _require_authenticated_api(request, language=language)
    if unauthorized is not None:
        return unauthorized

    texts = _webui_texts(language)
    kind = str(request.POST.get('kind', '')).strip().lower()
    file_filter = str(request.POST.get('filter', '')).strip().lower()
    if kind not in {'folder', 'file'}:
        return JsonResponse(
            {'ok': False, 'error': str(texts.get('picker_invalid_kind', 'Некорректный тип выбора пути.'))},
            status=400,
        )

    if str(request.POST.get('target', '') or '').strip() == 'main-source_folder' and request.FILES:
        return JsonResponse(
            {'ok': False, 'error': 'source_folder uploads must use the streaming recognition endpoint.'},
            status=400,
        )

    if request.FILES or _is_managed_result_target(request):
        try:
            return JsonResponse(_store_uploaded_path_selection(request, kind=kind))
        except (OSError, ValueError) as error:
            _LOG.exception('Failed to store WebUI uploaded path selection')
            return JsonResponse({'ok': False, 'error': str(error)}, status=400)

    if not _is_local_request(request):
        return JsonResponse({'ok': False, 'error': 'Remote clients must upload selected files through the browser.'}, status=400)

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
