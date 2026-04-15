# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
_spec_file = globals().get('__file__')
project_root = Path(_spec_file).resolve().parent if _spec_file else Path.cwd()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, '') or '').strip().lower()
    if not raw:
        return bool(default)
    return raw in {'1', 'true', 'yes', 'on'}


def _resolve_build_target() -> str:
    raw = str(os.getenv('NEURALIMAGE_BUILD_TARGET', '') or '').strip().lower()
    if raw in {'linux', 'windows'}:
        return raw
    if sys.platform.startswith('linux'):
        return 'linux'
    if sys.platform.startswith('win'):
        return 'windows'
    return 'native'


build_target = _resolve_build_target()
app_name = str(os.getenv('NEURALIMAGE_APP_NAME', 'NeuralImage') or 'NeuralImage').strip() or 'NeuralImage'

# Build-time flag: set via env to bundle optional Django WebUI assets/deps.
include_webui = _env_flag('NEURALIMAGE_INCLUDE_WEBUI', default=False)

datas = [
    # NOTE:
    # In PyInstaller >=6 onedir layout, all app files are already placed under
    # dist/<app>/_internal. Destination paths below must NOT include "_internal",
    # otherwise resources end up nested as _internal/_internal/... and runtime
    # file lookups like "_internal/resources/dark_modern.qss" fail.
    ('_internal/icon.png', '.'),
    ('_internal/icon.ico', '.'),
    ('_internal/settings_icon.png', '.'),
    ('_internal/resources/dark_modern.qss', 'resources'),
    ('_internal/resources/style.qss', 'resources'),
    ('_internal/resources/new_style.qss', 'resources'),
    ('_internal/resources/icons/check_light.svg', 'resources/icons'),
    ('_internal/resources/icons/chevron_down_light.svg', 'resources/icons'),
    ('_internal/resources/icons/chevron_up_light.svg', 'resources/icons'),
    ('resources/ui_texts_ru.json', 'resources'),
    ('resources/ui_texts_en.json', 'resources'),
    ('resources/changelog.md', 'resources'),
    ('resources/help.md', 'resources'),
    ('resources/changelog_ru.md', 'resources'),
    ('resources/changelog_en.md', 'resources'),
    ('resources/help_ru.md', 'resources'),
    ('resources/help_en.md', 'resources'),
    ('resources/conductors_workflow.json', 'resources'),
    ('resources/contacts_workflow.json', 'resources'),
    ('resources/memory_workflow.json', 'resources'),
]

update_client_path = project_root / 'resources' / 'update_client.json'
if update_client_path.exists():
    datas.append((str(update_client_path), 'resources'))

offline_timm_root = project_root / '_internal' / 'models' / 'timm'
if offline_timm_root.exists():
    for source_path in offline_timm_root.rglob('model.safetensors'):
        relative_parent = source_path.parent.relative_to(project_root / '_internal')
        datas.append((str(source_path), str(relative_parent)))

if include_webui:
    # WebUI assets for optional --web mode.
    datas += collect_data_files('webui', includes=['templates/**/*.html', 'static/**/*'])

hiddenimports = []
if include_webui:
    hiddenimports += [
        'django',
        'webui_project',
        'webui_project.settings',
    ]

base_excludes = []


if not include_webui:
    # Optional web mode is disabled in this build.
    base_excludes += [
        'django',
        'webui',
        'webui_project',
    ]

icon_path = None
if build_target == 'windows':
    icon_candidate = project_root / '_internal' / 'icon.ico'
    if icon_candidate.exists():
        icon_path = [str(icon_candidate)]
elif build_target == 'linux':
    icon_candidate = project_root / '_internal' / 'icon.png'
    if icon_candidate.exists():
        icon_path = [str(icon_candidate)]

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],          # <-- torch DLLs / pyds
    datas=datas ,
    hiddenimports=hiddenimports,
    # Disable local hook overrides and rely on PyInstaller's built-in torch hook behavior.
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hooks/rth_set_workdir.py'],  # ensure relative resource paths resolve from exe dir
    excludes=base_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=app_name,
)
