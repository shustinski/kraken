# -*- mode: python ; coding: utf-8 -*-

import sys
from importlib.util import find_spec
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

APP_NAME = 'NeuralImage'
INCLUDE_WEBUI = True
BUILD_TARGET = 'auto'  # Supported values: 'auto', 'linux', 'windows', 'native'.

block_cipher = None
_spec_file = globals().get('__file__')
project_root = Path(_spec_file).resolve().parent.parent if _spec_file else Path.cwd()


include_webui = bool(INCLUDE_WEBUI)


def _resolve_build_target() -> str:
    raw = str(BUILD_TARGET or 'auto').strip().lower()
    if raw in {'linux', 'windows', 'native'}:
        return raw
    if sys.platform.startswith('linux'):
        return 'linux'
    if sys.platform.startswith('win'):
        return 'windows'
    return 'native'


build_target = _resolve_build_target()
app_name = str(APP_NAME or 'NeuralImage').strip() or 'NeuralImage'

datas = [
    # NOTE:
    # In PyInstaller >=6 onedir layout, all app files are already placed under
    # dist/<app>/_internal. Destination paths below must NOT include "_internal",
    # otherwise resources end up nested as _internal/_internal/... and runtime
    # app-resource lookups fail.
    (str(project_root / 'resources' / 'internal' / 'icon.png'), '.'),
    (str(project_root / 'resources' / 'internal' / 'icon.ico'), '.'),
    (str(project_root / 'resources' / 'internal' / 'settings_icon.png'), '.'),
    (str(project_root / 'resources' / 'ui_texts_ru.json'), 'resources'),
    (str(project_root / 'resources' / 'ui_texts_en.json'), 'resources'),
    (str(project_root / 'resources' / 'changelog.md'), 'resources'),
    (str(project_root / 'resources' / 'help.md'), 'resources'),
    (str(project_root / 'resources' / 'changelog_ru.md'), 'resources'),
    (str(project_root / 'resources' / 'changelog_en.md'), 'resources'),
    (str(project_root / 'resources' / 'help_ru.md'), 'resources'),
    (str(project_root / 'resources' / 'help_en.md'), 'resources'),
    (str(project_root / 'resources' / 'conductors_workflow.json'), 'resources'),
    (str(project_root / 'resources' / 'contacts_workflow.json'), 'resources'),
    (str(project_root / 'resources' / 'memory_workflow.json'), 'resources'),
]

update_client_path = project_root / 'resources' / 'update_client.json'
if update_client_path.exists():
    datas.append((str(update_client_path), 'resources'))

offline_timm_root = project_root / 'resources' / 'internal' / 'models' / 'timm'
if offline_timm_root.exists():
    for source_path in offline_timm_root.rglob('model.safetensors'):
        relative_parent = source_path.parent.relative_to(project_root / 'resources' / 'internal')
        datas.append((str(source_path), str(relative_parent)))

if include_webui:
    # WebUI assets for optional --web mode.
    datas += collect_data_files('neuralimage.webui', includes=['templates/**/*.html', 'static/**/*'])
    datas += collect_data_files('django', includes=['contrib/admin/templates/**/*', 'contrib/admin/static/**/*'])
    datas += copy_metadata('django')

hiddenimports = []
if include_webui:
    hiddenimports += [
        'django',
        'django.core.management',
        'neuralimage.webui',
        'neuralimage.webui_project',
        'neuralimage.webui_project.settings',
        'neuralimage.webui_project.urls',
    ]
    hiddenimports += collect_submodules('django')
    hiddenimports += collect_submodules('neuralimage.webui')
    hiddenimports += collect_submodules('neuralimage.webui_project')
    if find_spec('ldap3') is not None:
        hiddenimports += collect_submodules('ldap3')

datas += collect_data_files(
    'kraken_core',
    includes=[
        'resources/styles/*.qss',
        'resources/styles/icons/*',
    ],
)

base_excludes = []


if not include_webui:
    # Optional web mode is disabled in this build.
    base_excludes += [
        'django',
        'neuralimage.webui',
        'neuralimage.webui_project',
    ]

icon_path = None
if build_target == 'windows':
    icon_candidate = project_root / 'resources' / 'internal' / 'icon.ico'
    if icon_candidate.exists():
        icon_path = [str(icon_candidate)]
elif build_target == 'linux':
    icon_candidate = project_root / 'resources' / 'internal' / 'icon.png'
    if icon_candidate.exists():
        icon_path = [str(icon_candidate)]

a = Analysis(
    [str(project_root / 'src' / 'neuralimage' / 'main.py')],
    pathex=[str(project_root / 'src')],
    binaries=[],          # <-- torch DLLs / pyds
    datas=datas ,
    hiddenimports=hiddenimports,
    hookspath=[str(project_root / 'packaging' / 'hooks')],
    hooksconfig={},
    runtime_hooks=[str(project_root / 'packaging' / 'hooks' / 'rth_set_workdir.py')],
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
