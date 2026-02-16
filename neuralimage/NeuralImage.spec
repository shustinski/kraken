# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_all

block_cipher = None
_spec_file = globals().get('__file__')
project_root = Path(_spec_file).resolve().parent if _spec_file else Path.cwd()

# Build-time flag: set to True to bundle optional Django WebUI assets/deps.
include_webui = False

# --- Torch collection (critical for c10.dll / torch\lib DLLs) ---
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

datas = [
    ('_internal/icon.png', '_internal'),
    ('_internal/icon.ico', '_internal'),
    ('_internal/settings_icon.png', '_internal'),
    ('_internal/resources/dark_modern.qss', '_internal/resources'),
    ('_internal/resources/style.qss', '_internal/resources'),
    ('_internal/resources/new_style.qss', '_internal/resources'),
    ('_internal/resources/icons/check_light.svg', '_internal/resources/icons'),
    ('_internal/resources/icons/chevron_down_light.svg', '_internal/resources/icons'),
    ('_internal/resources/icons/chevron_up_light.svg', '_internal/resources/icons'),
    ('resources/ui_texts_ru.json', 'resources'),
]


cuda_datas = []
cuda_bins = []
cuda_hidden = []

for pkg in [
    'nvidia.cublas',
    'nvidia.cudnn',
    'nvidia.cuda_runtime',
    'nvidia.nvrtc',
]:
    try:
        d, b, h = collect_all(pkg)
        cuda_datas += d
        cuda_bins += b
        cuda_hidden += h
    except Exception:
        pass

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


base_excludes = [
    # Compile stack is intentionally disabled for frozen app to keep build stable/fast.
    'torch._dynamo',
    'torch._inductor',
    'triton',
    # Not used by this desktop build.
    'tensorflow',
    'pytest',
    'torchaudio',
    'OpenGL',
    'tensorboard',
]
if not include_webui:
    # Optional web mode is disabled in this build.
    base_excludes += [
        'django',
        'webui',
        'webui_project',
    ]

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=torch_binaries + cuda_bins,          # <-- torch DLLs / pyds
    datas=datas + torch_datas + cuda_datas,
    hiddenimports=hiddenimports + torch_hiddenimports + cuda_hidden,
    # Disable local hook overrides (e.g., hooks/hook-torch.py) to use
    # PyInstaller's built-in torch hook behavior.
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project_root / 'rth_torch_dllpaths.py')],  # <-- IMPORTANT
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
    name='NeuralImage',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['_internal/icon.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='NeuralImage',
)
