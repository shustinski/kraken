# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

block_cipher = None
_spec_file = globals().get("__file__")
project_root = Path(_spec_file).resolve().parent.parent if _spec_file else Path.cwd()

entrypoint = project_root / "src" / "neuralimage" / "Validation_gradient_widget_lite" / "debug" / "main.py"
runtime_hook = project_root / "packaging" / "hooks" / "rth_set_workdir.py"
icon_path = project_root / "resources" / "internal" / "icon.ico"

datas = []
if icon_path.exists():
    datas.append((str(icon_path), "."))

hiddenimports = [
    "neuralimage.Validation_gradient_widget_lite",
    "neuralimage.Validation_gradient_widget_lite.debug.main",
]

# Keep this executable focused on the lite validation viewer. The repository
# module may use NumPy/OpenCV/SciPy paths, but it does not need the main
# NeuralImage training stack or WebUI.
excludes = [
    "neuralimage.application",
    "neuralimage.augmentations",
    "neuralimage.controller",
    "django",
    "neuralimage.infrastructure",
    "neuralimage.lib",
    "neuralimage.manage",
    "matplotlib",
    "neuralimage.model",
    "neuralimage.presenter",
    "pyqtgraph",
    "sklearn",
    "tensorflow",
    "timm",
    "torch",
    "torchaudio",
    "torchvision",
    "neuralimage.UI",
    "neuralimage.view",
    "neuralimage.webui",
    "neuralimage.webui_project",
]

a = Analysis(
    [str(entrypoint)],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(runtime_hook)] if runtime_hook.exists() else [],
    excludes=excludes,
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
    name="ValidationGradientWidgetLite",
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
    icon=[str(icon_path)] if icon_path.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ValidationGradientWidgetLite",
)
