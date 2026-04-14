from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


PROJECT_ROOT = Path(SPECPATH).resolve().parent
APP_NAME = "ViaLaNetPolygonWidget"
APP_ICON = PROJECT_ROOT / "polygon_widget" / "resources" / "styles" / "icons" / "icon.ico"

datas = collect_data_files(
    "polygon_widget",
    includes=[
        "pipeline_example.json",
        "resources/styles/*.qss",
        "resources/styles/icons/*",
    ],
)


a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(APP_ICON),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
