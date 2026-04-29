from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


# SPECPATH is the folder that contains this .spec (…/plugins/csliser/packaging/), not the .spec path.
PROJECT_ROOT = Path(SPECPATH).resolve().parent
APP_NAME = "CSliser"
APP_ICON = PROJECT_ROOT / "resources" / "icons" / "csliser.ico"

datas = collect_data_files(
    "kraken_core",
    includes=[
        "resources/styles/*.qss",
        "resources/styles/icons/*",
    ],
)
datas += [
    (str(PROJECT_ROOT / "resources" / "icons" / "csliser.ico"), "plugins/csliser/resources/icons"),
    (str(PROJECT_ROOT / "resources" / "icons" / "csliser.png"), "plugins/csliser/resources/icons"),
]


a = Analysis(
    [str(PROJECT_ROOT / "__main__.py")],
    pathex=[str(PROJECT_ROOT / "src")],
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
