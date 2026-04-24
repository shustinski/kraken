# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

project_root = Path(SPECPATH).resolve().parent.parent

block_cipher = None

added_files = [
    (str(project_root / "resources" / "icons" / "app_icon.ico"), "resources/icons"),
    (str(project_root / "src" / "krona" / "presentation" / "qt" / "ui_strings.json"), "krona/presentation/qt"),
    (str(project_root / "src" / "krona" / "presentation" / "qt" / "ui_strings_ru.json"), "krona/presentation/qt"),
    (str(project_root / "src" / "krona" / "presentation" / "qt" / "theme" / "dark.qss"), "krona/presentation/qt/theme"),
    (str(project_root / "src" / "krona" / "presentation" / "qt" / "theme" / "light.qss"), "krona/presentation/qt/theme"),
]
added_files += collect_data_files(
    "kraken_core",
    includes=[
        "resources/styles/*.qss",
        "resources/styles/icons/*",
        "resources/icons/*",
    ],
)

a = Analysis(
    [str(project_root / "src" / "krona" / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=added_files,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name="Krona",
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
    icon=str(project_root / "resources" / "icons" / "app_icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Krona",
)
