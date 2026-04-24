# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(SPECPATH)

block_cipher = None

added_files = [
    (str(project_root / "icons" / "app_icon.ico"), "icons"),
    (str(project_root / "logic_analyzer" / "presentation" / "qt" / "ui_strings.json"), "logic_analyzer/presentation/qt"),
    (str(project_root / "logic_analyzer" / "presentation" / "qt" / "ui_strings_ru.json"), "logic_analyzer/presentation/qt"),
    (str(project_root / "logic_analyzer" / "presentation" / "qt" / "theme" / "dark.qss"), "logic_analyzer/presentation/qt/theme"),
    (str(project_root / "logic_analyzer" / "presentation" / "qt" / "theme" / "light.qss"), "logic_analyzer/presentation/qt/theme"),
]

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
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
    name="LogicAnalyzer",
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
    icon=str(project_root / "icons" / "app_icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="LogicAnalyzer",
)
