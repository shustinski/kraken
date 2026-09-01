# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


analysis = Analysis(
    ["entries/desktop_entry.py"],
    pathex=[".."],
    binaries=[],
    datas=collect_data_files("kraken_core") + collect_data_files("kraken_hub"),
    hiddenimports=(
        collect_submodules("kraken_core")
        + collect_submodules("kraken_hub")
        + collect_submodules("kraken_manager")
        + ["keyring.backends.Windows"]
    ),
    hookspath=[],
    runtime_hooks=[],
    excludes=["fastapi", "uvicorn", "alembic", "sqlalchemy", "psycopg"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="KrakenDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="KrakenDesktop",
)
