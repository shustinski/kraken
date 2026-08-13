# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules


common_hidden = (
    collect_submodules("kraken_server")
    + collect_submodules("kraken_manager")
    + collect_submodules("sqlalchemy.dialects.postgresql")
    + collect_submodules("psycopg")
    + collect_submodules("alembic")
)
common_datas = [("../migrations", "migrations")]

server_analysis = Analysis(
    ["entries/server_entry.py"],
    pathex=[".."],
    binaries=[],
    datas=common_datas,
    hiddenimports=common_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["PyQt6"],
    noarchive=False,
)
server_pyz = PYZ(server_analysis.pure)
server_exe = EXE(
    server_pyz,
    server_analysis.scripts,
    [],
    exclude_binaries=True,
    name="KrakenServer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

admin_analysis = Analysis(
    ["entries/admin_entry.py"],
    pathex=[".."],
    binaries=[],
    datas=common_datas,
    hiddenimports=common_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["PyQt6"],
    noarchive=False,
)
admin_pyz = PYZ(admin_analysis.pure)
admin_exe = EXE(
    admin_pyz,
    admin_analysis.scripts,
    [],
    exclude_binaries=True,
    name="KrakenAdmin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    server_exe,
    server_analysis.binaries,
    server_analysis.datas,
    admin_exe,
    admin_analysis.binaries,
    admin_analysis.datas,
    strip=False,
    upx=True,
    name="KrakenServer",
)
