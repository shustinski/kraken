# ruff: noqa: F821 - PyInstaller injects its build API and DISTPATH into spec files

import shutil
from pathlib import Path

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

portable_root = Path(DISTPATH) / "KrakenServer"
blob_gateway = Path("../blob_gateway/target/release/kraken-blob-gateway.exe")
if not blob_gateway.is_file():
    raise FileNotFoundError("Build blob_gateway with cargo --release before packaging Kraken Server")
shutil.copy2(blob_gateway, portable_root / "KrakenBlobGateway.exe")
shutil.copytree("scripts", portable_root / "scripts", dirs_exist_ok=True)
shutil.copytree("config", portable_root / "config", dirs_exist_ok=True)
