# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

_HERE = Path(SPECPATH).resolve()
# The version is built in code from the release, the algorithm number and the beta
# counter (karakal/version.py); KARAKAL_VERSION can still override it.
sys.path.insert(0, str(_HERE / "src"))
from karakal.version import APP_VERSION as _default_version  # noqa: E402

_profile = os.environ.get("KARAKAL_BUILD_PROFILE", "dev").strip().lower()
_tester = _profile == "tester"
_version = os.environ.get("KARAKAL_VERSION", _default_version).strip() or _default_version
_profile_file = Path(os.environ.get("TEMP", ".")) / "build_profile.txt"
_datas = [
    ('src\\karakal\\resources\\icons\\karakal_light.ico', 'karakal/resources/icons'),
    ('src\\karakal\\resources\\icons\\karakal_light.png', 'karakal/resources/icons'),
    ('src\\karakal\\resources\\icons\\karakal.ico', 'karakal/resources/icons'),
    ('src\\karakal\\resources\\icons\\karakal.png', 'karakal/resources/icons'),
    ('resources\\update_client.json', 'resources'),
    ('CHANGELOG.md', 'resources'),
]
# Every build carries its profile and version: the window title, "What's new" and the
# updater read the version from here.
_profile_file.write_text(f"{'tester' if _tester else 'dev'}\n{_version}\n", encoding="utf-8")
_datas.append((str(_profile_file), "resources"))
_exe_name = f"karakal-{_version}" if _tester else "karakal"

a = Analysis(
    ['src\\karakal\\__main__.py'],
    pathex=['src'],
    binaries=[],
    datas=_datas,
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
    name=_exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='src\\karakal\\resources\\icons\\karakal_light.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=_exe_name,
)
