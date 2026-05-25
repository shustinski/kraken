from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


def _collect_src_python_modules(src_root: Path, package: str) -> list[str]:
    """Discover module names under ``src_root/package`` without importing them.

    ``collect_submodules`` only sees importable packages; if optional deps are
    missing at build time, nested modules (e.g. ``contour.graphics.*``) can be
    omitted even though the app imports them at runtime.
    """
    package_dir = src_root / package
    if not package_dir.is_dir():
        return []
    modules: list[str] = []
    for py_file in sorted(package_dir.rglob("*.py")):
        rel = py_file.relative_to(src_root)
        if rel.name == "__init__.py":
            continue
        modules.append(".".join(rel.with_suffix("").parts))
    return modules


PROJECT_ROOT = Path(SPECPATH).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
APP_NAME = "Contour"
APP_ICON = PROJECT_ROOT / "resources" / "icons" / "contour.ico"

datas = collect_data_files(
    "contour",
    includes=[
        "pipeline_example.json",
        "resources/ui_texts_*.json",
        "gamification/assets/gamification.qrc",
        "gamification/assets/pets/*/*/*.png",
    ],
)
datas += collect_data_files(
    "kraken_core",
    includes=[
        "resources/styles/*.qss",
        "resources/styles/icons/*",
    ],
)
datas += [
    (str(PROJECT_ROOT / "resources" / "icons" / "contour.ico"), "plugins/contour/resources/icons"),
    (str(PROJECT_ROOT / "resources" / "icons" / "contour.png"), "plugins/contour/resources/icons"),
]
hiddenimports = list(
    dict.fromkeys(
        collect_submodules("contour")
        + _collect_src_python_modules(SRC_ROOT, "contour")
    )
)


a = Analysis(
    [str(PROJECT_ROOT / "__main__.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
