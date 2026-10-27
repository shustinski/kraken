# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


analysis = Analysis(
    ["entries/desktop_entry.py"],
    pathex=[".."],
    binaries=[],
    datas=(collect_data_files("kraken_core") + collect_data_files("kraken_hub") + collect_data_files("kraken_manager")
           + collect_data_files("contour", includes=["resources/ui_texts_*.json"])),
    hiddenimports=(
        collect_submodules("kraken_core")
        + collect_submodules("kraken_hub")
        + collect_submodules("kraken_manager")
        + collect_submodules("kraken_agent")
        + ["keyring.backends.Windows", "contour.resources"]
    ),
    hookspath=[],
    runtime_hooks=[],
    # Plugin computations run in their own installed applications/Agent jobs.
    # Contour's reachable CIF codec is required by the native stitch worker;
    # exclude its UI/entry points rather than removing that shared reader.
    # Installed training packages must not pull multi-gigabyte CUDA runtimes
    # into the Desktop simply because they share the build environment.
    excludes=["fastapi", "uvicorn", "alembic", "sqlalchemy", "psycopg",
              "torch", "torchvision", "timm", "tensorflow", "jax", "jaxlib",
              "cv2", "matplotlib", "pandas", "sklearn",
              "neuralimage", "karakal", "krona", "csliser", "kategb",
              "contour.widget", "contour.application", "contour.view", "contour.presentation"],
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
