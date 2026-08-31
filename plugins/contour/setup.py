"""Native extension build for Contour's pinned third-party backends."""

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent
LIBOPENCIF_DIR = ROOT / "third_party" / "libopencif"

ext_modules = [
    Pybind11Extension(
        "contour._native.multi_separator",
        ["third_party/multi_separator/src/multi_separator.cxx"],
        include_dirs=["third_party/multi_separator/include"],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"0.0.1-437c651"')],
    ),
]

if (LIBOPENCIF_DIR / "libopencif.cc").exists():
    ext_modules.append(
        Pybind11Extension(
            "contour._native.cif_loader",
            [
                "third_party/libopencif/cif_loader.cxx",
                "third_party/libopencif/libopencif.cc",
            ],
            include_dirs=["third_party/libopencif"],
            cxx_std=14,
        )
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
