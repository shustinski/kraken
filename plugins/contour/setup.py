"""Native extension build for Contour's pinned third-party backends."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "contour._native.multi_separator",
            ["third_party/multi_separator/src/multi_separator.cxx"],
            include_dirs=["third_party/multi_separator/include"],
            cxx_std=17,
            define_macros=[("VERSION_INFO", '"0.0.1-437c651"')],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
