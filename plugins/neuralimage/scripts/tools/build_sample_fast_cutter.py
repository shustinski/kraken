from pathlib import Path
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    pyx_path = project_root / 'lib' / 'sample_fast_cutter_pyx.pyx'

    if not pyx_path.exists():
        print(f'File not found: {pyx_path}')
        return 1

    try:
        import numpy as np
        from setuptools import Extension, setup
        from Cython.Build import cythonize
    except Exception as exc:
        print('Missing build dependencies. Install them first:')
        print('  pip install cython setuptools wheel numpy')
        print(f'Details: {exc}')
        return 1

    ext_modules = [
        Extension(
            name='lib.sample_fast_cutter_pyx',
            sources=[str(pyx_path)],
            include_dirs=[np.get_include()],
        )
    ]

    setup(
        name='sample-fast-cutter-pyx',
        script_name='build_sample_fast_cutter.py',
        script_args=['build_ext', '--inplace'],
        ext_modules=cythonize(ext_modules, language_level='3'),
    )
    print('Build finished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
