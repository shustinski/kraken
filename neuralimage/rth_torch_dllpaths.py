import os
import sys
from pathlib import Path

_DLL_DIR_HANDLES = []

def _add_dir(p: Path) -> None:
    if not p or not p.is_dir():
        return
    try:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(p)))
    except Exception:
        pass

if sys.platform.startswith("win"):
    exe_dir = Path(sys.executable).resolve().parent

    # Typical onedir layout: <dist>/NeuralImage/torch/lib
    _add_dir(exe_dir / "torch" / "lib")

    # Your custom candidates: <dist>/NeuralImage/_internal and _internal/torch/lib
    _add_dir(exe_dir / "_internal")
    _add_dir(exe_dir / "_internal" / "torch" / "lib")

    # onefile extraction dir
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        mp = Path(meipass)
        _add_dir(mp)
        _add_dir(mp / "torch" / "lib")
        _add_dir(mp / "_internal")
        _add_dir(mp / "_internal" / "torch" / "lib")

    # PATH fallback (for transitive deps)
    extra = [
        exe_dir / "torch" / "lib",
        exe_dir / "_internal",
        exe_dir / "_internal" / "torch" / "lib",
    ]
    if meipass:
        mp = Path(meipass)
        extra += [
            mp,
            mp / "torch" / "lib",
            mp / "_internal",
            mp / "_internal" / "torch" / "lib",
        ]

    extra_str = [str(p) for p in extra if p.is_dir()]
    if extra_str:
        os.environ["PATH"] = ";".join(extra_str) + (";" + os.environ["PATH"] if os.environ.get("PATH") else "")