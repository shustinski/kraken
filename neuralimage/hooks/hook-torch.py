# Lean local override for PyInstaller torch hook.
# Purpose: avoid collect_submodules("torch"), which is extremely slow/heavy.

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False

datas = collect_data_files(
    "torch",
    excludes=[
        "**/*.h",
        "**/*.hpp",
        "**/*.cuh",
        "**/*.lib",
        "**/*.cpp",
        "**/*.pyi",
        "**/*.cmake",
    ],
)

binaries = collect_dynamic_libs("torch")

# Explicit, minimal set used by this app (training/inference/DDP).
hiddenimports = [
    "torch",
    "torch.nn",
    "torch.optim",
    "torch.cuda",
    "torch.distributed",
    "torch.multiprocessing",
    "torch.utils.data",
    "torch.utils.data.distributed",
    "torch.backends",
    "torch.backends.cudnn",
    "torch.amp",
]
