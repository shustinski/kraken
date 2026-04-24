from __future__ import annotations

import torch


def collect_torch_runtime_info() -> dict[str, object]:
    return {
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_arch_list': torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }


def main() -> None:
    info = collect_torch_runtime_info()
    print(info['torch_version'])
    print(info['cuda_version'])
    print(info['cuda_available'])
    print(info['cuda_arch_list'])


if __name__ == '__main__':
    main()
