from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


MODEL_HUB_IDS = {
    'swin_base_patch4_window7_224': 'timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k',
    'swin_large_patch4_window7_224': 'timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k',
}


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / '_internal' / 'models' / 'timm'


def download_model(model_name: str, output_root: Path) -> Path:
    hub_id = MODEL_HUB_IDS.get(model_name)
    if hub_id is None:
        raise ValueError(f'Unsupported model name: {model_name!r}')

    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / 'model.safetensors'

    downloaded_path = Path(
        hf_hub_download(
            repo_id=hub_id,
            filename='model.safetensors',
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
    )
    if downloaded_path.resolve() != target_path.resolve():
        shutil.copy2(downloaded_path, target_path)
    cache_dir = output_dir / '.cache'
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    return target_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Download offline timm Swin weights into the internal app folder.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=_default_output_dir(),
        help='Target root directory for offline timm weights.',
    )
    parser.add_argument(
        'models',
        nargs='*',
        default=sorted(MODEL_HUB_IDS),
        help='Model ids to download.',
    )
    args = parser.parse_args()

    for model_name in args.models:
        target_path = download_model(str(model_name), args.output_dir)
        print(f'{model_name}: {target_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
