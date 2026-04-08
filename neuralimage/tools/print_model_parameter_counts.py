from __future__ import annotations

import argparse
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.NeuralNetwork import get_registered_model_registry  # noqa: E402
from model.NeuralNetwork.registrator import ModelType, create_model  # noqa: E402


@dataclass(frozen=True)
class ModelParameterInfo:
    name: str
    model_type: str
    total_params: int
    trainable_params: int


def _build_model_instance(model_name: str, input_channels: int):
    registry = get_registered_model_registry()
    model_cls = registry[model_name]['model_class']
    signature = inspect.signature(model_cls.__init__)
    kwargs: dict[str, object] = {}

    for channel_kwarg in ('input_channels', 'in_channels', 'in_ch'):
        if channel_kwarg in signature.parameters:
            kwargs[channel_kwarg] = int(input_channels)
            break

    if 'deep_supervision' in signature.parameters:
        kwargs['deep_supervision'] = False

    return create_model(model_name, **kwargs)


def collect_model_parameter_counts(input_channels: int = 1) -> tuple[list[ModelParameterInfo], list[tuple[str, str]]]:
    registry = get_registered_model_registry()
    model_infos: list[ModelParameterInfo] = []
    failures: list[tuple[str, str]] = []

    for model_name, entry in sorted(registry.items(), key=lambda item: (item[1]['model_type'].value, item[0].lower())):
        try:
            model = _build_model_instance(model_name, input_channels)
            total_params = sum(parameter.numel() for parameter in model.parameters())
            trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            model_infos.append(
                ModelParameterInfo(
                    name=model_name,
                    model_type=entry['model_type'].value,
                    total_params=total_params,
                    trainable_params=trainable_params,
                )
            )
        except Exception as exc:  # pragma: no cover - failure details depend on local environment.
            failures.append((model_name, f'{type(exc).__name__}: {exc}'))

    return model_infos, failures


def _format_count(value: int) -> str:
    return f'{int(value):,}'.replace(',', ' ')


def _print_table(model_infos: list[ModelParameterInfo]) -> None:
    if not model_infos:
        print('No registered models found.')
        return

    type_width = max(len('type'), max(len(info.model_type) for info in model_infos))
    name_width = max(len('model'), max(len(info.name) for info in model_infos))
    total_width = max(len('total_params'), max(len(_format_count(info.total_params)) for info in model_infos))
    trainable_width = max(len('trainable_params'), max(len(_format_count(info.trainable_params)) for info in model_infos))

    header = (
        f'{"type":<{type_width}}  '
        f'{"model":<{name_width}}  '
        f'{"total_params":>{total_width}}  '
        f'{"trainable_params":>{trainable_width}}'
    )
    print(header)
    print('-' * len(header))
    for info in model_infos:
        print(
            f'{info.model_type:<{type_width}}  '
            f'{info.name:<{name_width}}  '
            f'{_format_count(info.total_params):>{total_width}}  '
            f'{_format_count(info.trainable_params):>{trainable_width}}'
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Print the parameter counts for all registered neural network models.'
    )
    parser.add_argument(
        '--input-channels',
        type=int,
        default=1,
        help='Number of input channels passed to each model constructor.',
    )
    args = parser.parse_args()

    model_infos, failures = collect_model_parameter_counts(input_channels=args.input_channels)
    _print_table(model_infos)

    total_params = sum(info.total_params for info in model_infos)
    trainable_params = sum(info.trainable_params for info in model_infos)
    print()
    print(f'models_ok={len(model_infos)}')
    print(f'total_params_sum={_format_count(total_params)}')
    print(f'trainable_params_sum={_format_count(trainable_params)}')

    if failures:
        print()
        print('failures:')
        for model_name, error in failures:
            print(f'- {model_name}: {error}')
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
