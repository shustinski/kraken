from types import SimpleNamespace

from lib.data_interfaces import SampleCutMode
from model.general_neural_handler import GeneralNeuralHandler


def _build_handler(dataloader_num_workers: int = -1, *, batch_size: int = 8, cut_mode: str = 'disk') -> GeneralNeuralHandler:
    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler.tranining_parameters = SimpleNamespace(
        dataloader_num_workers=dataloader_num_workers,
        batch_size=batch_size,
        cut_mode=cut_mode,
    )
    return handler


def test_resolve_dataloader_workers_uses_explicit_override(monkeypatch):
    monkeypatch.setattr('model.general_neural_handler._is_debugger_attached', lambda: False)

    handler = _build_handler(dataloader_num_workers=4)

    assert handler._resolve_dataloader_workers() == 4


def test_resolve_dataloader_workers_keeps_auto_mode_when_override_is_negative(monkeypatch):
    monkeypatch.setattr('model.general_neural_handler._is_debugger_attached', lambda: False)
    monkeypatch.setattr('model.general_neural_handler.os.cpu_count', lambda: 12)

    handler = _build_handler(dataloader_num_workers=-1, batch_size=8, cut_mode='disk')

    assert handler._resolve_dataloader_workers() == 8


def test_resolve_dataloader_workers_forces_zero_under_debugger(monkeypatch):
    monkeypatch.setattr('model.general_neural_handler._is_debugger_attached', lambda: True)

    handler = _build_handler(dataloader_num_workers=6)

    assert handler._resolve_dataloader_workers() == 0


def test_create_dataloader_disables_persistent_workers_for_epoch_mutating_datasets(monkeypatch):
    class _DatasetWithEpochState:
        def set_epoch(self):
            return

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return index

    captured_kwargs: list[dict[str, object]] = []

    def _fake_dataloader(dataset, **kwargs):
        captured_kwargs.append(dict(kwargs))
        return SimpleNamespace(dataset=dataset, **kwargs)

    handler = GeneralNeuralHandler.__new__(GeneralNeuralHandler)
    handler._need_stop = False
    handler._hard_mining_active = False
    handler.message_bus = SimpleNamespace(publish=lambda *_args, **_kwargs: None)
    handler.tranining_parameters = SimpleNamespace(
        shuffle=False,
        cut_mode=SampleCutMode.online,
        batch_size=8,
        hard_mining=SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(handler, '_resolve_dataloader_workers', lambda: 2)
    monkeypatch.setattr('model.general_neural_handler.DataLoader', _fake_dataloader)
    monkeypatch.setattr('model.general_neural_handler.torch.cuda.is_available', lambda: False)

    handler._create_dataloader(_DatasetWithEpochState(), _DatasetWithEpochState())

    assert len(captured_kwargs) == 2
    assert captured_kwargs[0]['persistent_workers'] is False
    assert captured_kwargs[1]['persistent_workers'] is False
