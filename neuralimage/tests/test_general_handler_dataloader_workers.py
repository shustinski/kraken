from types import SimpleNamespace

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
