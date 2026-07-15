from __future__ import annotations

from typing import Any


MUON_UNAVAILABLE_MESSAGE = 'Muon оптимизатор недоступен, выберите другой'


def resolve_muon_optimizer_class() -> Any | None:
    try:
        from torch import optim
    except Exception:
        return None

    return getattr(optim, 'Muon', None)


def is_muon_optimizer_available() -> bool:
    return resolve_muon_optimizer_class() is not None
