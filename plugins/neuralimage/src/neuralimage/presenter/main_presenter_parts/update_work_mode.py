"""main_presenter: update work mode."""
from __future__ import annotations
import neuralimage.presenter.state_sync as state_sync


class UpdateWorkModeMixin:
    def _update_work_mode(self) -> str:
        return state_sync.update_work_mode(self)
