"""model_train_and_recognition: stop."""
from __future__ import annotations


class StopMixin:
    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._pause_event.set()
