"""main_window: construction 2."""
from __future__ import annotations
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QDockWidget,
    QScrollArea,
    QWidget,
)
from neuralimage.lib.runtime_paths import resolve_internal_path
from neuralimage.lib.ui_texts import get_ui_language
from neuralimage.lib.version import get_app_title
from neuralimage.view.tic_tac_toe_dialog import TicTacToeDialog

from neuralimage.view.main_window_parts.responsive_preview_label import _load_persisted_ui_mode


class Construction2Mixin:
    def __init__(self, side_panel: QWidget | None = None):
        super().__init__()
        self._window_layout_restore_pending = True
        self._window_layout_save_timer = QTimer(self)
        self._window_layout_save_timer.setSingleShot(True)
        self._window_layout_save_timer.setInterval(300)
        self._window_layout_save_timer.timeout.connect(self._save_window_layout)
        self.setWindowTitle(get_app_title())
        self.setWindowIcon(QIcon(str(resolve_internal_path('icon.png'))))
        self.setGeometry(200, 200, 1200, 740)
        

        self.settings_dock = side_panel if isinstance(side_panel, QDockWidget) else None
        self._close_allowed = False
        self.log_scroll: QScrollArea | None = None

        self._batch_points_by_epoch: dict[int, list[tuple[float, float]]] = {}
        self._tic_tac_toe_dialog: TicTacToeDialog | None = None
        self._ram_mb: float | None = None
        self._vram_alloc_mb: float | None = None
        self._vram_reserved_mb: float | None = None
        self._vram_used_mb: float | None = None
        self._vram_total_mb: float | None = None
        self._train_speed_batches_per_sec: float | None = None
        self._recognition_speed_images_per_sec: float | None = None
        self._sample_count_value = 0
        self._sample_count_pending = False
        self._current_preview_sample_name = ""
        self._current_preview_mode = "train"
        self._last_status_message = ""
        self._recognition_started_at: float | None = None
        self._recognition_last_current = 0
        self._recognition_last_total = 0
        self._last_epoch_progress: tuple[int, int] = (0, 0)
        self._last_batch_progress: tuple[int, int] = (0, 0)
        self._last_recognition_progress: tuple[int, int] = (0, 0)
        self._active_progress_mode = "idle"
        self._last_validation_metrics: tuple[float, float, float] | None = None
        self._last_performance_metrics: dict[str, float] | None = None
        self._ui_language = get_ui_language()
        self._theme = "dark"
        self._file_menu = None
        self._settings_menu = None
        self._view_menu = None
        self._plugins_menu = None
        self._info_menu = None
        self._language_menu = None
        self._theme_menu = None
        self._open_config_action = None
        self._settings_sample_action = None
        self._settings_train_action = None
        self._settings_pred_action = None
        self._metrics_toggle_action = None
        self._log_toggle_action = None
        self._settings_toggle_action = None
        self._help_action = None
        self._changelog_action = None
        self._check_updates_action = None
        self._update_menu = None
        self._update_channel_menu = None
        self._update_channel_action_group = None
        self._update_channel_actions: dict[str, QAction] = {}
        self._available_update_channels: list[str] = []
        self._selected_update_channel = 'stable'
        self._open_validation_gradient_action = None
        self._ui_mode_menu = None
        self._ui_mode_simple_action = None
        self._ui_mode_advanced_action = None
        self._central_scroll: QScrollArea | None = None
        self._central_content: QWidget | None = None
        self._ui_mode = _load_persisted_ui_mode()
        self._current_work_mode = ''
        self._selected_simple_workflow: str | None = None
        self._queue_items: list[dict] = []

        self._setup_ui()
