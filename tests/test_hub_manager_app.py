from __future__ import annotations
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, get_ident
from time import monotonic, sleep
from types import SimpleNamespace
import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QWidget,
)
from kraken_core.frame_matrix.qt import FrameRect
from kraken_hub import manager_app, windows_credentials
from kraken_hub.composition import EmbeddedProjectService
from kraken_hub.manager_app import (
    DesktopController,
    _configure_administration_page,
    _count_regular_files,
    _development_session,
    _event_belongs_to_layer,
    _event_matches_node,
    _history_entries,
    _login,
    _my_work_panel,
    _performer_panel,
    _statistics_panel,
)
from kraken_hub.statistics_widgets import MetricChartWidget
from kraken_manager.domain.identity import Permission
from kraken_manager.domain.project import LayerType, RepresentationKind
from kraken_manager.infrastructure.reports import ActivityRecord
from kraken_manager.presentation.qt import (
    FrameSelection,
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
    ProjectListItem,
    ProjectManagerShell,
)
from kraken_manager.presentation.qt.widgets import ClickableLabel

from test_hub_manager_app_parts.review_return_preview import (
    test_review_return_preview_is_resizable,
    test_review_return_without_changed_files_refreshes_current_matrix,
)
from test_hub_manager_app_parts.running_agent_job import (
    test_non_running_agent_jobs_do_not_block_close,
    test_running_agent_job_still_blocks_close,
)
from test_hub_manager_app_parts.matrix_context_menu import (
    test_matrix_context_menu_exports_and_opens_vector_properties,
    test_matrix_context_menu_hides_properties_for_multiple_frames,
    test_matrix_context_menu_preserves_review_action,
)
from test_hub_manager_app_parts.event import (
    qapp,
    _event,
)
from test_hub_manager_app_parts.property_helpers_filter import test_property_helpers_filter_history_and_deduplicate_files
from test_hub_manager_app_parts.performer_table_supports import (
    test_performer_dialog_uses_color_picker,
    test_performer_table_supports_color_editing_and_context_actions,
)
from test_hub_manager_app_parts.statistics_panel_has import test_statistics_panel_has_human_readable_project_filtered_charts
from test_hub_manager_app_parts.delete_project_confirmation import test_delete_project_confirmation_calls_cache_only_service
from test_hub_manager_app_parts.controller_builds_layer import test_controller_builds_layer_and_representation_properties
from test_hub_manager_app_parts.frame_card_hides import test_frame_card_hides_mutating_actions_without_permissions
from test_hub_manager_app_parts.my_work_exposes import test_my_work_exposes_review_and_agent_recovery_actions
from test_hub_manager_app_parts.controller_builds_derived import test_controller_builds_derived_job_karakal_and_virtual_properties
from test_hub_manager_app_parts.image_representation_source import (
    test_login_creates_first_account_in_dialog,
    test_development_session_creates_and_reuses_vscode_account,
    test_login_autofills_only_after_windows_verification,
    test_login_does_not_autofill_when_windows_verification_is_cancelled,
    test_image_representation_source_picker_fills_selected_folder,
)
from test_hub_manager_app_parts.representation_files_are import test_representation_files_are_saved_without_blocking_qt
from test_hub_manager_app_parts.missing_agent_inputs import test_missing_agent_inputs_are_prepared_in_background_and_retried
from test_hub_manager_app_parts.remote_project_can import (
    test_remote_project_can_create_logical_layer,
    test_external_cif_import_uses_source_from_clicked_pipeline_lane,
)
from test_hub_manager_app_parts.created_layer_selected import test_created_layer_is_selected_and_its_pipeline_is_loaded
from test_hub_manager_app_parts.contour_vectorize_receives import (
    test_contour_vectorize_receives_staged_base_layer_path,
    test_local_administration_page_is_wired_for_desktop_session,
    test_layer_manager_reports_history_failure_without_opening_dialog,
)

pytest.importorskip("PyQt6")

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    test_review_return_preview_is_resizable,
    test_review_return_without_changed_files_refreshes_current_matrix,
    test_non_running_agent_jobs_do_not_block_close,
    test_running_agent_job_still_blocks_close,
    test_matrix_context_menu_exports_and_opens_vector_properties,
    test_matrix_context_menu_hides_properties_for_multiple_frames,
    test_matrix_context_menu_preserves_review_action,
    qapp,
    _event,
    test_property_helpers_filter_history_and_deduplicate_files,
    test_performer_dialog_uses_color_picker,
    test_performer_table_supports_color_editing_and_context_actions,
    test_statistics_panel_has_human_readable_project_filtered_charts,
    test_delete_project_confirmation_calls_cache_only_service,
    test_controller_builds_layer_and_representation_properties,
    test_frame_card_hides_mutating_actions_without_permissions,
    test_my_work_exposes_review_and_agent_recovery_actions,
    test_controller_builds_derived_job_karakal_and_virtual_properties,
    test_login_creates_first_account_in_dialog,
    test_development_session_creates_and_reuses_vscode_account,
    test_login_autofills_only_after_windows_verification,
    test_login_does_not_autofill_when_windows_verification_is_cancelled,
    test_image_representation_source_picker_fills_selected_folder,
    test_representation_files_are_saved_without_blocking_qt,
    test_missing_agent_inputs_are_prepared_in_background_and_retried,
    test_remote_project_can_create_logical_layer,
    test_external_cif_import_uses_source_from_clicked_pipeline_lane,
    test_created_layer_is_selected_and_its_pipeline_is_loaded,
    test_contour_vectorize_receives_staged_base_layer_path,
    test_local_administration_page_is_wired_for_desktop_session,
    test_layer_manager_reports_history_failure_without_opening_dialog,
)
