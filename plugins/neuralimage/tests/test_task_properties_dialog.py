import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication

from neuralimage.application.dto import MainWindowState, SettingsState
from neuralimage.view.task_properties_dialog import TaskPropertiesDialog


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_task_properties_dialog_renders_and_emits_restore(qapp):
    dialog = TaskPropertiesDialog(
        task_id=7,
        status='paused',
        paused=True,
        main_window_state=MainWindowState(
            work_mode='train_only',
            source_folder='source',
            result_folder='result',
            label_folder='labels',
            sample_folder='samples',
            model_path='model.pth',
            epochs=12,
        ),
        settings_state=SettingsState(
            step=88,
            sample_cut_mode='disk',
            train_patch_size=(128, 256),
            recognition_patch_size=(64, 96),
            sync_patch_sizes=False,
            show_batch_preview=False,
        ),
    )

    assert dialog.tree.topLevelItemCount() == 3
    assert dialog.tree.topLevelItem(0).childCount() >= 3

    restored: list[tuple[MainWindowState, SettingsState]] = []
    dialog.restore_requested.connect(lambda main_state, settings_state: restored.append((main_state, settings_state)))

    dialog.restore_button.click()

    assert len(restored) == 1
    restored_main, restored_settings = restored[0]
    assert restored_main.epochs == 12
    assert restored_settings.step == 88
    assert restored_settings.sample_cut_mode == 'disk'
