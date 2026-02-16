from presenter.workflow_mapper import resolve_work_mode
from presenter.workflow_mapper import build_workflow_parameters
from view.window_dataclasses import MainWindowState, SettingsState
from tests.helpers import make_test_dir


def test_resolve_work_mode_known_value():
    mode = resolve_work_mode('train_only')
    assert mode is not None
    assert mode.value == 'train_only'


def test_resolve_work_mode_unknown_value():
    assert resolve_work_mode('unknown') is None


def test_build_workflow_parameters_falls_back_to_adam_for_unknown_optimizer():
    source = make_test_dir("workflow_source")
    result = make_test_dir("workflow_result")
    sample = make_test_dir("workflow_sample")
    label = make_test_dir("workflow_label")

    main = MainWindowState(
        work_mode='train_only',
        source_folder=str(source),
        result_folder=str(result),
        sample_folder=str(sample),
        label_folder=str(label),
        epochs=1,
    )
    settings = SettingsState(
        optimizer_name='invalid_optimizer_name',
        warmup_enabled=True,
        warmup_epochs=4,
        warmup_start_factor=0.2,
        early_stopping_enabled=True,
        early_stopping_patience=7,
        early_stopping_min_delta=0.005,
        early_stopping_restore_best_weights=False,
    )

    _, training, _ = build_workflow_parameters(main, settings)

    assert training.optimizer.name.value == 'adam'
    assert training.warmup.enabled is True
    assert training.warmup.epochs == 4
    assert training.warmup.start_factor == 0.2
    assert training.early_stopping.enabled is True
    assert training.early_stopping.patience == 7
    assert training.early_stopping.min_delta == 0.005
    assert training.early_stopping.restore_best_weights is False
