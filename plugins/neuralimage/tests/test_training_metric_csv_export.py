import csv
import types

import neuralimage.model.NeuralNetwork.model_train_and_recognition as target
from tests.helpers import make_test_dir


def test_save_metric_charts_exports_png_and_csv():
    root = make_test_dir('training_metric_csv_export')
    trainer = target.TrainerProcess.__new__(target.TrainerProcess)
    trainer._save_path = root / 'model.pth'
    trainer._train_epoch_history = [(1.0, 0.8), (2.0, 0.4)]
    trainer._val_epoch_history = [(1.0, 0.9), (2.0, 0.5)]
    trainer._val_iou_history = [(1.0, 0.55), (2.0, 0.72)]
    trainer._val_dice_history = [(1.0, 0.63), (2.0, 0.81)]
    trainer._batch_points_by_epoch = {2: [(1.0, 0.7), (2.0, 0.45)]}

    trainer._save_metric_charts()

    loss_png = root / 'training_metrics_loss_by_epoch.png'
    loss_csv = root / 'training_metrics_loss_by_epoch.csv'
    quality_png = root / 'training_metrics_validation_quality.png'
    quality_csv = root / 'training_metrics_validation_quality.csv'
    batch_png = root / 'training_metrics_train_loss_by_batch.png'
    batch_csv = root / 'training_metrics_train_loss_by_batch.csv'

    assert loss_png.exists() is True
    assert loss_csv.exists() is True
    assert quality_png.exists() is True
    assert quality_csv.exists() is True
    assert batch_png.exists() is True
    assert batch_csv.exists() is True

    with loss_csv.open('r', encoding='utf-8', newline='') as file:
        rows = list(csv.reader(file))
    assert rows[0] == ['Epoch', 'Train Loss', 'Val Loss']
    assert rows[1] == ['1', '0.8', '0.9']
    assert rows[2] == ['2', '0.4', '0.5']


def test_run_single_training_epoch_saves_metric_charts_each_epoch():
    trainer = target.TrainerProcess.__new__(target.TrainerProcess)
    trainer._epochs = 3
    trainer._bus = types.SimpleNamespace(put=lambda *_args, **_kwargs: None)

    calls = {'save_metric_charts': 0}
    trainer._publish_epoch_start = lambda *args, **kwargs: None
    trainer._run_train_epoch = lambda *args, **kwargs: types.SimpleNamespace()
    trainer._reduce_epoch_train_stats = lambda *args, **kwargs: (1.0, 1)
    trainer._publish_epoch_train_metrics = lambda *args, **kwargs: None
    trainer._handle_validation = lambda *args, **kwargs: {'loss': 0.5}
    trainer._step_epoch_scheduler = lambda *args, **kwargs: None
    trainer._save_epoch_artifacts = lambda *args, **kwargs: None
    trainer._save_metric_charts = lambda: calls.__setitem__('save_metric_charts', calls['save_metric_charts'] + 1)
    trainer._publish_epoch_load_breakdown = lambda *args, **kwargs: None
    trainer._format_elapsed_duration = lambda *_args, **_kwargs: '00:00:01'

    runtime_state = types.SimpleNamespace(
        run_context=types.SimpleNamespace(train_size=4),
        active_profiler=None,
        early_stopping_state=types.SimpleNamespace(),
        early_stopping_config=types.SimpleNamespace(),
        is_main_process=True,
    )

    trainer._run_single_training_epoch(
        epoch=0,
        device=types.SimpleNamespace(type='cpu'),
        distributed=False,
        runtime_state=runtime_state,
    )

    assert calls['save_metric_charts'] == 1
