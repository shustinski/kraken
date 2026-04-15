import numpy as np

from application.dto import MainWindowState, SettingsState
from webui.services.training_session import TrainingSessionService


def test_training_session_snapshot_exposes_queue_progress_and_preview():
    service = TrainingSessionService(presenter=object())
    task = service._processing_session.enqueue_task(
        MainWindowState(work_mode='train_only'),
        SettingsState(),
    )

    service._on_metrics({'type': 'train_epoch_progress', 'current': 2, 'total': 5})
    service._on_metrics({'type': 'train_batch_progress', 'current': 8, 'total': 20})
    service._on_metrics({'type': 'recognition_progress', 'current': 3, 'total': 10})
    service._on_metrics({'type': 'system_memory', 'ram_mb': 2048, 'vram_allocated_mb': 512, 'vram_reserved_mb': 768})
    service._on_metrics({
        'type': 'train_perf',
        'epoch': 2,
        'batch_index': 8,
        'data_wait_ms': 1.0,
        'augmentation_ms': 2.0,
        'forward_ms': 3.0,
        'backward_ms': 4.0,
        'optimizer_ms': 5.0,
        'total_ms': 20.0,
    })
    service._on_metrics({
        'type': 'val_epoch',
        'epoch': 2,
        'loss': 0.2,
        'iou': 0.8,
        'dice': 0.85,
        'f1': 0.83,
    })
    service._on_metrics({
        'type': 'train_batch_preview',
        'sample_name': 'frame_a',
        'image': np.zeros((8, 8), dtype=np.uint8),
        'label': np.full((8, 8), 255, dtype=np.uint8),
        'outputs': np.full((8, 8), 127, dtype=np.uint8),
    })

    snapshot = service.snapshot()

    assert snapshot['queue'] == [{'task_id': task.task_id, 'work_mode': 'train_only', 'status': 'queued'}]
    assert snapshot['metrics']['progress']['epoch']['text'] == '40% (2/5)'
    assert snapshot['metrics']['progress']['batch']['text'] == '40% (8/20)'
    assert snapshot['metrics']['progress']['recognition']['text'] == '30% (3/10)'
    assert snapshot['metrics']['system_memory']['ram_mb'] == 2048.0
    assert snapshot['metrics']['validation_quality']['iou'] == 0.8
    assert snapshot['metrics']['train_speed_batches_per_sec'] == 50.0
    assert snapshot['metrics']['preview']['mode'] == 'train'
    assert snapshot['metrics']['preview']['sample_name'] == 'frame_a'
    assert snapshot['metrics']['preview']['image_url'].startswith('data:image/png;base64,')
    assert snapshot['metrics']['preview']['label_url'].startswith('data:image/png;base64,')
    assert snapshot['metrics']['preview']['output_url'].startswith('data:image/png;base64,')


def test_training_session_queue_actions_change_snapshot_state():
    service = TrainingSessionService(presenter=object())
    task = service._processing_session.enqueue_task(
        MainWindowState(work_mode='recognition_only'),
        SettingsState(),
    )

    ok, error = service.toggle_pause_task(task.task_id)
    assert ok is True
    assert error is None
    assert service.snapshot()['queue'][0]['status'] == 'paused'

    ok, error = service.remove_task(task.task_id)
    assert ok is True
    assert error is None
    assert service.snapshot()['queue'] == []
