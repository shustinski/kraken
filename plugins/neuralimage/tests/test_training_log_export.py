import neuralimage.model.NeuralNetwork.model_train_and_recognition as target
from tests.helpers import make_test_dir


def test_recording_queue_collects_logging_messages():
    forwarded: list[object] = []

    class _Queue:
        def put(self, item):
            forwarded.append(item)

    log_lines: list[str] = []
    queue = target._RecordingQueue(_Queue(), log_lines)
    queue.put(['logging', 'first line'])
    queue.put(['metrics', {'type': 'train_epoch'}])
    queue.put(['logging', 'second line'])

    assert len(forwarded) == 3
    assert len(log_lines) == 2
    assert log_lines[0].endswith('first line')
    assert log_lines[1].endswith('second line')


def test_save_training_log_writes_txt_file():
    root = make_test_dir('training_log_export')
    trainer = target.TrainerProcess.__new__(target.TrainerProcess)
    trainer._save_path = root / 'model.pth'
    trainer._training_log_lines = [
        '[2026-03-18 10:00:00] Start training',
        '[2026-03-18 10:00:05] Epoch [1/3] completed in 00:00:05.',
    ]

    log_path = trainer._save_training_log()

    assert log_path == root / 'training_log.txt'
    assert log_path.exists() is True
    log_text = log_path.read_text(encoding='utf-8')
    assert 'Start training' in log_text
    assert 'Epoch [1/3] completed in 00:00:05.' in log_text


def test_format_elapsed_duration_formats_hms():
    assert target.TrainerProcess._format_elapsed_duration(5.0) == '00:00:05'
    assert target.TrainerProcess._format_elapsed_duration(3661.0) == '01:01:01'
