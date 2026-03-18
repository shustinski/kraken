import csv

import model.NeuralNetwork.model_train_and_recognition as target
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
    batch_png = root / 'training_metrics_train_loss_epoch_0002.png'
    batch_csv = root / 'training_metrics_train_loss_epoch_0002.csv'

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
