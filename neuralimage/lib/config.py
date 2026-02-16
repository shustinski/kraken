import json
from copy import deepcopy
from pathlib import Path


class Config:
    _DEFAULT_DATA = {
        'work_mode': 'cif',
        'source_path': '',
        'result_path': '',
        'use_ready_model': False,
        'cr_samples_path': '',
        'cif_sample_jpg_path': '',
        'cif_sample_cif_path': '',
        'delete_after_training': False,
        'rotation_90': False,
        'rotation_180': False,
        'training_on_multiple_gpus': True,
        'prediction_on_multiple_gpus': 'parallel',
        'shift': 10,
        'model_type': 'small',
        'epochs': 30,
        'sample_preparation_type': 'cut',
        'sample_height': 256,
        'sample_wight': 256,
    }

    __data = deepcopy(_DEFAULT_DATA)
    __config_file = Path('data.json')

    @staticmethod
    def set_data(name: str, value):
        Config.__data[name] = value

    @staticmethod
    def get_data(name: str, default=None):
        return Config.__data.get(name, default)

    @staticmethod
    def save(path: str | Path | None = None):
        config_path = Path(path) if path is not None else Config.__config_file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open('w', encoding='utf-8') as file:
            json.dump(Config.__data, file, indent=4, ensure_ascii=False)

    @staticmethod
    def open(path: str | Path | None = None):
        config_path = Path(path) if path is not None else Config.__config_file
        if not config_path.exists():
            Config.save(config_path)
            return

        with config_path.open('r', encoding='utf-8') as file:
            data = json.load(file)

        merged = deepcopy(Config._DEFAULT_DATA)
        if isinstance(data, dict):
            merged.update(data)
        Config.__data = merged

    @staticmethod
    def reset_to_defaults():
        Config.__data = deepcopy(Config._DEFAULT_DATA)