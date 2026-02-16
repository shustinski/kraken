from copy import deepcopy

from lib.config import Config
from tests.helpers import make_test_dir


def test_config_set_get_data_roundtrip():
    key = 'unit_test_key'
    value = 123
    Config.set_data(key, value)
    assert Config.get_data(key) == value


def test_config_save_and_open_roundtrip(monkeypatch):
    tmp_path = make_test_dir("config_roundtrip")
    monkeypatch.chdir(tmp_path)
    backup = deepcopy(Config._Config__data)
    try:
        Config.set_data('source_path', 'abc')
        Config.save()

        Config.set_data('source_path', 'changed')
        Config.open()

        assert Config.get_data('source_path') == 'abc'
    finally:
        Config._Config__data = backup


def test_config_open_creates_file_if_missing(monkeypatch):
    tmp_path = make_test_dir("config_create")
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / 'data.json').exists()

    Config.open()

    assert (tmp_path / 'data.json').exists()

