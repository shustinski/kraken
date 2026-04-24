import importlib
import sys
import types

import pytest

pytest.importorskip('PyQt6')


def _import_main_presenter_with_stubs():
    nn_stub = types.ModuleType('model.NeuralNetwork')
    nn_stub.get_registered_models = lambda: {}
    nn_stub.get_registered_model_names_by_type = lambda: {}
    handler_stub = types.ModuleType('model.general_neural_handler')
    handler_stub.GeneralNeuralHandler = object
    images_stub = types.ModuleType('lib.images')
    images_stub.SampleWorker = object

    sys.modules['model.NeuralNetwork'] = nn_stub
    sys.modules['model.general_neural_handler'] = handler_stub
    sys.modules['lib.images'] = images_stub
    sys.modules.pop('presenter.main_presenter', None)
    return importlib.import_module('presenter.main_presenter')


def test_format_auto_answer_button_text_includes_countdown():
    module = _import_main_presenter_with_stubs()

    assert module._format_auto_answer_button_text('Нет', 15) == 'Нет (15)'
    assert module._format_auto_answer_button_text('Да', 1) == 'Да (1)'
    assert module._format_auto_answer_button_text('Нет', 0) == 'Нет'
