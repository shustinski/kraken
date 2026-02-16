from conftest import safe_import_or_skip

safe_import_or_skip('torch')
safe_import_or_skip('torchvision')
safe_import_or_skip('PIL')

from model.NeuralNetwork.dataset import summarise_list, index_in_list


def test_summarise_list_accumulates():
    assert summarise_list([1, 2, 3, 4]) == [1, 3, 6, 10]


def test_index_in_list_middle_and_tail():
    lookup = [1, 3, 6, 10]
    assert index_in_list(0, lookup) == (0, 0)
    assert index_in_list(4, lookup) == (2, 1)
    assert index_in_list(10, lookup) == (4, 0)

