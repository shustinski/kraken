import pytest

from model.NeuralNetwork.dataset import summarise_list, index_in_list


def test_summarise():
    test_list = [1,2,3,4,5]
    result = summarise_list(test_list)
    assert result == [1,3,6,10,15], f'Oh no, ошибка в {summarise_list.__name__}'

def test_index_in_list():
    test_list = [1,3,6,10,15]
    with pytest.raises(IndexError):
        index_in_list(20, test_list)
