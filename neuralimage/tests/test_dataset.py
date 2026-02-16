from model.NeuralNetwork.dataset import summarise_list, index_in_list


def test_summarise():
    test_list = [1,2,3,4,5]
    result = summarise_list(test_list)
    assert result == [1,3,6,10,15], f'Oh no, ошибка в {summarise_list.__name__}'

def test_index_in_list():
    test_index = 20
    test_list = [1,3,6,10,15]
    result = index_in_list(test_index,test_list)
    assert result == (4,5), f'Oh no, ошибка в {index_in_list.__name__}'
