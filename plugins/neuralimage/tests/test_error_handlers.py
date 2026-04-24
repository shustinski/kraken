from neuralimage.lib.error_handlers import CustomError, CustopErrorCollector


def test_custom_error_str_with_cause_and_func():
    err = CustomError('failed', func='run', cause=ValueError('bad'))
    text = str(err)
    assert 'bad' in text
    assert 'failed' in text
    assert 'run' in text


def test_error_collector_add_and_iterate():
    collector = CustopErrorCollector()
    collector.add('m1')
    collector.add('m2', func='f')

    assert collector.has_errors() is True
    assert len(collector) == 2
    assert [e.message for e in collector] == ['m1', 'm2']


def test_error_collector_str_empty_and_filled():
    collector = CustopErrorCollector()
    assert isinstance(str(collector), str)
    assert len(str(collector)) > 0

    collector.add('boom')
    text = str(collector)
    assert 'boom' in text
    assert '1.' in text

