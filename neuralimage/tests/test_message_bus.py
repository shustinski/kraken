from lib.message_bus import MessageBus


def test_subscribe_and_publish_calls_handler_once():
    bus = MessageBus()
    seen = []

    def handler(payload):
        seen.append(payload)

    bus.subscribe('topic', handler)
    bus.publish('topic', {'a': 1})

    assert seen == [{'a': 1}]


def test_unsubscribe_removes_handler():
    bus = MessageBus()
    seen = []

    def handler(payload):
        seen.append(payload)

    bus.subscribe('topic', handler)
    bus.unsubscribe('topic', handler)
    bus.publish('topic', 123)

    assert seen == []


def test_publish_unknown_topic_is_noop():
    bus = MessageBus()
    bus.publish('unknown', 'payload')


def test_unsubscribe_missing_handler_raises_value_error():
    bus = MessageBus()

    def handler(payload):
        return payload

    try:
        bus.unsubscribe('topic', handler)
    except ValueError:
        pass
    else:
        raise AssertionError('Expected ValueError when unsubscribing absent handler')

