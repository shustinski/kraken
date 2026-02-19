import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List


class AbstractMessageBus(ABC):
    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        pass

    @abstractmethod
    def unsubscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        pass

    @abstractmethod
    def publish(self, topic: str, payload: Any = None) -> None:
        pass


class MessageBus(AbstractMessageBus):
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        with self._lock:
            self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        with self._lock:
            handlers = self._subscribers.get(topic)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                return

    def publish(self, topic: str, payload: Any = None) -> None:
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))
        for handler in handlers:
            handler(payload)


general_bus = MessageBus()
