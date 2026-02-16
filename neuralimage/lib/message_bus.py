import threading
from collections import defaultdict
from typing import Callable, Any, Dict, List
from abc import ABC, abstractmethod

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
        # topic → список callbacks
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._lock = threading.RLock()  # RLock позволяет вложенные вызовы

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        """Подписаться на топик."""
        with self._lock:
            self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        """Отписаться от топика."""
        with self._lock:
            self._subscribers[topic].remove(handler)

    def publish(self, topic: str, payload: Any = None) -> None:
        """Отправить сообщение всем подписчикам топика."""
        with self._lock:
            for handler in self._subscribers.get(topic, []):
                handler(payload)


general_bus = MessageBus()
