from dataclasses import dataclass
from typing import Any, List, Optional

@dataclass
class CustomError:
    """Ошибка или предупреждение"""
    message: str                     # человекочитаемое сообщение
    func: Optional[str] = None       # номер строки (если известен)
    cause: Optional[BaseException] = None  # оригинальное исключение (если есть)

    def __str__(self) -> str:
        loc = []
        location = ", ".join(loc)
        base = f"{self.message}"
        if self.cause:
            base = f"{self.cause}: {base}"
        if self.func:
            base = f"{base} в функции ({self.func})"
        return base

class CustopErrorCollector:
    """Контейнер для накопления ParseError."""
    def __init__(self):
        self.errors: List[CustomError] = []

    def add(self,
            message: str,
            func: Optional[str] = None,
            cause: Optional[BaseException] = None) -> None:
        self.errors.append(
            CustomError(message, func, cause)
        )

    def has_errors(self) -> bool:
        return bool(self.errors)

    def __len__(self) -> int:
        return len(self.errors)

    def __iter__(self):
        return iter(self.errors)

    def __str__(self) -> str:
        if not self.errors:
            return 'В ходе не было обнаружено ошибок'
        return "\n".join(f"{i + 1}. {err}" for i, err in enumerate(self.errors))