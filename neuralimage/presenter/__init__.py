from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_presenter import MainPresenter
    from .web_presenter import WebPresenter

__all__ = ["MainPresenter", "WebPresenter"]


def __getattr__(name: str):
    if name == "MainPresenter":
        from .main_presenter import MainPresenter

        return MainPresenter
    if name == "WebPresenter":
        from .web_presenter import WebPresenter

        return WebPresenter
    raise AttributeError(name)
