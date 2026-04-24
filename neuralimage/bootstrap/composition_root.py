from application.ports import StateStore
from infrastructure.config.state_store import create_state_store


def create_desktop_state_store() -> StateStore:
    return create_state_store(default_backend='qsettings')


def create_web_state_store() -> StateStore:
    return create_state_store(default_backend='ini')


def create_main_presenter():
    from presenter import MainPresenter

    return MainPresenter(state_store=create_desktop_state_store())


def create_web_presenter():
    from presenter import WebPresenter

    return WebPresenter(state_store=create_web_state_store())
