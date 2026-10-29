"""Presentation labels for catalog storage profiles."""
LOCAL_STORAGE_LABEL = "Локальный файл"
REMOTE_STORAGE_LABEL = "Сервер PostgreSQL"


def storage_label(*, remote: bool) -> str:
    return REMOTE_STORAGE_LABEL if remote else LOCAL_STORAGE_LABEL
