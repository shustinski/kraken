from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from application.dto import MainWindowState, SettingsState
from lib.data_interfaces import WorkMode


_SANITIZE_PATTERN = re.compile(r'[^A-Za-z0-9._-]+')


def _sanitize_name(value: str, *, fallback: str = 'model') -> str:
    cleaned = _SANITIZE_PATTERN.sub('_', str(value or '').strip()).strip('._-')
    return cleaned or fallback


def build_training_artifact_dir(
    main_state: MainWindowState,
    settings_state: SettingsState,
    work_mode: WorkMode,
    *,
    timestamp: datetime | None = None,
) -> Path:
    current_time = timestamp or datetime.now()
    timestamp_text = current_time.strftime('%Y%m%d_%H%M%S')

    if work_mode == WorkMode.further_training and str(main_state.model_path).strip():
        model_path = Path(main_state.model_path)
        root_dir = model_path.parent if str(model_path.parent) else Path.cwd()
        base_name = _sanitize_name(model_path.stem, fallback='model')
    else:
        sample_path = Path(main_state.sample_folder)
        root_dir = sample_path.parent if str(sample_path) else Path.cwd()
        base_name = _sanitize_name(getattr(settings_state, 'model', ''), fallback='model')

    candidate = root_dir / f'{base_name}_{timestamp_text}'
    suffix = 1
    while candidate.exists():
        candidate = root_dir / f'{base_name}_{timestamp_text}_{suffix:02d}'
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate
