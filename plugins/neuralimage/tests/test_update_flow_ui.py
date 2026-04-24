import types

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication, QTextBrowser, QWidget

from neuralimage.lib.update_checker import ReleaseInfo, UpdateInfo
import neuralimage.presenter.update_flow as update_flow
@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_update_notification_dialog_uses_scrollable_text_browser(qapp):
    parent = QWidget()
    dialog = update_flow._UpdateNotificationDialog(
        parent,
        title='Update available',
        body_markdown='Intro\n\n## History\n\n### 6.1.0\n\n- Item',
        download_available=True,
        has_releases=True,
        texts={},
    )

    text_views = dialog.findChildren(QTextBrowser)

    assert len(text_views) == 1
    assert text_views[0].verticalScrollBar() is not None


def test_show_update_notification_routes_select_version_action(monkeypatch, qapp):
    calls = {'select_version': 0}

    class _DialogStub:
        def __init__(self, *_args, **_kwargs):
            self.selected_action = 'select_version'

        def exec(self):
            return 0

    presenter = types.SimpleNamespace(
        view=QWidget(),
        _selected_update_channel='stable',
        _start_update_download=lambda _release: None,
        _resolve_latest_release=lambda update_info: update_info,
        _show_release_selector=lambda _update_info: calls.__setitem__('select_version', calls['select_version'] + 1),
    )
    update_info = UpdateInfo(
        version='6.1.0',
        download_url='https://example.com/setup.exe',
        releases=(
            ReleaseInfo(version='6.1.0', download_url='https://example.com/setup.exe', notes='Fixes'),
        ),
    )

    monkeypatch.setattr(update_flow, '_UpdateNotificationDialog', _DialogStub)

    update_flow.show_update_notification(
        presenter,
        update_info,
        manual=False,
        app_version='6.0.0',
        collect_release_history_fn=lambda _info: '### 6.1.0\n\n- Fix',
    )

    assert calls['select_version'] == 1
