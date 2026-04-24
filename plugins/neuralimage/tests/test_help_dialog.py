import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QTextBrowser

from neuralimage.lib.ui_texts import set_ui_language
from neuralimage.view.help_dialog import HelpDialog, show_help_dialog


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_help_dialog_loads_russian_help_catalog(qapp):
    set_ui_language('ru')
    dialog = HelpDialog()

    assert dialog.windowTitle() == 'Справка по проекту'
    assert dialog.catalog_tree.topLevelItemCount() >= 3


def test_help_dialog_loads_english_help_catalog(qapp):
    set_ui_language('en')
    dialog = HelpDialog()

    assert dialog.windowTitle() == 'Project help'
    assert dialog.catalog_tree.topLevelItemCount() >= 3


def test_help_dialog_uses_single_modeless_window(qapp):
    dialog = show_help_dialog()
    second_dialog = show_help_dialog()
    qapp.processEvents()

    assert dialog is second_dialog
    assert dialog.isVisible()
    assert dialog.isModal() is False

    dialog.close()
    qapp.processEvents()


def test_help_dialog_article_views_do_not_scroll_vertically(qapp):
    set_ui_language('ru')
    dialog = HelpDialog()
    text_views = dialog.findChildren(QTextBrowser)

    assert text_views
    assert all(
        view.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        for view in text_views
    )
