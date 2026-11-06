from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QLabel

from kraken_hub.presentation.qt.cif_review_layers_dialog import CifReviewLayersDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_dialog_splits_selected_images_and_keeps_their_order(qapp) -> None:
    dialog = CifReviewLayersDialog(
        source_layer_name="Металл",
        images=(("img-a", "Слой A"), ("img-b", "Слой B"), ("img-c", "Слой C")),
        selected_ids=("img-c", "img-a"),
    )

    assert dialog.available.count() == 1
    assert dialog.available.item(0).text() == "Слой B"
    assert dialog.assigned.count() == 2
    assert dialog.ordered_ids() == ("img-c", "img-a")
    titles = [label.text() for label in dialog.findChildren(QLabel)]
    assert "Все изображения" in titles
    assert "Слой «Металл»" in titles
