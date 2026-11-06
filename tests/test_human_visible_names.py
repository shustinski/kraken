"""Internal identifiers stay available to actions without appearing in properties."""

from kraken_manager.presentation.qt.layer_management import ObjectPropertiesDialog, ObjectPropertiesSnapshot


def test_properties_hide_internal_identifiers_and_hashes(qtbot):
    identifier = "f00c53ee-1cc3-4196-9f5b-2d71ed1e1d82"
    digest = "f" * 64
    snapshot = ObjectPropertiesSnapshot(
        title="Contour",
        object_kind="job",
        properties=(
            ("Название", "Contour"),
            ("Идентификатор задания", identifier),
            ("Параметры", {"source_representation_id": identifier, "sha256": digest, "threshold": 0.5}),
        ),
        versions=({"filename": "0000.cif", "sha256": digest, "parent": identifier},),
    )
    dialog = ObjectPropertiesDialog(snapshot)
    qtbot.addWidget(dialog)
    text = "\n".join(
        table.item(row, column).text()
        for table in (dialog.properties_table, dialog.versions_table)
        for row in range(table.rowCount())
        for column in range(table.columnCount())
        if table.item(row, column) is not None
    )
    assert identifier not in text
    assert digest not in text
    assert "threshold" in text
    assert "0000.cif" in text
    assert dialog.snapshot.versions[0]["parent"] == identifier
