"""Server image preparation must batch frames instead of paying several transactions per frame."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from kraken_manager.application.frame_artifact_import import ImportFrameArtifactsHandler
from test_server_workspace_inputs_parts.server_layer import _server_layer


def _images(tmp_path: Path, count: int) -> Path:
    folder = tmp_path / "many-images"
    folder.mkdir()
    for number in range(count):
        Image.new("RGB", (8, 6), (number * 7 % 255, 0, 0)).save(folder / f"{number:04d}.png")
    return folder


def _counting(factory, counter: Counter):
    class Counting:
        def __getattr__(self, name):
            return getattr(factory, name)

        def __call__(self):
            counter["units_of_work"] += 1
            return factory()

    return Counting()


def test_preparation_uses_a_few_transactions_for_many_frames(tmp_path) -> None:
    server, local, _session, project, layer, _binding, representation, context = _server_layer(
        tmp_path, images=_images(tmp_path, 24), width=6, height=4,
    )
    counter: Counter = Counter()
    factory = local._uow(str(project.id))
    server._import_frame_artifacts = ImportFrameArtifactsHandler(_counting(factory, counter), local.profiles, local.clock)

    result = server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)

    assert result == {"prepared_frames": 24}
    assert counter["units_of_work"] <= 3
    cells = local.frame_cells(project.id, layer.id, representation.id)
    assert len(cells) == 24 and all(cell.sha256 for cell in cells)
    kinds = Counter(event.event_type for event in local.history(project.id))
    assert kinds["ArtifactSeriesCreated"] == 24
    assert kinds["ArtifactVersionCreated"] == 24

    history = local.history(project.id)
    counter.clear()
    assert server.materialize_representation_inputs(
        str(project.id), str(layer.id), str(representation.id), context,
    ) == result
    assert local.history(project.id) == history
    assert counter["units_of_work"] <= 1


def test_whole_action_is_authorized_once_and_written_in_bounded_batches(tmp_path, monkeypatch) -> None:
    from kraken_manager.application import frame_artifact_import as module

    monkeypatch.setattr(module, "TRANSACTION_BATCH_SIZE", 5)
    server, local, _session, project, layer, _binding, representation, context = _server_layer(
        tmp_path, images=_images(tmp_path, 12), width=4, height=3,
    )
    counter: Counter = Counter()
    handler = ImportFrameArtifactsHandler(
        _counting(local._uow(str(project.id)), counter), local.profiles, local.clock,
    )
    authorize = handler._load_project_and_authorize

    def counted(*args, **kwargs):
        counter["authorizations"] += 1
        return authorize(*args, **kwargs)

    monkeypatch.setattr(handler, "_load_project_and_authorize", counted)
    commands: list[int] = []

    def record(command):
        commands.append(len(command.items))
        return handler(command)

    server._import_frame_artifacts = record
    assert server.materialize_representation_inputs(
        str(project.id), str(layer.id), str(representation.id), context,
    ) == {"prepared_frames": 12}
    assert commands == [12]
    assert counter["authorizations"] == 1
    # authorized inside the first transaction; 3 series + 3 version transactions
    # (+1 only where the factory has no shared blob store, as in this fixture)
    assert counter["units_of_work"] == 6 + (getattr(local._uow(str(project.id)), "blobs", None) is None)
