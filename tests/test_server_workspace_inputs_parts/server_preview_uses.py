"""test_server_workspace_inputs: test server preview uses."""
from __future__ import annotations
from pathlib import Path
import pytest
from PyQt6.QtGui import QImage
from kraken_hub.matrix_source import KrakenMatrixAssetSource, KrakenMatrixDataSource
from kraken_core.frame_matrix import MatrixBounds, MatrixViewportRequest
from kraken_server.services import ForbiddenError, NotFoundError, ValidationError

from test_server_workspace_inputs_parts.server_layer import _server_layer

def test_server_source_preparation_resumes_after_missing_file(tmp_path):
    server, local, _session, project, layer, binding, representation, context = _server_layer(tmp_path)
    missing = Path(binding.image_directory) / "0002.jpg"
    contents = missing.read_bytes()
    missing.unlink()
    with pytest.raises(ValidationError, match="исходные изображения"):
        server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)
    missing.write_bytes(contents)
    assert server.materialize_representation_inputs(
        str(project.id),
        str(layer.id),
        str(representation.id),
        context,
    ) == {"prepared_frames": 2}
    assert len(local.frame_cells(project.id, layer.id, representation.id)) == 2

def test_server_source_preparation_rejects_wrong_project(tmp_path):
    server, _local, _session, project, layer, _binding, representation, context = _server_layer(tmp_path)
    from uuid import uuid4

    with pytest.raises(NotFoundError):
        server.materialize_representation_inputs(str(uuid4()), str(layer.id), str(representation.id), context)
    assert server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)

def test_server_source_preparation_preserves_import_permissions(tmp_path):
    from kraken_manager.domain.identity import Principal

    server, local, _session, project, layer, _binding, representation, context = _server_layer(tmp_path)
    history = local.history(project.id)
    server._actor = lambda _id: Principal.local(subject="reader", display_name="Reader")
    with pytest.raises(ForbiddenError):
        server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)
    assert local.history(project.id) == history
    assert local.frame_cells(project.id, layer.id, representation.id) == ()

def test_server_preview_uses_version_id_and_decodes_linked_image(tmp_path):
    server, local, _session, project, layer, _binding, representation, context = _server_layer(tmp_path)
    server.materialize_representation_inputs(str(project.id), str(layer.id), str(representation.id), context)

    class MatrixService:
        matrix_viewport = server.matrix_viewport

        def read_project_blob(self, project_id, version_id):
            version = local.get_artifact_version(project_id, version_id)
            assert version is not None
            return local.read_project_blob(project_id, version.sha256)

    service = MatrixService()
    source = KrakenMatrixDataSource(
        service,
        project_id=str(project.id),
        layer_id=str(layer.id),
        representation_ids=(str(representation.id),),
        matrix_width=2,
    )
    item = source.load_viewport(MatrixViewportRequest(MatrixBounds(1, 1, 2, 2))).items[0]
    assert item.asset is not None
    assert item.asset.source_key == item.metadata["artifact_version_id"]
    assert item.asset.source_key != item.metadata["asset_sha256"]
    output = KrakenMatrixAssetSource(service, project_id=str(project.id)).load_asset(item.asset, width=32, height=24)
    image = QImage.fromData(output)
    assert not image.isNull()
    assert (image.width(), image.height()) == (32, 24)
