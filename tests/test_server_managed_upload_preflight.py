"""Gateway preflight must use the real persistent command composition."""
import hashlib
from uuid import uuid4

import pytest

from kraken_server.services import CommandContext, ConflictError
from tests.test_admin_project_workflows import project

pytest_plugins = ["tests.test_admin_project_workflows"]


def test_persistent_upload_preflight_checks_revision_and_replays_existing_version(database):
    services, _, owner, _ = database
    pid = project(services, owner, "Gateway preflight regression")["project_id"]
    series = services.create_artifact_series(pid, {"name": "Uploaded", "scope": "project_attachment"},
        CommandContext(str(owner.id), str(uuid4()), None))
    content = b"managed gateway regression content"
    payload = {"filename": "test.bin", "media_type": "application/octet-stream",
               "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}
    context = CommandContext(str(owner.id), str(uuid4()), services.artifact_stream_revision(pid, series["id"]))
    assert services.prepare_managed_artifact_upload(pid, series["id"], payload, context) is None
    with pytest.raises(ConflictError):
        services.prepare_managed_artifact_upload(pid, series["id"], payload,
            CommandContext(str(owner.id), str(uuid4()), 42))
    saved = services.add_managed_artifact_version(pid, series["id"], payload, (content,), context)
    assert services.prepare_managed_artifact_upload(pid, series["id"], payload, context) == saved
    assert len(services.list_artifact_versions(pid, series["id"])) == 1
