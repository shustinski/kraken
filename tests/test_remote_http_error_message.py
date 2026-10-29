"""HTTP failures stay diagnosable: method, path, status, and server detail."""

from __future__ import annotations

import io
import urllib.error
import urllib.request

import pytest

from kraken_hub.remote_client_parts.extract_review_archive import RemoteServerError
from kraken_hub.remote_client_parts.request import RequestMixin


class _Client(RequestMixin):
    def __init__(self) -> None:
        self.base_url = "http://kraken.test"
        self.timeout = 1


def test_http_error_message_keeps_method_path_status_and_detail(monkeypatch) -> None:
    principal_id = "06e05318-bdd7-4cd6-9bcd-a2d3b384c520"
    path = f"/api/v1/projects/project/acl/{principal_id}"

    def fail(request, timeout):
        del timeout
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "missing",
            hdrs=None,
            fp=io.BytesIO(f'{{"detail":"{principal_id}"}}'.encode()),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(RemoteServerError) as caught:
        _Client().request("GET", path, token="token")
    message = str(caught.value)
    assert "GET" in message
    assert path in message
    assert "404" in message
    assert principal_id in message
    assert caught.value.status == 404
