"""Read-only snapshot of the protected user project (live bring-up baseline).

Authentication reuses the established live-server probe pattern:
``live_server_context.connect()`` decrypts the DPAPI-protected session file
``.test_runtime/live-server/server-session.secret`` and talks to the running
server through the standard ``kraken_hub`` remote client.

This probe performs GET requests only:

* ``GET /api/v1/session``               (inside ``connect()`` — session check)
* ``GET /api/v1/projects/{id}``         (project revision / state)
* ``GET /api/v1/projects/{id}/history`` (history length, fully paged)
* ``GET /api/v1/projects/{id}/jobs``    (job count / states / payload hashes)

No job or project state is ever modified: nothing is touched, retried,
cancelled, or resumed. The access token is never printed or written anywhere —
the receipt records identity only (principal id / subject / display name).

Job "payload hash" definition (also recorded in the receipt): the SHA-256 of
the exact raw JSON bytes of each job object as it appears in the ``items``
array of the ``/jobs`` response body (UTF-8, byte-for-byte as returned,
including the server's own JSON formatting).

Prints one JSON document on stdout. Exit code is 0 in both outcomes; the
``status`` field carries the result:

* ``{"status": "ok", ...baseline...}``
* ``{"status": "BLOCKED", "error_type": ..., "error": "<exact error text>"}``
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from typing import Any

PROTECTED_PROJECT_ID = "7c622c09-73fa-552f-9a52-e80b64aeb3ef"

PAYLOAD_SHA256_DEFINITION = (
    "SHA-256 of the exact raw JSON bytes of each job object as it appears in the "
    "'items' array of the GET /api/v1/projects/<project_id>/jobs response body "
    "(UTF-8, byte-for-byte as returned, including the server's own formatting)"
)


def _raw_item_spans(raw: bytes) -> list[bytes]:
    """Return the exact raw JSON span of each element of the items array."""
    text = raw.decode("utf-8")
    decoder = json.JSONDecoder()
    document, _ = decoder.raw_decode(text, 0)
    if not isinstance(document, dict) or not isinstance(document.get("items"), list):
        raise ValueError("jobs response is missing the items array")
    key_at = text.index('"items"')
    colon_at = text.index(":", key_at)
    value_at = colon_at + 1
    while value_at < len(text) and text[value_at].isspace():
        value_at += 1
    items, _ = decoder.raw_decode(text, value_at)
    spans: list[bytes] = []
    pos = value_at + 1  # skip the opening '['
    for _ in items:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        start = pos
        _, pos = decoder.raw_decode(text, pos)
        spans.append(text[start:pos].encode("utf-8"))
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text) and text[pos] == ",":
            pos += 1
    return spans


def _blocked(error_type: str, message: str) -> dict[str, Any]:
    return {"status": "BLOCKED", "error_type": error_type, "error": message}


def main() -> int:
    try:
        from live_server_context import connect

        remote = connect()
    except Exception as exc:  # noqa: BLE001 - the exact failure IS the report
        print(json.dumps(_blocked(type(exc).__name__, str(exc)), ensure_ascii=False))
        return 0

    project = remote.get_project(PROTECTED_PROJECT_ID)
    if project is None:
        print(json.dumps(
            _blocked("NotFoundError", f"Project {PROTECTED_PROJECT_ID} was not found on the server"),
            ensure_ascii=False))
        return 0

    try:
        history = remote.history(project.id)
        request = urllib.request.Request(
            f"{remote.base_url}/api/v1/projects/{project.id}/jobs",
            headers={"Authorization": f"Bearer {remote.auth.access_token}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except Exception as exc:  # noqa: BLE001 - the exact failure IS the report
        print(json.dumps(_blocked(type(exc).__name__, str(exc)), ensure_ascii=False))
        return 0

    try:
        parsed = json.loads(raw.decode("utf-8"))
        items = parsed["items"]
        spans = _raw_item_spans(raw)
        if len(items) != len(spans):
            raise ValueError("job item count differs from raw span count")
    except (ValueError, KeyError, UnicodeDecodeError) as exc:
        print(json.dumps(_blocked(type(exc).__name__, str(exc)), ensure_ascii=False))
        return 0

    jobs = [
        {
            "id": str(item.get("id", "")),
            "capability": str(item.get("capability", "")),
            "state": str(item.get("state", "")),
            "revision": item.get("revision"),
            "payload_sha256": hashlib.sha256(span).hexdigest(),
        }
        for item, span in zip(items, spans)
    ]
    principal = remote.auth.principal
    result = {
        "status": "ok",
        "principal": {
            "id": str(principal.id),
            "subject": principal.subject,
            "display_name": principal.display_name,
            "provider": principal.provider.value,
        },
        "project": {
            "id": str(project.id),
            "name": project.name,
            "width": project.width,
            "height": project.height,
            "state": project.state.value,
            "revision": project.revision,
        },
        "history_length": len(history),
        "jobs": {
            "count": len(jobs),
            "payload_sha256_definition": PAYLOAD_SHA256_DEFINITION,
            "items": jobs,
        },
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
