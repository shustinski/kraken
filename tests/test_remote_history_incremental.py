"""The workflow monitor can fetch history after its last event."""

from urllib.parse import parse_qs, urlsplit

from kraken_hub.infrastructure.remote.parts.submit_plugin_job import SubmitPluginJobMixin


def test_history_since_uses_last_position_as_cursor():
    class Client(SubmitPluginJobMixin):
        def __init__(self):
            self.paths = []

        def _call(self, method, path):
            self.paths.append(path)
            return {"items": [{"position": 42, "event_id": "new", "recorded_at": "2026-01-01T00:00:00+00:00"}],
                    "next_cursor": None}

    client = Client()
    events = client.history_since("project", 41)
    assert [event.position for event in events] == [42]
    assert parse_qs(urlsplit(client.paths[0]).query)["cursor"]
