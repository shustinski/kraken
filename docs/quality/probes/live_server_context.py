from pathlib import Path
from kraken_hub.infrastructure.remote.client import RemoteAuth, RemoteServerProjectService, RemoteHttpClient
from kraken_hub.infrastructure.remote.parts.principal_server_payload import _principal_from_server_payload
from kraken_server.configuration import unprotect_secret

RUN = Path(__file__).resolve().parents[3] / ".test_runtime/live-server"
URL = "http://127.0.0.1:8080"


def connect():
    token = unprotect_secret(RUN / "server-session.secret")
    http = RemoteHttpClient(URL)
    principal = _principal_from_server_payload(http.request("GET", "/api/v1/session", token=token))
    return RemoteServerProjectService(URL, auth=RemoteAuth(access_token=token, principal=principal),
                                      data_dir=RUN / "client", http=http)
