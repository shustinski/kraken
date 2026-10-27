"""Real Uvicorn startup with SCM calls replaced; never creates a service."""

import ctypes
import os
import socket
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows service host")


@pytest.fixture
def host(tmp_path, monkeypatch):
    from kraken_server import windows_service as service
    from kraken_server.configuration import ServerConfig

    configuration = SimpleNamespace(host="127.0.0.1", port=0, tls_cert_file=None, tls_key_file=None,
                                    apply_to_environment=lambda: None)
    monkeypatch.setattr(service, "_config_path", tmp_path / "not-read.toml")
    monkeypatch.setattr(service, "_service_exit_code", 0)
    monkeypatch.setattr(service, "_status_handle", None)
    monkeypatch.setattr(service, "_server", None)
    monkeypatch.setattr(ServerConfig, "load", lambda _path: configuration)
    monkeypatch.setenv("KRAKEN_SERVER_CONFIG", "temporary-fixture-restored-after-test")
    monkeypatch.setattr(service, "_configure_advapi", lambda: SimpleNamespace(
        RegisterServiceCtrlHandlerExW=lambda *_args: 1))
    statuses = []
    monkeypatch.setattr(service, "_set_status", lambda state, **options: statuses.append((state, options)))
    return service, configuration, statuses


def app_with_lifespan(*, fail=False, observations=None):
    async def app(scope, receive, send):
        assert scope["type"] == "lifespan"
        assert (await receive())["type"] == "lifespan.startup"
        if fail:
            await send({"type": "lifespan.startup.failed", "message": "deliberate readiness failure"})
            return
        observations.append("lifespan-ready")
        await send({"type": "lifespan.startup.complete"})
        assert (await receive())["type"] == "lifespan.shutdown"
        await send({"type": "lifespan.shutdown.complete"})
    return app


def test_running_is_reported_only_after_real_lifespan_and_listener_bind(host, monkeypatch):
    service, _configuration, statuses = host
    observations = []
    monkeypatch.setattr("kraken_server.runtime.create_app_from_environment",
                        lambda: app_with_lifespan(observations=observations))

    def report(state, **options):
        statuses.append((state, options))
        if state == service.SERVICE_RUNNING:
            assert observations == ["lifespan-ready"]
            assert service._server.started
            assert service._server.servers[0].sockets[0].getsockname()[1] > 0
            service._server.should_exit = True
    monkeypatch.setattr(service, "_set_status", report)
    service._run_service_main()
    assert [state for state, _ in statuses] == [service.SERVICE_START_PENDING,
                                               service.SERVICE_RUNNING, service.SERVICE_STOPPED]
    assert service._service_exit_code == 0


@pytest.mark.parametrize("failure", ["lifespan", "bind"])
def test_actual_startup_failure_never_reports_running_and_sets_failure_status(host, monkeypatch, failure):
    service, configuration, statuses = host
    monkeypatch.setattr("kraken_server.runtime.create_app_from_environment",
                        lambda: app_with_lifespan(fail=failure == "lifespan", observations=[]))
    with socket.socket() as occupied:
        if failure == "bind":
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                occupied.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            configuration.port = occupied.getsockname()[1]
        service._run_service_main()
    assert service.SERVICE_RUNNING not in [state for state, _ in statuses]
    assert statuses[-1] == (service.SERVICE_STOPPED, {"exit_code": 1})
    assert service._service_exit_code != 0


def test_return_without_startup_is_not_scm_success(host, monkeypatch):
    service, _configuration, statuses = host
    monkeypatch.setattr(service, "_create_server", lambda _configuration: SimpleNamespace(
        started=False, run=lambda: None))
    service._run_service_main()
    assert statuses[-1] == (service.SERVICE_STOPPED, {"exit_code": 1})
    assert service.SERVICE_RUNNING not in [state for state, _ in statuses]


def test_dispatcher_propagates_recorded_service_failure_to_process_exit(tmp_path, monkeypatch):
    from kraken_server import windows_service as service
    configuration = tmp_path / "server.toml"
    configuration.touch()
    monkeypatch.setattr(service, "_configure_advapi", lambda: SimpleNamespace(
        StartServiceCtrlDispatcherW=lambda _table: True))
    monkeypatch.setattr(service, "_service_exit_code", 1)
    with pytest.raises(SystemExit) as error:
        service.run_windows_service(configuration)
    assert error.value.code == 1


def test_callback_registration_failure_cannot_be_swallowed_as_process_success(monkeypatch):
    from kraken_server import windows_service as service
    def fail_before_status_registration():
        raise RuntimeError("deliberate early SCM API failure")
    monkeypatch.setattr(service, "_run_service_main", fail_before_status_registration)
    monkeypatch.setattr(service, "_service_exit_code", 0)
    service._service_main(0, None)
    assert service._service_exit_code == 1


def test_pending_checkpoints_and_control_masks_use_correct_scm_fields(monkeypatch):
    from kraken_server import windows_service as service
    captured = []
    def record(_handle, pointer):
        status = ctypes.cast(pointer, ctypes.POINTER(service._ServiceStatus)).contents
        captured.append((status.dwCurrentState, status.dwCheckPoint, status.dwControlsAccepted))
        return True
    monkeypatch.setattr(service.ctypes, "windll", SimpleNamespace(advapi32=SimpleNamespace(SetServiceStatus=record)))
    monkeypatch.setattr(service, "_status_handle", 1)
    monkeypatch.setattr(service, "_status_state", None)
    monkeypatch.setattr(service, "_status_checkpoint", 0)
    for state in (service.SERVICE_START_PENDING, service.SERVICE_START_PENDING,
                  service.SERVICE_RUNNING, service.SERVICE_STOP_PENDING, service.SERVICE_STOPPED):
        service._set_status(state)
    assert captured == [(2, 1, 0), (2, 2, 0), (4, 0, 5), (3, 1, 0), (1, 0, 0)]
