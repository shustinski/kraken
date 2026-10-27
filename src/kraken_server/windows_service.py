"""Minimal dependency-free Windows Service host for packaged Kraken Server."""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Any

SERVICE_NAME = "KrakenServer"
SERVICE_WIN32_OWN_PROCESS = 0x00000010
SERVICE_START_PENDING = 0x00000002
SERVICE_STOP_PENDING = 0x00000003
SERVICE_RUNNING = 0x00000004
SERVICE_STOPPED = 0x00000001
SERVICE_ACCEPT_STOP = 0x00000001
SERVICE_ACCEPT_SHUTDOWN = 0x00000004
SERVICE_CONTROL_STOP = 0x00000001
SERVICE_CONTROL_SHUTDOWN = 0x00000005
NO_ERROR = 0


class _ServiceStatus(ctypes.Structure):
    _fields_ = [
        ("dwServiceType", wintypes.DWORD),
        ("dwCurrentState", wintypes.DWORD),
        ("dwControlsAccepted", wintypes.DWORD),
        ("dwWin32ExitCode", wintypes.DWORD),
        ("dwServiceSpecificExitCode", wintypes.DWORD),
        ("dwCheckPoint", wintypes.DWORD),
        ("dwWaitHint", wintypes.DWORD),
    ]


_HANDLER = ctypes.WINFUNCTYPE(
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    ctypes.c_void_p,
)
_SERVICE_MAIN = ctypes.WINFUNCTYPE(None, wintypes.DWORD, ctypes.POINTER(wintypes.LPWSTR))


class _ServiceTableEntry(ctypes.Structure):
    _fields_ = [("lpServiceName", wintypes.LPWSTR), ("lpServiceProc", _SERVICE_MAIN)]


_status_handle: Any = None
_server: Any = None
_config_path: Path | None = None
_service_exit_code = 0
_status_checkpoint = 0
_status_state: int | None = None
_status_lock = threading.RLock()


def _configure_advapi() -> object:
    advapi32 = ctypes.windll.advapi32
    advapi32.RegisterServiceCtrlHandlerExW.argtypes = [
        wintypes.LPCWSTR,
        _HANDLER,
        ctypes.c_void_p,
    ]
    advapi32.RegisterServiceCtrlHandlerExW.restype = wintypes.HANDLE
    advapi32.SetServiceStatus.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ServiceStatus)]
    advapi32.SetServiceStatus.restype = wintypes.BOOL
    advapi32.StartServiceCtrlDispatcherW.argtypes = [ctypes.POINTER(_ServiceTableEntry)]
    advapi32.StartServiceCtrlDispatcherW.restype = wintypes.BOOL
    return advapi32


def _set_status(state: int, *, exit_code: int = 0, wait_hint: int = 0) -> None:
    with _status_lock:
        _set_status_locked(state, exit_code=exit_code, wait_hint=wait_hint)


def _set_status_locked(state: int, *, exit_code: int = 0, wait_hint: int = 0) -> None:
    global _status_checkpoint, _status_state
    if not _status_handle:
        return
    pending = state in {SERVICE_START_PENDING, SERVICE_STOP_PENDING}
    _status_checkpoint = _status_checkpoint + 1 if pending and state == _status_state else int(pending)
    _status_state = state
    accepted = SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN if state == SERVICE_RUNNING else 0
    status = _ServiceStatus(
        SERVICE_WIN32_OWN_PROCESS,
        state,
        accepted,
        exit_code,
        0,
        _status_checkpoint,
        wait_hint,
    )
    if not ctypes.windll.advapi32.SetServiceStatus(_status_handle, ctypes.byref(status)):
        raise ctypes.WinError()


@_HANDLER
def _control_handler(control: int, _event_type: int, _event_data: object, _context: object) -> int:
    if control in {SERVICE_CONTROL_STOP, SERVICE_CONTROL_SHUTDOWN}:
        _set_status(SERVICE_STOP_PENDING, wait_hint=15_000)
        if _server is not None:
            _server.should_exit = True
    return NO_ERROR


@_SERVICE_MAIN
def _service_main(_argument_count: int, _arguments: object) -> None:
    global _service_exit_code
    try:
        _run_service_main()
    except BaseException:
        # ctypes otherwise reports and discards callback exceptions. Preserve
        # failures that occur even before a status handle can be registered.
        _service_exit_code = 1
        logging.getLogger(__name__).exception("Windows service callback failed")


def _create_server(configuration: Any) -> Any:
    import uvicorn

    class ReadyServiceServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if not self.started or self.should_exit:
                raise RuntimeError("Server startup did not complete")
            # Uvicorn has completed ASGI lifespan startup and bound listeners.
            _set_status(SERVICE_RUNNING)

    return ReadyServiceServer(
        uvicorn.Config(
            "kraken_server.runtime:create_app_from_environment",
            host=configuration.host,
            port=configuration.port,
            factory=True,
            log_config=None,
            ssl_certfile=(None if configuration.tls_cert_file is None else str(configuration.tls_cert_file)),
            ssl_keyfile=(None if configuration.tls_key_file is None else str(configuration.tls_key_file)),
        )
    )


def _run_service_main() -> None:
    global _status_handle, _server, _service_exit_code
    _service_exit_code = 0
    advapi32 = _configure_advapi()
    _status_handle = advapi32.RegisterServiceCtrlHandlerExW(SERVICE_NAME, _control_handler, None)
    if not _status_handle:
        _service_exit_code = 1
        return
    progress_done = threading.Event()

    def report_progress():
        while not progress_done.wait(5):
            with _status_lock:
                if _status_state == SERVICE_START_PENDING:
                    _set_status(SERVICE_START_PENDING, wait_hint=30_000)

    progress = threading.Thread(target=report_progress, name="kraken-service-startup", daemon=True)
    try:
        _set_status(SERVICE_START_PENDING, wait_hint=30_000)
        progress.start()
        assert _config_path is not None
        os.environ["KRAKEN_SERVER_CONFIG"] = str(_config_path)
        from .configuration import ServerConfig

        configuration = ServerConfig.load(_config_path)
        configuration.apply_to_environment()
        _server = _create_server(configuration)
        _server.run()
        if not _server.started:
            raise RuntimeError("Server returned without completing startup")
        _set_status(SERVICE_STOPPED)
    except BaseException:
        _service_exit_code = 1
        _set_status(SERVICE_STOPPED, exit_code=1)
        logging.getLogger(__name__).exception("Windows service startup/runtime failed")
    finally:
        progress_done.set()
        if progress.is_alive():
            progress.join(timeout=2)
        _server = None


def run_windows_service(config_path: Path) -> None:
    if os.name != "nt":
        raise RuntimeError("Windows Service mode is available only on Windows")
    global _config_path
    _config_path = config_path.resolve(strict=True)
    advapi32 = _configure_advapi()
    table = (_ServiceTableEntry * 2)()
    table[0].lpServiceName = SERVICE_NAME
    table[0].lpServiceProc = _service_main
    if not advapi32.StartServiceCtrlDispatcherW(table):
        raise ctypes.WinError()
    if _service_exit_code:
        raise SystemExit(_service_exit_code)


__all__ = ["SERVICE_NAME", "run_windows_service"]
