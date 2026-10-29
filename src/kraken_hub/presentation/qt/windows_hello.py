"""Verify the current Windows user without unpacking PINs as passwords.

Uses IUserConsentVerifierInterop, the desktop Windows Hello API. Interface
IDs and vtable layouts come from UserConsentVerifierInterop.h, asyncinfo.h,
and windows.security.credentials.ui.h in the Windows SDK.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from uuid import UUID


class _GUID(ctypes.Structure):
    _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16),
                ("data3", ctypes.c_uint16), ("data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, value: str):
        return cls.from_buffer_copy(UUID(value).bytes_le)


_INTEROP = _GUID.parse("39e050c3-4e74-441a-8dc0-b81104df949c")
_OPERATION = _GUID.parse("fd596ffd-2318-558f-9dbe-d21df43764a5")
_ASYNC_INFO = _GUID.parse("00000036-0000-0000-c000-000000000046")
_HRESULT = ctypes.c_int32
_POINTER = ctypes.c_void_p


def _check(result: int) -> None:
    if result < 0:
        raise OSError(f"Windows Hello HRESULT 0x{result & 0xffffffff:08X}")


def _call(pointer, slot: int, *argument_types, result_type=_HRESULT):
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(_POINTER))).contents
    return ctypes.WINFUNCTYPE(result_type, _POINTER, *argument_types)(vtable[slot])


def _release(pointer) -> None:
    if pointer:
        _call(pointer, 2, result_type=ctypes.c_uint32)(pointer)


def _wait_for_verification(operation, info, *, timeout_ms: int = 120_000) -> bool:
    from PyQt6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    poll = QTimer()
    poll.setInterval(50)
    deadline = QTimer()
    deadline.setSingleShot(True)
    status = ctypes.c_int32(0)  # AsyncStatus.Started
    failure = []

    def check_status():
        try:
            _check(_call(info, 7, ctypes.POINTER(ctypes.c_int32))(info, ctypes.byref(status)))
        except OSError as exc:
            failure.append(exc)
            loop.quit()
            return
        if status.value != 0:
            loop.quit()

    poll.timeout.connect(check_status)
    deadline.timeout.connect(loop.quit)
    poll.start()
    deadline.start(timeout_ms)
    try:
        loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    finally:
        poll.stop()
        deadline.stop()
    if failure or status.value == 0:
        _call(info, 9)(info)  # IAsyncInfo.Cancel
        return False
    if status.value != 1:  # Only AsyncStatus.Completed may return a result.
        return False
    result = ctypes.c_int32(-1)
    _check(_call(operation, 8, ctypes.POINTER(ctypes.c_int32))(operation, ctypes.byref(result)))
    return result.value == 0  # UserConsentVerificationResult.Verified


def verify_user(parent_window: int) -> bool:
    if not parent_window:
        return False
    api = ctypes.WinDLL("combase")
    api.RoInitialize.argtypes = [ctypes.c_uint32]
    api.RoInitialize.restype = _HRESULT
    api.RoUninitialize.argtypes = []
    api.RoUninitialize.restype = None
    api.WindowsCreateString.argtypes = [wintypes.LPCWSTR, ctypes.c_uint32, ctypes.POINTER(_POINTER)]
    api.WindowsCreateString.restype = _HRESULT
    api.WindowsDeleteString.argtypes = [_POINTER]
    api.WindowsDeleteString.restype = _HRESULT
    api.RoGetActivationFactory.argtypes = [_POINTER, ctypes.POINTER(_GUID), ctypes.POINTER(_POINTER)]
    api.RoGetActivationFactory.restype = _HRESULT
    initialized = api.RoInitialize(0)  # Single-threaded UI apartment.
    if initialized != -2147417850:  # RPC_E_CHANGED_MODE: use the existing apartment.
        _check(initialized)
    strings = []
    factory, operation, info = _POINTER(), _POINTER(), _POINTER()

    def hstring(value):
        handle = _POINTER()
        _check(api.WindowsCreateString(value, len(value.encode("utf-16-le")) // 2, ctypes.byref(handle)))
        strings.append(handle)
        return handle

    try:
        name = hstring("Windows.Security.Credentials.UI.UserConsentVerifier")
        _check(api.RoGetActivationFactory(name, ctypes.byref(_INTEROP), ctypes.byref(factory)))
        message = hstring("Подтвердите личность, чтобы заполнить сохранённые данные Kraken.")
        _check(_call(factory, 6, wintypes.HWND, _POINTER, ctypes.POINTER(_GUID), ctypes.POINTER(_POINTER))(
            factory, parent_window, message, ctypes.byref(_OPERATION), ctypes.byref(operation),
        ))
        _check(_call(operation, 0, ctypes.POINTER(_GUID), ctypes.POINTER(_POINTER))(
            operation, ctypes.byref(_ASYNC_INFO), ctypes.byref(info),
        ))
        return _wait_for_verification(operation, info)
    finally:
        _release(info)
        _release(operation)
        _release(factory)
        for handle in reversed(strings):
            api.WindowsDeleteString(handle)
        if initialized >= 0:
            api.RoUninitialize()


def verify_windows_identity(parent_window: int = 0) -> bool:
    """Ask Windows Hello to verify the current user before releasing autofill."""
    if sys.platform != "win32":
        return False
    return verify_user(parent_window)
