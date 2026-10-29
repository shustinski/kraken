"""Windows-protected storage and OS reauthentication for desktop autofill."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

from kraken_hub.domain.credential_target import LEGACY_CREDENTIAL_TARGET, credential_target_name
from ctypes import wintypes

_TARGET = LEGACY_CREDENTIAL_TARGET
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2


class _FILETIME(ctypes.Structure):
    _fields_ = (("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD))


class _CREDENTIALW(ctypes.Structure):
    _fields_ = (
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", _FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    )


def _windows_api():
    if sys.platform != "win32":
        return None
    return (
        ctypes.WinDLL("advapi32", use_last_error=True),
        ctypes.WinDLL("credui", use_last_error=True),
        ctypes.OleDLL("ole32"),
        ctypes.WinDLL("secur32", use_last_error=True),
    )


def credential_target(data_directory: str | Path | None = None) -> str:
    """Preserve the legacy default profile; isolate explicitly separate profiles."""
    if data_directory is None:
        directory = None
        is_default = True
    else:
        directory = Path(data_directory).expanduser().resolve()
        is_default = directory == (Path.home() / ".kraken").resolve()
    return credential_target_name(
        is_default_profile=is_default,
        canonical_directory="" if directory is None else str(directory),
        casefold=sys.platform == "win32",
    )


def save_credentials(username: str, password: str, *, target: str | None = None) -> bool:
    """Replace the remembered Kraken credential in Windows Credential Manager."""
    apis = _windows_api()
    if apis is None:
        return False
    advapi, _credui, _ole32, _secur32 = apis
    encoded = password.encode("utf-16-le")
    blob = ctypes.create_string_buffer(encoded) if encoded else None
    credential = _CREDENTIALW()
    credential.Type = _CRED_TYPE_GENERIC
    credential.TargetName = _TARGET if target is None else target
    credential.CredentialBlobSize = len(encoded)
    if blob is not None:
        credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
    credential.Persist = _CRED_PERSIST_LOCAL_MACHINE
    credential.UserName = username
    advapi.CredWriteW.argtypes = (ctypes.POINTER(_CREDENTIALW), wintypes.DWORD)
    advapi.CredWriteW.restype = wintypes.BOOL
    try:
        return bool(advapi.CredWriteW(ctypes.byref(credential), 0))
    finally:
        if blob is not None:
            ctypes.memset(ctypes.addressof(blob), 0, ctypes.sizeof(blob))


def load_credentials(*, target: str | None = None) -> tuple[str, str] | None:
    """Read the last Kraken credential from the current Windows user's vault."""
    apis = _windows_api()
    if apis is None:
        return None
    advapi, _credui, _ole32, _secur32 = apis
    pointer = ctypes.POINTER(_CREDENTIALW)()
    advapi.CredReadW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_CREDENTIALW)),
    )
    advapi.CredReadW.restype = wintypes.BOOL
    advapi.CredFree.argtypes = (ctypes.c_void_p,)
    if not advapi.CredReadW(_TARGET if target is None else target, _CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
        return None
    try:
        credential = pointer.contents
        data = (
            ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            if credential.CredentialBlobSize
            else b""
        )
        return str(credential.UserName or ""), data.decode("utf-16-le")
    finally:
        advapi.CredFree(pointer)


def credentials_available(*, target: str | None = None) -> bool:
    """Check for a saved entry without copying its secret into Python memory."""
    apis = _windows_api()
    if apis is None:
        return False
    advapi, _credui, _ole32, _secur32 = apis
    pointer = ctypes.POINTER(_CREDENTIALW)()
    advapi.CredReadW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_CREDENTIALW)),
    )
    advapi.CredReadW.restype = wintypes.BOOL
    advapi.CredFree.argtypes = (ctypes.c_void_p,)
    if not advapi.CredReadW(_TARGET if target is None else target, _CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
        return False
    advapi.CredFree(pointer)
    return True


def verify_windows_identity(parent_window: int = 0) -> bool:
    """Ask Windows Hello to verify the current user before releasing autofill."""
    if sys.platform != "win32":
        return False
    from .windows_hello import verify_user

    return verify_user(parent_window)


__all__ = [
    "credential_target",
    "credentials_available",
    "load_credentials",
    "save_credentials",
    "verify_windows_identity",
]
