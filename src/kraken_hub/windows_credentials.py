"""Windows-protected storage and OS reauthentication for desktop autofill."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

_TARGET = "Kraken.ProjectManager.LocalAccount"
_ERROR_ACCOUNT_RESTRICTION = 1327
_CRED_TYPE_GENERIC = 1
_CRED_PERSIST_LOCAL_MACHINE = 2
_CREDUIWIN_ENUMERATE_CURRENT_USER = 0x200
_LOGON32_LOGON_INTERACTIVE = 2
_LOGON32_PROVIDER_DEFAULT = 0
_NAME_SAM_COMPATIBLE = 2
_MAX_CREDENTIAL_TEXT = 514


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


class _CREDUI_INFOW(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("hwndParent", wintypes.HWND),
        ("pszMessageText", wintypes.LPCWSTR),
        ("pszCaptionText", wintypes.LPCWSTR),
        ("hbmBanner", wintypes.HBITMAP),
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


def save_credentials(username: str, password: str) -> bool:
    """Replace the remembered Kraken credential in Windows Credential Manager."""
    apis = _windows_api()
    if apis is None:
        return False
    advapi, _credui, _ole32, _secur32 = apis
    encoded = password.encode("utf-16-le")
    blob = ctypes.create_string_buffer(encoded) if encoded else None
    credential = _CREDENTIALW()
    credential.Type = _CRED_TYPE_GENERIC
    credential.TargetName = _TARGET
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


def load_credentials() -> tuple[str, str] | None:
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
    if not advapi.CredReadW(_TARGET, _CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
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


def credentials_available() -> bool:
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
    if not advapi.CredReadW(_TARGET, _CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
        return False
    advapi.CredFree(pointer)
    return True


def verify_windows_identity(parent_window: int = 0) -> bool:
    """Ask Windows to verify the current OS user before releasing autofill data."""
    apis = _windows_api()
    if apis is None:
        return False
    advapi, credui, ole32, secur32 = apis
    current_user = ctypes.create_unicode_buffer(_MAX_CREDENTIAL_TEXT)
    current_user_size = wintypes.ULONG(len(current_user))
    secur32.GetUserNameExW.argtypes = (
        ctypes.c_int,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.ULONG),
    )
    secur32.GetUserNameExW.restype = wintypes.BOOL
    if not secur32.GetUserNameExW(
        _NAME_SAM_COMPATIBLE,
        current_user,
        ctypes.byref(current_user_size),
    ):
        return False

    info = _CREDUI_INFOW(
        ctypes.sizeof(_CREDUI_INFOW),
        parent_window or None,
        "Подтвердите личность, чтобы заполнить сохранённые данные Kraken.",
        "Kraken — проверка Windows",
        None,
    )
    credui.CredUIPromptForWindowsCredentialsW.argtypes = (
        ctypes.POINTER(_CREDUI_INFOW),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.ULONG),
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.ULONG),
        ctypes.POINTER(wintypes.BOOL),
        wintypes.DWORD,
    )
    credui.CredUIPromptForWindowsCredentialsW.restype = wintypes.DWORD
    credui.CredUnPackAuthenticationBufferW.argtypes = (
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    credui.CredUnPackAuthenticationBufferW.restype = wintypes.BOOL
    advapi.LogonUserW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    )
    advapi.LogonUserW.restype = wintypes.BOOL
    ole32.CoTaskMemFree.argtypes = (ctypes.c_void_p,)

    error = 0
    for _attempt in range(3):
        auth_package = wintypes.ULONG()
        output = ctypes.c_void_p()
        output_size = wintypes.ULONG()
        error = int(
            credui.CredUIPromptForWindowsCredentialsW(
                ctypes.byref(info),
                error,
                ctypes.byref(auth_package),
                None,
                0,
                ctypes.byref(output),
                ctypes.byref(output_size),
                None,
                _CREDUIWIN_ENUMERATE_CURRENT_USER,
            )
        )
        if error != 0:
            return False
        try:
            username = ctypes.create_unicode_buffer(_MAX_CREDENTIAL_TEXT)
            domain = ctypes.create_unicode_buffer(_MAX_CREDENTIAL_TEXT)
            password = ctypes.create_unicode_buffer(_MAX_CREDENTIAL_TEXT)
            username_size = wintypes.DWORD(len(username))
            domain_size = wintypes.DWORD(len(domain))
            password_size = wintypes.DWORD(len(password))
            unpacked = bool(
                credui.CredUnPackAuthenticationBufferW(
                    0,
                    output,
                    output_size.value,
                    username,
                    ctypes.byref(username_size),
                    domain,
                    ctypes.byref(domain_size),
                    password,
                    ctypes.byref(password_size),
                )
            )
            ctypes.memset(output, 0, output_size.value)
            if not unpacked:
                error = ctypes.get_last_error()
                continue
            if username.value.casefold() != current_user.value.casefold():
                error = 1326
                continue
            account_domain, separator, account_name = username.value.partition("\\")
            if not separator:
                error = 1326
                continue
            token = wintypes.HANDLE()
            accepted = bool(
                advapi.LogonUserW(
                    account_name,
                    account_domain,
                    password,
                    _LOGON32_LOGON_INTERACTIVE,
                    _LOGON32_PROVIDER_DEFAULT,
                    ctypes.byref(token),
                )
            )
            if accepted:
                ctypes.windll.kernel32.CloseHandle(token)
                return True
            error = ctypes.get_last_error()
            if error == _ERROR_ACCOUNT_RESTRICTION:
                return True
        finally:
            if output:
                ctypes.memset(output, 0, output_size.value)
                ole32.CoTaskMemFree(output)
            if "password" in locals():
                ctypes.memset(ctypes.addressof(password), 0, ctypes.sizeof(password))
    return False


__all__ = [
    "credentials_available",
    "load_credentials",
    "save_credentials",
    "verify_windows_identity",
]
