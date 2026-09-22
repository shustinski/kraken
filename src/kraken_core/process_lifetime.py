"""Keep every child process inside the lifetime of the process that created it."""

from __future__ import annotations

import ctypes
import os
import threading
from ctypes import wintypes

_SYNCHRONIZE = 0x00100000
_PROCESS_TERMINATE = 0x0001
_WAIT_FAILED = 0xFFFFFFFF
_INFINITE = 0xFFFFFFFF
_TH32CS_SNAPPROCESS = 0x00000002
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000
_CTRL_CLOSE_EVENT = 2
_CTRL_LOGOFF_EVENT = 5
_CTRL_SHUTDOWN_EVENT = 6

_ConsoleHandler = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
_job_handle: int | None = None
_console_handler: _ConsoleHandler | None = None
_bound = False


class _JobBasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JobExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobBasicLimit),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


def bind_child_process_lifetime() -> None:
    """Terminate descendant processes when this process or its console ends."""
    global _bound
    if os.name != "nt" or _bound or os.environ.get("PYTEST_CURRENT_TEST"):
        return
    _bound = True
    # Closing a window ends only the process attached to it. Workers started
    # without a window, including plugin and training processes, stay in this
    # job and are terminated with the process that created them.
    _arm_kill_on_job_close()
    _install_console_close_handler()
    _watch_parent_process()


def detached_creationflags(existing: int = 0) -> int:
    """Allow an installer to keep running after the application process exits."""
    if os.name != "nt":
        return existing
    return existing | _CREATE_BREAKAWAY_FROM_JOB


def _kernel32() -> ctypes.WinDLL:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.WaitForMultipleObjects.argtypes = [
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.WaitForMultipleObjects.restype = wintypes.DWORD
    kernel32.ExitProcess.argtypes = [wintypes.UINT]
    kernel32.ExitProcess.restype = None
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.SetConsoleCtrlHandler.argtypes = [_ConsoleHandler, wintypes.BOOL]
    kernel32.SetConsoleCtrlHandler.restype = wintypes.BOOL
    return kernel32


def _arm_kill_on_job_close() -> None:
    global _job_handle
    kernel32 = _kernel32()
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return
    limits = _JobExtendedLimit()
    limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | _JOB_OBJECT_LIMIT_BREAKAWAY_OK
    configured = kernel32.SetInformationJobObject(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    )
    assigned = configured and kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
    if not assigned:
        kernel32.CloseHandle(job)
        return
    _job_handle = int(job)


def _install_console_close_handler() -> None:
    global _console_handler
    kernel32 = _kernel32()

    def _on_console_control(control: int) -> bool:
        if control in {_CTRL_CLOSE_EVENT, _CTRL_LOGOFF_EVENT, _CTRL_SHUTDOWN_EVENT}:
            _terminate_descendants()
        return False

    _console_handler = _ConsoleHandler(_on_console_control)
    kernel32.SetConsoleCtrlHandler(_console_handler, True)


def _watch_parent_process() -> None:
    threading.Thread(
        target=_exit_when_an_ancestor_exits,
        name="kraken-process-lifetime",
        daemon=True,
    ).start()


def _exit_when_an_ancestor_exits() -> None:
    kernel32 = _kernel32()
    handles = _open_ancestor_handles()
    if not handles:
        return
    # Python on Windows keeps a launcher process between the console and the
    # interpreter, so the shell can die without being the immediate parent.
    array = (wintypes.HANDLE * len(handles))(*handles)
    waited = kernel32.WaitForMultipleObjects(len(handles), array, False, _INFINITE)
    for handle in handles:
        kernel32.CloseHandle(handle)
    if waited == _WAIT_FAILED:
        return
    _terminate_descendants()
    kernel32.ExitProcess(1)


def _open_ancestor_handles() -> list[int]:
    kernel32 = _kernel32()
    parents = dict(_process_parents())
    handles: list[int] = []
    pid = os.getppid()
    seen = {os.getpid()}
    while pid > 0 and pid not in seen and len(handles) < 64:
        seen.add(pid)
        handle = kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
        if handle:
            handles.append(int(handle))
        pid = parents.get(pid, 0)
    return handles


def _terminate_descendants() -> None:
    kernel32 = _kernel32()
    for pid in _descendant_pids(os.getpid()):
        process = kernel32.OpenProcess(_PROCESS_TERMINATE, False, pid)
        if not process:
            continue
        kernel32.TerminateProcess(process, 1)
        kernel32.CloseHandle(process)


def _descendant_pids(root_pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for pid, parent_pid in _process_parents():
        children.setdefault(parent_pid, []).append(pid)
    descendants: list[int] = []
    pending = list(children.get(root_pid, []))
    while pending:
        pid = pending.pop()
        descendants.append(pid)
        pending.extend(children.get(pid, []))
    return descendants


def _process_parents() -> list[tuple[int, int]]:
    kernel32 = _kernel32()
    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snapshot or int(snapshot) == ctypes.c_void_p(-1).value:
        return []
    entry = _ProcessEntry()
    entry.dwSize = ctypes.sizeof(entry)
    processes: list[tuple[int, int]] = []
    try:
        has_entry = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while has_entry:
            processes.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID)))
            has_entry = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return processes
