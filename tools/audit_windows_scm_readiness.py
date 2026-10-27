"""Read-only SCM preflight. Never registers, starts, stops or changes a service."""

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4


def scm_access():
    library = ctypes.WinDLL("advapi32", use_last_error=True)
    library.OpenSCManagerW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    library.OpenSCManagerW.restype = wintypes.HANDLE
    library.CloseServiceHandle.argtypes = [wintypes.HANDLE]
    library.CloseServiceHandle.restype = wintypes.BOOL
    result = {}
    for name, access in (("connect", 1), ("create_service_right", 3)):
        handle = library.OpenSCManagerW(None, None, access)
        result[name] = {"available": bool(handle), "winerror": 0 if handle else ctypes.get_last_error()}
        if handle:
            library.CloseServiceHandle(handle)
    return result


def query(name):
    process = subprocess.run(["sc.exe", "queryex", name], capture_output=True, text=True, timeout=10,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    return {"exit_code": process.returncode, "output": (process.stdout + process.stderr).strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--distribution", type=Path)
    arguments = parser.parse_args()
    if os.name != "nt":
        parser.error("This read-only preflight requires Windows")
    destination = arguments.report.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    audit_name = "KrakenServerAudit-" + uuid4().hex[:12]
    report = {"status": "preflight_only_not_acceptance", "mutations_executed": False,
              "production_service": query("KrakenServer"), "proposed_audit_name": audit_name,
              "audit_name_query": query(audit_name), "scm_access": scm_access(),
              "unique_name_runtime": "not yet tested; Admin commands always target KrakenServer",
              "proposed_steps": [
                  "Pin final stable binaries and copy to a fresh audit-only installation root",
                  "Provision only an owned PostgreSQL schema, absolute config/data roots and loopback ports",
                  "Record production service config/status and validate proposed audit name is absent",
                  "Register only the unique audit name as demand-start OWN_PROCESS after execution authorization",
                  "Require numeric SCM RUNNING plus authenticated HTTP and blob upload/download readiness",
                  "Stop only owned service, await STOPPED and verify its owned child exits",
                  "Replace only audit binaries while stopped, restart, and verify persisted authenticated data",
                  "Exercise failed startup and require nonzero SCM/process status without healthy RUNNING",
                  "Remove only owned service after verifying exact ImagePath and preserve production snapshot",
              ]}
    if arguments.distribution is not None:
        directory = arguments.distribution.resolve(strict=True)
        report["distribution"] = {}
        for name in ("KrakenServer.exe", "KrakenAdmin.exe", "KrakenBlobGateway.exe"):
            path = directory / name
            report["distribution"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                            "size_bytes": path.stat().st_size}
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "mutations_executed", "scm_access")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
