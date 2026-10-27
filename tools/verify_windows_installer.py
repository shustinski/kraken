"""Install/test/uninstall only a SHA-pinned, separately identified audit installer.

Server audit packages exercise current-user file installation, not elevated SCM
registration. The production service/reboot gate remains separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import winreg


def registration(app_id):
    key = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{app_id}_is1"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(hive, key, 0, winreg.KEY_READ | view) as opened:
                    return {
                        name: winreg.QueryValueEx(opened, name)[0] for name in ("InstallLocation", "UninstallString")
                    }
            except FileNotFoundError:
                pass
    return None


def service_hash():
    from kraken_admin.service_ownership import service_image_path

    value = service_image_path()
    return None if value is None else hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.build_config.read_text(encoding="utf-8-sig"))
    target, app_id = config["target"], config["audit_app_id"]
    if target not in ("Desktop", "Server") or not app_id.startswith(f"Kraken{target}Audit-"):
        raise ValueError("Only explicitly isolated audit installers may be exercised")
    installer = (Path(config["output"]) / f"Kraken{target}Setup-0.1.0.exe").resolve(strict=True)
    digest = hashlib.sha256(installer.read_bytes()).hexdigest()
    if digest != config["setup_sha256"]:
        raise ValueError("Compiled installer changed after its audit manifest was pinned")
    if registration(app_id) is not None:
        raise ValueError("Audit identity is already installed; preserve and investigate it")
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=False)
    install = report / "installed"
    if install.exists():
        raise ValueError("Audit installation directory must be fresh")
    before = service_hash()
    evidence = dict(
        status="running",
        target=target,
        variant=config["variant"],
        app_id=app_id,
        installer_sha256=digest,
        service_before_sha256=before,
    )
    started = time.monotonic()
    installed = False
    failure = None

    def run(command, log_name, timeout=180):
        with (report / log_name).open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        return result.returncode

    def require_own_registration():
        record = registration(app_id)
        if record is None or Path(record["InstallLocation"]).resolve() != install.resolve():
            raise RuntimeError("Installer registration does not match the owned audit directory")
        uninstall = install / "unins000.exe"
        if not uninstall.is_file() or str(uninstall).casefold() not in str(record["UninstallString"]).casefold():
            raise RuntimeError("Audit uninstall executable does not match its registration")
        return uninstall

    try:
        evidence["install_exit"] = run(
            [
                str(installer),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                "/CURRENTUSER",
                f"/DIR={install}",
                "/TASKS=",
                f"/GROUP={app_id}",
                f"/LOG={report / 'setup.log'}",
            ],
            "setup-output.log",
        )
        if evidence["install_exit"]:
            raise RuntimeError("Audit installer returned a failure; see setup log")
        require_own_registration()
        installed = True
        if service_hash() != before:
            raise RuntimeError("File-only audit installation changed an existing service")
        if target == "Desktop":
            executable = install / "KrakenDesktop.exe"
            evidence["acceptance_exit"] = run(
                [str(executable), "--self-test-dir", str(report / "desktop-acceptance")], "acceptance-output.log"
            )
            reports = list((report / "desktop-acceptance").glob("*/report.json"))
            if len(reports) != 1:
                raise RuntimeError("Expected one real installed Desktop acceptance report")
            acceptance = json.loads(reports[0].read_text(encoding="utf-8"))
            evidence["acceptance_checks"] = acceptance["checks"]
            if acceptance["status"] != "passed":
                raise RuntimeError("Installed Desktop self-test failed")
        else:
            verifier = Path(__file__).with_name("verify_windows_server_distribution.py")
            evidence["acceptance_exit"] = run(
                [
                    sys.executable,
                    str(verifier),
                    "--distribution",
                    str(install),
                    "--report",
                    str(report / "server-acceptance"),
                ],
                "acceptance-output.log",
                timeout=180,
            )
            acceptance = json.loads((report / "server-acceptance/report.json").read_text(encoding="utf-8"))
            evidence["acceptance_checks"] = acceptance["checks"]
            if acceptance["status"] != "passed":
                raise RuntimeError("Installed Server distribution acceptance failed")
        if evidence["acceptance_exit"]:
            raise RuntimeError("Installed application acceptance returned a failure")
        evidence["status"] = "passed"
    except BaseException as exc:
        failure = exc
        evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            if installed:
                uninstall = require_own_registration()
                evidence["uninstall_exit"] = run(
                    [
                        str(uninstall),
                        "/VERYSILENT",
                        "/SUPPRESSMSGBOXES",
                        "/NORESTART",
                        f"/LOG={report / 'uninstall.log'}",
                    ],
                    "uninstall-output.log",
                )
                evidence["registration_removed"] = registration(app_id) is None
                evidence["executable_removed"] = not (install / f"Kraken{target}.exe").exists()
                if (
                    evidence["uninstall_exit"]
                    or not evidence["registration_removed"]
                    or not evidence["executable_removed"]
                ):
                    raise RuntimeError("Audit uninstall did not remove its own registration/executable")
            evidence["service_after_sha256"] = service_hash()
            if evidence["service_after_sha256"] != before:
                raise RuntimeError("Audit uninstall changed a service belonging to another installation")
        except BaseException as exc:
            failure = failure or exc
            evidence.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
        evidence["seconds"] = round(time.monotonic() - started, 3)
        (report / "report.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if failure:
        raise failure
    print(
        json.dumps(
            {
                key: evidence[key]
                for key in ("status", "target", "seconds", "registration_removed", "executable_removed")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
