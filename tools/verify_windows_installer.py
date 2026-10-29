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
    from kraken_admin.infrastructure.windows_service import service_image_path

    value = service_image_path()
    return None if value is None else hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--desktop-reference-manifest", type=Path)
    parser.add_argument("--server-backup-pg-bin", type=Path)
    parser.add_argument("--server-backup-reference-manifest", type=Path)
    args = parser.parse_args()
    config = json.loads(args.build_config.read_text(encoding="utf-8-sig"))
    target, app_id = config["target"], config["audit_app_id"]
    if target not in ("Desktop", "Server") or not app_id.startswith(f"Kraken{target}Audit-"):
        raise ValueError("Only explicitly isolated audit installers may be exercised")
    if args.desktop_reference_manifest and target != "Desktop":
        raise ValueError("Reference deployment manifest applies only to Desktop")
    if bool(args.server_backup_pg_bin) != bool(args.server_backup_reference_manifest):
        raise ValueError("Server recovery acceptance requires both PostgreSQL tools and a real reference manifest")
    if args.server_backup_pg_bin and target != "Server":
        raise ValueError("Server recovery acceptance applies only to Server")
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
            probe_arguments = [] if args.desktop_reference_manifest is None else [
                "--self-test-reference-manifest", str(args.desktop_reference_manifest.resolve(strict=True))
            ]
            evidence["acceptance_exit"] = run(
                [str(executable), "--self-test-dir", str(report / "desktop-acceptance"), *probe_arguments], "acceptance-output.log"
            )
            reports = list((report / "desktop-acceptance").glob("*/report.json"))
            if len(reports) != 1:
                raise RuntimeError("Expected one real installed Desktop acceptance report")
            acceptance = json.loads(reports[0].read_text(encoding="utf-8"))
            evidence["acceptance_checks"] = acceptance["checks"]
            if acceptance["status"] != "passed":
                raise RuntimeError("Installed Desktop self-test failed")
            managed = acceptance.get("managed_matrix", {})
            if not managed.get("expected") or len(managed.get("previous_vector_version_ids", ())) < 2:
                raise RuntimeError("Installed Desktop did not prove nonempty matrix/head refresh/reopen")
            if not acceptance.get("source_files_unchanged"):
                raise RuntimeError("Installed Desktop did not verify source preservation")
            if not acceptance.get("settings_isolated"):
                raise RuntimeError("Installed Desktop did not verify GUI settings isolation")
            migration = acceptance.get("legacy_work_order_migration", {})
            if (migration.get("status") != "passed" or migration.get("imported") != 1
                    or migration.get("observations") != 2 or not migration.get("cache_replayed")
                    or not migration.get("backup_verified") or not migration.get("stale_backup_refused")
                    or not migration.get("latest_backup_restored")):
                raise RuntimeError("Installed Desktop did not pass legacy work migration acceptance")
            administration = acceptance.get("administration_operations", {})
            if (administration.get("status") != "passed" or not administration.get("gui_heartbeat")
                    or not administration.get("export_outside_gui") or not administration.get("busy_close_guarded")
                    or not administration.get("scan_passed")
                    or administration.get("restored_managed_pairs") != len(managed["expected"])):
                raise RuntimeError("Installed Desktop did not pass administration backup/scan/restore acceptance")
            evidence["managed_pairs"] = len(managed["expected"])
            evidence["fixture_kind"] = acceptance["fixture_kind"]
            evidence["reference_manifest_sha256"] = acceptance.get("reference_manifest_sha256")
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
            names = {check["name"] for check in acceptance.get("checks", ())}
            if not {"packaged_legacy_work_order_migration", "packaged_legacy_work_restart_persistence"} <= names:
                raise RuntimeError("Installed Server did not prove legacy work migration/restart persistence")
            if args.server_backup_pg_bin:
                evidence["backup_acceptance_exit"] = run([
                    sys.executable, str(Path(__file__).with_name("verify_windows_server_backup.py")),
                    "--distribution", str(install), "--pg-bin-dir", str(args.server_backup_pg_bin.resolve(strict=True)),
                    "--reference-manifest", str(args.server_backup_reference_manifest.resolve(strict=True)),
                    "--report", str(report / "server-backup-acceptance"),
                ], "backup-acceptance-output.log", timeout=300)
                backup = json.loads((report / "server-backup-acceptance/report.json").read_text(encoding="utf-8"))
                evidence["backup_acceptance"] = backup
                if (evidence["backup_acceptance_exit"] or backup["status"] != "passed"
                        or not backup.get("owned_databases_removed") or not backup.get("source_files_unchanged")):
                    raise RuntimeError("Installed Server backup/recovery acceptance failed")
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
