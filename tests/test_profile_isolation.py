"""Parallel audit/data profiles must not share remembered logins or GUI settings."""
import json
from pathlib import Path
import subprocess
import sys

from kraken_hub.infrastructure.credentials import windows as windows_credentials
from kraken_hub.presentation.qt.parts.development_session import _remember_credentials


def test_credential_targets_preserve_default_and_separate_profiles(tmp_path):
    legacy = windows_credentials.credential_target()
    assert windows_credentials.credential_target(Path.home() / ".kraken") == legacy
    a = windows_credentials.credential_target(tmp_path / "a")
    b = windows_credentials.credential_target(tmp_path / "b")
    assert a != b and a != legacy and b != legacy
    assert windows_credentials.credential_target(tmp_path / "a/../a") == a
    assert str(tmp_path) not in a


def test_remember_routes_to_the_selected_profile(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(windows_credentials, "save_credentials",
        lambda username, password, *, target: calls.append((username, target)))
    _remember_credentials("audit-a", "dummy", data_directory=tmp_path / "a")
    _remember_credentials("audit-b", "dummy", data_directory=tmp_path / "b")
    assert calls == [("audit-a", windows_credentials.credential_target(tmp_path / "a")),
                     ("audit-b", windows_credentials.credential_target(tmp_path / "b"))]


def test_explicit_settings_isolation_survives_process_restart(tmp_path):
    script = """
import json,sys
from pathlib import Path
from kraken_core.settings import QSettings
from PyQt6.QtCore import QSettings as NativeQSettings
from kraken_hub.composition.app import configure_settings_directory
assert QSettings('Kraken','KrakenHub').fileName() == NativeQSettings('Kraken','KrakenHub').fileName()
root=configure_settings_directory(sys.argv[1])
settings=QSettings('Kraken','KrakenHub')
assert Path(settings.fileName()).resolve().is_relative_to(root)
if len(sys.argv)>2:
 settings.setValue('program/server_url',sys.argv[2]); settings.sync()
print(json.dumps({'value':settings.value('program/server_url',''),'filename':settings.fileName()}))
"""
    def run(profile, *value):
        result = subprocess.run([sys.executable, "-c", script, str(profile), *value],
            check=True, capture_output=True, text=True, timeout=20)
        return json.loads(result.stdout)
    assert run(tmp_path / "a", "http://127.0.0.1:1")["value"] == "http://127.0.0.1:1"
    assert run(tmp_path / "b")["value"] == ""
    assert run(tmp_path / "a")["value"] == "http://127.0.0.1:1"
