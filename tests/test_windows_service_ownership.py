from unittest.mock import Mock

import pytest

from kraken_admin.service_ownership import require_owned_service


def test_exact_executable_owns_service_and_quoted_suffix_cannot_spoof(tmp_path):
    executable = tmp_path / "Kraken Server" / "KrakenServer.exe"
    assert require_owned_service(executable, image_path=f'"{executable}" --service --config "config.toml"')
    assert not require_owned_service(executable, image_path=None)
    for command in (
        f'"{executable}".other --service',
        f'"{executable}.other" --service',
        f'"{tmp_path / "other" / "KrakenServer.exe"}" --service',
        f"{executable} --service",
        '"',
    ):
        with pytest.raises(RuntimeError, match="another executable"):
            require_owned_service(executable, image_path=command)


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows SCM entry points")
def test_install_and_uninstall_never_mutate_foreign_service(tmp_path, monkeypatch):
    import kraken_admin.service_ownership as ownership
    from kraken_admin.cli_parts.install_service import _install_service
    from kraken_admin.cli_parts.api_request import _uninstall_service

    executable = tmp_path / "KrakenServer.exe"
    executable.touch()
    config = tmp_path / "server.toml"
    config.touch()
    monkeypatch.setattr(ownership, "service_image_path", lambda: f'"{tmp_path / "foreign.exe"}" --service')
    run = Mock()
    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(RuntimeError, match="another executable"):
        _install_service(executable, config, start=True)
    with pytest.raises(RuntimeError, match="another executable"):
        _uninstall_service(executable)
    run.assert_not_called()
    monkeypatch.setattr(ownership, "service_image_path", lambda: None)
    _uninstall_service(executable)
    run.assert_not_called()


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows SCM entry points")
def test_uninstall_owned_service_uses_exact_scm_name(tmp_path, monkeypatch):
    import kraken_admin.service_ownership as ownership
    from kraken_admin.cli_parts.api_request import _uninstall_service

    executable = tmp_path / "KrakenServer.exe"
    monkeypatch.setattr(ownership, "service_image_path", lambda: f'"{executable}" --service')
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr("subprocess.run", run)
    _uninstall_service(executable)
    assert [call.args[0] for call in run.call_args_list] == [
        ["sc.exe", "stop", "KrakenServer"],
        ["sc.exe", "delete", "KrakenServer"],
    ]


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows SCM entry points")
def test_install_owned_service_rejects_changed_config_before_any_scm_mutation(tmp_path, monkeypatch):
    import kraken_admin.service_ownership as ownership
    from kraken_admin.cli_parts.install_service import _install_service
    executable, config = tmp_path / "server.exe", tmp_path / "new.toml"
    executable.touch()
    config.touch()
    monkeypatch.setattr(ownership, "service_image_path",
                        lambda: f'"{executable}" --service --config "old.toml"')
    run = Mock()
    monkeypatch.setattr("subprocess.run", run)
    with pytest.raises(SystemExit, match="different startup arguments or --config"):
        _install_service(executable, config, start=True)
    run.assert_not_called()


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows SCM entry points")
@pytest.mark.parametrize("matching", [True, False])
def test_existing_service_race_rechecks_full_command_before_recovery_changes(tmp_path, monkeypatch, matching):
    import kraken_admin.service_ownership as ownership
    from kraken_admin.cli_parts.install_service import _install_service
    executable, config = tmp_path / "server.exe", tmp_path / "new.toml"
    executable.touch()
    config.touch()
    desired = f'"{executable}" --service --config "{config}"'
    raced = desired if matching else f'"{executable}" --service --config "old.toml"'
    image = Mock(side_effect=[None, None, raced])
    monkeypatch.setattr(ownership, "service_image_path", image)
    run = Mock(return_value=Mock(returncode=1073, stdout="1073", stderr=""))
    monkeypatch.setattr("subprocess.run", run)
    if matching:
        _install_service(executable, config, start=False)
        assert [call.args[0][1] for call in run.call_args_list] == ["create", "description", "failure"]
    else:
        with pytest.raises(SystemExit, match="different startup arguments or --config"):
            _install_service(executable, config, start=False)
        assert [call.args[0][1] for call in run.call_args_list] == ["create"]
