import json
from pathlib import Path

import lib.update_checker as update_checker
from lib.update_checker import (
    collect_release_history,
    compare_versions,
    download_update_installer,
    fetch_update_info,
    launch_update_installer,
    parse_update_payload,
    parse_version_parts,
    UpdateInfo,
    should_notify_version,
)


def test_parse_version_parts_ignores_suffixes() -> None:
    assert parse_version_parts('5.8.0-beta1') == (5, 8, 0, 1)


def test_compare_versions_normalizes_missing_parts() -> None:
    assert compare_versions('5.8', '5.8.0') == 0
    assert compare_versions('5.8.1', '5.8.0') == 1
    assert compare_versions('5.7.9', '5.8.0') == -1


def test_parse_update_payload_requires_version() -> None:
    assert parse_update_payload({}) is None
    parsed = parse_update_payload(
        {
            'version': '5.8.0',
            'download_url': 'http://localhost/setup-latest.exe',
            'mandatory': True,
            'releases': [
                {
                    'version': '5.8.0',
                    'download_url': 'http://localhost/setup-5.8.0.exe',
                    'notes': 'Latest release.',
                },
                {
                    'version': '5.7.0',
                    'download_url': 'http://localhost/setup-5.7.0.exe',
                    'notes': 'Previous release.',
                },
            ],
        }
    )
    assert parsed is not None
    assert parsed.download_url == 'http://localhost/setup-latest.exe'
    assert parsed.mandatory is True
    assert parsed.releases[0].download_url == 'http://localhost/setup-5.8.0.exe'
    assert parsed.releases[1].notes == 'Previous release.'


def test_should_notify_version_respects_current_and_last_notified() -> None:
    assert should_notify_version('5.8.0', '5.7.0', '') is True
    assert should_notify_version('5.8.0', '5.8.0', '') is False
    assert should_notify_version('5.8.0', '5.7.0', '5.8.0') is False
    assert should_notify_version('5.9.0', '5.7.0', '5.8.0') is True


def test_fetch_update_info_supports_local_manifest_path(tmp_path) -> None:
    manifest_path = tmp_path / 'version.json'
    manifest_path.write_text(
        json.dumps(
            {
                'version': '5.8.0',
                'download_url': str(tmp_path / 'NeuralImage-5.8.0.exe'),
                'release_notes': 'Local share update.',
                'releases': [
                    {
                        'version': '5.8.0',
                        'download_url': str(tmp_path / 'NeuralImage-5.8.0.exe'),
                        'notes': 'Local share update.',
                    },
                    {
                        'version': '5.7.0',
                        'download_url': str(tmp_path / 'NeuralImage-5.7.0.exe'),
                        'notes': 'Older release.',
                    },
                ],
            }
        ),
        encoding='utf-8',
    )
    update_info = fetch_update_info(str(manifest_path))
    assert update_info is not None
    assert update_info.version == '5.8.0'
    assert update_info.release_notes == 'Local share update.'
    assert len(update_info.releases) == 2


def test_fetch_update_info_returns_none_for_missing_manifest_path(tmp_path) -> None:
    missing_manifest = tmp_path / 'missing-version.json'
    assert fetch_update_info(str(missing_manifest)) is None


def test_download_update_installer_supports_local_file_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(update_checker.tempfile, 'gettempdir', lambda: str(tmp_path))
    source_installer = tmp_path / 'NeuralImage-5.8.0.exe'
    source_installer.write_bytes(b'installer payload')
    downloaded_path = download_update_installer(
        UpdateInfo(
            version='5.8.0',
            download_url=str(source_installer),
        )
    )
    assert downloaded_path.exists()
    assert downloaded_path.read_bytes() == b'installer payload'
    assert downloaded_path.name == 'NeuralImage-5.8.0.exe'


def test_launch_update_installer_creates_cleanup_launcher(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(update_checker.tempfile, 'gettempdir', lambda: str(tmp_path))
    installer_dir = tmp_path / 'NeuralImageUpdater'
    installer_dir.mkdir()
    installer_path = installer_dir / 'NeuralImage-5.8.0.exe'
    installer_path.write_bytes(b'installer payload')

    captured: dict[str, object] = {}

    def fake_popen(args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs

        class DummyProcess:
            pass

        return DummyProcess()

    monkeypatch.setattr(update_checker.subprocess, 'Popen', fake_popen)

    launch_update_installer(installer_path)

    command = captured['args']
    assert isinstance(command, list)
    assert command[:3] == ['cmd.exe', '/d', '/c']
    assert command[4] == str(installer_path)
    assert command[5] == str(installer_dir)

    launcher_path = Path(command[3])
    assert launcher_path.exists()
    launcher_body = launcher_path.read_text(encoding='ascii')
    assert '"%~1"' in launcher_body
    assert 'rmdir /s /q "%~2"' in launcher_body
    assert 'goto retry_cleanup' in launcher_body
    launcher_path.unlink(missing_ok=True)


def test_collect_release_history_returns_all_server_releases() -> None:
    update_info = parse_update_payload(
        {
            'version': '5.8.0',
            'releases': [
                {
                    'version': '5.8.0',
                    'download_url': '\\\\server\\share\\NeuralImage-5.8.0.exe',
                    'notes': 'Latest release.',
                },
                {
                    'version': '5.7.0',
                    'download_url': '\\\\server\\share\\NeuralImage-5.7.0.exe',
                    'notes': 'Rollback release.',
                },
            ],
        }
    )
    assert update_info is not None
    history = collect_release_history(update_info)
    assert '5.8.0' in history
    assert '5.7.0' in history
    assert 'Rollback release.' in history
