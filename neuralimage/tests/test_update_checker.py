from lib.update_checker import (
    compare_versions,
    parse_update_payload,
    parse_version_parts,
    should_notify_version,
    validate_downloaded_installer,
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
            'download_url': 'http://localhost/setup.exe',
            'sha256': 'a' * 64,
            'mandatory': True,
        }
    )
    assert parsed is not None
    assert parsed.sha256 == 'a' * 64
    assert parsed.mandatory is True


def test_should_notify_version_respects_current_and_last_notified() -> None:
    assert should_notify_version('5.8.0', '5.7.0', '') is True
    assert should_notify_version('5.8.0', '5.8.0', '') is False
    assert should_notify_version('5.8.0', '5.7.0', '5.8.0') is False
    assert should_notify_version('5.9.0', '5.7.0', '5.8.0') is True


def test_validate_downloaded_installer_accepts_matching_sha256(tmp_path) -> None:
    installer = tmp_path / 'NeuralImage-5.8.0.exe'
    installer.write_bytes(b'installer payload')
    assert (
        validate_downloaded_installer(
            installer,
            '340f4f42e5d28005ff7f01cc10e28f4aeb1f1ea60abeb75bf1aa49eab74a181b',
        )
        is True
    )
