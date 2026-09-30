"""Local TEMP share smoke for Karakal autoupdate (no Inno install)."""
from __future__ import annotations

import os
from pathlib import Path

from updater.client import (
    ReleaseInfo,
    compare_versions,
    download_update_installer,
    fetch_update_info,
    verify_installer_sha256,
)
from karakal.updater import (
    follow_update_root_moves,
    load_karakal_update_root,
    validate_karakal_update_root,
)


def main() -> None:
    share = Path(os.environ["TEMP"]) / "karakal_share"
    share_new = Path(os.environ["TEMP"]) / "karakal_share_new"
    cyr = Path(os.environ["TEMP"]) / "каракал обновления"

    ok, msg = validate_karakal_update_root(cyr)
    assert ok, msg

    resolved = follow_update_root_moves(share, channel="beta", persist=True)
    assert Path(resolved).resolve() == share_new.resolve(), (resolved, share_new)
    assert Path(load_karakal_update_root()).resolve() == share_new.resolve()

    info = fetch_update_info(str(share_new / "beta"), expected_channel="beta")
    assert info is not None and info.version == "0.1.0-beta2", info
    assert info.sha256
    assert compare_versions("0.1.0-beta1", "0.1.0-beta2") < 0

    path = download_update_installer(
        ReleaseInfo(version=info.version, download_url=info.download_url, sha256=info.sha256),
        app_id="karakal-smoke",
    )
    assert path.is_file()

    bad = Path(os.environ["TEMP"]) / "bad_installer.exe"
    bad.write_bytes(b"nope")
    try:
        verify_installer_sha256(bad, info.sha256)
        raise SystemExit("expected mismatch")
    except ValueError as exc:
        assert "mismatch" in str(exc).lower()

    info2 = fetch_update_info(str(cyr / "beta"), expected_channel="beta")
    assert info2 is not None and info2.version == "0.1.0-beta2"

    print("SMOKE_OK")
    print(f"moved_to_resolved={resolved}")
    print(f"beta_version={info.version}")
    print(f"sha256={info.sha256[:16]}...")
    print(f"cyr_ok={ok}")


if __name__ == "__main__":
    main()
