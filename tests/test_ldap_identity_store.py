"""Identity store coverage for LDAP logins and profile field updates."""
from __future__ import annotations

from kraken_manager.domain.identity import Principal

pytest_plugins = ["tests.test_admin_project_workflows"]


def _uuid(n: int) -> str:
    return f"{n:08d}-1111-2222-3333-444444444444"


def test_save_and_find_by_login_is_case_insensitive(database):
    services, _accounts, _owner, _config = database
    p = Principal.local(subject="Case.User", display_name="CU", principal_id=_uuid(1), login="Case.User")
    services.identities.save(p)
    found = services.identities.find_by_login("case.user")
    assert found is not None and str(found.id) == str(p.id)


def test_sync_avatar_respects_manual_source(database):
    services, _a, _o, _c = database
    pid = _uuid(2)
    services.identities.save(Principal.local(subject="av.user", display_name="AV", principal_id=pid))
    services.identities.set_manual_avatar(pid, "https://me/photo.png")
    services.identities.sync_avatar(pid, "https://ad/photo.png")
    assert services.identities.get(pid).avatar_url == "https://me/photo.png"


def test_sync_avatar_updates_when_not_manual(database):
    services, _a, _o, _c = database
    pid = _uuid(3)
    services.identities.save(Principal.local(subject="av2.user", display_name="AV2", principal_id=pid))
    services.identities.sync_avatar(pid, "https://ad/photo.png")
    assert services.identities.get(pid).avatar_url == "https://ad/photo.png"


def test_set_status(database):
    services, _a, _o, _c = database
    pid = _uuid(4)
    services.identities.save(Principal.local(subject="st.user", display_name="ST", principal_id=pid))
    services.identities.set_status(pid, "на встрече")
    assert services.identities.get(pid).status_text == "на встрече"


def test_save_conflict_does_not_clobber_avatar(database):
    services, _a, _o, _c = database
    pid = _uuid(5)
    first = Principal.local(subject="cw.user", display_name="CW", principal_id=pid, avatar_url="https://x/1.png", avatar_source="ldap")
    services.identities.save(first)
    services.identities.save(Principal.local(subject="cw.user", display_name="CW renamed", principal_id=pid))
    stored = services.identities.get(pid)
    assert stored.display_name == "CW renamed" and stored.avatar_url == "https://x/1.png"
