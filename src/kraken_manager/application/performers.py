"""Application helpers for the performer catalog."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from kraken_manager.application.errors import ConflictError
from kraken_manager.application.ports import PerformerStore
from kraken_manager.domain.identity import Performer, Principal, PrincipalProvider


# Stable, distinguishable colors.  A deterministic choice avoids assigning a
# different color when concurrent first-login requests race each other.
_GITLAB_PERFORMER_COLORS = (
    "#2E86AB",
    "#A23B72",
    "#3A7D44",
    "#C65D21",
    "#6C5CE7",
    "#B23A48",
    "#00796B",
    "#8D6E63",
)


def default_performer_color(principal: Principal) -> str:
    """Return a stable UI color for an authenticated principal."""

    digest = sha256(principal.external_key.encode("utf-8")).digest()
    return _GITLAB_PERFORMER_COLORS[int.from_bytes(digest[:2], "big") % len(_GITLAB_PERFORMER_COLORS)]


def ensure_gitlab_performer(principal: Principal, store: PerformerStore) -> Performer:
    """Create the linked performer on first GitLab login.

    Repeated calls are idempotent.  A changed GitLab display name is mirrored
    without replacing a user-selected color or reactivating an archived
    performer.  The unique principal constraint resolves concurrent first
    login requests safely.
    """

    if principal.provider is not PrincipalProvider.GITLAB:
        raise ValueError("automatic performers are created only for GitLab principals")

    existing = store.get_by_principal(principal.id)
    if existing is not None:
        if existing.name == principal.display_name:
            return existing
        return store.update(replace(existing, name=principal.display_name))

    performer = Performer.create(
        name=principal.display_name,
        color=default_performer_color(principal),
        principal_id=principal.id,
    )
    try:
        return store.create(performer)
    except ConflictError:
        # Another login request may have created the unique principal link
        # after our initial lookup.  Preserve genuine id/link conflicts.
        concurrent = store.get_by_principal(principal.id)
        if concurrent is None:
            raise
        return concurrent


__all__ = ["default_performer_color", "ensure_gitlab_performer"]
