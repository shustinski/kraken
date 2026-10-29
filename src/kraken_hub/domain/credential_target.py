"""Stable Credential Manager names for one desktop data profile."""

from __future__ import annotations

import hashlib


LEGACY_CREDENTIAL_TARGET = "Kraken.ProjectManager.LocalAccount"


def credential_target_name(
    *,
    is_default_profile: bool,
    canonical_directory: str,
    casefold: bool,
) -> str:
    """Keep the historical target for the default profile; hash every other one."""
    if is_default_profile:
        return LEGACY_CREDENTIAL_TARGET
    canonical = canonical_directory.casefold() if casefold else canonical_directory
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{LEGACY_CREDENTIAL_TARGET}.Profile.{digest}"
