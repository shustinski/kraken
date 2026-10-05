"""CHANGELOG.md of Karakal: the unreleased section, version sections, release notes.

Layout of the file::

    # Каракал — история изменений

    ## Не выпущено

    - what changed since the last published build

    ## 0.2.9103-beta0 — 2026-10-05

    - ...

Changes are written into «Не выпущено» while working. Publishing a beta turns that
section into the version section; publishing a stable release also gathers every
beta published since the previous stable release.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

UNRELEASED_TITLE = "Не выпущено"
_HEADING_RE = re.compile(r"^## (?P<title>.+?)\s*$")
_VERSION_RE = re.compile(r"^(?P<version>\d+(?:\.\d+)*(?:-[0-9A-Za-z.]+)?)\b")


@dataclass(slots=True)
class ChangelogSection:
    title: str
    body: str

    @property
    def version(self) -> str:
        match = _VERSION_RE.match(self.title)
        return match.group("version") if match else ""

    @property
    def is_unreleased(self) -> bool:
        return self.title.strip().lower() == UNRELEASED_TITLE.lower()

    @property
    def is_prerelease(self) -> bool:
        return "-" in self.version


def split_changelog(text: str) -> tuple[str, list[ChangelogSection]]:
    """Return the text before the first ``## `` heading and the sections after it."""

    head: list[str] = []
    sections: list[ChangelogSection] = []
    title: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if title is not None:
                sections.append(ChangelogSection(title, "\n".join(body).strip()))
            title = match.group("title")
            body = []
        elif title is None:
            head.append(line)
        else:
            body.append(line)
    if title is not None:
        sections.append(ChangelogSection(title, "\n".join(body).strip()))
    return "\n".join(head).strip(), sections


def join_changelog(head: str, sections: list[ChangelogSection]) -> str:
    parts = [head.strip()] if head.strip() else []
    for section in sections:
        parts.append(f"## {section.title}\n\n{section.body}".rstrip() if section.body else f"## {section.title}")
    return "\n\n".join(parts) + "\n"


def unreleased_notes(text: str) -> str:
    _head, sections = split_changelog(text)
    return next((section.body for section in sections if section.is_unreleased), "")


def release_changelog(text: str, version: str, date: str) -> tuple[str, str]:
    """Move «Не выпущено» under ``version``; return (new changelog text, release notes).

    A stable version (no ``-suffix``) also takes the notes of every beta published
    after the previous stable release, so the release notes tell the whole story.
    The unreleased section stays in the file, empty, for the next changes.
    """

    head, sections = split_changelog(text)
    if any(section.version == version for section in sections):
        raise ValueError(f"В CHANGELOG.md уже есть раздел {version}")
    unreleased = next((section for section in sections if section.is_unreleased), None)
    notes = unreleased.body.strip() if unreleased is not None else ""
    is_stable = "-" not in version
    if is_stable:
        betas: list[str] = []
        for section in sections:
            if section.is_unreleased:
                continue
            if not section.is_prerelease:
                break
            if section.body.strip():
                betas.append(f"### {section.version}\n\n{section.body.strip()}")
        notes = "\n\n".join(([notes] if notes else []) + betas)
    if not notes:
        raise ValueError("Раздел «Не выпущено» в CHANGELOG.md пуст: опишите изменения перед публикацией")
    released = ChangelogSection(f"{version} — {date}", unreleased.body.strip() if unreleased is not None else "")
    if is_stable and not released.body:
        released.body = "Релиз проверенных бета-версий."
    rest = [section for section in sections if not section.is_unreleased]
    new_sections = [ChangelogSection(UNRELEASED_TITLE, ""), released] + rest
    return join_changelog(head, new_sections), notes


def notes_for_version(text: str, version: str) -> str:
    """Notes of one released version, empty when the file does not have it."""

    _head, sections = split_changelog(text)
    return next((section.body for section in sections if section.version == version), "")
