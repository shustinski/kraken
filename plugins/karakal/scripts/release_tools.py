"""Helpers for scripts/publish.ps1: version to publish, release notes, beta number in code.

    python release_tools.py update-root
    python release_tools.py next-version --channel beta --share \\\\server\\share\\karakal
    python release_tools.py check-notes --channel beta
    python release_tools.py release-notes --version 0.2.9103-beta0 --out notes.md
    python release_tools.py set-beta --version 0.2.9103-beta1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "src"))

from karakal.changelog import release_changelog, unreleased_notes  # noqa: E402
from karakal.release import next_publish_version, set_code_beta  # noqa: E402

CHANGELOG = PLUGIN_ROOT / "CHANGELOG.md"
VERSION_FILE = PLUGIN_ROOT / "src" / "karakal" / "version.py"
UPDATE_CLIENT = PLUGIN_ROOT / "resources" / "update_client.json"


def _update_root() -> str:
    try:
        payload = json.loads(UPDATE_CLIENT.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("update_root", "") or "").strip() if isinstance(payload, dict) else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("update-root")
    next_version = commands.add_parser("next-version")
    next_version.add_argument("--channel", required=True, choices=("beta", "stable"))
    next_version.add_argument("--share", required=True)
    check = commands.add_parser("check-notes")
    check.add_argument("--channel", required=True, choices=("beta", "stable"))
    notes = commands.add_parser("release-notes")
    notes.add_argument("--version", required=True)
    notes.add_argument("--out", required=True)
    notes.add_argument("--date", default=date.today().isoformat())
    beta = commands.add_parser("set-beta")
    beta.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "update-root":
            print(_update_root())
        elif args.command == "next-version":
            print(next_publish_version(args.channel, args.share))
        elif args.command == "check-notes":
            # A stable release may consist of betas only; a beta needs its own notes.
            if args.channel == "beta" and not unreleased_notes(CHANGELOG.read_text(encoding="utf-8")).strip():
                raise ValueError("Раздел «Не выпущено» в CHANGELOG.md пуст: опишите изменения перед публикацией")
        elif args.command == "release-notes":
            text, release_notes = release_changelog(CHANGELOG.read_text(encoding="utf-8"), args.version, args.date)
            Path(args.out).write_text(release_notes + "\n", encoding="utf-8")
            CHANGELOG.write_text(text, encoding="utf-8")
        elif args.command == "set-beta":
            set_code_beta(VERSION_FILE, args.version)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
