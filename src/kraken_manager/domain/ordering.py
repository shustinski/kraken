"""Deterministic ordering of human-readable names containing numbers."""
import re


def natural_name_key(name: str) -> tuple:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", name)
    )
