"""File name suffixes that tell network masks from confidence maps of the same frame.

A frame ``NAME_01800`` has its mask in ``NAME_01800<mask suffix>.jpg`` and its confidence in
``NAME_01800<confidence suffix>.jpg``. Both may lie in one folder. By default masks have no
suffix and confidence maps end with ``_confidence``. Suffixes are matched without case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MASK_SUFFIX = ""
DEFAULT_CONFIDENCE_SUFFIX = "_confidence"


@dataclass(frozen=True, slots=True)
class FrameNaming:
    mask_suffix: str = DEFAULT_MASK_SUFFIX
    confidence_suffix: str = DEFAULT_CONFIDENCE_SUFFIX

    def normalized(self) -> "FrameNaming":
        return FrameNaming(
            mask_suffix=str(self.mask_suffix or "").strip(),
            confidence_suffix=str(self.confidence_suffix or "").strip(),
        )

    def is_confidence_file(self, path: Path | str) -> bool:
        return _strip_suffix(Path(path).stem, self.confidence_suffix) is not None

    def is_mask_file(self, path: Path | str) -> bool:
        """A mask is any image that is not a confidence map and carries the mask suffix."""

        if self.is_confidence_file(path):
            return False
        return not self.mask_suffix or _strip_suffix(Path(path).stem, self.mask_suffix) is not None

    def frame_stem(self, path: Path | str) -> str:
        """Frame name shared by the mask and the confidence map, without either suffix."""

        stem = Path(path).stem
        for suffix in (self.confidence_suffix, self.mask_suffix):
            stripped = _strip_suffix(stem, suffix)
            if stripped is not None:
                return stripped
        return stem


DEFAULT_FRAME_NAMING = FrameNaming()


def _strip_suffix(stem: str, suffix: str) -> str | None:
    suffix = str(suffix or "")
    if not suffix or len(stem) <= len(suffix) or not stem.lower().endswith(suffix.lower()):
        return None
    return stem[: -len(suffix)]


def validate_frame_naming(naming: FrameNaming) -> str:
    """Empty text when the suffixes can be used; otherwise what is wrong."""

    naming = naming.normalized()
    if naming.mask_suffix and naming.mask_suffix.lower() == naming.confidence_suffix.lower():
        return "same"
    for suffix in (naming.mask_suffix, naming.confidence_suffix):
        if any(char in suffix for char in '<>:"/\\|?*.'):
            return "characters"
    return ""


def confidence_lookup(index: dict[str, Path], naming: FrameNaming, *, same_folder_as_masks: bool) -> dict[str, Path]:
    """Confidence maps of one folder by frame stem.

    In the mask folder only files with the confidence suffix count. A separate confidence
    folder may also name its files like the masks.
    """

    suffixed: dict[str, Path] = {}
    plain: dict[str, Path] = {}
    for key, path in index.items():
        if naming.is_confidence_file(path):
            suffixed.setdefault(_frame_lookup_key(key, naming), path)
        elif not same_folder_as_masks:
            plain.setdefault(_frame_lookup_key(key, naming), path)
    return {**plain, **suffixed}


def frame_lookup_key(key: str, naming: FrameNaming) -> str:
    return _frame_lookup_key(key, naming)


def _frame_lookup_key(key: str, naming: FrameNaming) -> str:
    relative = Path(str(key))
    parent = relative.parent.as_posix()
    stem = naming.frame_stem(relative).lower()
    return stem if parent in {"", "."} else f"{parent.lower()}/{stem}"
