"""services: in memory gamification payload."""
from __future__ import annotations
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol
from contour.gamification.config import GamificationBalance
from contour.gamification.models import (
    CorrectionEvent,
    CorrectionType,
    PetType,
    Rarity,
)

class GamificationPayloadStore(Protocol):
    def load_payload(self) -> dict[str, Any] | None: ...

    def save_payload(self, payload: dict[str, Any]) -> None: ...

class InMemoryGamificationPayloadStore:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload

    def load_payload(self) -> dict[str, Any] | None:
        return self.payload

    def save_payload(self, payload: dict[str, Any]) -> None:
        self.payload = payload

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()

def _new_transaction_id() -> str:
    return uuid.uuid4().hex

def _enum_value(value: Any) -> str:
    return getattr(value, "value", str(value))

def _parse_pet_type(value: Any, default: PetType | None = None) -> PetType | None:
    if isinstance(value, PetType):
        return value
    try:
        return PetType(str(value))
    except ValueError:
        return default

def _parse_rarity(value: Any, default: Rarity | None = None) -> Rarity | None:
    if isinstance(value, Rarity):
        return value
    try:
        return Rarity(str(value))
    except ValueError:
        return default

def _parse_correction_type(value: Any, default: CorrectionType = CorrectionType.UNKNOWN) -> CorrectionType:
    if isinstance(value, CorrectionType):
        return value
    try:
        return CorrectionType(str(value))
    except ValueError:
        return default

def _clamped_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, default)

def _profile_level_for_xp(xp: int, balance: GamificationBalance) -> int:
    xp_per_level = max(1, int(balance.xp_per_level))
    return max(1, int(xp) // xp_per_level + 1)

class CorrectionEventClassifier:
    def classify(self, correction_event: CorrectionEvent) -> CorrectionType:
        if correction_event.correction_type is not None:
            return _parse_correction_type(correction_event.correction_type)
        if correction_event.has_real_mask_changes:
            return CorrectionType.UNKNOWN
        return CorrectionType.UNKNOWN
