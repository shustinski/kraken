"""synthetic_topology: plan ic window slots."""
from __future__ import annotations
import random


class PlanIcWindowSlotsMixin:
    def _plan_ic_window_slots(
        self,
        width: int,
        count: int,
        rng: random.Random,
        *,
        style: str,
        margin: int,
    ) -> list[tuple[int, int]]:
        usable_left = int(max(2, margin))
        usable_right = int(max(usable_left + 12, width - margin - 1))
        usable_width = max(12, usable_right - usable_left)
        windows: list[tuple[int, int]] = []
        if style == 'cell_array':
            slot_count = max(2, min(int(count), max(2, usable_width // 18)))
            pitch = usable_width / float(slot_count)
            for slot_index in range(slot_count):
                center = usable_left + ((slot_index + 0.5) * pitch)
                center += rng.uniform(-pitch * 0.08, pitch * 0.08)
                slot_width = pitch * rng.uniform(0.52, 0.82)
                x1 = int(round(center - (slot_width / 2.0)))
                x2 = int(round(center + (slot_width / 2.0)))
                windows.append((max(usable_left, x1), min(usable_right, x2)))
        else:
            slot_count = max(2, min(int(count), 6))
            cursor = usable_left
            for _slot_index in range(slot_count * 2):
                if len(windows) >= slot_count or cursor >= usable_right - 12:
                    break
                cursor += rng.randint(4, max(5, usable_width // max(10, slot_count * 5)))
                slot_width = rng.randint(max(12, usable_width // 10), max(18, usable_width // max(4, slot_count)))
                x1 = cursor
                x2 = min(usable_right, x1 + slot_width)
                if x2 - x1 < 10:
                    break
                windows.append((x1, x2))
                cursor = x2
        sanitized: list[tuple[int, int]] = []
        last_end = usable_left - 1
        for x1, x2 in sorted(windows):
            clipped_x1 = max(usable_left, int(x1), last_end + 4)
            clipped_x2 = min(usable_right, int(x2))
            if clipped_x2 - clipped_x1 < 10:
                continue
            sanitized.append((clipped_x1, clipped_x2))
            last_end = clipped_x2
        return sanitized
