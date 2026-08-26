from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter


@dataclass
class FilterApplicationProfile:
    image_path: str
    operations: tuple[str, ...]
    started_at: float = field(default_factory=perf_counter)
    wall_ms: float = 0.0
    phases_ms: dict[str, float] = field(default_factory=dict)
    step_timings: list[tuple[int, str, float]] = field(default_factory=list)

    @classmethod
    def begin(
        cls,
        *,
        image_path: str,
        pipeline_config: dict[str, object],
        queued_at: float | None = None,
    ) -> FilterApplicationProfile:
        raw_steps = pipeline_config.get("steps", [])
        steps = raw_steps if isinstance(raw_steps, list) else []
        operations = tuple(
            str(step.get("operation", ""))
            for step in steps
            if isinstance(step, dict) and bool(step.get("enabled", True))
        )
        session = cls(image_path=str(image_path), operations=operations)
        if queued_at is not None:
            session.record_phase(
                "queue_wait",
                max(0.0, (session.started_at - float(queued_at)) * 1000.0),
            )
        return session

    def record_phase(self, name: str, elapsed_ms: float) -> None:
        self.phases_ms[str(name)] = float(elapsed_ms)

    def record_step(self, index: int, operation: str, elapsed_ms: float) -> None:
        self.step_timings.append((int(index), str(operation), float(elapsed_ms)))

    def finish(self) -> None:
        if self.wall_ms > 0.0:
            return
        self.wall_ms = (perf_counter() - self.started_at) * 1000.0

    def format_summary(self, *, status: str) -> str:
        operations = ",".join(self.operations) if self.operations else "none"
        phases = " ".join(
            f"{name}={elapsed_ms:.3f}ms"
            for name, elapsed_ms in self.phases_ms.items()
        )
        return (
            f"[contour filter application profiling] status={status} "
            f"wall={self.wall_ms:.3f}ms filters={len(self.operations)} "
            f"operations={operations!r} image={Path(self.image_path).name!r}"
            f"{(' ' + phases) if phases else ''}"
        )

    def format_stats(self) -> str:
        prefix = "[contour filter application profiling stats]"
        if not self.step_timings:
            return f"{prefix} no enabled filter steps"
        lines = [prefix]
        for index, operation, elapsed_ms in self.step_timings:
            lines.append(
                f"step={index + 1} operation={operation!r} wall={elapsed_ms:.3f}ms"
            )
        return "\n".join(lines)
