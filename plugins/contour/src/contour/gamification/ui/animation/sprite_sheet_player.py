from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


@dataclass(frozen=True, slots=True)
class SpriteSheetAnimation:
    """Configuration for a horizontal sprite sheet animation.

    To add a new sprite sheet, provide a loaded ``QPixmap``, set ``frame_count``,
    and optionally override ``frame_width`` / ``frame_height``. If frame width is
    omitted, it is computed as ``sheet.width() // frame_count``. Different
    animation states can reuse one sheet by setting ``start_frame`` and
    ``end_frame`` to a frame range.
    """

    pixmap: QPixmap
    frame_count: int
    fps: int
    loop: bool = True
    frame_width: int | None = None
    frame_height: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None

    @property
    def effective_frame_width(self) -> int:
        if self.frame_width is not None:
            return max(0, int(self.frame_width))
        if self.frame_count <= 0:
            return 0
        return self.pixmap.width() // self.frame_count

    @property
    def effective_frame_height(self) -> int:
        if self.frame_height is not None:
            return max(0, int(self.frame_height))
        return self.pixmap.height()

    @property
    def effective_start_frame(self) -> int:
        return max(0, int(self.start_frame or 0))

    @property
    def effective_end_frame(self) -> int:
        default_end = max(0, self.frame_count - 1)
        if self.end_frame is None:
            return default_end
        return min(default_end, max(0, int(self.end_frame)))

    @property
    def is_valid(self) -> bool:
        return (
            not self.pixmap.isNull()
            and self.frame_count > 0
            and self.effective_frame_width > 0
            and self.effective_frame_height > 0
            and self.effective_start_frame <= self.effective_end_frame
        )


class SpriteSheetPlayer(QWidget):
    """Lightweight QLabel-based sprite sheet player for PyQt6 widgets."""

    def __init__(self, parent: QWidget | None = None, *, scaled_size: int = 132) -> None:
        super().__init__(parent)
        self._animation: SpriteSheetAnimation | None = None
        self._frames: list[QPixmap] = []
        self._scaled_frames_cache: dict[tuple[int, int], list[QPixmap]] = {}
        self._current_frame_index = 0
        self._playing = False
        self._paused = False
        self._visibility_paused = False
        self._scaled_size = max(32, int(scaled_size))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.next_frame)
        self._image_label = QLabel("Animation missing")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(self._scaled_size, self._scaled_size)
        self._image_label.setScaledContents(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._image_label, 1)

    @property
    def current_frame_index(self) -> int:
        return self._current_frame_index

    @property
    def is_playing(self) -> bool:
        return self._playing and not self._paused

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def timer_active(self) -> bool:
        return self._timer.isActive()

    @property
    def current_frame(self) -> QPixmap | None:
        if not self._frames:
            return None
        return self._frames[self._current_frame_index]

    def sizeHint(self) -> QSize:
        return QSize(self._scaled_size, self._scaled_size)

    def set_animation(self, animation: SpriteSheetAnimation | None) -> None:
        self._timer.stop()
        self._animation = animation
        self._frames = self._extract_frames(animation)
        self._scaled_frames_cache.clear()
        self._playing = False
        self._paused = False
        self._visibility_paused = False
        self._current_frame_index = 0
        if not self._frames:
            self._image_label.setText("Animation missing")
            self._image_label.setPixmap(QPixmap())
            return
        self._image_label.setText("")
        self._show_current_frame()

    def play(self) -> None:
        if not self._frames or self._animation is None:
            self.set_animation(None)
            return
        self._playing = True
        self._paused = False
        self._visibility_paused = False
        self.set_frame(0)
        self._start_timer_if_visible()

    def stop(self) -> None:
        self._timer.stop()
        self._playing = False
        self._paused = False
        self._visibility_paused = False
        if self._frames:
            self.set_frame(0)

    def pause(self) -> None:
        if not self._playing:
            return
        self._paused = True
        self._timer.stop()

    def resume(self) -> None:
        if not self._frames:
            return
        self._playing = True
        self._paused = False
        self._start_timer_if_visible()

    def set_frame(self, index: int) -> None:
        if not self._frames:
            self._current_frame_index = 0
            return
        self._current_frame_index = max(0, min(int(index), len(self._frames) - 1))
        self._show_current_frame()

    def next_frame(self) -> None:
        if not self._frames or self._animation is None:
            return
        next_index = self._current_frame_index + 1
        if next_index >= len(self._frames):
            if self._animation.loop:
                next_index = 0
            else:
                self.stop()
                return
        self.set_frame(next_index)

    def hideEvent(self, event) -> None:
        self._visibility_paused = self._timer.isActive()
        self._timer.stop()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._playing and not self._paused:
            self._start_timer_if_visible()
        self._visibility_paused = False

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() != QEvent.Type.WindowStateChange:
            return
        window = self.window()
        if window is not None and window.isMinimized():
            self._visibility_paused = self._timer.isActive()
            self._timer.stop()
        elif self._playing and not self._paused:
            self._start_timer_if_visible()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scaled_frames_cache.clear()
        self._show_current_frame()

    def _start_timer_if_visible(self) -> None:
        if self._animation is None or len(self._frames) <= 1:
            self._timer.stop()
            return
        if not self.isVisible():
            self._timer.stop()
            return
        interval_ms = max(16, round(1000 / max(1, int(self._animation.fps))))
        self._timer.start(interval_ms)

    def _show_current_frame(self) -> None:
        frame = self.current_frame
        if frame is None or frame.isNull():
            return
        target = self._image_label.contentsRect().size()
        if target.width() <= 0 or target.height() <= 0:
            target = QSize(self._scaled_size, self._scaled_size)
        scaled_frames = self._scaled_frames_for(target)
        if scaled_frames:
            self._image_label.setPixmap(scaled_frames[self._current_frame_index])

    def _scaled_frames_for(self, target: QSize) -> list[QPixmap]:
        cache_key = (max(1, target.width()), max(1, target.height()))
        cached = self._scaled_frames_cache.get(cache_key)
        if cached is not None:
            return cached
        scaled = [
            frame.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            for frame in self._frames
        ]
        self._scaled_frames_cache[cache_key] = scaled
        return scaled

    @staticmethod
    def _extract_frames(animation: SpriteSheetAnimation | None) -> list[QPixmap]:
        if animation is None or not animation.is_valid:
            return []
        frame_width = animation.effective_frame_width
        frame_height = animation.effective_frame_height
        frames: list[QPixmap] = []
        for frame_index in range(animation.effective_start_frame, animation.effective_end_frame + 1):
            source_x = frame_index * frame_width
            if source_x + frame_width > animation.pixmap.width():
                break
            frames.append(animation.pixmap.copy(source_x, 0, frame_width, frame_height))
        return frames


__all__ = ["SpriteSheetAnimation", "SpriteSheetPlayer"]
