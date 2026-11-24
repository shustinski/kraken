"""Opt-in Qt bridge. All widget access occurs on the GUI thread."""

from __future__ import annotations

import base64
import json
import math
from collections import deque

from PyQt6.QtCore import QBuffer, QEvent, QIODevice, QObject, QPoint, Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)

MAX_NODES = 2000
_ACTIVE_ENDPOINTS: set[str] = set()


def _text(widget: QWidget) -> str:
    if isinstance(widget, QLineEdit) and widget.echoMode() != QLineEdit.EchoMode.Normal:
        return "[redacted]"
    for method in ("text", "currentText", "toPlainText"):
        getter = getattr(widget, method, None)
        if callable(getter):
            try:
                return str(getter())[:300]
            except RuntimeError, TypeError:
                pass
    return ""


def _windows() -> list[QWidget]:
    app = QApplication.instance()
    if app is None:
        return []
    return [window for window in app.topLevelWidgets() if window.isVisible()]


def _walk(window: QWidget, window_index: int) -> tuple[list[dict], dict[str, tuple[QWidget | QAction, QWidget]]]:
    nodes: list[dict] = []
    targets: dict[str, tuple[QWidget | QAction, QWidget]] = {}

    def visit(widget: QWidget, path: str) -> None:
        if len(nodes) >= MAX_NODES:
            return
        key = f"w{window_index}:{path}"
        rect = widget.rect()
        global_pos = widget.mapToGlobal(rect.topLeft())
        nodes.append(
            {
                "id": key,
                "class": type(widget).__name__,
                "object_name": widget.objectName(),
                "text": _text(widget),
                "visible": widget.isVisible(),
                "enabled": widget.isEnabled(),
                "focused": widget.hasFocus(),
                "sensitive": isinstance(widget, QLineEdit) and widget.echoMode() != QLineEdit.EchoMode.Normal,
                "bounds": [global_pos.x(), global_pos.y(), rect.width(), rect.height()],
            }
        )
        targets[key] = (widget, widget)
        if isinstance(widget, QMainWindow):
            for index, action in enumerate(widget.menuBar().actions()):
                visit_action(action, f"{path}/menu{index}", widget.menuBar())
        if isinstance(widget, QMenu):
            for index, action in enumerate(widget.actions()):
                visit_action(action, f"{path}/action{index}", widget)
        for index, child in enumerate(widget.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)):
            if child is not widget and child.isVisible() and not child.isWindow():
                visit(child, f"{path}/{index}")

    def visit_action(action: QAction, path: str, host: QMenu | QMenuBar) -> None:
        if len(nodes) >= MAX_NODES or action.isSeparator():
            return
        key = f"w{window_index}:{path}"
        nodes.append(
            {
                "id": key,
                "class": "QAction",
                "object_name": action.objectName(),
                "text": action.text(),
                "visible": action.isVisible() and host.isVisible(),
                "enabled": action.isEnabled(),
                "focused": False,
                "bounds": None,
            }
        )
        targets[key] = (action, host)
        submenu = action.menu()
        if submenu is not None:
            for index, subaction in enumerate(submenu.actions()):
                visit_action(subaction, f"{path}/{index}", submenu)

    visit(window, "0")
    return nodes, targets


class _InteractionRecorder(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.events: deque[dict] = deque(maxlen=1000)
        self.skip_target: QObject | None = None
        self.skip_kind: QEvent.Type | None = None
        self.skip_count = 0

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if not isinstance(watched, QWidget):
            return False
        kind = event.type()
        if kind not in (QEvent.Type.MouseButtonRelease, QEvent.Type.KeyPress):
            return False
        if watched is self.skip_target and kind == self.skip_kind and self.skip_count:
            self.skip_count -= 1
            if self.skip_count == 0:
                self.skip_target = None
                self.skip_kind = None
            return False
        if kind == QEvent.Type.KeyPress:
            if isinstance(watched, QLineEdit) and watched.echoMode() != QLineEdit.EchoMode.Normal:
                return False
            value = event.text()
            if not value or not value.isprintable():
                return False
            action = "type"
        else:
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            value = ""
            action = "click"
        for index, window in enumerate(_windows()):
            nodes, targets = _walk(window, index)
            for node in nodes:
                target = targets[node["id"]][0]
                if target is watched:
                    self.events.append(
                        {
                            "id": node["id"],
                            "action": action,
                            "value": value,
                            "text": node["text"],
                            "class": node["class"],
                            "object_name": node["object_name"],
                            "x": event.position().x() / max(1, watched.width()) if action == "click" else None,
                            "y": event.position().y() / max(1, watched.height()) if action == "click" else None,
                        }
                    )
                    return False
        return False


class InspectorBridge:
    """Local authenticated request endpoint, constructed in QApplication's thread."""

    def __init__(self, socket_name: str, token: str) -> None:
        if not socket_name or not token:
            raise ValueError("socket name and token are required")
        if socket_name in _ACTIVE_ENDPOINTS:
            raise RuntimeError("Inspector endpoint already active")
        self._token = token
        self._server = QLocalServer()
        if not self._server.listen(socket_name):
            raise RuntimeError(f"Inspector endpoint unavailable: {self._server.errorString()}")
        _ACTIVE_ENDPOINTS.add(socket_name)
        self._socket_name = socket_name
        self._connections: set[QLocalSocket] = set()
        self._server.newConnection.connect(self._accept)
        self._recorder = _InteractionRecorder()
        QApplication.instance().installEventFilter(self._recorder)

    def close(self) -> None:
        QApplication.instance().removeEventFilter(self._recorder)
        for socket in tuple(self._connections):
            socket.disconnectFromServer()
            socket.deleteLater()
        self._connections.clear()
        self._server.close()
        QLocalServer.removeServer(self._socket_name)
        _ACTIVE_ENDPOINTS.discard(self._socket_name)

    def _accept(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            self._connections.add(socket)
            socket.readyRead.connect(lambda s=socket: self._read(s))
            socket.disconnected.connect(lambda s=socket: self._connections.discard(s))

    def _read(self, socket: QLocalSocket) -> None:
        if not socket.canReadLine():
            if socket.bytesAvailable() > 65536:
                socket.disconnectFromServer()
            return
        raw = bytes(socket.readLine())
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            result = self._dispatch(request)
        except (ValueError, TypeError) as exc:
            result = {"error": "invalid_request", "detail": str(exc)}
        except Exception as exc:
            result = {"error": "bridge_failure", "detail": f"{type(exc).__name__}: {exc}"}
        socket.write(json.dumps(result, ensure_ascii=False).encode() + b"\n")
        socket.flush()

    def _dispatch(self, request: dict) -> dict:
        if request.get("token") != self._token:
            return {"error": "unauthorized"}
        operation = request.get("op")
        windows = _windows()
        if operation == "inspect":
            return {
                "windows": [
                    {"id": f"w{index}", "title": window.windowTitle(), "nodes": _walk(window, index)[0]}
                    for index, window in enumerate(windows)
                ]
            }
        if operation == "events":
            events = list(self._recorder.events)
            self._recorder.events.clear()
            return {"events": events}
        if operation == "screenshot":
            index = int(request.get("window", 0))
            if index < 0 or index >= len(windows):
                return {"error": "window_not_found"}
            image = windows[index].grab().toImage()
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            if not image.save(buffer, "PNG"):
                return {"error": "screenshot_failed"}
            return {"png_base64": base64.b64encode(bytes(buffer.data())).decode("ascii")}
        if operation == "action":
            targets = {}
            for index, window in enumerate(windows):
                targets.update(_walk(window, index)[1])
            entry = targets.get(str(request.get("id", "")))
            if entry is None:
                return {"error": "element_not_found"}
            target, host = entry
            if not target.isEnabled() or not target.isVisible() or not host.isVisible():
                return {"error": "element_unavailable"}
            action = str(request.get("action", ""))
            value = str(request.get("value", ""))
            x = float(request.get("x", 0.5))
            y = float(request.get("y", 0.5))
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
                return {"error": "invalid_position"}
            if action not in {"click", "double_click", "right_click", "type", "key", "scroll"}:
                return {"error": "unsupported_action"}
            if isinstance(target, QAction) and action != "click":
                return {"error": "unsupported_target"}
            if action == "type" and not isinstance(target, (QLineEdit, QTextEdit, QPlainTextEdit, QComboBox)):
                return {"error": "unsupported_target"}
            if action == "type" and isinstance(target, QLineEdit) and target.echoMode() != QLineEdit.EchoMode.Normal:
                return {"error": "sensitive_field"}
            if action == "key" and value not in {"Enter", "Escape", "Tab", "Backspace"}:
                return {"error": "unsupported_key"}
            if action == "scroll" and (not isinstance(target, QAbstractItemView) or not value.lstrip("-").isdigit()):
                return {"error": "unsupported_target"}
            # Queue the real event so actions opening a nested modal loop do not
            # hold the response hostage. The next inspection observes its effect.
            QTimer.singleShot(0, lambda: self._perform(target, host, action, value, x, y))
            return {"ok": True, "scheduled": True}
        return {"error": "unknown_operation"}

    def _perform(self, target: QWidget | QAction, host: QWidget, action: str, value: str, x: float, y: float) -> None:
        receiver = host if isinstance(target, QAction) else target
        if isinstance(target, QAbstractItemView):
            receiver = target.viewport()
        self._recorder.skip_target = receiver
        self._recorder.skip_kind = QEvent.Type.KeyPress if action in {"type", "key"} else QEvent.Type.MouseButtonRelease
        self._recorder.skip_count = len(value) if action == "type" else 1
        try:
            self._send_event(target, host, action, value, x, y)
        finally:
            if self._recorder.skip_target is receiver:
                self._recorder.skip_target = None
                self._recorder.skip_kind = None
                self._recorder.skip_count = 0

    @staticmethod
    def _send_event(target: QWidget | QAction, host: QWidget, action: str, value: str, x: float, y: float) -> None:
        if isinstance(target, QAction):
            if action == "click" and isinstance(host, (QMenu, QMenuBar)) and host.isVisible():
                QTest.mouseClick(host, Qt.MouseButton.LeftButton, pos=host.actionGeometry(target).center())
            return
        if not target.isVisible() or not target.isEnabled():
            return
        if action == "type":
            target.setFocus()
            QTest.keyClicks(target, value)
        elif action == "key":
            target.setFocus()
            keys = {
                "Enter": Qt.Key.Key_Return,
                "Escape": Qt.Key.Key_Escape,
                "Tab": Qt.Key.Key_Tab,
                "Backspace": Qt.Key.Key_Backspace,
            }
            key = keys.get(value)
            if key is not None:
                QTest.keyClick(target, key)
        elif action == "scroll":
            if isinstance(target, QAbstractItemView):
                target.verticalScrollBar().setValue(target.verticalScrollBar().value() + int(value))
        else:
            button = Qt.MouseButton.RightButton if action == "right_click" else Qt.MouseButton.LeftButton
            receiver = target.viewport() if isinstance(target, QAbstractItemView) else target
            point = QPoint(
                min(receiver.width() - 1, int(receiver.width() * x)),
                min(receiver.height() - 1, int(receiver.height() * y)),
            )
            if action == "double_click":
                QTest.mouseDClick(receiver, button, pos=point)
            else:
                QTest.mouseClick(receiver, button, pos=point)
